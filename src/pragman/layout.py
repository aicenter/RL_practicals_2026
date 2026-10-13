"""ASCII maps: parsing, validation, BFS distances, and a few ready-made layouts.

A *map* describes what is static -- walls, goal cells, where pellets sit, and
the supports of the initial distributions. Where everybody actually is at some
moment is a *position*, and that lives in :class:`pragman.state.PragmanState`.
Keeping the two apart is what lets a map carry several ``P`` cells (meaning "the
agent starts uniformly at random on one of these") while a position carries
exactly one.

| Char | Meaning                                                         |
|------|-----------------------------------------------------------------|
| ``#``| wall                                                            |
| ``.``| empty floor                                                     |
| ``o``| pellet                                                          |
| ``P``| agent start; one or more, uniform over them                     |
| ``G``| ghost spawn; the first ``n_ghosts`` in row-major order are used |
| ``X``| goal cell                                                       |
| ``O``| reserved for power pellets in a later version; rejected here    |
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .errors import LayoutError
from .state import (
    ACTIONS,
    AGENT_CHAR,
    GHOST_CHAR,
    GOAL_CHAR,
    LEGAL_CHARS,
    PELLET_CHAR,
    POWER_PELLET_CHAR,
    WALL_CHAR,
    Action,
    Cell,
    as_action,
    split_ascii,
)

__all__ = [
    "Layout",
    "NAMED_LAYOUTS",
    "EMPTY_5X5",
    "SLIPPERY_5X5",
    "MAZE_8X8",
    "HAUNTED_5X5",
    "PELLETS_8X8",
    "CLASSIC_11X11",
]


@dataclass(frozen=True)
class Layout:
    """A parsed ASCII map.

    Construct one from a drawing::

        layout = Layout.from_ascii('''
            #######
            #P.o..#
            #.###.#
            #..G.X#
            #######
        ''')

    Everything except ``ascii`` is derived at construction, so a ``Layout`` is
    a value: two layouts are equal exactly when their (normalized) drawings are.

    Attributes:
        ascii: the normalized drawing, one row per line, no trailing newline.
        height: number of rows.
        width: number of columns.
        walls: boolean array of shape ``(height, width)``, ``True`` for a wall.
        floor_cells: every non-wall cell, in row-major order.
        pellet_cells: the cells drawn as ``o``.
        agent_starts: the cells drawn as ``P``; the agent starts uniformly on one.
        ghost_starts: the cells drawn as ``G``, in row-major order. The first
            ``cfg.n_ghosts`` of them are used, deterministically.
        goal_cells: the cells drawn as ``X``.
    """

    ascii: str
    height: int = field(init=False, repr=False)
    width: int = field(init=False, repr=False)
    walls: np.ndarray = field(init=False, repr=False)
    floor_cells: tuple[Cell, ...] = field(init=False, repr=False)
    pellet_cells: tuple[Cell, ...] = field(init=False, repr=False)
    agent_starts: tuple[Cell, ...] = field(init=False, repr=False)
    ghost_starts: tuple[Cell, ...] = field(init=False, repr=False)
    goal_cells: tuple[Cell, ...] = field(init=False, repr=False)
    _distance_cache: dict[Cell, dict[Cell, int]] = field(
        init=False, repr=False, default_factory=dict
    )

    # -- construction ---------------------------------------------------------

    def __post_init__(self) -> None:
        rows = _parse_rows(self.ascii)
        set_ = object.__setattr__  # the dataclass is frozen; this is the sanctioned way
        set_(self, "ascii", "\n".join(rows))
        set_(self, "height", len(rows))
        set_(self, "width", len(rows[0]))
        set_(self, "walls", np.array([[c == WALL_CHAR for c in row] for row in rows], dtype=bool))
        for name, chars in (
            ("floor_cells", LEGAL_CHARS - {WALL_CHAR}),
            ("pellet_cells", {PELLET_CHAR}),
            ("agent_starts", {AGENT_CHAR}),
            ("ghost_starts", {GHOST_CHAR}),
            ("goal_cells", {GOAL_CHAR}),
        ):
            set_(self, name, _cells_with(rows, chars))
        _validate(self)

    @classmethod
    def from_ascii(cls, text: str | Sequence[str]) -> Layout:
        """Parse a drawing given as a string or as a list of rows.

        Leading and trailing blank lines are dropped and common indentation is
        removed, so a triple-quoted string indented inside a function still
        works. Everything else must line up: all rows the same width, only
        legal characters, at least one ``P``, and at least one other walkable
        cell to go to.

        Raises:
            LayoutError: naming the offending row and column.
        """
        return cls("\n".join(split_ascii(text)))

    def __repr__(self) -> str:  # the drawing itself is too big for a repr
        return (
            f"<Layout {self.height}x{self.width} starts={len(self.agent_starts)} "
            f"ghost_starts={len(self.ghost_starts)} pellets={len(self.pellet_cells)} "
            f"goals={len(self.goal_cells)}>"
        )

    # `walls` is a numpy array, so the dataclass-generated __eq__ would raise
    # "truth value of an array is ambiguous" and __hash__ would fail. Both are
    # therefore defined on the normalized drawing, which determines everything
    # else anyway. PragmanConfig's hashability depends on this.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return self.ascii == other.ascii

    def __hash__(self) -> int:
        return hash(self.ascii)

    # -- geometry -------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``."""
        return (self.height, self.width)

    @property
    def agent_start(self) -> Cell:
        """The first ``P`` cell in row-major order.

        Convenience for the common single-``P`` map. When a map has several
        ``P``s the initial distribution is uniform over :attr:`agent_starts`,
        and this property is only the first of them.
        """
        return self.agent_starts[0]

    def in_bounds(self, cell: Cell) -> bool:
        """Is this cell on the grid at all?"""
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def is_wall(self, cell: Cell) -> bool:
        """Is this cell a wall? Anything off the grid counts as a wall.

        That is why a border of ``#`` is conventional but not required.
        """
        if not self.in_bounds(cell):
            return True
        return bool(self.walls[cell[0], cell[1]])

    def move(self, cell: Cell, action: int | Action) -> Cell:
        """Where does *action* take you from *cell*?

        A move into a wall or off the grid leaves you where you were. This one
        rule covers both the agent (event 2 of the step) and the ghosts (event
        6), which is why they never need to special-case each other.
        """
        drow, dcol = as_action(action).delta
        target = (cell[0] + drow, cell[1] + dcol)
        return cell if self.is_wall(target) else target

    def legal_actions(self, cell: Cell) -> tuple[Action, ...]:
        """The actions from *cell* that actually change position.

        Used for the uniform part of a ghost's distribution. It can be empty in
        a walled-in pocket, and callers must cope with that.
        """
        return tuple(a for a in ACTIONS if self.move(cell, a) != cell)

    def neighbours(self, cell: Cell) -> tuple[Cell, ...]:
        """The cells reachable from *cell* in one move (excluding *cell* itself)."""
        return tuple(self.move(cell, a) for a in self.legal_actions(cell))

    # -- distances ------------------------------------------------------------

    def distances(self, from_cell: Cell) -> dict[Cell, int]:
        """Breadth-first shortest-path distances from *from_cell*, in steps.

        Only reachable cells appear in the result, so an unreachable cell is
        "not in the dictionary" rather than infinity -- write
        ``distances.get(cell, math.inf)`` if you need a number.

        Results are cached per source cell (the grid never changes), which is
        what makes :class:`pragman.ghosts.ChaserGhost` a dictionary lookup
        rather than a search. The returned dictionary is the cached one: treat
        it as read-only.
        """
        cached = self._distance_cache.get(from_cell)
        if cached is not None:
            return cached
        if self.is_wall(from_cell):
            raise LayoutError(f"cannot measure distances from wall cell {from_cell}")
        distances = {from_cell: 0}
        queue = deque([from_cell])
        while queue:
            cell = queue.popleft()
            for neighbour in self.neighbours(cell):
                if neighbour not in distances:
                    distances[neighbour] = distances[cell] + 1
                    queue.append(neighbour)
        self._distance_cache[from_cell] = distances
        return distances

    def unreachable_cells(self) -> frozenset[Cell]:
        """Floor cells the agent can never get to from any of its start cells.

        There is no connectivity *requirement* -- a sealed pocket is a legal and
        rather useful teaching example -- so this is offered instead, for
        factories and tests that do want to assert connectivity.
        """
        reachable: set[Cell] = set()
        for start in self.agent_starts:
            reachable.update(self.distances(start))
        return frozenset(cell for cell in self.floor_cells if cell not in reachable)


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_rows(text: str) -> list[str]:
    """Normalize a drawing into rows and reject anything malformed."""
    rows = split_ascii(text)
    if not rows:
        raise LayoutError("layout is empty: a map needs at least one row")
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise LayoutError(
                f"ragged layout: row {index} has width {len(row)} but row 0 has width {width}"
            )
    for row_index, row in enumerate(rows):
        for col_index, char in enumerate(row):
            if char == POWER_PELLET_CHAR:
                raise LayoutError(
                    f"'O' (power pellet) at row {row_index}, column {col_index} is reserved "
                    "for a future version of pragman and is not supported yet; "
                    "use 'o' for an ordinary pellet"
                )
            if char not in LEGAL_CHARS:
                raise LayoutError(
                    f"unknown character {char!r} at row {row_index}, column {col_index}; "
                    "legal characters are '#' wall, '.' floor, 'o' pellet, 'P' agent start, "
                    "'G' ghost spawn, 'X' goal"
                )
    return rows


def _cells_with(rows: list[str], chars: set[str]) -> tuple[Cell, ...]:
    """Every cell holding one of *chars*, in row-major order."""
    return tuple(
        (row_index, col_index)
        for row_index, row in enumerate(rows)
        for col_index, char in enumerate(row)
        if char in chars
    )


def _validate(layout: Layout) -> None:
    """The checks that need the parsed layout rather than the raw rows."""
    if not layout.agent_starts:
        raise LayoutError(
            "layout has no 'P': a map needs at least one agent start cell "
            "(several mean the agent starts uniformly at random on one of them)"
        )
    walkable_elsewhere = set(layout.floor_cells) - set(layout.agent_starts)
    if not walkable_elsewhere:
        raise LayoutError(
            "layout has no walkable cell besides the agent start cells, "
            "so there is nowhere to go"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Named layouts
#
# Plain module-level strings, so that you can copy one, change two characters,
# and pass it straight to PragmanConfig(layout=...).
# ──────────────────────────────────────────────────────────────────────────────

#: 5x5 open room, no walls, no ghosts: 25 states, small enough to do by hand.
EMPTY_5X5 = """\
P....
.....
.....
.....
....X"""

#: Two corridors joined in the middle. The walls are what makes slipping
#: interesting: a sideways slip in a corridor simply bumps you into a wall.
SLIPPERY_5X5 = """\
P....
.###.
.....
.###.
....X"""

#: 8x8 maze, about fifty walkable cells, no ghosts. Also the Dyna-Q world.
MAZE_8X8 = """\
P.......
.###.#..
........
.#..##..
.#......
...##...
.#......
......#X"""

#: 5x5 with a single ghost that starts well away from both agent and goal.
HAUNTED_5X5 = """\
P...X
.....
..#..
.....
..G.."""

#: 8x8 filled with pellets and one ghost: the state space explodes, which is
#: exactly the point -- this is where function approximation earns its keep.
PELLETS_8X8 = """\
########
#Pooooo#
#o##o#o#
#oooooo#
#o#oo#o#
#ooGooo#
#oooooo#
########"""

#: The workhorse: an 11x11 Pac-Man maze with a central ghost house holding four
#: ghosts (the first ``n_ghosts`` of them are used) and pellets everywhere else.
CLASSIC_11X11 = """\
###########
#ooooooooo#
#o#ooooo#o#
#o#o###o#o#
#oo#G.G#oo#
#oo#...#oo#
#oo#G.G#oo#
#o#o#.#o#o#
#o#ooooo#o#
#ooooPoooo#
###########"""

#: Every named layout, by the name of the factory that uses it.
NAMED_LAYOUTS: dict[str, str] = {
    "empty_5x5": EMPTY_5X5,
    "slippery_5x5": SLIPPERY_5X5,
    "maze_8x8": MAZE_8X8,
    "haunted_5x5": HAUNTED_5X5,
    "pellets_8x8": PELLETS_8X8,
    "classic_11x11": CLASSIC_11X11,
}
