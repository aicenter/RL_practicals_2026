"""Cells, actions, the state of the world, and recorded episodes.

Nothing in this module knows how the world *changes* -- that is
:mod:`pragman.dynamics`. Here we only define what a situation *is*.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import LayoutError

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .config import PragmanConfig
    from .layout import Layout

#: A grid position, ``(row, col)``, with row 0 at the top and col 0 on the left.
Cell = tuple[int, int]


class Action(IntEnum):
    """The four moves. ``Discrete(4)``, in this order.

    There is no ``STAY``: walking into a wall already expresses "do nothing",
    and four actions keep hand-drawn dynamic-programming tables down to four
    arrows per cell.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> tuple[int, int]:
        """The ``(drow, dcol)`` this action adds to a cell."""
        return _DELTAS[self]

    @property
    def opposite(self) -> Action:
        """The reverse move (``UP.opposite is DOWN``)."""
        return Action((self + 2) % 4)

    @property
    def perpendicular(self) -> tuple[Action, Action]:
        """The two moves at right angles to this one, in ``Action`` order.

        These are the two directions an agent can slip into; see
        :func:`pragman.dynamics.slip_distribution`.
        """
        return _PERPENDICULAR[self]


_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

_PERPENDICULAR: dict[Action, tuple[Action, Action]] = {
    Action.UP: (Action.RIGHT, Action.LEFT),
    Action.DOWN: (Action.RIGHT, Action.LEFT),
    Action.RIGHT: (Action.UP, Action.DOWN),
    Action.LEFT: (Action.UP, Action.DOWN),
}

#: All four actions, in ``Action`` order. Handy for iteration.
ACTIONS: tuple[Action, ...] = tuple(Action)


def as_action(value: int | Action) -> Action:
    """Coerce an integer (or numpy integer) coming from a policy into an :class:`Action`.

    Raises:
        ValueError: if the value is not one of 0, 1, 2, 3.
    """
    try:
        return Action(int(value))
    except ValueError as exc:  # pragma: no cover - trivial re-wording
        legal = ", ".join(f"{a.value} ({a.name})" for a in ACTIONS)
        raise ValueError(f"{value!r} is not a legal action; legal actions are {legal}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# The state
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PragmanState:
    """One complete situation: where everybody is and which pellets are left.

    Frozen and hashable, so a state can be a dictionary key directly -- in the
    tabular model, in Dyna-Q's model table, and in an MCTS tree -- without any
    wrapper class. ``Q[info["state"]]`` in a lab script does what you expect.

    Note what is *not* here. There is no step counter: the episode clock belongs
    to the experiment, not to the Markov decision process, so the environment
    owns it (``env.elapsed_steps``). And ghosts carry no direction, which keeps a
    "random" ghost Markov in position alone -- you can write its transition
    matrix by hand in Week 4.

    Attributes:
        agent: the agent's cell.
        ghosts: one cell per ghost. **Index-aligned and never sorted**:
            ``state.ghosts[i]`` is always the ghost governed by
            ``cfg.ghost_policies[i]``, which spawned at the *i*-th ``G`` cell of
            the layout in row-major order. Ghosts may differ in personality, so
            they are not interchangeable and must not be canonicalized by sorting.
        pellets: the pellets that are still on the board.
        terminated: whether this state is terminal (death, goal, or last pellet).
            Terminal states are absorbing with zero reward.
    """

    agent: Cell
    ghosts: tuple[Cell, ...] = ()
    pellets: frozenset[Cell] = frozenset()
    terminated: bool = False

    def __post_init__(self) -> None:
        # Accept hand-written fixtures such as PragmanState((1, 1), [(2, 2)], {(3, 3)})
        # and store them in the canonical hashable types.
        if not isinstance(self.agent, tuple):
            object.__setattr__(self, "agent", tuple(self.agent))
        if not isinstance(self.ghosts, tuple):
            object.__setattr__(self, "ghosts", tuple(tuple(g) for g in self.ghosts))
        if not isinstance(self.pellets, frozenset):
            object.__setattr__(self, "pellets", frozenset(self.pellets))

    @property
    def pellets_left(self) -> int:
        """How many pellets remain on the board."""
        return len(self.pellets)

    # -- ASCII positions ------------------------------------------------------

    def ascii(self, layout: Layout) -> str:
        """Render this state as the "position" form of the ASCII notation.

        A *map* (:meth:`pragman.layout.Layout.from_ascii`) says what is static;
        a *position* says where everything is right now. The layout is needed
        here because the state itself does not know where the walls and goal
        cells are.

        Overlaps are drawn by precedence ``P > G > o > X > .``, which makes this
        form **lossy in three ways** -- it is for display, tests, and
        hand-written fixtures, while :class:`PragmanState` stays canonical:

        1. A pellet underneath the agent or a ghost cannot be shown, so
           :meth:`from_ascii` will read that cell as pellet-free.
        2. Ghost *identity* is not shown. :meth:`from_ascii` recovers ghosts in
           row-major order, so a state whose ghosts are not in row-major order
           (or whose ghosts share a cell) will not round-trip exactly.
        3. ``terminated`` is not shown and reads back as ``False``.

        Note this is a *method* taking a layout, so write ``state.ascii(layout)``.
        """
        grid = [list(row) for row in layout.ascii.splitlines()]
        # Repaint every non-wall cell from scratch, lowest precedence first.
        for (row, col) in layout.floor_cells:
            grid[row][col] = GOAL_CHAR if (row, col) in layout.goal_cells else FLOOR_CHAR
        for (row, col) in sorted(self.pellets):
            grid[row][col] = PELLET_CHAR
        for (row, col) in self.ghosts:
            grid[row][col] = GHOST_CHAR
        grid[self.agent[0]][self.agent[1]] = AGENT_CHAR
        return "\n".join("".join(row) for row in grid)

    #: ``state.to_ascii(layout)`` reads better in some code; same function.
    to_ascii = ascii

    @classmethod
    def from_ascii(
        cls,
        text: str | Sequence[str],
        layout: Layout,
        n_ghosts: int | None = None,
        terminated: bool = False,
    ) -> PragmanState:
        """Parse the "position" form of the ASCII notation against a layout.

        Unlike a map, a position describes one concrete situation, so exactly
        one ``P`` is required here (and exactly ``n_ghosts`` ``G``s, when you
        say how many ghosts you expect). Ghosts are returned in row-major order
        because the drawing cannot say which ghost is which; see :meth:`ascii`
        for the three ways this notation loses information.

        Args:
            text: the drawing, as a string or a list of rows.
            layout: the map it is drawn on. Its walls must match the drawing.
            n_ghosts: if given, the number of ``G`` cells required.
            terminated: value for the state's ``terminated`` flag, which the
                drawing cannot express.

        Raises:
            LayoutError: naming the offending row and column.
        """
        rows = split_ascii(text)
        if len(rows) != layout.height or any(len(r) != layout.width for r in rows):
            raise LayoutError(
                f"state ascii: the drawing is {len(rows)}x"
                f"{max((len(r) for r in rows), default=0)} but the layout is "
                f"{layout.height}x{layout.width}"
            )

        agents: list[Cell] = []
        ghosts: list[Cell] = []
        pellets: list[Cell] = []
        for row, line in enumerate(rows):
            for col, char in enumerate(line):
                cell = (row, col)
                is_wall = layout.is_wall(cell)
                if char == WALL_CHAR:
                    if not is_wall:
                        raise LayoutError(
                            f"state ascii: wall '#' at row {row}, column {col}, "
                            "but the layout has floor there"
                        )
                    continue
                if char == POWER_PELLET_CHAR:
                    raise LayoutError(
                        f"state ascii: 'O' (power pellet) at row {row}, column {col} "
                        "is reserved for a future version and is not supported"
                    )
                if char not in LEGAL_CHARS:
                    raise LayoutError(
                        f"state ascii: unknown character {char!r} at row {row}, column {col}; "
                        f"legal characters are {_legal_chars_help()}"
                    )
                if is_wall:
                    raise LayoutError(
                        f"state ascii: {char!r} at row {row}, column {col}, "
                        "but the layout has a wall there"
                    )
                if char == AGENT_CHAR:
                    agents.append(cell)
                elif char == GHOST_CHAR:
                    ghosts.append(cell)
                elif char == PELLET_CHAR:
                    pellets.append(cell)
                elif char == GOAL_CHAR and cell not in layout.goal_cells:
                    raise LayoutError(
                        f"state ascii: goal 'X' at row {row}, column {col}, "
                        "but the layout has no goal cell there"
                    )

        if len(agents) != 1:
            raise LayoutError(
                f"state ascii: a position must contain exactly one 'P', found {len(agents)} "
                "(a *map* may have several, meaning a uniform start distribution, but a "
                "position says where the agent is now)"
            )
        if n_ghosts is not None and len(ghosts) != n_ghosts:
            raise LayoutError(
                f"state ascii: expected exactly {n_ghosts} 'G' cells, found {len(ghosts)}"
            )
        return cls(agents[0], tuple(ghosts), frozenset(pellets), terminated)


# The five characters a position can contain, plus the two the parser rejects
# loudly. Kept here (rather than imported from layout.py) so that state.py has
# no import-time dependency on layout.py.
WALL_CHAR = "#"
FLOOR_CHAR = "."
PELLET_CHAR = "o"
AGENT_CHAR = "P"
GHOST_CHAR = "G"
GOAL_CHAR = "X"
POWER_PELLET_CHAR = "O"
LEGAL_CHARS = frozenset({WALL_CHAR, FLOOR_CHAR, PELLET_CHAR, AGENT_CHAR, GHOST_CHAR, GOAL_CHAR})


def _legal_chars_help() -> str:
    return "'#' wall, '.' floor, 'o' pellet, 'P' agent, 'G' ghost, 'X' goal"


def split_ascii(text: str | Sequence[str]) -> list[str]:
    """Split a drawing into rows, dropping blank leading/trailing lines and indentation."""
    if not isinstance(text, str):
        text = "\n".join(text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    margin = min(indents, default=0)
    return [line[margin:] for line in lines]


# ──────────────────────────────────────────────────────────────────────────────
# Episodes
# ──────────────────────────────────────────────────────────────────────────────

#: Bumped if the on-disk JSON format ever changes incompatibly.
EPISODE_FORMAT_VERSION = 1


@dataclass
class Episode:
    """One recorded trajectory, together with the world it was recorded in.

    Human demonstrations (Week 12), preference-labelling clips (Week 11), and
    offline datasets (Week 12) all use this one container, so that three labs do
    not invent three incompatible tuple formats.

    The config is stored, not merely the layout, because a trajectory without
    its world is not replayable: behavioural cloning needs the observation
    encoding to turn states into network inputs, inverse RL needs the dynamics
    to compute feature expectations, offline RL needs the reward parameters to
    know what the logged returns mean, and re-rendering a clip needs the layout.

    Attributes:
        config: the world this was recorded in.
        states: length ``T + 1``; ``states[t]`` is the state *before* ``actions[t]``.
        actions: length ``T``.
        rewards: length ``T``; ``rewards[t]`` is the reward for ``actions[t]``.
        terminated: whether the episode ended in a terminal state.
        truncated: whether it ended because of the step limit.
    """

    config: PragmanConfig
    states: list[PragmanState]
    actions: list[int]
    rewards: list[float]
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(
                f"an episode needs one more state than actions, got {len(self.states)} states "
                f"and {len(self.actions)} actions"
            )
        if len(self.rewards) != len(self.actions):
            raise ValueError(
                f"an episode needs one reward per action, got {len(self.rewards)} rewards "
                f"and {len(self.actions)} actions"
            )
        if self.terminated and self.truncated:
            raise ValueError("an episode cannot be both terminated and truncated")

    def __len__(self) -> int:
        """The number of steps, ``T``."""
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        """Undiscounted return of the episode."""
        return float(sum(self.rewards))

    def observations(self) -> list:
        """The observation the agent saw at each state, ``len(self) + 1`` of them."""
        # Imported lazily: observations.py imports config.py, which imports this
        # module, so a top-level import here would be circular.
        from .observations import observe

        return [observe(state, self.config) for state in self.states]

    def frames(self, **frame_kwargs: Any) -> list:
        """Re-render the episode as RGB frames, one per state.

        Rendering is pure -- :func:`pragman.render.frame` takes a state and a
        layout, never a live environment -- which is what makes a recorded clip
        re-renderable weeks after it was collected.
        """
        from .render import frame, ghost_names_from_config  # lazy, as observations() is

        frame_kwargs.setdefault("cell_pixels", self.config.cell_pixels)
        frame_kwargs.setdefault("palette", self.config.palette)
        # Without this, a re-rendered clip from a mixed-personality config would
        # show identical ghosts -- exactly the Week 11 preference-labelling case.
        frame_kwargs.setdefault("ghost_names", ghost_names_from_config(self.config))
        return [frame(state, self.config.layout, **frame_kwargs) for state in self.states]

    # -- serialization --------------------------------------------------------

    def to_json_dict(self, config: bool = True) -> dict[str, Any]:
        """This episode as plain JSON-able data (used by :meth:`save`)."""
        data: dict[str, Any] = {
            "states": [state_to_json(s) for s in self.states],
            "actions": [int(a) for a in self.actions],
            "rewards": [float(r) for r in self.rewards],
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }
        if config:
            data["config"] = self.config.to_json_dict()
        return data

    @classmethod
    def from_json_dict(
        cls, data: dict[str, Any], config: PragmanConfig | None = None
    ) -> Episode:
        """Inverse of :meth:`to_json_dict`; ``config`` overrides an embedded one."""
        from .config import PragmanConfig  # lazy: see observations()

        if config is None:
            config = PragmanConfig.from_json_dict(data["config"])
        return cls(
            config=config,
            states=[state_from_json(s) for s in data["states"]],
            actions=[int(a) for a in data["actions"]],
            rewards=[float(r) for r in data["rewards"]],
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
        )

    def save(self, path: str | Path) -> None:
        """Write the episode to a JSON file (no new dependencies, human-readable)."""
        payload = {
            "format": "pragman-episode",
            "version": EPISODE_FORMAT_VERSION,
            **self.to_json_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Episode:
        """Read back an episode written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        _check_format(payload, "pragman-episode")
        return cls.from_json_dict(payload)


def save_episodes(episodes: Iterable[Episode], path: str | Path) -> None:
    """Write many episodes to one JSON file, storing each distinct config once.

    A dataset of thousands of episodes usually shares a handful of worlds, so
    the configs are de-duplicated and each episode just stores an index into the
    config table. Training across several layouts therefore stays a single file.
    """
    configs: list[Any] = []
    seen: dict[Any, int] = {}
    records = []
    for episode in episodes:
        key = episode.config
        if key not in seen:
            seen[key] = len(configs)
            configs.append(episode.config.to_json_dict())
        record = episode.to_json_dict(config=False)
        record["config"] = seen[key]
        records.append(record)
    payload = {
        "format": "pragman-episodes",
        "version": EPISODE_FORMAT_VERSION,
        "configs": configs,
        "episodes": records,
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_episodes(path: str | Path) -> list[Episode]:
    """Read back a file written by :func:`save_episodes`."""
    from .config import PragmanConfig  # lazy: see Episode.observations()

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_format(payload, "pragman-episodes")
    configs = [PragmanConfig.from_json_dict(c) for c in payload["configs"]]
    return [
        Episode.from_json_dict(record, config=configs[record["config"]])
        for record in payload["episodes"]
    ]


def state_to_json(state: PragmanState) -> dict[str, Any]:
    """A state as plain JSON-able data. Pellets are sorted, so files are stable."""
    return {
        "agent": list(state.agent),
        "ghosts": [list(g) for g in state.ghosts],
        "pellets": [list(p) for p in sorted(state.pellets)],
        "terminated": bool(state.terminated),
    }


def state_from_json(data: dict[str, Any]) -> PragmanState:
    """Inverse of :func:`state_to_json`."""
    return PragmanState(
        agent=(int(data["agent"][0]), int(data["agent"][1])),
        ghosts=tuple((int(r), int(c)) for r, c in data["ghosts"]),
        pellets=frozenset((int(r), int(c)) for r, c in data["pellets"]),
        terminated=bool(data["terminated"]),
    )


def _check_format(payload: dict[str, Any], expected: str) -> None:
    found = payload.get("format")
    if found != expected:
        # The two formats are easy to confuse, and a lab that is handed "a demo
        # file" should be told which loader it wants rather than guessing.
        swap = {
            "pragman-episode": ("pragman-episodes", "load_episodes(path)"),
            "pragman-episodes": ("pragman-episode", "Episode.load(path)"),
        }
        other, loader = swap.get(expected, (None, None))
        if found == other:
            raise ValueError(
                f"this is a {found!r} file, not a {expected!r} one -- read it with {loader}"
            )
        raise ValueError(f"expected a {expected!r} file, found format {found!r}")
    version = payload.get("version")
    if version != EPISODE_FORMAT_VERSION:
        raise ValueError(
            f"{expected} file has version {version}, this version of pragman writes "
            f"and reads version {EPISODE_FORMAT_VERSION}"
        )
