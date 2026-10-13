"""The one object every lab script builds: :class:`PragmanConfig`.

Eighteen fields, grouped, all with defaults except the layout. This is the
object that appears in student boilerplate, so it is deliberately boring.
Change one field with the standard library::

    from dataclasses import replace
    harder = replace(cfg, slip_prob=0.2)

Note what is *not* here: the discount factor. Gamma belongs to the agent, not
to the environment, so it is an argument of your algorithm (your value
iteration, say), not a config field.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from .errors import ConfigError
from .ghosts import (
    MIX_ORDER,
    GhostPolicy,
    ghost_policy_by_name,
    policy_from_json,
    policy_to_json,
)
from .layout import Layout
from .state import Cell

__all__ = ["PragmanConfig"]

START_CHOICES = ("layout", "random")
PELLET_CHOICES = ("layout", "all_floor", "none")
OBSERVATION_CHOICES = ("index", "coords", "channels", "pixels")
#: Palettes, shared by the ``pixels`` encoding and every render mode.
PALETTE_CHOICES = ("classic", "high_contrast")


@dataclass(frozen=True)
class PragmanConfig:
    """Everything that defines one Prag-man world.

    Attributes:
        layout: the map. A ``str`` or a list of rows is accepted and parsed
            into a :class:`~pragman.layout.Layout` for you.
        start: ``"layout"`` starts uniformly on the layout's ``P`` cells;
            ``"random"`` starts uniformly on any floor cell without a ghost on
            it, which helps exploration and gives Week 8 a non-degenerate rho_0.
        n_ghosts: how many ghosts; they take the first ``n_ghosts`` ``G`` cells
            of the layout, in row-major order.
        ghost_policy: ``"random"``, ``"chaser"``, ``"proximity"``, ``"mix"``, a
            policy object, or one entry per ghost. Normalized into
            :attr:`ghost_policies`.
        ghost_epsilon: probability that a ghost ignores its personality and
            moves uniformly at random. The default is 0.1 rather than 0 so that
            a chaser can actually be shaken off: with eps = 0 a chaser on an
            open grid is inescapable once adjacent, which makes the optimal
            policy turn on parity arguments rather than on anything a student
            would recognize as strategy.
        pellets: ``"layout"`` uses the ``o`` cells, ``"all_floor"`` fills every
            floor cell except goal cells, ``"none"`` leaves the maze empty.
        slip_prob: probability that the action is replaced by one of the two
            perpendicular ones, split evenly. ``0.2`` reproduces the familiar
            0.8 / 0.1 / 0.1 gridworld.
        living_reward: paid on every step that does not end the episode.
        pellet_reward: paid for eating a pellet.
        clear_bonus: paid on top for eating the *last* pellet.
        goal_reward: paid for reaching a goal cell.
        death_reward: paid for being caught by a ghost (negative).
        max_steps: truncation limit, or ``None`` never to truncate. The
            environment owns this clock; it is not part of the state.
        observation: ``"index"``, ``"coords"``, ``"channels"``, or ``"pixels"``.
        view_radius: egocentric window radius for ``channels``/``pixels``;
            ``None`` shows the whole grid.
        cell_pixels: block size in the ``pixels`` encoding.
        allow_state_injection: gates ``reset(options={"state": ...})`` and
            ``restore_state()``. A signpost saying "you are about to cheat",
            not a sandbox.
        ghost_policies: derived, length ``n_ghosts``: the policy object
            governing each ghost. ``state.ghosts[i]`` is governed by
            ``ghost_policies[i]``, for the whole episode.
        ghost_policy_names: derived, the names of those policies. Config
            equality and hashing use these names, so that two configs built in
            different ways ("chaser" versus ``ChaserGhost()``) compare equal.
    """

    # --- world ---------------------------------------------------------------
    layout: Layout
    start: str = "layout"
    n_ghosts: int = 0
    ghost_policy: str | GhostPolicy | Sequence[str | GhostPolicy] = field(
        default="random", compare=False
    )
    ghost_epsilon: float = 0.1
    pellets: str = "layout"
    slip_prob: float = 0.0

    # --- rewards -------------------------------------------------------------
    living_reward: float = -0.01
    pellet_reward: float = 0.1
    clear_bonus: float = 1.0
    goal_reward: float = 1.0
    death_reward: float = -1.0

    # --- episode -------------------------------------------------------------
    max_steps: int | None = 100

    # --- observation ---------------------------------------------------------
    observation: str = "channels"
    view_radius: int | None = None
    cell_pixels: int = 8
    palette: str = "classic"

    # --- access --------------------------------------------------------------
    allow_state_injection: bool = True

    # --- derived (not arguments) ---------------------------------------------
    ghost_policies: tuple[GhostPolicy, ...] = field(init=False, repr=False, default=())
    ghost_policy_names: tuple[str, ...] = field(
        init=False, repr=False, compare=False, default=()
    )

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # the dataclass is frozen; this is the sanctioned way

        # -- the layout, which may arrive as a drawing ------------------------
        if isinstance(self.layout, Layout):
            pass
        elif isinstance(self.layout, str) or (
            isinstance(self.layout, (list, tuple))
            and all(isinstance(row, str) for row in self.layout)
        ):
            set_(self, "layout", Layout.from_ascii(self.layout))
        else:
            raise ConfigError(
                "config field 'layout' must be a Layout, a string of ASCII art, or a list "
                f"of row strings, not {type(self.layout).__name__}"
            )

        # -- the world --------------------------------------------------------
        _check_choice("start", self.start, START_CHOICES)
        _check_choice("pellets", self.pellets, PELLET_CHOICES)
        _check_probability("slip_prob", self.slip_prob)
        _check_probability("ghost_epsilon", self.ghost_epsilon)

        if not isinstance(self.n_ghosts, int) or isinstance(self.n_ghosts, bool):
            raise ConfigError(f"config field 'n_ghosts' must be an int, got {self.n_ghosts!r}")
        if self.n_ghosts < 0:
            raise ConfigError(f"config field 'n_ghosts' must be >= 0, got {self.n_ghosts}")
        available = len(self.layout.ghost_starts)
        if self.n_ghosts > available:
            raise ConfigError(
                f"config field 'n_ghosts' is {self.n_ghosts} but the layout has only "
                f"{available} 'G' cell(s); add more 'G's to the map or lower n_ghosts"
            )

        # -- rewards ----------------------------------------------------------
        for name in ("living_reward", "pellet_reward", "clear_bonus", "goal_reward",
                     "death_reward"):
            _check_finite(name, getattr(self, name))

        # -- episode ----------------------------------------------------------
        if self.max_steps is not None:
            if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
                raise ConfigError(
                    f"config field 'max_steps' must be a positive int or None, "
                    f"got {self.max_steps!r}"
                )
            if self.max_steps < 1:
                raise ConfigError(
                    f"config field 'max_steps' must be >= 1 or None, got {self.max_steps}"
                )

        # -- observation ------------------------------------------------------
        _check_choice("observation", self.observation, OBSERVATION_CHOICES)
        if self.view_radius is not None:
            if self.observation in ("index", "coords"):
                raise ConfigError(
                    f"config field 'view_radius' is meaningless with observation="
                    f"{self.observation!r}: an egocentric window only applies to the "
                    "'channels' and 'pixels' encodings; set view_radius=None"
                )
            if not isinstance(self.view_radius, int) or isinstance(self.view_radius, bool):
                raise ConfigError(
                    f"config field 'view_radius' must be a non-negative int or None, "
                    f"got {self.view_radius!r}"
                )
            if self.view_radius < 0:
                raise ConfigError(
                    f"config field 'view_radius' must be >= 0 or None, got {self.view_radius}"
                )
        if not isinstance(self.cell_pixels, int) or isinstance(self.cell_pixels, bool):
            raise ConfigError(
                f"config field 'cell_pixels' must be a positive int, got {self.cell_pixels!r}"
            )
        if self.cell_pixels < 1:
            raise ConfigError(f"config field 'cell_pixels' must be >= 1, got {self.cell_pixels}")
        _check_choice("palette", self.palette, PALETTE_CHOICES)

        # -- access -----------------------------------------------------------
        if not isinstance(self.allow_state_injection, bool):
            raise ConfigError(
                "config field 'allow_state_injection' must be True or False, "
                f"got {self.allow_state_injection!r}"
            )

        # -- the derived ghost policies ---------------------------------------
        policies = _resolve_ghost_policies(self.ghost_policy, self.n_ghosts)
        set_(self, "ghost_policies", policies)
        set_(self, "ghost_policy_names", tuple(p.name for p in policies))

    # -- convenience for the environment and the model ------------------------

    def start_cells(self) -> tuple[Cell, ...]:
        """The cells the agent may start on; rho_0 is uniform over them.

        With ``start="layout"`` these are the map's ``P`` cells. With
        ``start="random"`` they are all floor cells that no ghost starts on.
        """
        if self.start == "layout":
            return self.layout.agent_starts
        # Goal cells are excluded for the same reason reset() clears a pellet
        # from the agent's start cell: spawning *on* the goal would create a
        # non-terminal state in which the agent stands on X and has to leave and
        # come back to collect the reward, which is a confusing artefact rather
        # than a lesson. Ghost spawns are excluded so nobody starts dead.
        occupied = set(self.initial_ghosts()) | set(self.layout.goal_cells)
        return tuple(cell for cell in self.layout.floor_cells if cell not in occupied)

    @property
    def n_pellets(self) -> int:
        """How many pellets this world starts with.

        Worth having as a property because it is the natural denominator for a
        training log -- "ate 45 of 51" says more than a return does -- and
        because ``len(cfg.initial_pellets())`` reads like it might be doing
        something.
        """
        return len(self.initial_pellets())

    def initial_ghosts(self) -> tuple[Cell, ...]:
        """The ghosts' spawn cells: the first ``n_ghosts`` ``G`` cells, row-major.

        Deterministic on purpose. Randomizing *which* spawns are used would
        need a sampling-without-replacement rule that no lab wants; it can be
        added later without a redesign, since rho_0 already exists.
        """
        return self.layout.ghost_starts[: self.n_ghosts]

    def initial_pellets(self) -> frozenset[Cell]:
        """The pellets on the board before the agent's own cell is cleared.

        With ``pellets="all_floor"`` goal cells get no pellet (they end the
        episode anyway) but ghost spawns do. The agent's start cell is cleared
        by :meth:`~pragman.env.PragmanEnv.reset`, because which cell that is
        depends on the draw from rho_0.
        """
        if self.pellets == "none":
            return frozenset()
        if self.pellets == "all_floor":
            return frozenset(self.layout.floor_cells) - set(self.layout.goal_cells)
        return frozenset(self.layout.pellet_cells)

    # -- serialization --------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Plain JSON-able data, for :class:`~pragman.state.Episode` files.

        The layout is stored as its drawing and each ghost as its name plus
        its settings, so a ``ProximityChaser(wake_distance=6)`` comes back
        unchanged. A *custom* ghost must be registered in the loading process.
        """
        data: dict[str, Any] = {}
        for spec in fields(self):
            if not spec.init:
                continue
            value = getattr(self, spec.name)
            if spec.name == "layout":
                value = value.ascii
            elif spec.name == "ghost_policy":
                value = [policy_to_json(policy) for policy in self.ghost_policies]
            data[spec.name] = value
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> PragmanConfig:
        """Inverse of :meth:`to_json_dict`.

        Raises:
            ConfigError: naming the policy, if an episode was recorded with a
                custom ghost that has not been registered in this process. It
                is better to fail loudly than to substitute a random ghost and
                silently change the dynamics an offline dataset came from.
        """
        data = dict(data)
        data["layout"] = Layout.from_ascii(data["layout"])
        data["ghost_policy"] = [policy_from_json(entry) for entry in data["ghost_policy"]]
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _check_choice(name: str, value: Any, options: Sequence[str]) -> None:
    if value not in options:
        legal = ", ".join(repr(option) for option in options)
        raise ConfigError(f"config field {name!r} must be one of {legal}, got {value!r}")


def _check_probability(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"config field {name!r} must be a number in [0, 1], got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"config field {name!r} must be in [0, 1], got {value!r}")


def _check_finite(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"config field {name!r} must be a finite number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ConfigError(f"config field {name!r} must be finite, got {value!r}")


def _resolve_ghost_policies(
    spec: str | GhostPolicy | Sequence[str | GhostPolicy], n_ghosts: int
) -> tuple[GhostPolicy, ...]:
    """Turn the ``ghost_policy`` field into one policy object per ghost.

    A single value (name or object) is replicated; ``"mix"`` cycles through
    random, chaser, proximity in the ghosts' row-major spawn order, so "which
    ghost is which" is answerable by looking at the map; an explicit sequence
    is used as given and must have exactly ``n_ghosts`` entries.
    """
    if isinstance(spec, str):
        if spec == "mix":
            names = [MIX_ORDER[i % len(MIX_ORDER)] for i in range(n_ghosts)]
            return tuple(ghost_policy_by_name(name) for name in names)
        return tuple(ghost_policy_by_name(spec) for _ in range(n_ghosts))

    if _is_policy(spec):
        return tuple(spec for _ in range(n_ghosts))  # type: ignore[misc]

    if isinstance(spec, Sequence):
        entries = list(spec)
        if len(entries) != n_ghosts:
            raise ConfigError(
                f"config field 'ghost_policy' lists {len(entries)} policies but n_ghosts is "
                f"{n_ghosts}; give one entry per ghost, or a single value to use for all"
            )
        return tuple(
            ghost_policy_by_name(entry) if isinstance(entry, str) else _as_policy(entry)
            for entry in entries
        )

    raise ConfigError(
        "config field 'ghost_policy' must be a policy name ('random', 'chaser', "
        f"'proximity', 'mix'), a policy object, or a sequence of those, not {spec!r}"
    )


def _is_policy(candidate: Any) -> bool:
    return hasattr(candidate, "distribution") and hasattr(candidate, "name")


def _as_policy(candidate: Any) -> GhostPolicy:
    if not _is_policy(candidate):
        raise ConfigError(
            f"{candidate!r} is not a ghost policy: it needs a 'name' and a "
            "'distribution(cfg, state, ghost_index)' method"
        )
    return candidate
