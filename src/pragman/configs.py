"""The ten ready-made worlds, one row of the course map each.

Every factory returns a :class:`~pragman.config.PragmanConfig` and takes
keyword overrides, so a lab can start from a known-good world and change one
thing::

    cfg  = pragman.configs.maze_8x8()                  # as the notes describe it
    cfg  = pragman.configs.maze_8x8(slip_prob=0.3)     # ... but slipperier
    env  = pragman.make("maze_8x8", slip_prob=0.3)     # the same thing, in one line

============================  =======  =========================================
factory                       week     what it is for
============================  =======  =========================================
:func:`empty_5x5`             4        25 states, deterministic: hand-checkable DP
:func:`slippery_5x5`          4-5      the same, with slips
:func:`maze_8x8`              5-6, 10  ~52 states; also the Dyna-Q world
:func:`haunted_5x5`           5-6, 10  ~600 states: DP and Dyna-Q *with* a ghost
:func:`pellets_8x8`           6-8      the state space explodes; REINFORCE lives here
:func:`classic_11x11`         7-9      the workhorse
:func:`pixels_11x11`          7        88x88x3, CPU-trainable
:func:`planning_11x11`        10       deterministic, injection on: MCTS
:func:`partial_11x11`         13       ``view_radius=2``: partial observability
:func:`single_ghost_11x11`    13       one ghost; for experiments with reward functions
============================  =======  =========================================

Two rules of thumb behind the numbers. **Ghost difficulty is deliberately
conservative in Weeks 7-9**: ``classic_11x11`` ships with *random* ghosts,
because two competent chasers plus sparse pellets is a hard exploration problem
that may not train on a laptop CPU inside a 90-minute practical. Chasers are the
"now make it harder" extension, and ``pragman.make("classic_11x11",
ghost_policy="mix")`` is the most vivid one to watch. And
``allow_state_injection`` is switched **off** in the three model-free worlds
(``pellets_8x8``, ``classic_11x11``, ``pixels_11x11``), where reading the true
state out of the simulator would be cheating rather than planning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

from .config import PragmanConfig
from .errors import ConfigError
from .ghosts import ChaserGhost
from .layout import NAMED_LAYOUTS
from .state import Action, PragmanState

__all__ = [
    "empty_5x5",
    "slippery_5x5",
    "maze_8x8",
    "haunted_5x5",
    "pellets_8x8",
    "classic_11x11",
    "pixels_11x11",
    "planning_11x11",
    "partial_11x11",
    "single_ghost_11x11",
    "pellet_distance_shaping",
    "FACTORIES",
    "names",
    "by_name",
]

#: The 11x11 maze holds 51 pellets, so 100 steps could not clear it even if the
#: agent never wasted one. Every world built on that map therefore gets a longer
#: clock of 300 steps, since a task whose terminal bonus is unreachable by
#: construction teaches the wrong lesson.
CLASSIC_MAX_STEPS = 300


# ──────────────────────────────────────────────────────────────────────────────
# Weeks 4-6: small enough to enumerate, so the observation is a plain index
# ──────────────────────────────────────────────────────────────────────────────


def empty_5x5(**overrides: Any) -> PragmanConfig:
    """Week 4. An open 5x5 room with a goal in the far corner: 25 states.

    Deterministic, ghost-free, and small enough that the whole value table fits
    on a slide -- and small enough to check dynamic programming by hand.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["empty_5x5"],
        observation="index",
    )


def slippery_5x5(**overrides: Any) -> PragmanConfig:
    """Weeks 4-5. Two corridors, and a 20% chance of slipping sideways.

    ``slip_prob=0.2`` is the familiar 0.8 / 0.1 / 0.1 gridworld: with the walls
    in place, a sideways slip in a corridor simply bumps you into a wall, which
    is what makes "the same action, a different outcome" visible in one step.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["slippery_5x5"],
        observation="index",
        slip_prob=0.2,
    )


def maze_8x8(**overrides: Any) -> PragmanConfig:
    """Weeks 5-6 and 10. A 52-cell maze with a gentle 10% slip.

    Big enough that a policy has to plan around walls, small enough to
    enumerate -- which is why Week 10 uses it to compare Dyna-Q's *learned*
    model against ``env.model``, the ground truth.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["maze_8x8"],
        observation="index",
        slip_prob=0.1,
    )


def haunted_5x5(**overrides: Any) -> PragmanConfig:
    """Weeks 5-6 and 10. One random ghost: about 600 states.

    Twenty-four cells for the agent times twenty-four for the ghost. Still
    enumerable, so it is the smallest world in which dynamic programming has to
    reason about somebody else moving -- and the first where the optimal policy
    is visibly about risk rather than about distance.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["haunted_5x5"],
        observation="index",
        n_ghosts=1,
        ghost_policy="random",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Weeks 6-9: too big to enumerate, so the agent gets planes of features
# ──────────────────────────────────────────────────────────────────────────────


def pellets_8x8(**overrides: Any) -> PragmanConfig:
    """Weeks 6-8. Twenty-nine pellets and one ghost, so the table stops fitting.

    Twenty-nine pellets mean 2**29 pellet configurations: asking for
    ``observation="index"`` here raises
    :class:`~pragman.errors.StateSpaceTooLarge`, which is a more convincing
    argument for function approximation than any slide. This is also where
    REINFORCE lives in Week 8.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["pellets_8x8"],
        observation="channels",
        n_ghosts=1,
        ghost_policy="random",
        allow_state_injection=False,
    )


def classic_11x11(**overrides: Any) -> PragmanConfig:
    """Weeks 7-9. The workhorse: the full maze, 51 pellets, two random ghosts.

    Deep Q-learning (Week 7), policy gradients (Week 8) and PPO (Week 9) are
    all trained here, so it is the one world worth knowing by heart. The ghosts
    are random on purpose (see the module docstring); ``ghost_policy="mix"`` or
    ``"chaser"`` is the "now make it harder" setting.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["classic_11x11"],
        observation="channels",
        n_ghosts=2,
        ghost_policy="random",
        max_steps=CLASSIC_MAX_STEPS,
        allow_state_injection=False,
    )


def pixels_11x11(**overrides: Any) -> PragmanConfig:
    """Week 7. :func:`classic_11x11` seen as an 88x88x3 picture.

    Eleven cells at eight pixels each, in colour: small enough for a
    convolutional network to train on a laptop CPU, big enough that the agent
    has to *see* rather than be told. It is, however, hungry in a way the
    channel encoding is not: one board is 23 kB against 0.5 kB, so a replay
    buffer holding 100 000 transitions of ``classic_11x11`` holds fewer than
    9 000 of these in the same memory. Expect to shrink the buffer, and expect
    roughly a third of the throughput. Swapping this for ``classic_11x11``
    changes one config field and nothing else in the training script, because
    both encodings are channel-last.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["classic_11x11"],
        observation="pixels",
        n_ghosts=2,
        ghost_policy="random",
        max_steps=CLASSIC_MAX_STEPS,
        allow_state_injection=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Weeks 10-13: planning, partial observability, and reward shaping
# ──────────────────────────────────────────────────────────────────────────────


def planning_11x11(**overrides: Any) -> PragmanConfig:
    """Week 10. Fully deterministic, with state injection on: the MCTS world.

    Every source of chance is switched off: ``slip_prob=0``, ``ghost_epsilon=0``
    and -- the one that is easy to miss -- ``tie_break="first"`` in the chasers.
    A chaser that splits its mass between two equally close moves leaves roughly
    two nodes in five branching, so tree search would be a search over chance
    nodes, which is a lecture of its own. Here ``successors`` returns a single
    outcome for every state and action, and the tree is a plain tree. Pass
    ``ghost_policy="chaser"`` to get the stochastic version back as an exercise.
    The two ghosts chase, which is what makes the search worth doing at all: a
    plan that ignores them dies.

    ``allow_state_injection`` is on, so ``env.clone_state()`` and
    ``env.restore_state(s)`` can wind a simulator back to a tree node. Note
    that the random stream is deliberately *not* part of the state, so a
    restored environment continues rather than repeats.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["classic_11x11"],
        observation="channels",
        n_ghosts=2,
        ghost_policy=ChaserGhost(tie_break="first"),
        slip_prob=0.0,
        ghost_epsilon=0.0,
        max_steps=CLASSIC_MAX_STEPS,
        allow_state_injection=True,
    )


def partial_11x11(**overrides: Any) -> PragmanConfig:
    """Week 13. The same maze through a 5x5 window: partial observability.

    ``view_radius=2`` crops the observation to the cells within two steps of the
    agent, padded with wall beyond the edge of the board and *not* rotated (the
    agent has no heading). The two ghosts are
    :class:`~pragman.ghosts.ProximityChaser`, which gives the lecture its
    story: the ghost you cannot see is the one that only wakes when you come
    close, so the observation is no longer a state and a memoryless policy has
    to gamble.
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["classic_11x11"],
        observation="channels",
        n_ghosts=2,
        ghost_policy="proximity",
        view_radius=2,
        max_steps=CLASSIC_MAX_STEPS,
    )


def single_ghost_11x11(**overrides: Any) -> PragmanConfig:
    """Week 13. The classic maze with a single random ghost.

    An ordinary world -- the maze, one ghost, the true reward -- meant for
    experiments in which the reward is swapped out, for instance::

        cfg = pragman.configs.single_ghost_11x11()
        env = pragman.ReplaceReward(
            pragman.PragmanEnv(cfg), pragman.configs.pellet_distance_shaping(cfg)
        )
    """
    return _build(
        overrides,
        layout=NAMED_LAYOUTS["classic_11x11"],
        observation="channels",
        n_ghosts=1,
        ghost_policy="random",
        max_steps=CLASSIC_MAX_STEPS,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reward shaping
# ──────────────────────────────────────────────────────────────────────────────


def pellet_distance_shaping(
    cfg: PragmanConfig,
    weight: float = 0.2,
    one_sided: bool = True,
) -> Callable[[PragmanState, int | Action, PragmanState, dict[str, Any]], float]:
    """A shaping reward that encourages moving towards pellets.

    Returns a function with the signature
    :class:`~pragman.wrappers.ReplaceReward` expects, ``(state, action,
    next_state, info) -> float``, which adds to the environment's true reward a
    bonus for *getting closer to the nearest pellet*, measured in breadth-first
    steps through the maze::

        r_shaped = r_true + weight * max(0, d(s) - d(s'))

    where ``d`` is the distance from the agent to the nearest remaining pellet.
    The true reward is read from ``info["true_reward"]``, which ``ReplaceReward``
    puts there before calling this function -- so the shaping *adds to* the task
    reward rather than replacing it, and both returns can be logged side by side.

    Args:
        cfg: the world, for its layout's (cached) distance tables.
        weight: how much a step of progress towards a pellet is worth.
        one_sided: reward approaching a pellet without penalizing moving away
            (the default). Set it to ``False`` for the symmetric version
            ``weight * (d(s) - d(s'))``.
    """
    layout = cfg.layout

    def shaped_reward(
        state: PragmanState,
        action: int | Action,
        next_state: PragmanState,
        info: dict[str, Any],
    ) -> float:
        try:
            true_reward = float(info["true_reward"])
        except KeyError:
            raise KeyError(
                "pellet_distance_shaping() adds to the environment's own reward, which it "
                "expects in info['true_reward']; pragman.ReplaceReward puts it there, so "
                "use this function through that wrapper"
            ) from None
        progress = _pellet_distance(layout, state) - _pellet_distance(layout, next_state)
        if one_sided:
            progress = max(0.0, progress)
        return true_reward + float(weight) * progress

    return shaped_reward


def _pellet_distance(layout, state: PragmanState) -> float:
    """Breadth-first distance from the agent to the nearest remaining pellet.

    Zero when the board is clear (or when every remaining pellet is walled off
    from the agent, which only a hand-made layout can arrange). The layout
    caches one distance table per source cell, so this is a dictionary lookup
    after the first visit to a cell.
    """
    if not state.pellets:
        return 0.0
    distances = layout.distances(state.agent)
    reachable = [distances[cell] for cell in state.pellets if cell in distances]
    return float(min(reachable)) if reachable else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# The registry the rest of the package looks worlds up in
# ──────────────────────────────────────────────────────────────────────────────

#: Every factory, by the name ``pragman.make`` and the Gymnasium ids use.
FACTORIES: dict[str, Callable[..., PragmanConfig]] = {
    factory.__name__: factory
    for factory in (
        empty_5x5,
        slippery_5x5,
        maze_8x8,
        haunted_5x5,
        pellets_8x8,
        classic_11x11,
        pixels_11x11,
        planning_11x11,
        partial_11x11,
        single_ghost_11x11,
    )
}


def names() -> tuple[str, ...]:
    """The names of all ten worlds, in course order."""
    return tuple(FACTORIES)


def by_name(name: str) -> Callable[..., PragmanConfig]:
    """Look a factory up by name.

    Raises:
        ConfigError: listing the available worlds, rather than a bare KeyError.
    """
    try:
        return FACTORIES[name]
    except (KeyError, TypeError):
        available = ", ".join(FACTORIES)
        raise ConfigError(
            f"unknown Prag-man world {name!r}; the ready-made worlds are {available} "
            "(or build a PragmanConfig of your own)"
        ) from None


_CONFIG_FIELDS = frozenset(spec.name for spec in fields(PragmanConfig) if spec.init)


def _build(overrides: dict[str, Any], **defaults: Any) -> PragmanConfig:
    """Merge a factory's defaults with the caller's overrides.

    Checking the override names here rather than letting ``PragmanConfig``
    raise a ``TypeError`` means that ``pragman.make("maze_8x8", slipp_prob=0.2)``
    tells you which field you misspelt instead of complaining about an
    unexpected keyword argument.
    """
    unknown = sorted(set(overrides) - _CONFIG_FIELDS)
    if unknown:
        legal = ", ".join(sorted(_CONFIG_FIELDS))
        raise ConfigError(
            f"unknown config field(s) {unknown}; PragmanConfig takes {legal}"
        )
    return PragmanConfig(**{**defaults, **overrides})
