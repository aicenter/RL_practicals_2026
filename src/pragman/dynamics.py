"""The pure core of the environment: how one step happens.

Everything here is a function of ``(cfg, state, ...)`` with no hidden state, so
you can call it from an MCTS rollout, from a dynamic-programming sweep, or from
a unit test, without ever constructing an environment.

Two compositions are built from the same three pieces:

* :func:`transition` *samples* a slip and a ghost move and calls :func:`apply`
  -- this is what ``PragmanEnv.step`` uses;
* :func:`successors` *enumerates* the same product exactly -- this is what the
  white-box tabular model uses.

They are two different compositions of the same factors, which is precisely why
it is worth testing that they agree empirically.

**The order of events inside a step.** One rule governs everything: *once
``terminated`` is set, nothing further moves; only rewards already earned in
that step stand.*

1. **Slip.** With probability ``slip_prob`` the action is replaced by one of
   the two perpendicular actions, each with probability ``slip_prob / 2``.
   (Sampled by :func:`transition`, enumerated by :func:`successors`; by the
   time :func:`apply` is called the slip has already happened.)
2. **Agent moves.** Into a wall or off the grid means staying put.
3. **Death A.** If the agent's new cell holds a ghost: ``death_reward``,
   terminate. Touching a ghost always kills, so a ghost sitting on a pellet or
   on the goal is genuinely dangerous.
4. **Pellet.** If the new cell holds a pellet: remove it, add ``pellet_reward``.
   If none remain, add ``clear_bonus`` and terminate.
5. **Goal.** If the new cell is a goal cell: add ``goal_reward``, terminate.
6. **Ghosts move simultaneously.** Every ghost's distribution is computed from
   the *same* post-agent-move, pre-ghost-move state, and then they all move at
   once. Ghosts do not collide with one another and may share a cell.
7. **Death B.** If a ghost's new cell equals the agent's cell: ``death_reward``,
   terminate. A swap needs no separate rule -- see the note in :func:`apply_moves`.
8. **Living reward** is added if and only if the step did not terminate.

Truncation is not part of this module at all: the environment counts steps and
sets ``truncated`` itself, because a time limit is a property of the experiment
rather than of the Markov decision process.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np

from .config import PragmanConfig
from .ghosts import check_distribution, mix_epsilon, resolve_illegal
from .state import Action, Cell, PragmanState, as_action

__all__ = [
    "slip_distribution",
    "ghost_distributions",
    "apply",
    "apply_moves",
    "sample_moves",
    "transition",
    "successors",
]

#: What :func:`apply`, :func:`apply_moves`, and :func:`transition` return.
Outcome = tuple[PragmanState, float, bool]

#: One row of what :func:`successors` returns.
Successor = tuple[float, PragmanState, float, bool]

_K = TypeVar("_K")


# ──────────────────────────────────────────────────────────────────────────────
# The two sources of randomness
# ──────────────────────────────────────────────────────────────────────────────


def slip_distribution(cfg: PragmanConfig, action: int | Action) -> dict[Action, float]:
    """Which action actually gets executed, given the one the agent chose.

    With probability ``1 - slip_prob`` it is the chosen action; otherwise it is
    one of the two perpendicular ones, each with probability ``slip_prob / 2``.
    ``slip_prob = 0.2`` therefore reproduces the familiar 0.8 / 0.1 / 0.1
    gridworld. Actions with zero probability are left out.
    """
    action = as_action(action)
    if cfg.slip_prob == 0.0:
        return {action: 1.0}
    distribution = {action: 1.0 - cfg.slip_prob}
    for sideways in action.perpendicular:
        distribution[sideways] = distribution.get(sideways, 0.0) + cfg.slip_prob / 2.0
    return {a: p for a, p in sorted(distribution.items()) if p > 0.0}


def ghost_distributions(
    cfg: PragmanConfig, state: PragmanState
) -> tuple[dict[Action, float], ...]:
    """Each ghost's action distribution, epsilon-mixing already applied.

    The ghosts all see the same *state*, which is the post-agent-move,
    pre-ghost-move situation: they move simultaneously, so no ghost can react
    to another ghost's move within the same step.

    Every policy's raw output is checked to be a distribution, and then mixed
    with uniform noise as ``(1 - ghost_epsilon) * policy + ghost_epsilon *
    uniform(legal moves)``. Both rules live in the environment so that no
    policy -- built in or written by a student -- has to implement them.
    """
    if len(state.ghosts) != len(cfg.ghost_policies):
        raise ValueError(
            f"the state has {len(state.ghosts)} ghost(s) but the config expects "
            f"{len(cfg.ghost_policies)} (cfg.n_ghosts = {cfg.n_ghosts})"
        )
    distributions = []
    for index, policy in enumerate(cfg.ghost_policies):
        raw = policy.distribution(cfg, state, index)
        check_distribution(raw, policy)
        legal = cfg.layout.legal_actions(state.ghosts[index])
        distributions.append(mix_epsilon(raw, legal, cfg.ghost_epsilon))
    return tuple(distributions)


# ──────────────────────────────────────────────────────────────────────────────
# One step, given the moves
# ──────────────────────────────────────────────────────────────────────────────


def apply(
    cfg: PragmanConfig,
    state: PragmanState,
    agent_action: int | Action,
    ghost_actions: Sequence[int | Action] = (),
) -> Outcome:
    """Play out one step, given the moves everybody makes. Pure and deterministic.

    Args:
        cfg: the world.
        state: the situation before the step.
        agent_action: the action *after* any slip. :func:`transition` and
            :func:`successors` deal with the slip; by the time we get here the
            realized direction is known.
        ghost_actions: one action per ghost, index-aligned with
            ``state.ghosts``.

    Returns:
        ``(next_state, reward, terminated)``.
    """
    layout = cfg.layout
    ghost_actions = tuple(ghost_actions)
    if len(ghost_actions) != len(state.ghosts):
        # Checked here, rather than left to zip(strict=True), so that the
        # message names the actual problem: a caller who forgets a ghost would
        # otherwise get "zip() argument 2 is shorter than argument 1".
        raise ValueError(
            f"got {len(ghost_actions)} ghost action(s) for {len(state.ghosts)} ghost(s); "
            "ghost actions are index-aligned with state.ghosts, one per ghost"
        )
    agent_cell = layout.move(state.agent, agent_action)
    ghost_cells = tuple(
        layout.move(cell, action)
        for cell, action in zip(state.ghosts, ghost_actions, strict=True)
    )
    return apply_moves(cfg, state, agent_cell, ghost_cells)


def apply_moves(
    cfg: PragmanConfig,
    state: PragmanState,
    agent_cell: Cell,
    ghost_cells: tuple[Cell, ...],
) -> Outcome:
    """One step, given where everybody *ends up* rather than which way they went.

    This is the real body of the step, shared by :func:`apply` and
    :func:`successors`. Working in destination cells rather than in actions
    costs nothing (a move into a wall has already become "stay put") and saves
    the enumeration from producing duplicates it would only have to merge.

    See the module docstring for the numbered order of events.
    """
    if len(ghost_cells) != len(state.ghosts):
        raise ValueError(
            f"got {len(ghost_cells)} ghost move(s) for a state with "
            f"{len(state.ghosts)} ghost(s); they are index-aligned, so there must be one each"
        )
    # Terminal states are absorbing with zero reward, so dynamic programming is
    # a plain contraction with no special cases.
    if state.terminated:
        return state, 0.0, True

    reward = 0.0
    pellets = state.pellets

    # 3. Death A -- the agent walked into a ghost. Nothing else in this step
    #    happens, which is why the ghosts stay exactly where they were.
    if agent_cell in state.ghosts:
        dead = PragmanState(agent_cell, state.ghosts, pellets, True)
        return dead, reward + cfg.death_reward, True

    # 4. Pellet.
    if agent_cell in pellets:
        pellets = pellets - {agent_cell}
        reward += cfg.pellet_reward
        if not pellets:
            reward += cfg.clear_bonus
            return PragmanState(agent_cell, state.ghosts, pellets, True), reward, True

    # 5. Goal.
    if agent_cell in cfg.layout.goal_cells:
        reward += cfg.goal_reward
        return PragmanState(agent_cell, state.ghosts, pellets, True), reward, True

    # 6. The ghosts have already moved (simultaneously, all from `state`).
    # 7. Death B -- a ghost caught the agent.
    #
    #    A *swap* (the agent leaving A for B while a ghost leaves B for A) needs
    #    no clause of its own: stepping onto B while a ghost stood there is
    #    death A, which returned above. The assertion below records that
    #    reasoning, so that reordering the events trips a test rather than
    #    quietly letting the agent walk through a ghost.
    caught = any(cell == agent_cell for cell in ghost_cells)
    assert not any(
        new_cell == state.agent and old_cell == agent_cell
        for old_cell, new_cell in zip(state.ghosts, ghost_cells, strict=True)
    ), "a swap should already have been caught as death A"
    if caught:
        dead = PragmanState(agent_cell, ghost_cells, pellets, True)
        return dead, reward + cfg.death_reward, True

    # 8. Living reward, only because the step did not end the episode.
    reward += cfg.living_reward
    return PragmanState(agent_cell, ghost_cells, pellets, False), reward, False


# ──────────────────────────────────────────────────────────────────────────────
# Sampling: what PragmanEnv.step does
# ──────────────────────────────────────────────────────────────────────────────


def sample_moves(
    cfg: PragmanConfig,
    state: PragmanState,
    action: int | Action,
    rng: np.random.Generator,
) -> tuple[Action, tuple[Action, ...]]:
    """Draw the realized agent action (after slipping) and one action per ghost.

    Split out from :func:`transition` so that a caller who needs to know
    *whether the agent slipped* -- the environment puts it in ``info`` -- can
    sample first and call :func:`apply` itself.
    """
    agent_action = _sample(slip_distribution(cfg, action), rng)
    ghost_actions = tuple(
        _sample(distribution, rng) for distribution in ghost_distributions(cfg, state)
    )
    return agent_action, ghost_actions


def transition(
    cfg: PragmanConfig,
    state: PragmanState,
    action: int | Action,
    rng: np.random.Generator,
) -> Outcome:
    """Sample one step. This is the *sampling* model, used by ``PragmanEnv.step``.

    Args:
        cfg: the world.
        state: the situation before the step.
        action: the action the agent *chose*; it may slip.
        rng: a ``numpy.random.Generator``. The environment passes its own
            ``self.np_random``; an MCTS rollout passes whatever it likes. The
            generator is deliberately not part of the state, so restoring a
            state does not rewind the random stream.

    Returns:
        ``(next_state, reward, terminated)``.
    """
    if state.terminated:
        return state, 0.0, True
    agent_action, ghost_actions = sample_moves(cfg, state, action, rng)
    return apply(cfg, state, agent_action, ghost_actions)


def _sample(distribution: Mapping[_K, float], rng: np.random.Generator) -> _K:
    """Draw one key from a small dictionary-shaped distribution."""
    outcomes = list(distribution)
    if len(outcomes) == 1:
        return outcomes[0]
    probabilities = np.array([distribution[key] for key in outcomes], dtype=float)
    probabilities /= probabilities.sum()  # guard against harmless floating-point drift
    return outcomes[int(rng.choice(len(outcomes), p=probabilities))]


# ──────────────────────────────────────────────────────────────────────────────
# Enumeration: what the tabular model uses
# ──────────────────────────────────────────────────────────────────────────────


def successors(
    cfg: PragmanConfig, state: PragmanState, action: int | Action
) -> tuple[Successor, ...]:
    """Every possible outcome of one step, with its exact probability.

    This is the *white-box* model: the same three factors as :func:`transition`
    (the slip and one distribution per ghost), enumerated instead of sampled.

    Duplicate outcomes are merged -- several combinations of ghost moves lead
    to the same board, and after a terminating agent move the ghosts do not
    move at all -- so the probabilities of the returned rows sum to 1. Rows are
    merged on ``(next_state, reward, terminated)``, so a state reachable with
    two different rewards would occupy two rows; no configuration in the course
    does that, but a caller should accumulate rather than assign. The order is
    deterministic (sorted by board position), so a printout is reproducible.

    Returns:
        A tuple of ``(probability, next_state, reward, terminated)`` rows.
    """
    if state.terminated:
        return ((1.0, state, 0.0, True),)  # terminal states are absorbing

    layout = cfg.layout
    # Where the agent may end up, merging slips that bump into the same wall.
    agent_cells = resolve_illegal(slip_distribution(cfg, action), layout, state.agent)
    # ... and where each ghost may end up, merged the same way.
    ghost_cells = [
        resolve_illegal(distribution, layout, cell)
        for distribution, cell in zip(ghost_distributions(cfg, state), state.ghosts, strict=True)
    ]

    merged: dict[tuple[PragmanState, float, bool], float] = {}
    for agent_cell, agent_probability in agent_cells.items():
        for combination in itertools.product(*[list(d.items()) for d in ghost_cells]):
            probability = agent_probability
            for _, ghost_probability in combination:
                probability *= ghost_probability
            outcome = apply_moves(
                cfg, state, agent_cell, tuple(cell for cell, _ in combination)
            )
            merged[outcome] = merged.get(outcome, 0.0) + probability

    return tuple(
        (probability, next_state, reward, terminated)
        for (next_state, reward, terminated), probability in sorted(
            merged.items(), key=lambda item: _sort_key(item[0][0])
        )
    )


def _sort_key(state: PragmanState) -> tuple:
    """A total order on states, so that :func:`successors` is reproducible."""
    return (state.agent, state.ghosts, sorted(state.pellets), state.terminated)
