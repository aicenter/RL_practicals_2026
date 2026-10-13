"""The white-box model.

The agreement of the sampler with the white-box model is the test that keeps
the whole thing honest. ``transition`` (which the
environment samples from) and ``successors`` (which the tabular model is built
from) are two *different compositions of the same three factors* -- the slip and
one distribution per ghost -- so agreement between them is a real check and not a
tautology.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pragman import configs
from pragman.config import PragmanConfig
from pragman.dynamics import successors, transition
from pragman.env import PragmanEnv
from pragman.errors import StateSpaceTooLarge
from pragman.model import (
    DEFAULT_MAX_STATES,
    build_tabular_model,
    enumerate_states,
    initial_state,
)
from pragman.state import ACTIONS, Action, PragmanState

# ──────────────────────────────────────────────────────────────────────────────
# How big the enumerable worlds are
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "factory, expected",
    [
        # 5x5 open room, no ghosts: one state per cell.
        ("empty_5x5", 25),
        # 8x8 maze: one state per walkable cell.
        ("maze_8x8", 52),
        # 24 walkable cells for the agent times 24 for the ghost.
        ("haunted_5x5", 24 * 24),
    ],
)
def test_the_enumerable_worlds_have_the_state_counts_the_course_map_promises(factory, expected):
    """The course map's state counts are a teaching claim, so they are asserted."""
    states, index = enumerate_states(getattr(configs, factory)())
    assert len(states) == expected
    assert len(index) == expected
    assert index[states[7]] == 7  # the inverse mapping really is the inverse


@pytest.mark.parametrize("factory", ["pellets_8x8", "classic_11x11"])
def test_pellet_worlds_blow_the_cap_and_say_why(factory):
    """"DP does not scale" is demonstrated, not asserted.

    The message has to be worth reading: it points at ``successors()``, which
    gives the exact one-step model for a *single* state, and at asynchronous
    dynamic programming.
    """
    with pytest.raises(StateSpaceTooLarge) as excinfo:
        enumerate_states(getattr(configs, factory)())

    error = excinfo.value
    message = str(error)
    assert error.count > DEFAULT_MAX_STATES
    assert error.max_states == DEFAULT_MAX_STATES
    assert "state space too large" in message
    assert "2**n_pellets" in message
    assert "successors(cfg, state, action)" in message
    assert "asynchronous" in message


def test_the_cap_is_a_keyword_argument():
    """A student with memory to spare can raise it."""
    with pytest.raises(StateSpaceTooLarge) as excinfo:
        enumerate_states(configs.empty_5x5(), max_states=10)
    assert excinfo.value.max_states == 10
    assert len(enumerate_states(configs.empty_5x5(), max_states=25)[0]) == 25


# ──────────────────────────────────────────────────────────────────────────────
# Successors is a distribution, in a reproducible order
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ["empty_5x5", "slippery_5x5", "maze_8x8", "haunted_5x5"])
def test_successor_probabilities_sum_to_one_everywhere(factory):
    """Every state, every action.

    The probabilities are raw products of floats, so they are compared with a
    tolerance rather than exactly -- which is also the advice for anyone using
    ``successors`` in a lab.
    """
    cfg = getattr(configs, factory)()
    states, _index = enumerate_states(cfg)
    for state in states:
        for action in ACTIONS:
            rows = successors(cfg, state, action)
            total = sum(probability for probability, *_ in rows)
            assert total == pytest.approx(1.0, abs=1e-9), (state, action)
            assert all(probability > 0.0 for probability, *_ in rows)


def test_successors_is_deterministically_ordered():
    """A student's printout is reproducible run to run."""
    cfg = configs.haunted_5x5()
    states, _index = enumerate_states(cfg)
    for state in states[:40]:
        for action in ACTIONS:
            assert successors(cfg, state, action) == successors(cfg, state, action)


def test_successors_merges_duplicate_outcomes():
    """Rows are merged on ``(next_state, reward, terminated)``.

    Two ghost moves that bump into the same wall are one outcome, not two.
    """
    cfg = PragmanConfig(
        layout="#####\n#P.G#\n#####",
        n_ghosts=1,
        ghost_policy="random",
        ghost_epsilon=0.0,
        observation="coords",
    )
    state = PragmanState((1, 1), ((1, 3),))
    rows = successors(cfg, state, Action.UP)
    # The ghost's only legal move is LEFT, and the agent bumps the wall above:
    # one outcome, probability 1.
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(1.0)
    assert rows[0][1] == PragmanState((1, 1), ((1, 2),))

    # And distinct boards stay distinct rows.
    open_room = PragmanConfig(layout="P..\n...\n..G", n_ghosts=1, observation="coords")
    busy = successors(open_room, PragmanState((1, 1), ((2, 2),)), Action.UP)
    assert len({row[1] for row in busy}) == len(busy)


# ──────────────────────────────────────────────────────────────────────────────
# The sampler and the white-box model agree
# ──────────────────────────────────────────────────────────────────────────────


def test_the_empirical_transition_matches_the_enumerated_one():
    """Sample ``transition`` and compare with ``successors``.

    The world is ``haunted_5x5`` made slippery, so that *both* sources of
    randomness -- the agent's slip and the ghost's move, including its epsilon
    mixing -- are exercised at once.
    """
    cfg = replace(configs.haunted_5x5(), slip_prob=0.2, ghost_epsilon=0.1)
    states, _index = enumerate_states(cfg)
    rng = np.random.default_rng(20260923)

    # A handful of states spread through the enumeration order, none terminal.
    sample = [state for state in states[::37] if not state.terminated][:6]
    assert len(sample) == 6

    draws = 2_000
    for state in sample:
        for action in ACTIONS:
            expected = {}
            for probability, next_state, reward, terminated in successors(cfg, state, action):
                expected[(next_state, round(reward, 9), terminated)] = probability

            counts: dict = {}
            for _ in range(draws):
                next_state, reward, terminated = transition(cfg, state, action, rng)
                key = (next_state, round(reward, 9), terminated)
                counts[key] = counts.get(key, 0) + 1

            # Nothing was sampled that the model says is impossible ...
            assert set(counts) <= set(expected), (state, action)
            # ... and every outcome's frequency matches its probability.
            for key, probability in expected.items():
                assert counts.get(key, 0) / draws == pytest.approx(probability, abs=0.05), (
                    state,
                    action,
                    key,
                )


# ──────────────────────────────────────────────────────────────────────────────
# The tabular model is a well-formed MDP
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ["empty_5x5", "slippery_5x5", "maze_8x8", "haunted_5x5"])
def test_the_tabular_model_is_a_well_formed_mdp(factory):
    """``P`` rows stochastic, ``rho0`` a distribution, ``R`` finite."""
    model = build_tabular_model(getattr(configs, factory)())
    S, A = model.n_states, model.n_actions

    assert A == 4
    assert model.P.shape == (S, A, S)
    assert model.R.shape == (S, A)
    assert model.rho0.shape == (S,)
    assert model.terminal.shape == (S,)

    assert np.all(model.P >= 0.0)
    assert np.allclose(model.P.sum(axis=2), 1.0)
    assert model.rho0.sum() == pytest.approx(1.0)
    assert np.all(model.rho0 >= 0.0)
    assert np.all(np.isfinite(model.R))


@pytest.mark.parametrize("factory", ["maze_8x8", "haunted_5x5"])
def test_terminal_states_are_absorbing_with_zero_reward(factory):
    """So value iteration is a plain contraction with no special cases."""
    model = build_tabular_model(getattr(configs, factory)())
    terminal = np.flatnonzero(model.terminal)
    assert terminal.size > 0

    for s in terminal:
        assert model.states[s].terminated is True
        assert np.allclose(model.R[s], 0.0)
        for a in range(model.n_actions):
            assert model.P[s, a, s] == pytest.approx(1.0)


def test_rho0_puts_all_its_mass_on_states_reset_can_produce():
    """rho_0 is uniform over the cells ``reset()`` draws."""
    cfg = configs.haunted_5x5()
    model = build_tabular_model(cfg)
    start_cells = cfg.start_cells()

    support = {model.states[s] for s in np.flatnonzero(model.rho0)}
    assert support == {initial_state(cfg, cell) for cell in start_cells}
    assert np.allclose(model.rho0[np.flatnonzero(model.rho0)], 1.0 / len(start_cells))


def test_the_model_ignores_max_steps():
    """The episode clock is not part of the MDP."""
    short = build_tabular_model(configs.maze_8x8(max_steps=5))
    long = build_tabular_model(configs.maze_8x8(max_steps=None))
    assert short.states == long.states
    assert np.array_equal(short.P, long.P)


def test_the_start_distribution_changes_only_rho0():
    """``start="random"`` spreads rho_0 over every cell the agent may legally
    begin on -- all 25 but the goal, which is excluded so that nobody starts
    standing on the reward -- and leaves the states, ``P`` and ``R`` alone.
    """
    corner = build_tabular_model(configs.empty_5x5())
    everywhere = build_tabular_model(configs.empty_5x5(start="random"))
    assert np.count_nonzero(everywhere.rho0) == 24
    assert np.isclose(everywhere.rho0.sum(), 1.0)

    assert set(corner.states) == set(everywhere.states)
    order = [everywhere.index[state] for state in corner.states]
    assert np.array_equal(corner.P, everywhere.P[order][:, :, order])
    assert np.array_equal(corner.R, everywhere.R[order])


# ──────────────────────────────────────────────────────────────────────────────
# The agent's start cell is cleared of its pellet
# ──────────────────────────────────────────────────────────────────────────────

# Every floor cell holds a pellet under `pellets="all_floor"`, including the one
# the agent spawns on -- which is the case this rule exists for.
#
#   row 1:  #P..#
#   row 2:  #...#
PELLET_ROOM = """\
#####
#P..#
#...#
#####"""


def test_reset_clears_the_pellet_under_the_agent():
    """Pellets are collected by *moving* onto a cell, so a spawn pellet would sit
    there uneaten underneath the agent."""
    cfg = PragmanConfig(layout=PELLET_ROOM, pellets="all_floor", observation="coords")
    assert (1, 1) in cfg.initial_pellets()  # it is there before reset

    env = PragmanEnv(cfg)
    env.reset(seed=0)
    state = env.state
    assert state.agent == (1, 1)
    assert state.agent not in state.pellets
    assert state.pellets == frozenset({(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)})


def test_reset_clears_the_pellet_wherever_start_random_drops_the_agent():
    """The rule matters most under ``start="random"`` in a full maze."""
    cfg = PragmanConfig(
        layout=PELLET_ROOM, pellets="all_floor", start="random", observation="coords"
    )
    env = PragmanEnv(cfg)

    landed = set()
    for seed in range(40):
        env.reset(seed=seed)
        state = env.state
        landed.add(state.agent)
        assert state.agent not in state.pellets
        assert len(state.pellets) == len(cfg.initial_pellets()) - 1
    assert landed == set(cfg.layout.floor_cells)  # it really does start anywhere


def test_the_enumerated_start_states_are_cleared_the_same_way():
    """``model.initial_state`` must clear the pellet exactly as ``reset`` does.

    If it did not, every state ``reset()`` produces would be missing from the
    enumeration, and ``rho0`` would key on a state that never occurs.
    """
    cfg = PragmanConfig(
        layout=PELLET_ROOM, pellets="all_floor", start="random", observation="coords"
    )
    model = build_tabular_model(cfg)
    for s in np.flatnonzero(model.rho0):
        state = model.states[s]
        assert state.agent not in state.pellets

    env = PragmanEnv(cfg)
    for seed in range(10):
        env.reset(seed=seed)
        assert env.state in model.index
