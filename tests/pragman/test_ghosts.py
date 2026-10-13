"""Ghost personalities and ghost identity.

Probably the most bug-prone code in the package.

Two things are worth knowing before reading these tests:

* A policy returns a *distribution*; the environment samples. So a personality
  is asserted against ``policy.distribution(...)`` directly, or through
  ``dynamics.ghost_distributions`` with ``ghost_epsilon=0``. With the default
  epsilon of 0.1 the environment mixes in uniform noise, which spreads mass over
  every legal move and would hide the personality entirely.
* Ghosts are **index-aligned and never sorted**: ``state.ghosts[i]`` is governed
  by ``cfg.ghost_policies[i]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from pragman.config import PragmanConfig
from pragman.dynamics import ghost_distributions, successors, transition
from pragman.errors import ConfigError
from pragman.ghosts import (
    MIX_ORDER,
    ChaserGhost,
    ProximityChaser,
    RandomGhost,
    mix_epsilon,
    resolve_illegal,
    uniform,
)
from pragman.model import build_tabular_model
from pragman.state import Action, PragmanState

# An open 3x3 room. From the bottom-right corner the two ways towards the agent
# in the top-left corner are exactly tied, which is what makes it the fixture
# for the tie-breaking tests.
#
#   row 0:  P..
#   row 1:  ...
#   row 2:  ..G
OPEN_ROOM = """\
P..
...
..G"""

# A ten-cell corridor. Here a chaser has exactly one distance-decreasing move and
# a wanderer has two, so the two personalities are visibly different.
#
#   col:    0123456789
#   row 0:  #P......G#
CORRIDOR = """\
#P......G#"""


def world(layout: str, **overrides) -> PragmanConfig:
    """A config with the noise switched off unless a test asks for it."""
    overrides.setdefault("observation", "coords")
    overrides.setdefault("n_ghosts", 1)
    overrides.setdefault("ghost_epsilon", 0.0)
    overrides.setdefault("max_steps", None)
    return PragmanConfig(layout=layout, **overrides)


@dataclass(frozen=True)
class StubGhost:
    """A user-supplied policy that always wants one particular move.

    Used to check the two rules the *environment* applies to every policy, so
    that no policy has to implement them itself.
    """

    action: Action = Action.UP
    name: ClassVar[str] = "stub"

    def distribution(self, cfg, state, ghost_index):
        return {self.action: 1.0}


@dataclass(frozen=True)
class BrokenGhost:
    """A policy whose distribution does not sum to 1."""

    name: ClassVar[str] = "broken"

    def distribution(self, cfg, state, ghost_index):
        return {Action.UP: 0.25, Action.DOWN: 0.25}


# ──────────────────────────────────────────────────────────────────────────────
# RandomGhost
# ──────────────────────────────────────────────────────────────────────────────


def test_random_ghost_is_uniform_over_legal_moves():
    """"Random ghost" means what everyone thinks it means."""
    cfg = world(OPEN_ROOM)
    agent_away = PragmanState((0, 0), ((1, 1),))
    assert RandomGhost().distribution(cfg, agent_away, 0) == {
        Action.UP: 0.25,
        Action.RIGHT: 0.25,
        Action.DOWN: 0.25,
        Action.LEFT: 0.25,
    }

    # In the bottom-right corner only UP and LEFT go anywhere.
    in_corner = PragmanState((0, 0), ((2, 2),))
    assert RandomGhost().distribution(cfg, in_corner, 0) == {Action.UP: 0.5, Action.LEFT: 0.5}


# ──────────────────────────────────────────────────────────────────────────────
# ChaserGhost
# ──────────────────────────────────────────────────────────────────────────────


def test_chaser_puts_all_its_mass_on_the_one_closing_move():
    """In a corridor there is one distance-decreasing move, and it gets everything.

    Asserted against the *raw* distribution: the environment's default epsilon
    of 0.1 would otherwise leave 5% on walking away.
    """
    cfg = world(CORRIDOR)
    state = PragmanState.from_ascii("#P...G...#", cfg.layout, n_ghosts=1)  # ghost four steps away
    distances = cfg.layout.distances(state.agent)
    assert distances[state.ghosts[0]] == 4

    assert ChaserGhost().distribution(cfg, state, 0) == {Action.LEFT: 1.0}


def test_every_move_a_chaser_considers_strictly_decreases_the_distance():
    """The general statement, checked from every cell of the open room."""
    cfg = world(OPEN_ROOM)
    layout = cfg.layout
    for ghost_cell in layout.floor_cells:
        state = PragmanState((0, 0), (ghost_cell,))
        if ghost_cell == state.agent:
            continue
        distances = layout.distances(state.agent)
        here = distances[ghost_cell]

        distribution = ChaserGhost().distribution(cfg, state, 0)
        assert sum(distribution.values()) == pytest.approx(1.0)
        for action, probability in distribution.items():
            assert probability > 0.0
            assert distances[layout.move(ghost_cell, action)] == here - 1


def test_chaser_splits_its_mass_uniformly_over_ties():
    """From the far corner, UP and LEFT are equally good, so each gets half."""
    cfg = world(OPEN_ROOM)
    state = PragmanState((0, 0), ((2, 2),))
    assert ChaserGhost().distribution(cfg, state, 0) == {Action.UP: 0.5, Action.LEFT: 0.5}
    assert ChaserGhost(tie_break="uniform") == ChaserGhost()  # the default


def test_tie_break_first_makes_the_chaser_deterministic():
    """``tie_break="first"`` takes the lowest action index.

    This is what keeps ``planning_11x11`` free of chance nodes: with ties broken
    uniformly, roughly two nodes in five would still branch.
    """
    cfg = world(OPEN_ROOM, ghost_policy=ChaserGhost(tie_break="first"))
    state = PragmanState((0, 0), ((2, 2),))
    assert ChaserGhost(tie_break="first").distribution(cfg, state, 0) == {Action.UP: 1.0}

    # ... and the whole world is then deterministic: one successor per action.
    for action in Action:
        rows = successors(cfg, state, action)
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(1.0)


def test_tie_break_must_be_uniform_or_first():
    with pytest.raises(ConfigError) as excinfo:
        ChaserGhost(tie_break="lowest")
    assert "tie_break" in str(excinfo.value)


def test_a_chaser_walled_off_from_the_agent_wanders():
    """With no path to the agent there is nothing to minimize, so it goes uniform."""
    sealed = world(
        """\
        ####
        #P.#
        ####
        #G.#
        ####""",
        n_ghosts=1,
    )
    state = PragmanState((1, 1), ((3, 1),))
    assert ChaserGhost().distribution(sealed, state, 0) == {Action.RIGHT: 1.0}
    # (1, 1) is unreachable from (3, 1), so every move scores infinity and the
    # ghost falls back to uniform over its one legal move.


# ──────────────────────────────────────────────────────────────────────────────
# ProximityChaser switches at exactly wake_distance
# ──────────────────────────────────────────────────────────────────────────────


def test_proximity_chaser_wakes_at_exactly_wake_distance():
    """``distance <= wake_distance`` chases; one step further it wanders."""
    policy = ProximityChaser(wake_distance=4)
    cfg = world(CORRIDOR, ghost_policy=policy)
    layout = cfg.layout

    asleep = PragmanState.from_ascii("#P....G..#", layout, n_ghosts=1)  # distance 5
    awake = PragmanState.from_ascii("#P...G...#", layout, n_ghosts=1)  # distance 4
    assert layout.distances(asleep.agent)[asleep.ghosts[0]] == 5
    assert layout.distances(awake.agent)[awake.ghosts[0]] == 4

    assert policy.distribution(cfg, awake, 0) == ChaserGhost().distribution(cfg, awake, 0)
    assert policy.distribution(cfg, asleep, 0) == RandomGhost().distribution(cfg, asleep, 0)
    # Concretely: chase left, or wander both ways.
    assert policy.distribution(cfg, awake, 0) == {Action.LEFT: 1.0}
    assert policy.distribution(cfg, asleep, 0) == {Action.RIGHT: 0.5, Action.LEFT: 0.5}


def test_wake_distance_is_a_setting_not_a_constant():
    """A keener ghost wakes further away."""
    cfg = world(CORRIDOR, ghost_policy=ProximityChaser(wake_distance=6))
    state = PragmanState.from_ascii("#P....G..#", cfg.layout, n_ghosts=1)  # distance 5
    assert ProximityChaser(wake_distance=6).distribution(cfg, state, 0) == {Action.LEFT: 1.0}


# ──────────────────────────────────────────────────────────────────────────────
# The two rules the environment applies to every policy
# ──────────────────────────────────────────────────────────────────────────────


def test_epsilon_mixing_sums_to_one_and_leaves_an_escape():
    """``(1 - eps) * policy + eps * uniform(legal)``."""
    cfg = world(CORRIDOR, ghost_policy="chaser", ghost_epsilon=0.1)
    state = PragmanState.from_ascii("#P...G...#", cfg.layout, n_ghosts=1)

    (mixed,) = ghost_distributions(cfg, state)
    assert sum(mixed.values()) == pytest.approx(1.0)
    # The chaser wanted LEFT with certainty; 10% of the mass is now split over
    # the two legal moves, so 5% of the time it walks away from you.
    assert mixed[Action.LEFT] == pytest.approx(0.95)
    assert mixed[Action.RIGHT] == pytest.approx(0.05)


def test_epsilon_mixing_is_applied_to_user_supplied_policies_too():
    """A student's own ghost gets the same treatment, without implementing it."""
    cfg = world(CORRIDOR, ghost_policy=StubGhost(Action.LEFT), ghost_epsilon=0.2)
    state = PragmanState.from_ascii("#P...G...#", cfg.layout, n_ghosts=1)

    (mixed,) = ghost_distributions(cfg, state)
    assert sum(mixed.values()) == pytest.approx(1.0)
    assert mixed[Action.LEFT] == pytest.approx(0.9)  # 0.8 * 1 + 0.2 * 0.5
    assert mixed[Action.RIGHT] == pytest.approx(0.1)


def test_epsilon_zero_leaves_the_policy_alone():
    dist = {Action.UP: 1.0}
    assert mix_epsilon(dist, [Action.UP, Action.DOWN], 0.0) == {Action.UP: 1.0}


def test_mass_on_an_illegal_move_resolves_to_staying_put():
    """Mass on a move into a wall is legal and means "stay where you are"."""
    cfg = world(CORRIDOR, ghost_policy=StubGhost(Action.UP))  # UP is a wall everywhere here
    state = PragmanState.from_ascii("#P...G...#", cfg.layout, n_ghosts=1)

    # In cell space, all the mass lands back on the ghost's own cell.
    (dist,) = ghost_distributions(cfg, state)
    assert resolve_illegal(dist, cfg.layout, state.ghosts[0]) == {(0, 5): 1.0}

    # And so the ghost simply does not move.
    rows = successors(cfg, state, Action.RIGHT)
    assert len(rows) == 1
    probability, next_state, _reward, _terminated = rows[0]
    assert probability == pytest.approx(1.0)
    assert next_state.ghosts == ((0, 5),)


def test_two_wall_bumping_actions_merge_into_one_outcome():
    """Duplicates are merged, which is what keeps ``successors`` small."""
    layout = world(CORRIDOR).layout
    dist = {Action.UP: 0.5, Action.DOWN: 0.5}  # both are walls in the corridor
    assert resolve_illegal(dist, layout, (0, 5)) == {(0, 5): 1.0}


def test_a_distribution_that_does_not_sum_to_one_names_the_policy():
    """Checked once, with a clear error naming the policy."""
    cfg = world(CORRIDOR, ghost_policy=BrokenGhost())
    state = PragmanState.from_ascii("#P...G...#", cfg.layout, n_ghosts=1)

    with pytest.raises(ConfigError) as excinfo:
        ghost_distributions(cfg, state)
    message = str(excinfo.value)
    assert "broken" in message and "0.5" in message


def test_uniform_over_nothing_still_sums_to_one():
    """A ghost sealed into a single cell has no legal move; every action stays put."""
    assert uniform([]) == {a: 0.25 for a in Action}


# ──────────────────────────────────────────────────────────────────────────────
# Mixed personalities
# ──────────────────────────────────────────────────────────────────────────────

# Five ghost spawns, because no shipped layout has that many.
#
#   row 0:  G.G.G
#   row 1:  ..P..
#   row 2:  .G.G.
FIVE_SPAWNS = """\
G.G.G
..P..
.G.G."""


@pytest.mark.parametrize("n_ghosts", [1, 2, 3, 4, 5])
def test_mix_assigns_random_chaser_proximity_cyclically(n_ghosts):
    """``"mix"`` cycles in the ghosts' row-major spawn order."""
    cfg = PragmanConfig(
        layout=FIVE_SPAWNS, n_ghosts=n_ghosts, ghost_policy="mix", observation="coords"
    )
    assert MIX_ORDER == ("random", "chaser", "proximity")
    expected = [MIX_ORDER[i % len(MIX_ORDER)] for i in range(n_ghosts)]
    assert [policy.name for policy in cfg.ghost_policies] == expected
    assert len(cfg.ghost_policies) == n_ghosts


def test_mix_follows_the_maps_row_major_spawn_order():
    """"Which ghost is which" is answerable by looking at the map."""
    cfg = PragmanConfig(
        layout=FIVE_SPAWNS, n_ghosts=4, ghost_policy="mix", observation="coords"
    )
    assert cfg.initial_ghosts() == ((0, 0), (0, 2), (0, 4), (2, 1))
    assert cfg.ghost_policy_names == ("random", "chaser", "proximity", "random")


def test_an_explicit_sequence_is_honoured_as_given():
    """A per-ghost list generalizes ``"mix"`` and costs no extra machinery."""
    cfg = PragmanConfig(
        layout=FIVE_SPAWNS,
        n_ghosts=3,
        ghost_policy=["chaser", "chaser", RandomGhost()],
        observation="coords",
    )
    assert cfg.ghost_policies == (ChaserGhost(), ChaserGhost(), RandomGhost())


def test_a_single_value_is_replicated():
    cfg = PragmanConfig(
        layout=FIVE_SPAWNS, n_ghosts=3, ghost_policy="chaser", observation="coords"
    )
    assert cfg.ghost_policies == (ChaserGhost(),) * 3


def test_a_sequence_of_the_wrong_length_raises():
    """One entry per ghost, or a single value for all."""
    with pytest.raises(ConfigError) as excinfo:
        PragmanConfig(
            layout=FIVE_SPAWNS,
            n_ghosts=3,
            ghost_policy=["chaser", "random"],
            observation="coords",
        )
    message = str(excinfo.value)
    assert "'ghost_policy'" in message and "2 policies" in message and "n_ghosts is 3" in message


def test_an_unknown_policy_name_lists_the_known_ones():
    with pytest.raises(ConfigError) as excinfo:
        PragmanConfig(
            layout=FIVE_SPAWNS, n_ghosts=1, ghost_policy="blinky", observation="coords"
        )
    message = str(excinfo.value)
    assert "'blinky'" in message and "chaser" in message and "proximity" in message


# ──────────────────────────────────────────────────────────────────────────────
# Each ghost is governed by its own policy, for the whole episode
# ──────────────────────────────────────────────────────────────────────────────

# A long corridor with a ghost spawn at each end and the agent in the middle.
# Both ghosts have two legal moves, so a chaser and a wanderer look different.
#
#   col:    0123456789...
#   row 0:  #.G...P...G.#
DUEL = """\
#.G...P...G.#"""


def duel_world(**overrides) -> PragmanConfig:
    overrides.setdefault("ghost_policy", ["chaser", "random"])
    overrides.setdefault("ghost_epsilon", 0.0)
    return PragmanConfig(
        layout=DUEL,
        n_ghosts=2,
        slip_prob=0.0,
        max_steps=None,
        observation="coords",
        **overrides,
    )


def test_only_the_chaser_closes_the_distance():
    """Ghost 0 is the chaser and ghost 1 the wanderer, and it shows.

    The agent walks into the wall above it, so it never moves; over many samples
    the chaser's distance falls by exactly one every step while the wanderer's
    is a coin flip.
    """
    cfg = duel_world()
    layout = cfg.layout
    state = PragmanState((0, 6), ((0, 2), (0, 10)))
    distances = layout.distances(state.agent)
    assert distances[state.ghosts[0]] == 4 and distances[state.ghosts[1]] == 4

    rng = np.random.default_rng(20260923)
    deltas = [[], []]
    for _ in range(400):
        next_state, _reward, terminated = transition(cfg, state, Action.UP, rng)
        assert not terminated
        assert next_state.agent == (0, 6)  # UP is a wall: the agent stays put
        for i in (0, 1):
            deltas[i].append(distances[next_state.ghosts[i]] - distances[state.ghosts[i]])

    assert set(deltas[0]) == {-1}  # the chaser always closes in
    assert set(deltas[1]) == {-1, 1}  # the wanderer goes either way
    assert np.mean(deltas[1]) == pytest.approx(0.0, abs=0.15)


def test_ghost_identity_survives_a_whole_episode():
    """``state.ghosts[i]`` stays the ghost of ``cfg.ghost_policies[i]``."""
    cfg = duel_world()
    layout = cfg.layout
    state = PragmanState((0, 6), ((0, 2), (0, 10)))
    rng = np.random.default_rng(0)

    for step in range(3):
        state, _reward, terminated = transition(cfg, state, Action.UP, rng)
        assert not terminated
        # The chaser is index 0 throughout, wherever the wanderer happens to be.
        assert state.ghosts[0] == (0, 3 + step)
        assert layout.distances(state.agent)[state.ghosts[0]] == 3 - step


def test_ghost_positions_are_never_sorted():
    """Sorting would make the chaser and the wanderer interchangeable.

    Here the chaser (index 0) sits to the *right* of the wanderer, so the tuple
    is not in row-major order -- and must stay that way.
    """
    cfg = duel_world()
    state = PragmanState((0, 6), ((0, 10), (0, 2)))  # chaser on the right
    rng = np.random.default_rng(1)

    next_state, _reward, _terminated = transition(cfg, state, Action.UP, rng)
    assert next_state.ghosts[0] == (0, 9)  # the chaser closed in, from the right
    assert next_state.ghosts[0] > next_state.ghosts[1]  # still not sorted
    assert sorted(next_state.ghosts) != list(next_state.ghosts)


def test_two_states_differing_only_in_which_ghost_is_where_are_distinct():
    """Swapping two ghosts gives a different key in ``model.index``.

    The tiny room below is four cells with two ghost spawns, small enough to
    enumerate: 4 agent cells x 4 x 4 ghost cells, minus what is unreachable.
    """
    cfg = PragmanConfig(
        layout="####\n#P.#\n#GG#\n####",
        n_ghosts=2,
        ghost_policy=["chaser", "random"],
        observation="coords",
        max_steps=None,
    )
    model = build_tabular_model(cfg)

    here = PragmanState((1, 1), ((2, 1), (2, 2)))
    swapped = PragmanState((1, 1), ((2, 2), (2, 1)))
    assert here != swapped
    assert here in model.index and swapped in model.index
    assert model.index[here] != model.index[swapped]


def test_a_state_with_the_wrong_number_of_ghosts_is_refused():
    """The state and the config must agree on how many ghosts there are."""
    cfg = duel_world()
    with pytest.raises(ValueError) as excinfo:
        ghost_distributions(cfg, PragmanState((0, 6), ((0, 2),)))
    assert "1 ghost" in str(excinfo.value) and "2" in str(excinfo.value)


def test_the_default_epsilon_would_hide_all_of_this():
    """Why these tests use the raw distribution, without ``ghost_epsilon``.

    With the shipped ``ghost_epsilon=0.1`` even a chaser has mass on walking
    away, which is the point -- a chaser you cannot shake off makes the optimal
    policy an exercise in parity rather than in strategy.
    """
    cfg = duel_world(ghost_epsilon=0.1)
    state = PragmanState((0, 6), ((0, 2), (0, 10)))
    chaser, _wanderer = ghost_distributions(cfg, state)
    assert set(chaser) == {Action.RIGHT, Action.LEFT}
    assert chaser[Action.RIGHT] == pytest.approx(0.95)
    assert chaser[Action.LEFT] == pytest.approx(0.05)
    assert math.isclose(sum(chaser.values()), 1.0)
