"""One step of the world: the pure core in ``pragman.dynamics``.

The order of events is what these tests are really about, and it comes down to
one rule: *once ``terminated`` is set, nothing further moves; only
rewards already earned in that step stand.* Every death, bonus, and short-circuit
below is a consequence of that sentence.

Most tests call :func:`pragman.dynamics.apply` with the moves spelled out, so
that nothing depends on a random draw and the board can be read off the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from pragman.config import PragmanConfig
from pragman.dynamics import apply, apply_moves, slip_distribution, successors, transition
from pragman.state import Action, PragmanState

# Rewards, so that the arithmetic in the assertions is readable.
LIVING = -0.01
PELLET = 0.1
CLEAR = 1.0
GOAL = 1.0
DEATH = -1.0


def world(layout: str, **overrides) -> PragmanConfig:
    """A config with every source of randomness switched off unless asked for."""
    overrides.setdefault("observation", "coords")
    overrides.setdefault("slip_prob", 0.0)
    overrides.setdefault("ghost_epsilon", 0.0)
    overrides.setdefault("max_steps", None)
    return PragmanConfig(layout=layout, **overrides)


# ──────────────────────────────────────────────────────────────────────────────
# Walls block movement, and off the grid is a wall too
# ──────────────────────────────────────────────────────────────────────────────

# Deliberately borderless: the agent starts in the top-left corner with a wall
# to its right and the edge of the world above and to its left.
#
#   row 0:  P#.
#   row 1:  ...
NO_BORDER = "P#.\n..."


@pytest.mark.parametrize(
    "action, expected",
    [
        (Action.UP, (0, 0)),  # off the top edge
        (Action.LEFT, (0, 0)),  # off the left edge
        (Action.RIGHT, (0, 0)),  # into the wall at (0, 1)
        (Action.DOWN, (1, 0)),  # the only way out
    ],
)
def test_walls_and_edges_block_the_agent(action, expected):
    """A blocked move is a no-op, not an error and not a wrap-around."""
    cfg = world(NO_BORDER)
    state = PragmanState((0, 0))

    next_state, reward, terminated = apply(cfg, state, action)
    assert next_state.agent == expected
    assert reward == pytest.approx(LIVING)  # a wasted step still costs the living reward
    assert terminated is False


def test_a_blocked_move_is_blocked_in_the_white_box_model_too():
    """``successors`` must agree with ``apply``: same rule, one implementation."""
    cfg = world(NO_BORDER)
    state = PragmanState((0, 0))
    rows = successors(cfg, state, Action.RIGHT)
    assert len(rows) == 1
    probability, next_state, reward, terminated = rows[0]
    assert (probability, next_state.agent, terminated) == (1.0, (0, 0), False)
    assert reward == pytest.approx(LIVING)


def test_a_ghost_bumps_into_walls_exactly_as_the_agent_does():
    """Event 6: "exactly as the agent's does"."""
    cfg = world("P#.\n..G", n_ghosts=1)
    state = PragmanState((0, 0), ((1, 2),))
    # UP from (1, 2) is (0, 2), which is floor; RIGHT is off the grid.
    assert apply(cfg, state, Action.DOWN, [Action.UP])[0].ghosts == ((0, 2),)
    assert apply(cfg, state, Action.DOWN, [Action.RIGHT])[0].ghosts == ((1, 2),)


# ──────────────────────────────────────────────────────────────────────────────
# Slip frequencies
# ──────────────────────────────────────────────────────────────────────────────

# An open 5x5 room with no goal and no pellets: nothing can end an episode, so
# every sample is a clean draw from the slip distribution.
OPEN_ROOM = "P....\n.....\n.....\n.....\n....."


def test_slip_distribution_is_the_familiar_gridworld():
    """``slip_prob=0.2`` reproduces the 0.8 / 0.1 / 0.1 textbook gridworld."""
    cfg = world(OPEN_ROOM, slip_prob=0.2)
    assert slip_distribution(cfg, Action.UP) == {
        Action.UP: 0.8,
        Action.RIGHT: 0.1,
        Action.LEFT: 0.1,
    }
    # The two perpendicular actions, never the reverse one.
    assert Action.DOWN not in slip_distribution(cfg, Action.UP)
    assert slip_distribution(cfg, Action.RIGHT) == {
        Action.UP: 0.1,
        Action.RIGHT: 0.8,
        Action.DOWN: 0.1,
    }


def test_no_slip_means_no_slip():
    assert slip_distribution(world(OPEN_ROOM), Action.LEFT) == {Action.LEFT: 1.0}


def test_slip_frequencies_match_slip_prob():
    """20 000 samples from the middle of an open room.

    The agent asks to go UP; it should end up above 80% of the time, and to
    either side 10% each. Tolerance is 0.01, comfortably above the ~0.003
    standard error of 20 000 Bernoulli draws.
    """
    cfg = world(OPEN_ROOM, slip_prob=0.2)
    state = PragmanState((2, 2))
    rng = np.random.default_rng(20260923)

    counts = {(1, 2): 0, (2, 3): 0, (2, 1): 0}  # UP, RIGHT, LEFT
    samples = 20_000
    for _ in range(samples):
        next_state, _reward, terminated = transition(cfg, state, Action.UP, rng)
        assert not terminated
        counts[next_state.agent] += 1

    assert counts[(1, 2)] / samples == pytest.approx(0.8, abs=0.01)
    assert counts[(2, 3)] / samples == pytest.approx(0.1, abs=0.01)
    assert counts[(2, 1)] / samples == pytest.approx(0.1, abs=0.01)


def test_a_sideways_slip_into_a_wall_simply_bumps():
    """Slipping is resolved *before* the move, so a slip into a wall is a no-op."""
    cfg = world(
        """\
        ###
        #P#
        #.#
        ###""",
        slip_prob=0.2,
    )
    state = PragmanState((1, 1))
    rows = successors(cfg, state, Action.DOWN)
    # 0.8 goes down; the two 0.1 sideways slips both bump into a wall and merge.
    outcomes = {row[1].agent: row[0] for row in rows}
    assert outcomes == {(2, 1): pytest.approx(0.8), (1, 1): pytest.approx(0.2)}


# ──────────────────────────────────────────────────────────────────────────────
# The three ways a ghost kills you
# ──────────────────────────────────────────────────────────────────────────────

# A three-cell corridor with one ghost spawn.
#
#   col:    01234
#   row 0:  #####
#   row 1:  #P.G#
#   row 2:  #####
GHOST_CORRIDOR = """\
#####
#P.G#
#####"""


def test_death_A_walking_into_a_ghost():
    """Event 3: the agent's new cell holds a ghost.

    Touching a ghost always kills, so a ghost sitting on a pellet or on the goal
    is genuinely dangerous.
    """
    cfg = world(GHOST_CORRIDOR, n_ghosts=1)
    state = PragmanState((1, 2), ((1, 3),))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [Action.LEFT])
    assert terminated is True
    assert reward == pytest.approx(DEATH)  # no living reward on a terminating step
    assert next_state.agent == (1, 3)
    assert next_state.terminated is True


def test_death_B_a_ghost_walking_into_you():
    """Event 7: a ghost's new cell equals the agent's."""
    cfg = world(GHOST_CORRIDOR, n_ghosts=1)
    state = PragmanState((1, 1), ((1, 2),))

    # The agent walks into the wall above it and stays; the ghost steps onto it.
    next_state, reward, terminated = apply(cfg, state, Action.UP, [Action.LEFT])
    assert terminated is True
    assert reward == pytest.approx(DEATH)
    assert next_state.agent == (1, 1) and next_state.ghosts == ((1, 1),)


def test_a_swap_kills_the_agent_and_leaves_the_ghost_standing():
    """Event 7's note: a swap needs no rule of its own.

    The agent steps onto the ghost's cell while the ghost steps onto the
    agent's. Stepping onto a ghost's *pre-move* cell is already death A, which
    short-circuits everything after it -- so the observable consequence is that
    the agent dies where the ghost was and the ghost never moves.
    """
    cfg = world(GHOST_CORRIDOR, n_ghosts=1)
    state = PragmanState((1, 1), ((1, 2),))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [Action.LEFT])
    assert terminated is True
    assert reward == pytest.approx(DEATH)
    assert next_state.agent == (1, 2)
    assert next_state.ghosts == ((1, 2),)  # the ghost stayed where it was


def test_the_agent_cannot_walk_through_a_ghost():
    """The corollary of the swap rule: there is no way past a ghost in a corridor."""
    cfg = world(GHOST_CORRIDOR, n_ghosts=1)
    state = PragmanState((1, 1), ((1, 2),))
    for ghost_action in Action:
        next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [ghost_action])
        assert terminated is True and reward == pytest.approx(DEATH)


# ──────────────────────────────────────────────────────────────────────────────
# Pellets, the clear bonus, the goal, and the one compound reward
# ──────────────────────────────────────────────────────────────────────────────

# Two pellets in a corridor.
#
#   row 1:  #Poo#
PELLET_CORRIDOR = """\
#####
#Poo#
#####"""


def test_eating_a_pellet_removes_it_and_pays():
    """Event 4. Not the last pellet, so the episode continues."""
    cfg = world(PELLET_CORRIDOR)
    state = PragmanState((1, 1), (), frozenset({(1, 2), (1, 3)}))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT)
    assert next_state.pellets == frozenset({(1, 3)})
    assert reward == pytest.approx(PELLET + LIVING)  # 0.09
    assert terminated is False


def test_the_last_pellet_pays_the_clear_bonus_and_ends_the_episode():
    """Event 4's second half: clearing the maze is an achievement."""
    cfg = world(PELLET_CORRIDOR)
    state = PragmanState((1, 2), (), frozenset({(1, 3)}))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT)
    assert next_state.pellets == frozenset()
    assert terminated is True
    assert reward == pytest.approx(PELLET + CLEAR)  # 1.1, and no living reward


def test_walking_over_an_already_eaten_cell_pays_nothing_extra():
    cfg = world(PELLET_CORRIDOR)
    state = PragmanState((1, 2), (), frozenset({(1, 3)}))
    next_state, reward, terminated = apply(cfg, state, Action.LEFT)
    assert reward == pytest.approx(LIVING)
    assert next_state.pellets == frozenset({(1, 3)}) and terminated is False


# A corridor ending in a goal cell.
#
#   row 1:  #P.X#
GOAL_CORRIDOR = """\
#####
#P.X#
#####"""


def test_reaching_the_goal_pays_and_terminates():
    """Event 5. ``goal_reward`` and ``clear_bonus`` are the symmetric pair."""
    cfg = world(GOAL_CORRIDOR)
    state = PragmanState((1, 2))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT)
    assert next_state.agent == (1, 3)
    assert reward == pytest.approx(GOAL)  # no living reward on the terminating step
    assert terminated is True


# The one compound reward left in the whole environment: eat a pellet, then be
# caught by a ghost in the same step.
#
#   col:    012345
#   row 1:  #PoGo#
PELLET_THEN_GHOST = """\
######
#PoGo#
######"""


def test_pellet_then_death_B_is_the_one_compound_reward():
    """Rewards already earned in a step stand, even if the step then kills you.

    The agent steps right onto the pellet at (1, 2) -- which is *not* the last
    one, so the episode does not end there -- and the ghost at (1, 3) steps left
    onto it. The reward is ``pellet_reward + death_reward``, with no living
    reward, because the step terminated.
    """
    cfg = world(PELLET_THEN_GHOST, n_ghosts=1)
    state = PragmanState((1, 1), ((1, 3),), frozenset({(1, 2), (1, 4)}))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [Action.LEFT])
    assert terminated is True
    assert reward == pytest.approx(PELLET + DEATH)  # -0.9
    assert next_state.agent == (1, 2) and next_state.ghosts == ((1, 2),)
    assert next_state.pellets == frozenset({(1, 4)})  # the pellet really was eaten


# ──────────────────────────────────────────────────────────────────────────────
# Termination short-circuits everything after it
# ──────────────────────────────────────────────────────────────────────────────

# A corridor whose middle cell is the goal, with a ghost beyond it.
#
#   row 1:  #PXG#
GOAL_THEN_GHOST = """\
#####
#PXG#
#####"""


def test_no_ghost_moves_after_a_terminating_agent_move():
    """Reaching the goal ends the step; the ghost's chosen move is never played."""
    cfg = world(GOAL_THEN_GHOST, n_ghosts=1)
    state = PragmanState((1, 1), ((1, 3),))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [Action.LEFT])
    assert terminated is True and reward == pytest.approx(GOAL)
    # Had the ghost moved, it would be on (1, 2) -- and would have killed the
    # agent, turning a +1 into a -1. It does not move.
    assert next_state.ghosts == ((1, 3),)


def test_no_ghost_moves_after_the_last_pellet_either():
    """The same short-circuit for the clear bonus."""
    cfg = world("#####\n#PoG#\n#####", n_ghosts=1)
    state = PragmanState((1, 1), ((1, 3),), frozenset({(1, 2)}))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [Action.LEFT])
    assert terminated is True and reward == pytest.approx(PELLET + CLEAR)
    assert next_state.ghosts == ((1, 3),)


def test_no_ghost_moves_after_death_A():
    """Walking into a ghost stops the step, so the other ghost stays too."""
    cfg = world("######\n#PG.G#\n######", n_ghosts=2)
    state = PragmanState((1, 1), ((1, 2), (1, 4)))

    next_state, reward, terminated = apply(cfg, state, Action.RIGHT, [Action.RIGHT, Action.LEFT])
    assert terminated is True and reward == pytest.approx(DEATH)
    assert next_state.ghosts == ((1, 2), (1, 4))  # neither ghost moved


def test_a_terminal_state_is_absorbing_with_zero_reward():
    """No special-casing, so value iteration is a plain contraction."""
    cfg = world(GOAL_CORRIDOR)
    dead_end = PragmanState((1, 3), (), frozenset(), terminated=True)
    rng = np.random.default_rng(0)

    assert apply(cfg, dead_end, Action.LEFT) == (dead_end, 0.0, True)
    assert transition(cfg, dead_end, Action.LEFT, rng) == (dead_end, 0.0, True)
    assert successors(cfg, dead_end, Action.LEFT) == ((1.0, dead_end, 0.0, True),)


# ──────────────────────────────────────────────────────────────────────────────
# The core is pure
# ──────────────────────────────────────────────────────────────────────────────


def test_apply_does_not_mutate_the_state_it_was_given():
    """Everything in ``dynamics`` is a function of ``(cfg, state, ...)``."""
    cfg = world(PELLET_CORRIDOR)
    state = PragmanState((1, 1), (), frozenset({(1, 2), (1, 3)}))
    before = (state.agent, state.ghosts, state.pellets, state.terminated)

    apply(cfg, state, Action.RIGHT)
    assert (state.agent, state.ghosts, state.pellets, state.terminated) == before


def test_apply_insists_on_one_move_per_ghost():
    """Ghost moves are index-aligned with ``state.ghosts``.

    Forgetting a ghost is an error rather than a ghost that quietly stands
    still, which would be a very hard bug to see in a rollout.
    """
    cfg = world(GHOST_CORRIDOR, n_ghosts=1)
    state = PragmanState((1, 1), ((1, 3),))
    with pytest.raises(ValueError) as excinfo:
        apply(cfg, state, Action.RIGHT, [])
    assert "index-aligned" in str(excinfo.value)

    with pytest.raises(ValueError):
        apply_moves(cfg, state, (1, 2), ((1, 3), (1, 3)))
