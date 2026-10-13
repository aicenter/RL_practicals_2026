"""The environment layer, plus the wrapper.

What is pinned here: the episode clock (truncation, and the refusal to step an
episode that has ended), the state-access API (clone / restore / injection), the
convenience attributes, and :class:`pragman.ReplaceReward`,
which is the mechanism behind Weeks 11-13.
"""

from __future__ import annotations

import numpy as np
import pytest

import pragman
from pragman import Action, PragmanState
from pragman.errors import ConfigError, StateInjectionDisabled

# ``empty_5x5`` is an open room: the agent starts at (0, 0), the goal is the far
# corner (4, 4), there are no ghosts, no pellets and no slipping.
#
#     P . . . .
#     . . . . .
#     . . . . .
#     . . . . .
#     . . . . X
#
# So DOWN x4 then RIGHT x4 is the shortest path, and its last step is the only
# one that terminates.
TO_THE_GOAL = [Action.DOWN] * 4 + [Action.RIGHT] * 4


# ── the clock: truncation and termination ─────────────────────────────────────


def test_truncates_at_exactly_max_steps():
    """Truncation on step ``max_steps``, not before, not after.

    Walking LEFT into the wall from (0, 0) is a no-op, so nothing but the clock
    can end this episode.
    """
    env = pragman.make("empty_5x5", max_steps=5)
    env.reset(seed=0)

    for step in range(1, 5):
        _obs, _reward, terminated, truncated, _info = env.step(Action.LEFT)
        assert (terminated, truncated) == (False, False), f"ended early on step {step}"
        assert env.elapsed_steps == step

    _obs, _reward, terminated, truncated, _info = env.step(Action.LEFT)
    assert (terminated, truncated) == (False, True)
    assert env.elapsed_steps == 5


def test_max_steps_none_never_truncates():
    """``max_steps=None`` means the clock never runs out."""
    env = pragman.make("empty_5x5", max_steps=None)
    env.reset(seed=0)
    for _ in range(250):
        _obs, _reward, terminated, truncated, _info = env.step(Action.LEFT)
        assert not terminated and not truncated
    assert env.elapsed_steps == 250


def test_terminated_and_truncated_are_never_both_true():
    """The two flags are exclusive, however the episode ends.

    ``classic_11x11`` has ghosts, pellets, a clear bonus and a 300-step clock,
    so a long random walk meets every ending the environment has.
    """
    env = pragman.make("classic_11x11")
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    endings = {"terminated": 0, "truncated": 0}
    for _ in range(3000):
        _obs, _reward, terminated, truncated, _info = env.step(int(rng.integers(4)))
        assert not (terminated and truncated)
        if terminated or truncated:
            endings["terminated" if terminated else "truncated"] += 1
            env.reset()
    assert endings["terminated"] and endings["truncated"], "expected to see both endings"


def test_no_living_reward_on_a_terminating_step():
    """The living reward is paid only if the step survives.

    Reaching the goal therefore pays exactly ``goal_reward``, with no -0.01
    deducted, and each earlier step pays exactly the living reward.
    """
    env = pragman.make("empty_5x5")
    env.reset(seed=0)
    rewards = [env.step(action)[1] for action in TO_THE_GOAL]
    assert rewards[:-1] == [-0.01] * 7
    assert rewards[-1] == pytest.approx(1.0)  # goal_reward, undiminished


def test_step_after_the_episode_has_ended_raises():
    """Stepping a finished episode raises rather than returning nonsense."""
    env = pragman.make("empty_5x5")
    env.reset(seed=0)
    for action in TO_THE_GOAL:
        _obs, _reward, terminated, _truncated, _info = env.step(action)
    assert terminated

    with pytest.raises(RuntimeError, match="episode has ended; call reset"):
        env.step(Action.UP)

    # The same applies after truncation.  Strictly, only termination would need
    # it, but Gymnasium requires a reset either way, and a silently
    # continuing episode would be the harder bug to find.
    env = pragman.make("empty_5x5", max_steps=2)
    env.reset(seed=0)
    env.step(Action.LEFT)
    env.step(Action.LEFT)
    with pytest.raises(RuntimeError, match="episode has ended; call reset"):
        env.step(Action.LEFT)


def test_reset_restarts_the_clock():
    env = pragman.make("empty_5x5")
    env.reset(seed=0)
    env.step(Action.DOWN)
    assert env.elapsed_steps == 1
    env.reset(seed=0)
    assert env.elapsed_steps == 0


# ── what info carries ─────────────────────────────────────────────────────────


def test_info_carries_the_true_state_and_its_three_flags():
    env = pragman.make("pellets_8x8")
    _obs, info = env.reset(seed=0)
    assert set(info) == {"state", "pellets_left", "death", "slipped"}
    assert info["state"].pellets_left == info["pellets_left"] > 0
    assert info["death"] is False and info["slipped"] is False

    _obs, _reward, _terminated, _truncated, info = env.step(Action.UP)
    assert isinstance(info["state"], PragmanState)
    assert info["pellets_left"] == info["state"].pellets_left


def test_walking_into_a_ghost_is_death_and_says_so():
    """The agent moves first, so this kills every time.

    The position below is hand-written: the agent stands to
    the left of the ghost, and steps RIGHT onto it.
    """
    env = pragman.make("haunted_5x5", observation="coords")
    env.reset(
        seed=0,
        options={
            "state": [
                "....X",
                ".....",
                "..#..",
                ".....",
                ".PG..",
            ]
        },
    )
    _obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert terminated and not truncated
    assert reward == pytest.approx(-1.0)  # death_reward, with no living reward on top
    assert info["death"] is True
    assert info["state"].agent == (4, 2) == info["state"].ghosts[0]


def test_slipped_flag_reports_that_the_world_moved_you_elsewhere():
    """``info["slipped"]`` is the flag a stuck student finds first.

    With ``slip_prob=1`` the chosen action is *always* replaced by one of the
    two perpendicular ones, so every step reports a slip.
    """
    env = pragman.make("empty_5x5", slip_prob=1.0)
    env.reset(seed=0)
    for _ in range(5):
        _obs, _reward, _terminated, _truncated, info = env.step(Action.DOWN)
        assert info["slipped"] is True

    env = pragman.make("empty_5x5")  # slip_prob = 0 by default
    env.reset(seed=0)
    assert env.step(Action.DOWN)[4]["slipped"] is False


# ── the convenience attributes ────────────────────────────────────────────────


def test_the_attributes_a_student_reaches_for():
    """Students hold an ``env``, not a ``cfg``."""
    env = pragman.make("empty_5x5")
    assert env.cfg == pragman.configs.empty_5x5()
    assert env.layout is env.cfg.layout
    assert env.elapsed_steps == 0

    _obs, info = env.reset(seed=0)
    assert env.state == info["state"]
    # env.successors is the white-box model for one state, and agrees with the
    # pure function it delegates to.
    assert env.successors(info["state"], Action.DOWN) == pragman.successors(
        env.cfg, info["state"], Action.DOWN
    )
    # env.model is the whole (P, R, rho0) table, built once and cached.
    assert env.model is env.model
    assert env.model.P.shape == (25, 4, 25)


# ── clone / restore ───────────────────────────────────────────────────────────


def test_clone_then_restore_replays_the_same_trajectory():
    """Save the state, play on, restore, and replay identically.

    This holds on a *deterministic* world only, and deliberately so: **the
    random number generator is not part of the state**, so
    restoring does not rewind the random stream.  ``planning_11x11`` has no
    randomness at all -- no slipping, ghost_epsilon = 0, and chasers that break
    ties by lowest action index -- which is exactly why Week 10 plans in it.
    """
    env = pragman.make("planning_11x11")
    env.reset(seed=0)
    for action in (Action.LEFT, Action.LEFT, Action.UP):
        env.step(action)

    saved = env.clone_state()
    continuation = [Action.UP, Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

    def play():
        """Play the continuation and write down everything that happened."""
        transcript = []
        for action in continuation:
            _obs, reward, terminated, truncated, info = env.step(action)
            transcript.append((reward, terminated, truncated, info["state"]))
            if terminated or truncated:
                break  # the chasers do catch the agent; that is part of the replay
        return transcript

    first = play()
    assert len(first) >= 4, "the replay is only interesting if several steps happened"
    assert env.restore_state(saved) == saved
    assert env.clone_state() == saved
    assert play() == first


def test_restore_state_takes_the_clock_explicitly():
    """The clock is the environment's, so restoring says what it is.

    It defaults to 0 because the usual caller is a planner asking "suppose the
    world were like this", which wants a fresh horizon.
    """
    env = pragman.make("empty_5x5", max_steps=10)
    env.reset(seed=0)
    env.step(Action.DOWN)
    saved = env.clone_state()

    env.restore_state(saved)
    assert env.elapsed_steps == 0

    env.restore_state(saved, elapsed_steps=9)
    assert env.elapsed_steps == 9
    _obs, _reward, terminated, truncated, _info = env.step(Action.LEFT)
    assert (terminated, truncated) == (False, True)  # the tenth step of the clock


# ── state injection through reset ─────────────────────────────────────────────


def test_reset_with_a_state_yields_exactly_that_state_s_observation():
    """For both a state object and the hand-written position form."""
    env = pragman.make("haunted_5x5")
    state = PragmanState(agent=(1, 1), ghosts=((3, 3),), pellets=frozenset())

    observation, info = env.reset(options={"state": state})
    assert info["state"] == state
    assert observation == pragman.observe(state, env.cfg)
    assert env.elapsed_steps == 0

    drawn, info = env.reset(
        options={
            "state": [
                "....X",
                ".P...",
                "..#..",
                "...G.",
                ".....",
            ]
        }
    )
    assert info["state"] == state  # the same position, drawn instead of constructed
    assert drawn == observation


def test_injecting_an_impossible_state_under_index_observations_raises():
    """A state with no tabular index fails with an explanation.

    ``haunted_5x5`` is observed as an index, so every state must be one of the
    enumerated ones; here the agent has been placed inside the central wall.
    """
    env = pragman.make("haunted_5x5")  # observation="index"
    walled_in = PragmanState(agent=(2, 2), ghosts=((4, 2),), pellets=frozenset())
    with pytest.raises(ValueError, match="not reachable"):
        env.reset(options={"state": walled_in})


def test_injection_is_refused_when_the_config_forbids_it():
    """Both doors are locked on the model-free worlds.

    ``StateInjectionDisabled`` is a ``ConfigError`` -- not a ``PermissionError``,
    which would suggest the operating system had an opinion -- and the message
    says which flag to flip.
    """
    env = pragman.make("classic_11x11")  # allow_state_injection=False
    _obs, info = env.reset(seed=0)
    state = info["state"]

    assert issubclass(StateInjectionDisabled, ConfigError)
    with pytest.raises(StateInjectionDisabled, match="allow_state_injection"):
        env.reset(options={"state": state})
    with pytest.raises(StateInjectionDisabled, match="allow_state_injection"):
        env.restore_state(state)

    # The flag is a signpost, not a sandbox: the same world, built with the flag
    # on, injects happily.
    permissive = pragman.make("classic_11x11", allow_state_injection=True)
    permissive.reset(seed=0)
    assert permissive.restore_state(state) == state


# ── ReplaceReward ─────────────────────────────────────────────────────────────


def constant_reward(state, action, next_state, info):
    """A reward function that ignores the world entirely."""
    return 42.0


def test_replace_reward_pays_its_own_reward_and_keeps_the_true_one():
    """The wrapper swaps the number, and files the original in ``info``.

    Keeping the environment's own reward in ``info["true_reward"]`` lets a
    training run log the true return alongside the one the agent is optimizing.
    """
    env = pragman.ReplaceReward(pragman.make("empty_5x5"), constant_reward)
    _obs, info = env.reset(seed=0)
    assert "true_reward" not in info  # nothing has been paid yet

    _obs, reward, _terminated, _truncated, info = env.step(Action.DOWN)
    assert reward == 42.0
    assert info["true_reward"] == pytest.approx(-0.01)

    for action in TO_THE_GOAL[1:]:
        _obs, reward, terminated, _truncated, info = env.step(action)
    assert terminated
    assert reward == 42.0
    assert info["true_reward"] == pytest.approx(1.0)  # the goal, under the true reward


def test_replace_reward_leaves_the_dynamics_alone():
    """"Only the number changes": same seed, same actions, same states and endings."""
    actions = np.random.default_rng(0).integers(0, 4, size=40)

    plain = pragman.make("haunted_5x5")
    plain.reset(seed=5)
    wrapped = pragman.ReplaceReward(pragman.make("haunted_5x5"), constant_reward)
    wrapped.reset(seed=5)

    for action in actions:
        _o, _r, terminated, truncated, plain_info = plain.step(int(action))
        _o, reward, wrapped_terminated, wrapped_truncated, wrapped_info = wrapped.step(int(action))
        assert wrapped_info["state"] == plain_info["state"]
        assert (wrapped_terminated, wrapped_truncated) == (terminated, truncated)
        assert reward == 42.0 and wrapped_info["true_reward"] == _r
        if terminated or truncated:
            break


def test_replace_reward_sees_the_transition_it_is_scoring():
    """The reward function is handed ``(state, action, next_state, info)``."""
    seen = []

    def record(state, action, next_state, info):
        seen.append((state.agent, int(action), next_state.agent))
        return 0.0

    env = pragman.ReplaceReward(pragman.make("empty_5x5"), record)
    env.reset(seed=0)
    env.step(Action.DOWN)
    env.step(Action.RIGHT)
    assert seen == [((0, 0), Action.DOWN, (1, 0)), ((1, 0), Action.RIGHT, (1, 1))]


def test_replace_reward_does_not_forward_attributes():
    """Gymnasium 1.x wrappers do not proxy attributes -- so ``.unwrapped`` it is.

    Pinned because it is the first thing that surprises somebody who wraps a
    Prag-man environment and then reaches for ``env.cfg``.
    """
    env = pragman.ReplaceReward(pragman.make("empty_5x5"), constant_reward)
    with pytest.raises(AttributeError):
        _ = env.cfg
    assert env.unwrapped.cfg == pragman.configs.empty_5x5()
    assert env.unwrapped.layout is env.unwrapped.cfg.layout
