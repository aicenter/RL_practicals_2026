"""Gymnasium conformance.

These are the tests that say "Prag-man is an ordinary Gymnasium environment".
Everything here is derived from the Gymnasium API and the ten ready-made
worlds; nothing here knows how the dynamics are implemented.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

import pragman

#: All ten ready-made worlds.
FACTORIES = pragman.configs.names()

#: The two worlds with no randomness anywhere: no slip, and either no ghosts at
#: all (``empty_5x5``) or ghosts whose every knob is switched off
#: (``planning_11x11``: ghost_epsilon=0 and chasers that break ties by lowest
#: action index).  Different seeds cannot make these two behave differently, so
#: the divergence half of the seeding test does not apply to them.
DETERMINISTIC = ("empty_5x5", "planning_11x11")
STOCHASTIC = tuple(name for name in FACTORIES if name not in DETERMINISTIC)


def rollout(env, actions):
    """Play a fixed action sequence and return a comparable transcript."""
    transcript = []
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(int(action))
        transcript.append(
            (np.asarray(observation).tobytes(), reward, terminated, truncated, info["state"])
        )
        if terminated or truncated:
            break
    return transcript


def fixed_actions(n=40, seed=0):
    """A fixed action sequence -- the *agent* is never the source of randomness here."""
    return np.random.default_rng(seed).integers(0, 4, size=n)


# ── the environment checker ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", FACTORIES)
def test_check_env_passes_for_every_factory(name):
    """``check_env`` passes for every world.

    Built with ``render_mode=None`` on purpose, so that the checker never tries
    the ``"human"`` mode and pygame (an optional extra) is never imported.
    """
    check_env(pragman.make(name, render_mode=None))


def test_action_space_is_four_directions():
    """``Discrete(4)`` -- UP, RIGHT, DOWN, LEFT, and no STAY."""
    env = pragman.make("empty_5x5")
    assert env.action_space == gym.spaces.Discrete(4)
    assert [int(a) for a in pragman.Action] == [0, 1, 2, 3]


def test_metadata_advertises_the_three_render_modes():
    """The metadata a Gymnasium user reads before rendering."""
    env = pragman.make("empty_5x5")
    assert env.metadata["render_modes"] == ["human", "rgb_array", "ansi"]
    assert env.metadata["render_fps"] == 8


# ── observations lie in the declared space ────────────────────────────────────


@pytest.mark.parametrize("name", FACTORIES)
def test_every_observation_lies_in_the_observation_space(name):
    """Over a random rollout, in every world, on reset and on step."""
    env = pragman.make(name)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation), f"{name}: reset observation escaped"

    for action in fixed_actions(n=60, seed=1):
        observation, _reward, terminated, truncated, info = env.step(int(action))
        assert env.observation_space.contains(observation), f"{name}: step observation escaped"
        assert isinstance(info["state"], pragman.PragmanState)
        if terminated or truncated:
            observation, info = env.reset()
            assert env.observation_space.contains(observation)


# ── seeding ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", FACTORIES)
def test_same_seed_reproduces_the_trajectory(name):
    """``reset(seed=k)`` twice, same actions, same trajectory."""
    actions = fixed_actions(seed=2)
    first = pragman.make(name)
    first.reset(seed=123)
    second = pragman.make(name)
    second.reset(seed=123)
    assert rollout(first, actions) == rollout(second, actions)


@pytest.mark.parametrize("name", STOCHASTIC)
def test_different_seeds_eventually_diverge(name):
    """Only the *stochastic* worlds may differ by seed.

    ``empty_5x5`` and ``planning_11x11`` are excluded because they have no
    randomness at all -- for them, identical behaviour under every seed is the
    correct answer, not a seeding bug.
    """
    actions = fixed_actions(seed=3)
    reference = pragman.make(name)
    reference.reset(seed=0)
    baseline = rollout(reference, actions)

    def differs(seed):
        env = pragman.make(name)
        env.reset(seed=seed)
        return rollout(env, actions) != baseline

    assert any(differs(seed) for seed in range(1, 8)), f"{name} ignores its seed"


# ── registration ──────────────────────────────────────────────────────────────


def test_importing_pragman_registers_every_id():
    """Registration is an import side effect, so ids exist now.

    Gymnasium 1.0 removed the entry-point plugin mechanism, which is why
    ``import pragman`` has to come before ``gym.make("Pragman/...")``.
    """
    assert set(pragman.ENV_IDS) == set(FACTORIES)
    for name, identifier in pragman.ENV_IDS.items():
        assert identifier in gym.registry, f"{name} is not registered as {identifier}"


@pytest.mark.parametrize("name", FACTORIES)
def test_gym_make_matches_the_factory(name):
    """The id builds the same world the factory does.

    Two other properties worth pinning here.  ``gym.make`` returns the
    *wrapped* environment, so the config is reached through ``.unwrapped``,
    while ``pragman.make`` hands back the bare :class:`PragmanEnv`.  And every
    id registers ``max_episode_steps=None``: Prag-man owns its own clock, and a
    ``TimeLimit`` wrapper on top would double-apply it.
    """
    wrapped = gym.make(pragman.ENV_IDS[name])
    assert wrapped.spec.max_episode_steps is None
    assert wrapped.unwrapped.cfg == pragman.configs.by_name(name)()
    assert isinstance(wrapped.unwrapped, pragman.PragmanEnv)
    assert isinstance(pragman.make(name), pragman.PragmanEnv)
    assert type(pragman.make(name)) is pragman.PragmanEnv  # no wrappers at all


def test_gym_make_accepts_config_overrides():
    """Every world ships with every knob, through either door."""
    wrapped = gym.make(pragman.ENV_IDS["maze_8x8"], slip_prob=0.3)
    assert wrapped.unwrapped.cfg.slip_prob == 0.3
    assert pragman.make("maze_8x8", slip_prob=0.3).cfg.slip_prob == 0.3


# ── vectorization ─────────────────────────────────────────────────────────────


def test_make_vec_shapes_and_pinned_autoreset_mode():
    """``make_vec`` pins next-step autoreset, and says so."""
    envs = pragman.make_vec("empty_5x5", 3)
    assert envs.num_envs == 3
    assert envs.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP

    observations, infos = envs.reset(seed=0)
    assert observations.shape == (3,)  # Discrete observations, one per copy
    assert envs.single_observation_space == pragman.make("empty_5x5").observation_space

    observations, rewards, terminations, truncations, infos = envs.step(np.array([1, 1, 1]))
    assert rewards.shape == terminations.shape == truncations.shape == (3,)
    envs.close()


def test_make_vec_autoresets_on_the_following_step():
    """The trap Week 9's rollout buffer has to know about.

    Under ``AutoresetMode.NEXT_STEP`` the step that ends an episode returns the
    *final* observation and reward; the step after it ignores the action, resets
    the copy, and returns the *first* observation of the new episode with reward
    0 and both flags False.  Walking into the left wall of ``empty_5x5`` can
    only ever run out the clock, so truncation here is exact.
    """
    envs = pragman.make_vec("empty_5x5", 2, max_steps=4)
    envs.reset(seed=1)
    left = np.array([3, 3])

    for step in range(1, 4):
        _obs, rewards, terminations, truncations, _infos = envs.step(left)
        assert not terminations.any() and not truncations.any(), f"ended early at step {step}"
        assert rewards.tolist() == [-0.01, -0.01]  # the living reward

    _obs, rewards, terminations, truncations, infos = envs.step(left)
    assert truncations.tolist() == [True, True]
    assert terminations.tolist() == [False, False]

    _obs, rewards, terminations, truncations, infos = envs.step(left)
    assert rewards.tolist() == [0.0, 0.0]  # the reset step pays nothing
    assert not terminations.any() and not truncations.any()
    assert infos["state"][0].agent == (0, 0)  # back at the start
    envs.close()


def test_make_vec_reports_the_true_state_as_an_object_array():
    """``info["state"]`` survives vectorization as an object array."""
    envs = pragman.make_vec("haunted_5x5", 2)
    _observations, infos = envs.reset(seed=0)
    states = infos["state"]
    assert isinstance(states, np.ndarray) and states.dtype == object
    assert all(isinstance(state, pragman.PragmanState) for state in states)

    _obs, _rewards, _term, _trunc, infos = envs.step(np.array([0, 1]))
    assert isinstance(infos["state"][0], pragman.PragmanState)
    envs.close()
