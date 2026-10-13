"""What the agent sees.

The four encodings, the channel-last layout, the egocentric window, and the
promise that the pure ``observe(state, cfg)`` gives exactly what ``env.step``
gave for that same state.
"""

from __future__ import annotations

import numpy as np
import pytest

import pragman
from pragman import PragmanConfig, PragmanState
from pragman.errors import StateSpaceTooLarge

# A board small enough to read.  Rows are printed top to bottom, so the agent
# starts at (0, 0), the pellet sits at (0, 2), the only wall is (1, 1), the
# ghost spawns at (1, 2) and the goal is the far corner (2, 2).
#
#     P . o
#     . # G
#     . . X
TINY = ["P.o", ".#G", "..X"]

# The same room with no wall and no interior structure, used for the crops.
#
#     P . o
#     . . .
#     G . X
OPEN = ["P.o", "...", "G.X"]


def tiny_config(**overrides):
    settings = {"observation": "channels", **overrides}
    return PragmanConfig(layout=TINY, n_ghosts=1, **settings)


def planes(observation, names, wanted):
    """The plane called *wanted*, looked up by name rather than by index."""
    return observation[:, :, names.index(wanted)]


# ── channels: the hand-written expectation ────────────────────────────────────


def test_channel_planes_match_a_hand_written_board():
    """Binary planes on a 3x3 fixture, channel-last ``(H, W, C)``.

    Every plane is written out by hand below, so a reader can check the encoding
    against the drawing at the top of this file without running anything.
    """
    cfg = tiny_config()
    state = PragmanState(agent=(0, 0), ghosts=((1, 2),), pellets=frozenset({(0, 2)}))

    names = pragman.channel_names(cfg)
    assert names == ("walls", "agent", "ghosts", "pellets", "goal")

    expected = {
        "walls": [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]],
        "agent": [[1, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]],
        "ghosts": [[0, 0, 0],
                   [0, 0, 1],
                   [0, 0, 0]],
        "pellets": [[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]],
        "goal": [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1]],
    }

    observation = pragman.observe(state, cfg)
    assert observation.shape == (3, 3, 5)  # (H, W, C): features last
    assert observation.dtype == np.uint8
    for name, plane in expected.items():
        assert np.array_equal(planes(observation, names, name), np.array(plane)), name
    # ... and nothing else is in the array.
    assert np.array_equal(observation, np.stack([np.array(expected[n]) for n in names], axis=-1))


def test_the_goal_plane_only_exists_when_the_layout_has_a_goal():
    """``C`` is a property of the config, so skeletons read it off.

    ``pellets_8x8`` is a pellet world with no goal cell, so it has four planes,
    while the 3x3 fixture above has five.
    """
    with_goal = tiny_config()
    assert pragman.channel_names(with_goal)[-1] == "goal"
    assert pragman.observation_space(with_goal).shape[-1] == 5

    without_goal = pragman.make("pellets_8x8")
    assert without_goal.channel_names == ("walls", "agent", "ghosts", "pellets")
    assert without_goal.observation_space.shape[-1] == 4
    assert len(without_goal.channel_names) == without_goal.observation_space.shape[-1]


def test_ghost_free_worlds_keep_an_empty_ghost_plane():
    """The four basic planes are always there.

    A network written for a haunted world therefore still accepts observations
    from a ghost-free one -- the ghost plane is simply all zeros.
    """
    cfg = PragmanConfig(layout=TINY, n_ghosts=0, observation="channels")
    names = pragman.channel_names(cfg)
    assert "ghosts" in names
    observation = pragman.observe(PragmanState(agent=(0, 0)), cfg)
    assert not planes(observation, names, "ghosts").any()


def test_mixed_personalities_get_one_plane_each():
    """Ghosts that behave differently must look different.

    Without this, ``ghost_policy="mix"`` would quietly become a
    partial-observability problem instead of the harder-dynamics problem it is
    meant to be.  ``mix`` assigns random, chaser, proximity cyclically in
    row-major spawn order, and each ghost is drawn on the plane belonging to its
    own personality.
    """
    env = pragman.make("classic_11x11", ghost_policy="mix", n_ghosts=3)
    names = env.channel_names
    assert names == ("walls", "agent", "ghosts_random", "ghosts_chaser",
                     "ghosts_proximity", "pellets")
    assert env.cfg.ghost_policy_names == ("random", "chaser", "proximity")

    observation, info = env.reset(seed=0)
    for ghost, personality in zip(info["state"].ghosts, env.cfg.ghost_policy_names, strict=True):
        plane = planes(observation, names, f"ghosts_{personality}")
        assert plane[ghost] == 1, f"the {personality} ghost is missing from its own plane"
        assert plane.sum() == 1  # and it is alone there

    # A homogeneous world shares one plane: all its ghosts behave alike.
    assert pragman.make("classic_11x11").channel_names.count("ghosts") == 1


# ── the egocentric window ─────────────────────────────────────────────────────


def test_egocentric_crop_at_a_corner_is_padded_with_wall():
    """At a corner, half the window is off the board -- and is wall.

    With ``view_radius=1`` the agent at (0, 0) sees the 3x3 window spanning rows
    -1..1 and columns -1..1.  The row and the column beyond the edge come back
    as wall, which is also what ``Layout.is_wall`` says about out-of-bounds
    cells, and the window is *not* rotated: the agent has no heading.
    """
    cfg = PragmanConfig(layout=OPEN, n_ghosts=1, observation="channels", view_radius=1)
    state = PragmanState(agent=(0, 0), ghosts=((2, 0),), pellets=frozenset({(0, 2)}))
    names = pragman.channel_names(cfg)

    observation = pragman.observe(state, cfg)
    assert observation.shape == (3, 3, len(names))

    assert np.array_equal(
        planes(observation, names, "walls"),
        np.array([[1, 1, 1],    # off the top of the board
                  [1, 0, 0],    # off the left edge, then floor (0,0) and (0,1)
                  [1, 0, 0]]),  # off the left edge, then floor (1,0) and (1,1)
    )
    assert np.array_equal(
        planes(observation, names, "agent"),
        np.array([[0, 0, 0],
                  [0, 1, 0],    # the agent is always at the centre of its window
                  [0, 0, 0]]),
    )
    # The pellet at (0, 2) and the ghost at (2, 0) are outside the window, so the
    # agent cannot see them.  That is the Week 13 hook, not a bug.
    assert not planes(observation, names, "pellets").any()
    assert not planes(observation, names, "ghosts").any()
    assert not planes(observation, names, "goal").any()


def test_a_window_that_covers_the_board_is_the_board():
    """Centred on the middle cell, a radius-1 window on a 3x3 board
    is exactly the uncropped encoding."""
    state = PragmanState(agent=(1, 1), ghosts=((2, 0),), pellets=frozenset({(0, 2)}))
    full = pragman.observe(
        state, PragmanConfig(layout=OPEN, n_ghosts=1, observation="channels")
    )
    cropped = pragman.observe(
        state, PragmanConfig(layout=OPEN, n_ghosts=1, observation="channels", view_radius=1)
    )
    assert np.array_equal(cropped, full)


def test_view_radius_shrinks_the_declared_space_too():
    """``view_radius=r`` gives a ``(2r+1, 2r+1)`` observation.

    ``partial_11x11`` is the shipped example: an 11x11 maze seen through a 5x5
    window.
    """
    env = pragman.make("partial_11x11")
    assert env.cfg.view_radius == 2
    assert env.observation_space.shape[:2] == (5, 5)
    observation, _info = env.reset(seed=0)
    assert observation.shape[:2] == (5, 5)


# ── the two non-array encodings ───────────────────────────────────────────────


def test_coords_are_the_positions_and_nothing_else():
    """Agent row and column, then each ghost's, as int32.

    Deliberately **not Markov** when pellets are on the board: two positions
    that differ only in which pellets remain encode identically.  It is the
    cheapest way to show a class what "the observation is not the state" costs.
    """
    cfg = tiny_config(observation="coords")
    state = PragmanState(agent=(0, 0), ghosts=((1, 2),), pellets=frozenset({(0, 2)}))

    observation = pragman.observe(state, cfg)
    assert observation.dtype == np.int32
    assert observation.tolist() == [0, 0, 1, 2]
    assert pragman.observation_space(cfg).shape == (4,)  # 2 + 2 * n_ghosts

    eaten = PragmanState(agent=(0, 0), ghosts=((1, 2),), pellets=frozenset())
    assert np.array_equal(pragman.observe(eaten, cfg), observation)


def test_index_is_one_integer_per_reachable_state():
    """``Discrete(S)``, distinct states get distinct indices."""
    env = pragman.make("empty_5x5")  # observation="index"
    assert env.observation_space.n == 25

    # Twenty-four cells the agent can stand on and go on playing, plus the goal
    # cell, which exists only as the terminal state that ends the episode.
    def state_at(cell):
        return PragmanState(agent=cell, terminated=cell == (4, 4))

    seen = {pragman.observe(state_at((row, col)), env.cfg) for row in range(5)
            for col in range(5)}
    assert seen == set(range(25))


def test_index_refuses_a_state_this_world_cannot_reach():
    """A hand-written state that does not fit gets an explanation."""
    cfg = pragman.configs.haunted_5x5()
    with pytest.raises(ValueError, match="not reachable"):
        pragman.observe(PragmanState(agent=(2, 2), ghosts=((4, 2),)), cfg)


def test_index_refuses_a_world_too_big_to_enumerate():
    """Asking for a table you could not build says so loudly.

    Twenty-nine pellets mean 2**29 pellet configurations, which is the whole
    argument for function approximation in Week 6 -- demonstrated rather than
    asserted.
    """
    with pytest.raises(StateSpaceTooLarge):
        pragman.make("pellets_8x8", observation="index")


# ── observe() agrees with the environment ─────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["empty_5x5", "haunted_5x5", "classic_11x11", "pixels_11x11", "partial_11x11"]
)
def test_observe_reproduces_what_the_environment_returned(name):
    """The pure function and the live environment cannot drift apart.

    Week 12's inverse-RL lab encodes states that were never visited in an
    environment, and Week 11 re-encodes recorded episodes; both call
    ``observe(state, cfg)`` directly, so it must give exactly what ``step``
    gave.
    """
    env = pragman.make(name)
    observation, info = env.reset(seed=0)
    assert np.array_equal(observation, pragman.observe(info["state"], env.cfg))

    rng = np.random.default_rng(0)
    for _ in range(40):
        observation, _reward, terminated, truncated, info = env.step(int(rng.integers(4)))
        assert np.array_equal(observation, pragman.observe(info["state"], env.cfg))
        if terminated or truncated:
            observation, info = env.reset()
            assert np.array_equal(observation, pragman.observe(info["state"], env.cfg))


def test_observe_is_pure():
    """Same state in, same array out, and no shared buffer."""
    cfg = tiny_config()
    state = PragmanState(agent=(0, 0), ghosts=((1, 2),), pellets=frozenset({(0, 2)}))
    first, second = pragman.observe(state, cfg), pragman.observe(state, cfg)
    assert np.array_equal(first, second)
    first[0, 0, 0] = 9  # scribbling on one observation must not affect the next
    assert not np.array_equal(first, pragman.observe(state, cfg))
    assert np.array_equal(second, pragman.observe(state, cfg))


def test_channels_and_pixels_agree_on_where_the_agent_is():
    """The two array encodings differ only in how a cell is drawn.

    Same axis order, same board, so swapping one for the other is a one-field
    config change -- which is the Week 7 demonstration.
    """
    channels = pragman.make("classic_11x11")
    pixels = pragman.make("pixels_11x11")
    assert channels.observation_space.shape[:2] == (11, 11)
    assert pixels.observation_space.shape == (11 * 8, 11 * 8, 3)
    assert channels.cfg.cell_pixels == 8


# ──────────────────────────────────────────────────────────────────────────────
# The "index" observation and the tabular model must agree on what state 7 is
# ──────────────────────────────────────────────────────────────────────────────


def test_index_observation_matches_the_tabular_model_index():
    """``observe(state, cfg)`` returns exactly ``model.index[state]``.

    Every tabular lab leans on this: a student fills ``Q[obs, action]`` from the
    environment and then compares it against ``model.R`` or a DP solution
    indexed the same way. The two enumerations are computed separately (the
    encoder caches its own), so nothing but determinism keeps them aligned --
    which is precisely why it is worth pinning.
    """
    import pragman
    from pragman.model import build_tabular_model

    cfg = pragman.configs.haunted_5x5()
    model = build_tabular_model(cfg)
    env = pragman.PragmanEnv(cfg)

    obs, info = env.reset(seed=0)
    assert obs == model.index[info["state"]]

    for _ in range(50):
        obs, _reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs == model.index[info["state"]], "encoder and model disagree on a state's index"
        if terminated or truncated:
            obs, info = env.reset()
            assert obs == model.index[info["state"]]
