"""The ten ready-made worlds -- the factory table of :mod:`pragman.configs`.

Each row of that table is a promise made to a particular week of the course, so
each column is asserted here: the observation encoding, how many ghosts and of
which personality, whether state injection is allowed, and the episode clock.
When a factory drifts, a lab skeleton breaks -- and this is where it shows.
"""

from __future__ import annotations

import numpy as np
import pytest

import pragman
from pragman import Action, PragmanConfig, PragmanState
from pragman.errors import ConfigError

# The course map, transcribed.  ``max_steps`` is 300 for every world built on
# the 11x11 map ("as for every world on this map") and the config default of 100
# elsewhere; ``allow_state_injection`` is off in the three model-free worlds.
#
#   name              observation  ghosts  personalities                injection  max_steps
FACTORY_TABLE = [
    ("empty_5x5",     "index",     0,      (),                          True,      100),
    ("slippery_5x5",  "index",     0,      (),                          True,      100),
    ("maze_8x8",      "index",     0,      (),                          True,      100),
    ("haunted_5x5",   "index",     1,      ("random",),                 True,      100),
    ("pellets_8x8",   "channels",  1,      ("random",),                 False,     100),
    ("classic_11x11", "channels",  2,      ("random", "random"),        False,     300),
    ("pixels_11x11",  "pixels",    2,      ("random", "random"),        False,     300),
    ("planning_11x11", "channels", 2,      ("chaser", "chaser"),        True,      300),
    ("partial_11x11", "channels",  2,      ("proximity", "proximity"),  True,      300),
    ("single_ghost_11x11", "channels", 1,  ("random",),                 True,      300),
]


def test_exactly_the_ten_worlds_of_the_course_map_exist():
    assert list(pragman.configs.names()) == [row[0] for row in FACTORY_TABLE]


@pytest.mark.parametrize(
    "name, observation, n_ghosts, personalities, injection, max_steps",
    FACTORY_TABLE,
    ids=[row[0] for row in FACTORY_TABLE],
)
def test_the_factory_table(name, observation, n_ghosts, personalities, injection, max_steps):
    """Every column of the course map, for every world."""
    cfg = pragman.configs.by_name(name)()

    assert cfg.observation == observation
    assert cfg.n_ghosts == n_ghosts
    assert cfg.ghost_policy_names == personalities
    assert len(cfg.ghost_policies) == n_ghosts
    assert cfg.allow_state_injection is injection
    assert cfg.max_steps == max_steps

    # The environment agrees with the config it was built from.
    env = pragman.make(name)
    assert env.cfg == cfg
    assert env.observation_space == pragman.observation_space(cfg)


def test_the_small_worlds_are_small_enough_to_enumerate():
    """The state counts Weeks 4-6 rely on.

    ``empty_5x5`` has 25 states (24 cells to stand on plus the terminal goal),
    ``maze_8x8`` about fifty, and ``haunted_5x5`` 576 --
    24 agent cells times 24 ghost cells.
    """
    assert pragman.make("empty_5x5").observation_space.n == 25
    assert 45 <= pragman.make("maze_8x8").observation_space.n <= 60  # "~50 states"
    assert pragman.make("haunted_5x5").observation_space.n == 576


def test_the_stochasticity_knobs_of_weeks_4_to_6():
    """``slippery_5x5`` slips 20%, ``maze_8x8`` a gentle 10%."""
    assert pragman.configs.empty_5x5().slip_prob == 0.0
    assert pragman.configs.slippery_5x5().slip_prob == 0.2
    assert pragman.configs.maze_8x8().slip_prob == 0.1


def test_pixels_11x11_is_the_88_by_88_picture():
    """Eleven cells at eight pixels each, in colour."""
    env = pragman.make("pixels_11x11")
    assert env.observation_space.shape == (88, 88, 3)
    assert env.cfg.cell_pixels == 8
    assert env.cfg.layout == pragman.configs.classic_11x11().layout  # the same maze


def test_partial_11x11_is_the_same_maze_through_a_window():
    """``view_radius=2``, which is the Week 13 hook."""
    cfg = pragman.configs.partial_11x11()
    assert cfg.view_radius == 2
    assert cfg.layout == pragman.configs.classic_11x11().layout


# ── planning_11x11 is deterministic in all three respects ─────────────────────


def test_planning_11x11_has_every_source_of_chance_switched_off():
    """The third one is the one that is easy to miss.

    ``slip_prob=0`` and ``ghost_epsilon=0`` are obvious; a ``ChaserGhost`` that
    splits its mass between moves that *tie* for closest would still leave the
    tree search branching on chance nodes, so the chasers here break ties by
    lowest action index.
    """
    cfg = pragman.configs.planning_11x11()
    assert cfg.slip_prob == 0.0
    assert cfg.ghost_epsilon == 0.0
    assert all(policy.tie_break == "first" for policy in cfg.ghost_policies)


def test_planning_11x11_has_exactly_one_successor_per_state_and_action():
    """What keeps the Week 10 MCTS lab off chance nodes.

    This is the property that is easy to regress -- turn one knob back on and
    ``successors`` starts returning a distribution again -- so it is checked
    over a real rollout rather than on the initial state alone.
    """
    env = pragman.make("planning_11x11")
    rng = np.random.default_rng(0)
    _obs, info = env.reset(seed=0)
    state = info["state"]

    for _ in range(200):
        for action in range(4):
            outcomes = env.successors(state, action)
            assert len(outcomes) == 1, f"{len(outcomes)} outcomes from {state} under {action}"
            probability, _next_state, _reward, _terminated = outcomes[0]
            assert probability == pytest.approx(1.0)
        _obs, _reward, terminated, truncated, info = env.step(int(rng.integers(4)))
        state = info["state"]
        if terminated or truncated:
            _obs, info = env.reset()
            state = info["state"]


def test_planning_11x11_allows_the_injection_mcts_needs():
    """Injection is on here, and off in the model-free worlds."""
    env = pragman.make("planning_11x11")
    _obs, info = env.reset(seed=0)
    assert env.restore_state(info["state"]) == info["state"]


# ── overrides: every knob on every world ─────────────────────────────────────


def test_any_field_can_be_overridden_on_any_world():
    """``pragman.make("classic_11x11", ghost_policy="mix")``."""
    harder = pragman.make("classic_11x11", ghost_policy="mix")
    assert harder.cfg.ghost_policy_names == ("random", "chaser")

    slippery = pragman.configs.maze_8x8(slip_prob=0.3)
    assert slippery.slip_prob == 0.3
    assert pragman.configs.maze_8x8().slip_prob == 0.1  # the factory is unchanged


def test_start_random_is_available_on_every_world():
    """"anywhere" without editing a map.

    It helps exploration for DQN and policy gradients, and gives Week 8 a
    non-degenerate rho_0 to point at when deriving the visitation measure.
    """
    env = pragman.make("classic_11x11", start="random")
    starts = set()
    for seed in range(25):
        _obs, info = env.reset(seed=seed)
        starts.add(info["state"].agent)
    assert len(starts) > 1
    assert all(not env.layout.is_wall(cell) for cell in starts)


def test_a_typo_names_the_thing_that_was_mistyped():
    """The error message is part of the ergonomics."""
    with pytest.raises(ConfigError, match="unknown Prag-man world"):
        pragman.make("classsic_11x11")
    with pytest.raises(ConfigError, match="slipp_prob"):
        pragman.make("maze_8x8", slipp_prob=0.2)


# ── pellet_distance_shaping ───────────────────────────────────────────────────

# A corridor with a single pellet at the far end, so the distance to the nearest
# pellet is something a reader can count:
#
#     # # # # # # #
#     # P . . . o #
#     # # # # # # #
CORRIDOR = ["#######", "#P...o#", "#######"]


def corridor_states():
    """The agent at home, and one step closer to the pellet."""
    pellets = frozenset({(1, 5)})
    return (
        PragmanState(agent=(1, 1), pellets=pellets),  # distance to the pellet: 4
        PragmanState(agent=(1, 2), pellets=pellets),  # distance to the pellet: 3
    )


def test_one_sided_shaping_pays_for_approaching_and_nothing_for_retreating():
    """Approaching the pellet pays ``weight``; moving away pays nothing."""
    cfg = PragmanConfig(layout=CORRIDOR, observation="coords")
    shaping = pragman.configs.pellet_distance_shaping(cfg, weight=0.2)  # one_sided=True
    home, forward = corridor_states()
    no_true_reward = {"true_reward": 0.0}

    approach = shaping(home, Action.RIGHT, forward, no_true_reward)
    retreat = shaping(forward, Action.LEFT, home, no_true_reward)
    assert approach == pytest.approx(0.2)
    assert retreat == pytest.approx(0.0)


def test_the_signed_form_also_charges_for_retreating():
    """With ``one_sided=False``, moving away from the pellet costs ``weight``."""
    cfg = PragmanConfig(layout=CORRIDOR, observation="coords")
    shaping = pragman.configs.pellet_distance_shaping(cfg, weight=0.2, one_sided=False)
    home, forward = corridor_states()
    no_true_reward = {"true_reward": 0.0}

    approach = shaping(home, Action.RIGHT, forward, no_true_reward)
    retreat = shaping(forward, Action.LEFT, home, no_true_reward)
    assert approach == pytest.approx(0.2)
    assert retreat == pytest.approx(-0.2)


def test_the_shaping_adds_to_the_true_reward_and_says_so_when_it_cannot():
    """The shaping *adds to* the task reward, so both returns can be logged."""
    cfg = PragmanConfig(layout=CORRIDOR, observation="coords")
    shaping = pragman.configs.pellet_distance_shaping(cfg, weight=0.2)
    home, forward = corridor_states()

    assert shaping(home, Action.RIGHT, forward, {"true_reward": -0.01}) == pytest.approx(0.19)
    with pytest.raises(KeyError, match="true_reward"):
        shaping(home, Action.RIGHT, forward, {})
