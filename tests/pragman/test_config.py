"""PragmanConfig: defaults, validation, and value semantics.

The config is the object that appears in every student's boilerplate,
so its error messages are part of the teaching material: each one must name the
offending field *and* the values it will accept.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pragman.config import PragmanConfig
from pragman.errors import ConfigError
from pragman.ghosts import ChaserGhost, ProximityChaser, RandomGhost
from pragman.layout import Layout

# A three-cell corridor with one ghost spawn and one goal.
#
#   row 0:  ######
#   row 1:  #PoGX#
#   row 2:  ######
CORRIDOR = """\
######
#PoGX#
######"""


def world(**overrides) -> PragmanConfig:
    """A config on ``CORRIDOR``, with overrides."""
    overrides.setdefault("observation", "coords")
    return PragmanConfig(layout=CORRIDOR, **overrides)


# ──────────────────────────────────────────────────────────────────────────────
# Defaults and construction
# ──────────────────────────────────────────────────────────────────────────────


def test_the_only_required_field_is_the_layout():
    """Every other field has a default."""
    cfg = PragmanConfig(layout=CORRIDOR)

    assert cfg.start == "layout"
    assert cfg.n_ghosts == 0
    assert cfg.ghost_epsilon == 0.1  # not 0: a chaser must be shakeable off
    assert cfg.pellets == "layout"
    assert cfg.slip_prob == 0.0
    assert (cfg.living_reward, cfg.pellet_reward, cfg.clear_bonus) == (-0.01, 0.1, 1.0)
    assert (cfg.goal_reward, cfg.death_reward) == (1.0, -1.0)
    assert cfg.max_steps == 100
    assert cfg.observation == "channels"
    assert cfg.view_radius is None
    assert cfg.cell_pixels == 8
    assert cfg.palette == "classic"
    assert cfg.allow_state_injection is True


def test_a_drawing_is_parsed_into_a_layout():
    """``layout=`` accepts a string or a list of rows and normalizes it."""
    assert world().layout == Layout.from_ascii(CORRIDOR)
    assert PragmanConfig(layout=CORRIDOR.splitlines()).layout == Layout.from_ascii(CORRIDOR)

    with pytest.raises(ConfigError) as excinfo:
        PragmanConfig(layout=42)
    assert "'layout'" in str(excinfo.value)


def test_replace_is_the_supported_way_to_tweak_one_field():
    """``dataclasses.replace`` re-runs validation, so a tweak cannot smuggle in nonsense."""
    cfg = world()
    slippery = replace(cfg, slip_prob=0.2)
    assert slippery.slip_prob == 0.2 and cfg.slip_prob == 0.0

    with pytest.raises(ConfigError):
        replace(cfg, slip_prob=2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Every invalid field raises ConfigError, naming the field
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [-0.1, 1.5, "a lot", None])
def test_invalid_slip_prob_raises(value):
    with pytest.raises(ConfigError) as excinfo:
        world(slip_prob=value)
    assert "'slip_prob'" in str(excinfo.value)


@pytest.mark.parametrize("value", [-0.1, 1.5, "sometimes"])
def test_invalid_ghost_epsilon_raises(value):
    with pytest.raises(ConfigError) as excinfo:
        world(ghost_epsilon=value)
    assert "'ghost_epsilon'" in str(excinfo.value)


def test_invalid_observation_raises_and_lists_the_choices():
    with pytest.raises(ConfigError) as excinfo:
        world(observation="pixel")  # the legal spelling is "pixels"
    message = str(excinfo.value)
    assert "'observation'" in message
    for choice in ("'index'", "'coords'", "'channels'", "'pixels'"):
        assert choice in message


def test_invalid_start_raises_and_lists_the_choices():
    with pytest.raises(ConfigError) as excinfo:
        world(start="anywhere")
    message = str(excinfo.value)
    assert "'start'" in message and "'layout'" in message and "'random'" in message


def test_invalid_pellets_raises_and_lists_the_choices():
    with pytest.raises(ConfigError) as excinfo:
        world(pellets="every_other")
    message = str(excinfo.value)
    assert "'pellets'" in message
    for choice in ("'layout'", "'all_floor'", "'none'"):
        assert choice in message


@pytest.mark.parametrize("observation", ["index", "coords"])
def test_view_radius_is_meaningless_without_a_grid_observation(observation):
    """An egocentric window only applies to ``channels``/``pixels``."""
    with pytest.raises(ConfigError) as excinfo:
        PragmanConfig(layout=CORRIDOR, observation=observation, view_radius=2)
    message = str(excinfo.value)
    assert "'view_radius'" in message and observation in message


def test_view_radius_is_allowed_with_channels_and_pixels():
    assert PragmanConfig(layout=CORRIDOR, observation="channels", view_radius=2).view_radius == 2
    assert PragmanConfig(layout=CORRIDOR, observation="pixels", view_radius=0).view_radius == 0
    with pytest.raises(ConfigError):
        PragmanConfig(layout=CORRIDOR, observation="channels", view_radius=-1)


def test_n_ghosts_may_not_exceed_the_layouts_G_cells():
    """``CORRIDOR`` has one ``G``, so asking for two says so."""
    assert world(n_ghosts=1).initial_ghosts() == ((1, 3),)

    with pytest.raises(ConfigError) as excinfo:
        world(n_ghosts=2)
    message = str(excinfo.value)
    assert "'n_ghosts'" in message and "only 1 'G' cell" in message


@pytest.mark.parametrize("value", [-1, 1.5, "two", True])
def test_invalid_n_ghosts_raises(value):
    with pytest.raises(ConfigError) as excinfo:
        world(n_ghosts=value)
    assert "'n_ghosts'" in str(excinfo.value)


@pytest.mark.parametrize("value", [0, -5, 2.5, "forever"])
def test_invalid_max_steps_raises(value):
    with pytest.raises(ConfigError) as excinfo:
        world(max_steps=value)
    assert "'max_steps'" in str(excinfo.value)


def test_max_steps_none_means_never_truncate():
    assert world(max_steps=None).max_steps is None


def test_invalid_palette_raises():
    with pytest.raises(ConfigError) as excinfo:
        world(palette="neon")
    message = str(excinfo.value)
    assert "'palette'" in message and "'classic'" in message and "'high_contrast'" in message


def test_non_finite_rewards_raise():
    for field in ("living_reward", "pellet_reward", "clear_bonus", "goal_reward", "death_reward"):
        with pytest.raises(ConfigError) as excinfo:
            world(**{field: float("inf")})
        assert repr(field) in str(excinfo.value)


def test_cell_pixels_must_be_a_positive_int():
    with pytest.raises(ConfigError):
        world(cell_pixels=0)
    with pytest.raises(ConfigError):
        world(cell_pixels=1.5)


def test_allow_state_injection_must_be_a_bool():
    with pytest.raises(ConfigError) as excinfo:
        world(allow_state_injection="yes")
    assert "'allow_state_injection'" in str(excinfo.value)


# ──────────────────────────────────────────────────────────────────────────────
# The derived initial distributions
# ──────────────────────────────────────────────────────────────────────────────


def test_start_cells_follow_the_start_field():
    """``"layout"`` uses the ``P`` cells; ``"random"`` uses every free floor cell.

    "Free" excludes ghost spawns (nobody starts dead) and goal cells (spawning on
    the goal would make a non-terminal state in which the agent stands on X and
    must leave and come back to collect the reward -- an artefact, not a lesson).
    """
    assert world(n_ghosts=1).start_cells() == ((1, 1),)

    anywhere = world(n_ghosts=1, start="random")
    assert anywhere.start_cells() == ((1, 1), (1, 2))  # (1, 3) a ghost, (1, 4) the goal


def test_initial_pellets_follow_the_pellets_field():
    """``"all_floor"`` fills every floor cell except the goals."""
    assert world().initial_pellets() == frozenset({(1, 2)})
    assert world(pellets="none").initial_pellets() == frozenset()
    # The goal at (1, 4) gets no pellet; the ghost spawn at (1, 3) does.
    assert world(pellets="all_floor").initial_pellets() == frozenset(
        {(1, 1), (1, 2), (1, 3)}
    )


# ──────────────────────────────────────────────────────────────────────────────
# A config is a value, hashable, and compares its ghosts by settings
# ──────────────────────────────────────────────────────────────────────────────


def test_configs_are_hashable_and_usable_as_dict_keys():
    """A config keys the config table of ``save_episodes`` and the env cache."""
    a = world(n_ghosts=1, ghost_policy="chaser")
    b = world(n_ghosts=1, ghost_policy="chaser")
    other = world(n_ghosts=1, ghost_policy="random")

    assert a == b and hash(a) == hash(b)
    assert a != other

    table = {a: "chased", other: "wandered"}
    assert table[b] == "chased"
    assert len(table) == 2


def test_a_named_policy_and_the_object_it_names_compare_equal():
    """``ghost_policy="chaser"`` and ``ChaserGhost()`` build the same world."""
    assert world(n_ghosts=1, ghost_policy="chaser") == world(
        n_ghosts=1, ghost_policy=ChaserGhost()
    )
    assert world(n_ghosts=1, ghost_policy="random") == world(
        n_ghosts=1, ghost_policy=RandomGhost()
    )


def test_ghost_policies_compare_by_settings_not_only_by_name():
    """``ProximityChaser(wake_distance=6)`` is *not* the default.

    Comparing by name alone was the v1.1 plan and is wrong: it would make these
    two configs equal, and an ``Episode`` serialized by name would reload with
    different dynamics.
    """
    default = world(n_ghosts=1, ghost_policy="proximity")
    keener = world(n_ghosts=1, ghost_policy=ProximityChaser(wake_distance=6))

    assert default != keener
    assert len({default: 1, keener: 2}) == 2
    assert default.ghost_policy_names == keener.ghost_policy_names == ("proximity",)


def test_a_config_serializes_and_reloads_unchanged():
    """The JSON form keeps the layout drawing and each ghost's settings."""
    cfg = world(n_ghosts=1, ghost_policy=ProximityChaser(wake_distance=6), slip_prob=0.2)
    assert PragmanConfig.from_json_dict(cfg.to_json_dict()) == cfg
