"""States, ASCII positions, and recorded episodes.

The central idea under test is the distinction between a *map*, which says what
is static, and a *position*, which says where everybody is right now. The position form
is deliberately **lossy in three ways**, and the tests below pin each loss
explicitly so that it stays a documented property rather than a later surprise.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from pragman.config import PragmanConfig
from pragman.env import PragmanEnv
from pragman.errors import ConfigError, LayoutError
from pragman.ghosts import ProximityChaser, RandomGhost, uniform
from pragman.layout import Layout
from pragman.state import (
    ACTIONS,
    Action,
    Episode,
    PragmanState,
    load_episodes,
    save_episodes,
)

# One ghost, one pellet, one goal -- small enough to read, rich enough to lose
# information in all three documented ways.
#
#   row 0:  #####
#   row 1:  #P.G#
#   row 2:  #o.X#
#   row 3:  #####
ROOM = Layout.from_ascii(
    """\
    #####
    #P.G#
    #o.X#
    #####"""
)

# The same room with a second ghost spawn, for the ghost-order test.
TWO_GHOSTS = Layout.from_ascii(
    """\
    #####
    #P.G#
    #G.X#
    #####"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────


def test_action_order_is_the_one_discrete_4_promises():
    """UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3.

    Tie-breaking towards the lowest index shows up in the greedy policies of
    Week 4, so the order is part of the contract rather than an accident.
    """
    assert (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT) == (0, 1, 2, 3)
    assert ACTIONS == (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)


def test_action_geometry():
    """Deltas use (row, col) with row 0 at the top, so UP subtracts from the row."""
    assert Action.UP.delta == (-1, 0)
    assert Action.RIGHT.delta == (0, 1)
    assert Action.DOWN.delta == (1, 0)
    assert Action.LEFT.delta == (0, -1)
    assert Action.UP.opposite is Action.DOWN
    assert Action.LEFT.opposite is Action.RIGHT
    # The two directions a slip can send you; see dynamics.slip_distribution.
    assert Action.UP.perpendicular == (Action.RIGHT, Action.LEFT)
    assert Action.RIGHT.perpendicular == (Action.UP, Action.DOWN)


# ──────────────────────────────────────────────────────────────────────────────
# The state itself
# ──────────────────────────────────────────────────────────────────────────────


def test_a_state_is_a_frozen_hashable_value():
    """States are dictionary keys in the tabular model, in Dyna-Q, and in MCTS."""
    a = PragmanState((1, 1), ((1, 3),), frozenset({(2, 1)}))
    b = PragmanState((1, 1), ((1, 3),), frozenset({(2, 1)}))
    assert a == b and hash(a) == hash(b)
    assert {a: "seen"}[b] == "seen"
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.agent = (0, 0)  # frozen: the state is a value, not a mutable board


def test_hand_written_fixtures_are_normalized():
    """Lists and sets in a hand-written fixture become the canonical hashable types."""
    state = PragmanState([1, 1], [[1, 3]], {(2, 1)})
    assert state.agent == (1, 1)
    assert state.ghosts == ((1, 3),)
    assert state.pellets == frozenset({(2, 1)})
    assert state.pellets_left == 1
    assert state.terminated is False  # the default


def test_terminated_is_part_of_the_state_identity():
    """A terminal board is a different state from the same board mid-episode."""
    board = dict(agent=(1, 1), ghosts=((1, 3),), pellets=frozenset())
    assert PragmanState(**board, terminated=True) != PragmanState(**board, terminated=False)


# ──────────────────────────────────────────────────────────────────────────────
# The position form, and exactly how it loses information
# ──────────────────────────────────────────────────────────────────────────────


def test_ascii_draws_the_position_over_the_map():
    """``state.ascii(layout)`` needs the layout: a position cannot draw its own walls."""
    state = PragmanState((1, 1), ((1, 3),), frozenset({(2, 1)}))
    assert state.ascii(ROOM) == "#####\n#P.G#\n#o.X#\n#####"


def test_ascii_precedence_is_P_over_G_over_o_over_X():
    """Overlaps are drawn by precedence ``P > G > o > X > .``."""
    # Agent on the goal, ghost on the pellet: both cover what is underneath.
    state = PragmanState((2, 3), ((2, 1),), frozenset({(2, 1)}))
    assert state.ascii(ROOM) == "#####\n#...#\n#G.P#\n#####"


def test_position_round_trips_when_nothing_is_hidden():
    """The round-trip is exact for a state free of all three losses."""
    state = PragmanState((1, 2), ((1, 3),), frozenset({(2, 1)}), terminated=False)
    assert PragmanState.from_ascii(state.ascii(ROOM), ROOM, n_ghosts=1) == state


def test_loss_one_a_pellet_under_the_agent_is_lost():
    """A pellet the agent is standing on cannot be drawn, and reads back absent."""
    state = PragmanState((2, 1), ((1, 3),), frozenset({(2, 1)}))
    drawing = state.ascii(ROOM)
    assert drawing == "#####\n#..G#\n#P.X#\n#####"  # the 'o' is nowhere to be seen

    recovered = PragmanState.from_ascii(drawing, ROOM, n_ghosts=1)
    assert recovered.pellets == frozenset()
    assert recovered != state


def test_loss_one_a_pellet_under_a_ghost_is_lost():
    """The same holds for a ghost sitting on a pellet -- a genuinely dangerous cell."""
    state = PragmanState((1, 1), ((2, 1),), frozenset({(2, 1)}))
    recovered = PragmanState.from_ascii(state.ascii(ROOM), ROOM, n_ghosts=1)
    assert recovered.pellets == frozenset()
    assert recovered.ghosts == ((2, 1),)


def test_loss_two_ghost_order_is_not_recoverable():
    """The drawing cannot say which ghost is which, so ``from_ascii`` returns row-major.

    Ghost order matters (``state.ghosts[i]`` is governed by
    ``cfg.ghost_policies[i]``), so a state whose ghosts are *not* in row-major
    order does not round-trip.
    """
    state = PragmanState((1, 1), ((2, 1), (1, 3)), frozenset())  # not row-major
    recovered = PragmanState.from_ascii(state.ascii(TWO_GHOSTS), TWO_GHOSTS, n_ghosts=2)
    assert recovered.ghosts == ((1, 3), (2, 1))  # row-major, i.e. swapped
    assert recovered != state

    # ... whereas the row-major twin of the same board does round-trip.
    row_major = PragmanState((1, 1), ((1, 3), (2, 1)), frozenset())
    assert PragmanState.from_ascii(row_major.ascii(TWO_GHOSTS), TWO_GHOSTS, n_ghosts=2) == row_major


def test_loss_three_terminated_is_not_drawable():
    """``terminated`` reads back ``False`` unless passed explicitly."""
    state = PragmanState((1, 2), ((1, 3),), frozenset({(2, 1)}), terminated=True)
    drawing = state.ascii(ROOM)

    assert PragmanState.from_ascii(drawing, ROOM, n_ghosts=1).terminated is False
    assert PragmanState.from_ascii(drawing, ROOM, n_ghosts=1, terminated=True) == state


# ──────────────────────────────────────────────────────────────────────────────
# A position is stricter than a map, with its own errors
# ──────────────────────────────────────────────────────────────────────────────


def test_a_position_needs_exactly_one_P():
    """Several ``P``s are legal in a *map* and illegal in a *position*.

    The message is distinct from the layout parser's, so a student who drew a
    position when they meant a map is told which of the two they have.
    """
    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii("#####\n#P.P#\n#o.X#\n#####", ROOM)
    message = str(excinfo.value)
    assert message.startswith("state ascii:")
    assert "exactly one 'P'" in message and "found 2" in message

    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii("#####\n#...#\n#o.X#\n#####", ROOM)
    assert "found 0" in str(excinfo.value)


def test_a_position_needs_exactly_n_ghosts():
    """``n_ghosts`` is checked when given, so a fixture cannot silently lose a ghost."""
    drawing = "#####\n#P.G#\n#o.X#\n#####"
    assert PragmanState.from_ascii(drawing, ROOM, n_ghosts=1).ghosts == ((1, 3),)

    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii(drawing, ROOM, n_ghosts=2)
    assert "expected exactly 2 'G'" in str(excinfo.value)


def test_a_position_must_agree_with_the_maps_walls():
    """Drawing floor where the map has a wall (or the reverse) is an error."""
    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii("#####\n#P.G#\n#o.X#\n#.###", ROOM)
    assert "row 3" in str(excinfo.value)

    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii("#####\n#P#G#\n#o.X#\n#####", ROOM)
    assert "row 1" in str(excinfo.value) and "column 2" in str(excinfo.value)


def test_a_position_of_the_wrong_size_is_an_error():
    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii("###\n#P#\n###", ROOM)
    assert "3x3" in str(excinfo.value) and "4x5" in str(excinfo.value)


def test_a_position_may_not_put_a_goal_where_the_map_has_none():
    with pytest.raises(LayoutError) as excinfo:
        PragmanState.from_ascii("#####\n#PXG#\n#o.X#\n#####", ROOM)
    assert "no goal cell" in str(excinfo.value)


# ──────────────────────────────────────────────────────────────────────────────
# Episodes carry their world
# ──────────────────────────────────────────────────────────────────────────────

# A corridor with a ghost, used for the recorded episodes below.
CORRIDOR = """\
######
#P..G#
######"""

# A second, visibly different world, so that `save_episodes` has two configs to
# keep apart.
OTHER_WORLD = """\
#####
#P.X#
#####"""


@dataclass(frozen=True)
class SneakyGhost:
    """A custom policy that is deliberately never registered (see the episode tests below)."""

    name: ClassVar[str] = "sneaky"

    def distribution(self, cfg, state, ghost_index):
        return uniform(cfg.layout.legal_actions(state.ghosts[ghost_index]))


def record(cfg: PragmanConfig, actions, seed: int = 0) -> tuple[Episode, list]:
    """Run *actions* in a fresh environment and return the episode and its observations."""
    env = PragmanEnv(cfg)
    observation, _ = env.reset(seed=seed)
    states = [env.state]
    observations = [observation]
    rewards: list[float] = []
    taken: list[int] = []
    terminated = truncated = False
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)
        taken.append(int(action))
        rewards.append(float(reward))
        states.append(env.state)
        observations.append(observation)
        if terminated or truncated:
            break
    episode = Episode(cfg, states, taken, rewards, terminated, truncated)
    return episode, observations


def test_episode_round_trips_with_its_ghost_settings_intact(tmp_path):
    """A ``ProximityChaser(wake_distance=6)`` must survive save/load.

    Storing a policy by *name alone* would reload it as the default
    ``wake_distance=4`` and silently change the dynamics the dataset came from,
    which must never happen.
    """
    cfg = PragmanConfig(
        layout=CORRIDOR,
        n_ghosts=1,
        ghost_policy=ProximityChaser(wake_distance=6),
        observation="coords",
    )
    episode, _ = record(cfg, [Action.RIGHT, Action.RIGHT])

    path = tmp_path / "episode.json"
    episode.save(path)
    reloaded = Episode.load(path)

    assert reloaded.config.ghost_policies == (ProximityChaser(wake_distance=6),)
    assert reloaded.config.ghost_policies[0].wake_distance == 6
    assert reloaded.config == cfg
    assert reloaded == episode


def test_a_reloaded_config_is_not_the_default_proximity_chaser(tmp_path):
    """The negative half of the test above: the default really is different."""
    cfg = PragmanConfig(
        layout=CORRIDOR,
        n_ghosts=1,
        ghost_policy=ProximityChaser(wake_distance=6),
        observation="coords",
    )
    default = PragmanConfig(
        layout=CORRIDOR, n_ghosts=1, ghost_policy="proximity", observation="coords"
    )
    assert cfg != default

    path = tmp_path / "episode.json"
    record(cfg, [Action.RIGHT])[0].save(path)
    assert Episode.load(path).config != default


def test_episode_observations_match_what_the_environment_returned(tmp_path):
    """``episode.observations()`` reproduces the record-time observations."""
    cfg = PragmanConfig(layout=CORRIDOR, n_ghosts=1, observation="coords")
    episode, observations = record(cfg, [Action.RIGHT, Action.RIGHT, Action.LEFT])

    recomputed = episode.observations()
    assert len(recomputed) == len(episode) + 1
    for seen, again in zip(observations, recomputed, strict=True):
        assert np.array_equal(seen, again)

    # And still after a round trip through disk, since the config came too.
    path = tmp_path / "episode.json"
    episode.save(path)
    for seen, again in zip(observations, Episode.load(path).observations(), strict=True):
        assert np.array_equal(seen, again)


def test_save_episodes_stores_a_shared_config_once(tmp_path):
    """Two episodes from one world share a single config record."""
    cfg = PragmanConfig(layout=CORRIDOR, n_ghosts=1, observation="coords")
    first, _ = record(cfg, [Action.RIGHT], seed=0)
    second, _ = record(cfg, [Action.RIGHT, Action.RIGHT], seed=1)

    path = tmp_path / "dataset.json"
    save_episodes([first, second], path)

    import json

    payload = json.loads(path.read_text())
    assert len(payload["configs"]) == 1
    assert [record_["config"] for record_ in payload["episodes"]] == [0, 0]

    reloaded = load_episodes(path)
    assert reloaded == [first, second]
    # The de-duplication must not make them *the same object's* episodes only by
    # accident: both must still describe the world they were recorded in.
    assert all(episode.config == cfg for episode in reloaded)


def test_save_episodes_keeps_two_layouts_apart(tmp_path):
    """A dataset spanning two worlds stays one file, with two config records."""
    corridor = PragmanConfig(layout=CORRIDOR, n_ghosts=1, observation="coords")
    other = PragmanConfig(layout=OTHER_WORLD, observation="coords")
    first, _ = record(corridor, [Action.RIGHT])
    second, _ = record(other, [Action.RIGHT, Action.RIGHT])

    path = tmp_path / "two_worlds.json"
    save_episodes([first, second, first], path)

    import json

    payload = json.loads(path.read_text())
    assert len(payload["configs"]) == 2
    assert [record_["config"] for record_ in payload["episodes"]] == [0, 1, 0]

    reloaded = load_episodes(path)
    assert reloaded[0].config.layout == corridor.layout
    assert reloaded[1].config.layout == other.layout
    assert reloaded == [first, second, first]


def test_loading_an_unregistered_custom_ghost_names_it(tmp_path):
    """An episode recorded with a custom ghost fails loudly on load.

    Substituting a random ghost would quietly change the dynamics an offline
    dataset came from, so the error names the policy instead.
    """
    cfg = PragmanConfig(
        layout=CORRIDOR, n_ghosts=1, ghost_policy=SneakyGhost(), observation="coords"
    )
    episode, _ = record(cfg, [Action.RIGHT])

    path = tmp_path / "sneaky.json"
    episode.save(path)  # saving is fine: the name and settings are written out

    with pytest.raises(ConfigError) as excinfo:
        Episode.load(path)
    message = str(excinfo.value)
    assert "sneaky" in message
    assert "register_ghost_policy" in message


def test_episode_rejects_mismatched_lengths():
    """``states`` is ``T + 1`` long and ``rewards`` is ``T`` long."""
    cfg = PragmanConfig(layout=OTHER_WORLD, observation="coords")
    state = PragmanState((1, 1))
    with pytest.raises(ValueError):
        Episode(cfg, [state], [0], [0.0])  # one state too few
    with pytest.raises(ValueError):
        Episode(cfg, [state, state], [0], [0.0, 0.0])  # one reward too many
    with pytest.raises(ValueError):
        Episode(cfg, [state, state], [0], [0.0], terminated=True, truncated=True)


def test_episode_total_reward_and_length():
    cfg = PragmanConfig(layout=OTHER_WORLD, observation="coords")
    episode, _ = record(cfg, [Action.RIGHT, Action.RIGHT])
    assert len(episode) == len(episode.actions) == len(episode.rewards)
    assert episode.total_reward == pytest.approx(sum(episode.rewards))


def test_a_random_ghost_policy_is_a_value():
    """Policies are frozen dataclasses, which is what keeps a config hashable."""
    assert RandomGhost() == RandomGhost()
    assert hash(RandomGhost()) == hash(RandomGhost())
    assert ProximityChaser(wake_distance=6) != ProximityChaser(wake_distance=4)
