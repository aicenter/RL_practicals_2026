"""Layouts: parsing, validation, geometry, and BFS distances.

These tests cover the parser and the layout as a value, plus the parts that
everything else leans on (``move`` treating out-of-bounds as wall,
the BFS tables the chasers use).

Every fixture is a hand-drawn map small enough to read at a glance.
"""

from __future__ import annotations

import numpy as np
import pytest

from pragman.config import PragmanConfig
from pragman.env import PragmanEnv
from pragman.errors import LayoutError
from pragman.layout import NAMED_LAYOUTS, Layout
from pragman.model import build_tabular_model
from pragman.state import Action

# A map with one of everything, used by the parsing tests.
#
#     col:  0123456
#   row 0:  #######
#   row 1:  #P.o..#
#   row 2:  #.###.#
#   row 3:  #..G.X#
#   row 4:  #######
SAMPLER = """\
#######
#P.o..#
#.###.#
#..G.X#
#######"""


# ──────────────────────────────────────────────────────────────────────────────
# Parsing: what a map says
# ──────────────────────────────────────────────────────────────────────────────


def test_parsing_reads_every_character_role():
    """Each ASCII role lands in its own attribute, in row-major order."""
    layout = Layout.from_ascii(SAMPLER)

    assert layout.shape == (5, 7)
    assert layout.height == 5 and layout.width == 7
    assert layout.agent_starts == ((1, 1),)
    assert layout.agent_start == (1, 1)
    assert layout.pellet_cells == ((1, 3),)
    assert layout.ghost_starts == ((3, 3),)
    assert layout.goal_cells == ((3, 5),)
    # Floor is everything that is not a wall, whatever is drawn on it.
    assert (1, 1) in layout.floor_cells and (3, 5) in layout.floor_cells
    assert (0, 0) not in layout.floor_cells
    assert layout.walls[0, 0] and not layout.walls[1, 1]


def test_indentation_and_blank_lines_are_stripped():
    """A triple-quoted map indented inside a function still parses."""
    indented = """
        ###
        #P#
        #.#
        ###
    """
    assert Layout.from_ascii(indented).ascii == "###\n#P#\n#.#\n###"


def test_a_list_of_rows_is_accepted():
    """``from_ascii`` takes a list of row strings as well as one string."""
    assert Layout.from_ascii(["#P#", "#.#"]) == Layout.from_ascii("#P#\n#.#")


def test_ghost_starts_are_row_major():
    """``n_ghosts`` ghosts take the first ``n_ghosts`` ``G`` cells, row-major.

    This ordering is the one the whole package uses -- it is what makes "which
    ghost is which" answerable by looking at the map.
    """
    layout = Layout.from_ascii(
        """\
        G.G
        .P.
        G.G"""
    )
    assert layout.ghost_starts == ((0, 0), (0, 2), (2, 0), (2, 2))


# ──────────────────────────────────────────────────────────────────────────────
# The parser rejects malformed maps
# ──────────────────────────────────────────────────────────────────────────────


def test_ragged_rows_raise_naming_the_row():
    """Rows must all have the same width, and the error says which row is wrong."""
    with pytest.raises(LayoutError) as excinfo:
        Layout.from_ascii("###\n#P.#\n###")
    message = str(excinfo.value)
    assert "row 1" in message and "ragged" in message


def test_a_map_without_P_raises():
    """At least one ``P`` is required: a map with no start has no initial distribution."""
    with pytest.raises(LayoutError) as excinfo:
        Layout.from_ascii("###\n#.#\n###")
    assert "'P'" in str(excinfo.value)


def test_unknown_character_is_rejected_by_name():
    """Unknown characters are named, with their row and column."""
    with pytest.raises(LayoutError) as excinfo:
        Layout.from_ascii("####\n#P%#\n####")
    message = str(excinfo.value)
    assert "'%'" in message and "row 1" in message and "column 2" in message


def test_power_pellet_is_reserved_for_v2():
    """``O`` parses to a clear "not yet" rather than to a silent pellet."""
    with pytest.raises(LayoutError) as excinfo:
        Layout.from_ascii("####\n#PO#\n####")
    message = str(excinfo.value)
    assert "'O'" in message and "power pellet" in message


def test_a_map_with_nowhere_to_go_raises():
    """There must be a walkable cell besides the start cells."""
    with pytest.raises(LayoutError):
        Layout.from_ascii("###\n#P#\n###")


# ──────────────────────────────────────────────────────────────────────────────
# Several `P`s are legal and mean a uniform start
# ──────────────────────────────────────────────────────────────────────────────

# Two start cells, at opposite ends of a three-cell corridor.
TWO_STARTS = """\
#####
#P.P#
#####"""


def test_two_start_cells_are_legal():
    """Several ``P``s denote a uniform initial distribution, not an error."""
    layout = Layout.from_ascii(TWO_STARTS)
    assert layout.agent_starts == ((1, 1), (1, 3))
    # `agent_start` is the convenience for single-P maps: the first one.
    assert layout.agent_start == (1, 1)


def test_two_start_cells_give_rho0_one_half_each():
    """rho_0 is uniform over the ``P`` cells, and both states are enumerated."""
    cfg = PragmanConfig(layout=TWO_STARTS, observation="coords")
    model = build_tabular_model(cfg)

    start_states = [model.states[s] for s in np.nonzero(model.rho0)[0]]
    assert {state.agent for state in start_states} == {(1, 1), (1, 3)}
    assert np.allclose(model.rho0[np.nonzero(model.rho0)], 0.5)
    assert model.rho0.sum() == pytest.approx(1.0)
    # Both really are in the table, not merely in rho_0.
    assert {state.agent for state in model.states} >= {(1, 1), (1, 3)}


def test_resets_land_on_both_start_cells():
    """Over many seeds ``reset()`` draws both ``P`` cells."""
    env = PragmanEnv(PragmanConfig(layout=TWO_STARTS, observation="coords"))
    drawn = set()
    for seed in range(30):
        env.reset(seed=seed)
        drawn.add(env.state.agent)
    assert drawn == {(1, 1), (1, 3)}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry: movement and out-of-bounds
# ──────────────────────────────────────────────────────────────────────────────

# Deliberately borderless, to pin "off the grid counts as wall".
#
#   row 0:  P#.
#   row 1:  ...
NO_BORDER = "P#.\n..."


def test_off_the_grid_counts_as_a_wall():
    """A border of ``#`` is conventional, not required."""
    layout = Layout.from_ascii(NO_BORDER)
    assert layout.is_wall((-1, 0)) and layout.is_wall((2, 0)) and layout.is_wall((0, 3))
    assert not layout.in_bounds((-1, 0))
    assert not layout.is_wall((0, 0))


def test_move_bumps_into_walls_and_edges():
    """From the corner ``P`` of ``NO_BORDER``, only DOWN goes anywhere."""
    layout = Layout.from_ascii(NO_BORDER)
    assert layout.move((0, 0), Action.UP) == (0, 0)  # off the top edge
    assert layout.move((0, 0), Action.LEFT) == (0, 0)  # off the left edge
    assert layout.move((0, 0), Action.RIGHT) == (0, 0)  # into the wall at (0, 1)
    assert layout.move((0, 0), Action.DOWN) == (1, 0)
    assert layout.legal_actions((0, 0)) == (Action.DOWN,)
    assert layout.neighbours((0, 0)) == ((1, 0),)


def test_legal_actions_in_the_open():
    """An interior cell of an open room has all four moves."""
    layout = Layout.from_ascii("P..\n...\n...")
    assert layout.legal_actions((1, 1)) == (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)


# ──────────────────────────────────────────────────────────────────────────────
# BFS distances: the chasers' workhorse
# ──────────────────────────────────────────────────────────────────────────────

# A U-shaped corridor: the two ends are two cells apart as the crow flies but
# four steps apart through the maze, which is the whole point of using BFS
# rather than Manhattan distance for the chasers.
U_SHAPE = """\
#####
#P#.#
#...#
#####"""


def test_distances_follow_the_maze_not_the_crow():
    """``distances`` is breadth-first through walkable cells."""
    layout = Layout.from_ascii(U_SHAPE)
    distances = layout.distances((1, 1))
    assert distances[(1, 1)] == 0
    assert distances[(2, 1)] == 1
    assert distances[(2, 2)] == 2
    assert distances[(2, 3)] == 3
    assert distances[(1, 3)] == 4  # two columns away, four steps around the wall


def test_unreachable_cells_are_absent_from_the_distance_table():
    """An unreachable pocket is legal; it simply never appears in the BFS result."""
    sealed = Layout.from_ascii(
        """\
        ####
        #P.#
        ####
        #..#
        ####"""
    )
    assert (3, 1) not in sealed.distances((1, 1))
    assert sealed.unreachable_cells() == frozenset({(3, 1), (3, 2)})


def test_a_connected_map_has_no_unreachable_cells():
    assert Layout.from_ascii(U_SHAPE).unreachable_cells() == frozenset()


def test_distances_from_a_wall_raise():
    with pytest.raises(LayoutError):
        Layout.from_ascii(U_SHAPE).distances((0, 0))


# ──────────────────────────────────────────────────────────────────────────────
# A Layout is a value
# ──────────────────────────────────────────────────────────────────────────────


def test_layout_round_trips_through_its_ascii():
    """``Layout.from_ascii(l.ascii) == l`` -- the drawing determines everything."""
    for name, drawing in NAMED_LAYOUTS.items():
        layout = Layout.from_ascii(drawing)
        assert Layout.from_ascii(layout.ascii) == layout, name


def test_layouts_are_hashable_and_compare_by_drawing():
    """A ``Layout`` holds a numpy array, so ``__eq__``/``__hash__`` are defined by hand.

    Without them the dataclass-generated ``__eq__`` would raise "truth value of
    an array is ambiguous" -- and ``PragmanConfig``'s hashability depends on this.
    """
    a = Layout.from_ascii(SAMPLER)
    b = Layout.from_ascii("\n\n" + SAMPLER + "\n")  # same map, sloppier drawing
    other = Layout.from_ascii(U_SHAPE)

    assert a == b and hash(a) == hash(b)
    assert a != other
    assert a != "not a layout"

    table = {a: "sampler", other: "u-shape"}
    assert table[b] == "sampler"
    assert len(table) == 2


def test_named_layouts_all_parse_and_are_connected():
    """The shipped maps are valid, and nothing in them is walled off from the start."""
    for name, drawing in NAMED_LAYOUTS.items():
        layout = Layout.from_ascii(drawing)
        assert layout.agent_starts, name
        assert layout.unreachable_cells() == frozenset(), name
