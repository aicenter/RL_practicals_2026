"""Rendering.

The renderer is numpy-only and pure, which is what lets the ``"pixels"``
observation *be* the rendered frame.  The optional extras (pygame, imageio,
matplotlib) are not installed here, so the tests for those paths only check that
the failure explains itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import pragman
from pragman import PragmanConfig, PragmanState
from pragman.render import PALETTES, egocentric_crop

# A board with one of everything, small enough to count pixels on.
#
#     # # # # #
#     # P . o #
#     # . # G #
#     # . . X #
#     # # # # #
BOARD = ["#####", "#P.o#", "#.#G#", "#..X#", "#####"]


def board_config(**overrides):
    settings = {"observation": "pixels", "n_ghosts": 1, **overrides}
    return PragmanConfig(layout=BOARD, **settings)


def board_state():
    return PragmanState(agent=(1, 1), ghosts=((2, 3),), pellets=frozenset({(1, 3)}))


# ── the identity that keeps the two call sites honest ─────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {},                                # the defaults
        {"palette": "high_contrast"},      # the lecture-hall palette
        {"ghost_policy": "mix"},           # ghosts coloured by personality
        {"cell_pixels": 13},               # a non-default block size
    ],
    ids=["default", "high_contrast", "mixed_ghosts", "cell_pixels_13"],
)
def test_pixels_observation_is_byte_identical_to_rgb_array(overrides):
    """The picture you learn from is exactly the picture you project.

    Both call sites must pass the same ``cell_pixels``, the same palette and the
    same ghost colours; this is the test that catches it when they drift.  Note
    that ``palette`` is a *config* field for precisely this reason -- a display
    setting the renderer could override on its own would break the identity.
    """
    env = pragman.make("pixels_11x11", render_mode="rgb_array", **overrides)
    observation, _info = env.reset(seed=3)
    rendered = env.render()

    assert rendered.dtype == np.uint8
    assert np.array_equal(observation, rendered)

    # ... and it keeps holding as the world moves.
    rng = np.random.default_rng(0)
    for _ in range(20):
        observation, _reward, terminated, truncated, _info = env.step(int(rng.integers(4)))
        assert np.array_equal(observation, env.render())
        if terminated or truncated:
            break


def test_with_a_view_radius_the_observation_is_a_crop_of_that_very_frame():
    """``view_radius`` narrows what the *agent* sees, not the hall.

    ``render()`` still shows the whole board; the observation is the egocentric
    window cut out of exactly that frame.
    """
    env = pragman.make("pixels_11x11", render_mode="rgb_array", view_radius=2)
    observation, info = env.reset(seed=4)
    full = env.render()

    assert full.shape == (88, 88, 3)                     # the whole 11x11 board
    assert observation.shape == (5 * 8, 5 * 8, 3)        # (2r+1) cells of 8 pixels
    assert np.array_equal(
        observation,
        egocentric_crop(
            full,
            info["state"].agent,
            env.cfg.view_radius,
            cell_pixels=env.cfg.cell_pixels,
            palette=env.cfg.palette,
        ),
    )


# ── renderer smoke tests ──────────────────────────────────────────────────────


def test_frame_has_the_shape_and_dtype_it_promises():
    """``(H*k, W*k, 3)`` uint8, at whatever cell size you ask for."""
    layout = board_config().layout
    for cell_pixels in (1, 8, 24):
        image = pragman.frame(board_state(), layout, cell_pixels=cell_pixels)
        assert image.shape == (5 * cell_pixels, 5 * cell_pixels, 3)
        assert image.dtype == np.uint8


def test_frame_is_pure_and_deterministic():
    """Same arguments in, byte-identical array out.

    This is what allows a recorded ``Episode`` to be re-rendered afterwards for
    Week 11's preference-labelling clips -- rendering never consults a live
    environment.
    """
    layout = board_config().layout
    first = pragman.frame(board_state(), layout, cell_pixels=8)
    second = pragman.frame(board_state(), layout, cell_pixels=8)
    assert np.array_equal(first, second)
    first[:] = 0  # a fresh array each time, not a shared buffer
    assert np.array_equal(pragman.frame(board_state(), layout, cell_pixels=8), second)


def test_the_frame_actually_draws_walls_the_agent_the_ghost_and_the_pellet():
    """A picture is only useful if the pieces are where the board says they are.

    Each cell is a ``k x k`` block, so cell (r, c) occupies rows ``r*k:(r+1)*k``.
    We check the colour in the middle of a few blocks rather than every pixel.
    """
    palette = PALETTES["classic"]
    k = 8
    image = pragman.frame(board_state(), board_config().layout, cell_pixels=k)

    def centre(row, col):
        return tuple(int(v) for v in image[row * k + k // 2, col * k + k // 2])

    assert centre(0, 0) == palette["wall"]      # the border
    assert centre(1, 2) == palette["floor"]     # empty floor
    assert centre(1, 1) == palette["agent"]     # the agent's disc
    assert centre(1, 3) == palette["pellet"]    # the pellet's dot
    assert centre(2, 3) != palette["floor"]     # the ghost's block (eyes may be central)
    assert centre(3, 3) == palette["goal"]      # the goal cell


def test_the_two_palettes_really_are_different_pictures():
    """``classic`` for students, ``high_contrast`` for projectors."""
    assert set(PALETTES) == {"classic", "high_contrast"}
    layout = board_config().layout
    classic = pragman.frame(board_state(), layout, palette="classic")
    contrast = pragman.frame(board_state(), layout, palette="high_contrast")
    assert classic.shape == contrast.shape
    assert not np.array_equal(classic, contrast)

    with pytest.raises(ValueError, match="unknown palette"):
        pragman.frame(board_state(), layout, palette="chartreuse")


def test_ghosts_are_coloured_by_personality():
    """The hall can see which ghost is hunting.

    Colours follow the personality, not the index, so swapping the two ghosts'
    personalities swaps their colours.
    """
    layout = pragman.Layout.from_ascii("\n".join(["#####", "#P.G#", "#.#G#", "#..X#", "#####"]))
    state = PragmanState(agent=(1, 1), ghosts=((1, 3), (2, 3)), pellets=frozenset())
    k = 8

    def ghost_colour(image, cell):
        row, col = cell
        return tuple(int(v) for v in image[row * k + k - 2, col * k + k // 2])

    chaser_first = pragman.frame(state, layout, cell_pixels=k, ghost_names=("chaser", "random"))
    random_first = pragman.frame(state, layout, cell_pixels=k, ghost_names=("random", "chaser"))
    assert ghost_colour(chaser_first, (1, 3)) == PALETTES["classic"]["ghost_chaser"]
    assert ghost_colour(random_first, (1, 3)) == PALETTES["classic"]["ghost_random"]
    assert ghost_colour(chaser_first, (2, 3)) == ghost_colour(random_first, (1, 3))


def test_egocentric_crop_pads_with_the_wall_colour():
    """Beyond the edge of the board there is wall, in pixels too."""
    layout = board_config().layout
    image = pragman.frame(board_state(), layout, cell_pixels=8, palette="classic")
    window = egocentric_crop(image, agent=(1, 1), view_radius=2, cell_pixels=8)

    assert window.shape == (5 * 8, 5 * 8, 3)
    # The agent is at (1, 1), so the window spans rows -1..3: its whole first
    # row of cells lies off the board and must be solid wall.
    assert np.all(window[:8] == PALETTES["classic"]["wall"])
    assert np.all(window[:, :8] == PALETTES["classic"]["wall"])


def test_overlays_draw_on_top_without_changing_the_shape():
    """Grid, heatmap and arrows are numpy, and off by default."""
    layout = board_config().layout
    plain = pragman.frame(board_state(), layout, cell_pixels=16)

    for overlays in (
        {"grid": True},
        {"heatmap": np.arange(25, dtype=float).reshape(5, 5)},
        {"arrows": np.zeros((5, 5), dtype=int)},
    ):
        decorated = pragman.frame(board_state(), layout, cell_pixels=16, overlays=overlays)
        assert decorated.shape == plain.shape and decorated.dtype == np.uint8
        assert not np.array_equal(decorated, plain), overlays

    with pytest.raises(ValueError, match="unknown overlay"):
        pragman.frame(board_state(), layout, overlays={"text": "hello"})


# ── the ANSI mode ─────────────────────────────────────────────────────────────


def test_ansi_without_colour_is_the_ascii_position_form():
    """The glyphs are those of the position notation.

    One character per cell, exactly ``layout.height`` lines, no trailing
    newline -- and the same drawing ``state.ascii(layout)`` produces.
    """
    layout = board_config().layout
    state = board_state()
    text = pragman.ansi(state, layout, colour=False)

    assert text == state.ascii(layout)
    assert text.splitlines() == ["#####", "#P.o#", "#.#G#", "#..X#", "#####"]
    assert len(text.splitlines()) == layout.height
    assert all(len(line) == layout.width for line in text.splitlines())
    assert not text.endswith("\n")


def test_ansi_with_colour_keeps_one_line_per_row():
    """The escape sequences add bytes, not lines."""
    layout = board_config().layout
    coloured = pragman.ansi(board_state(), layout)
    assert len(coloured.splitlines()) == layout.height
    assert "\x1b[" in coloured


def test_env_render_modes_return_what_the_mode_promises():
    """``rgb_array`` gives the frame, ``ansi`` gives a string."""
    picture = pragman.make("haunted_5x5", render_mode="rgb_array")
    picture.reset(seed=0)
    assert isinstance(picture.render(), np.ndarray)

    text = pragman.make("haunted_5x5", render_mode="ansi")
    _obs, info = text.reset(seed=0)
    assert isinstance(text.render(), str)
    assert pragman.ansi(info["state"], text.layout, colour=False) == info["state"].ascii(
        text.layout
    )


def test_an_unknown_render_mode_is_refused():
    with pytest.raises(ValueError, match="render_mode"):
        pragman.make("empty_5x5", render_mode="sculpture")


# ── the optional extras, which are deliberately not installed ─────────────────
#
# pygame, imageio and matplotlib are extras.  The numpy paths
# above must work without them; these three tests only pin that asking for an
# extra produces an error that says how to install it, rather than a bare
# ModuleNotFoundError from somewhere deep in the call stack.

def test_human_mode_explains_that_it_needs_pygame():
    """``render_mode="human"`` is the ``pragman[play]`` extra."""
    try:
        import pygame  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - only on a machine with the extra installed
        pytest.skip("pygame is installed, so there is no missing-extra message to check")

    env = pragman.make("empty_5x5", render_mode="human")
    with pytest.raises(ImportError, match=r"pip install 'pragman\[play\]'"):
        env.reset(seed=0)


def test_gif_export_explains_that_it_needs_imageio():
    """GIF export is the ``pragman[viz]`` extra."""
    try:
        import imageio  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover
        pytest.skip("imageio is installed, so there is no missing-extra message to check")

    frames = [pragman.frame(board_state(), board_config().layout)]
    with pytest.raises(ImportError, match=r"pip install 'pragman\[viz\]'"):
        pragman.save_gif(frames, "unwritten.gif")


def test_the_figure_helpers_explain_that_they_need_matplotlib():
    """``viz.py`` figures are the ``pragman[viz]`` extra."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover
        pytest.skip("matplotlib is installed, so there is no missing-extra message to check")

    env = pragman.make("empty_5x5")
    with pytest.raises(ImportError, match=r"pip install 'pragman\[viz\]'"):
        pragman.plot_values(env, np.zeros(25))
