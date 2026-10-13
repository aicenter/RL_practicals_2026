"""Rendering Prag-man states as pictures.

This module is the *numpy-only* core of the visual side of Prag-man.  It has no
dependency beyond numpy, so it always works, and it is deliberately small enough
to read in one sitting.

Three things live here:

``frame``
    Turn a state into an ``(H*k, W*k, 3)`` uint8 image.  This is the single
    function that knows what a Prag-man board *looks like*: it is used by
    ``render_mode="rgb_array"``, by the ``"pixels"`` observation encoding, and by
    the GIF exporter.  It is a **pure function** -- same arguments in, byte-identical
    array out -- which is what lets the pixels observation and the renderer be
    tested for byte-identity.
``ansi``
    Turn a state into a coloured string for ``render_mode="ansi"``.
``PygameWindow``
    The little window behind ``render_mode="human"``.  pygame is an optional
    extra and is imported lazily, so importing this module never needs it.

Design notes for the curious student
------------------------------------
* Everything is drawn with numpy *slicing* and a handful of cached boolean
  masks.  There is not a single loop over pixels anywhere in this file: the
  board is painted at cell resolution and then blown up with ``np.repeat``.
* The renderer is deliberately **text-free**.  Drawing digits in pure numpy
  would mean shipping a bitmap font, so numeric annotations (value functions,
  visit counts) live in :mod:`pragman.viz`, which may use matplotlib.
* Ghosts are coloured by *personality*, not by index, so that a lecture-hall
  audience can see at a glance which one is hunting and which one is wandering.

Example
-------
>>> img = frame(state, layout, cell_pixels=24, palette="high_contrast")   # doctest: +SKIP
>>> img.shape                                                             # doctest: +SKIP
(264, 264, 3)
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from pragman.config import PragmanConfig
    from pragman.layout import Layout
    from pragman.state import PragmanState

__all__ = [
    "Colour",
    "Palette",
    "PALETTES",
    "frame",
    "ansi",
    "save_gif",
    "PygameWindow",
    "egocentric_crop",
    "ghost_names_from_config",
    "require_optional",
]

Colour = tuple[int, int, int]
Palette = dict[str, Colour]
Cell = tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────────────
#
# Plain dictionaries of RGB triples, so that a lecturer who dislikes a colour can
# edit one line.  Every palette must define every key below; ``_resolve_palette``
# checks that, because a missing key would otherwise surface as a KeyError deep
# inside the drawing code.

PALETTE_KEYS = (
    "wall",
    "floor",
    "grid",
    "goal",
    "pellet",
    "agent",
    "agent_outline",
    "ghost",  # used when the personality is unknown or not supplied
    "ghost_random",
    "ghost_chaser",
    "ghost_proximity",
    "ghost_eye",
    "heat_low",
    "heat_high",
)

PALETTES: dict[str, Palette] = {
    # The arcade look: navy walls on a black floor, a yellow agent.
    "classic": {
        "wall": (24, 24, 168),
        "floor": (0, 0, 0),
        "grid": (56, 56, 56),
        "goal": (0, 190, 90),
        "pellet": (255, 184, 151),
        "agent": (255, 222, 0),
        "agent_outline": (255, 222, 0),  # no visible outline: yellow on black is enough
        "ghost": (255, 105, 180),
        "ghost_random": (0, 235, 235),
        "ghost_chaser": (255, 40, 40),
        "ghost_proximity": (255, 168, 60),
        "ghost_eye": (255, 255, 255),
        "heat_low": (58, 12, 90),
        "heat_high": (252, 232, 100),
    },
    # Colourblind-safe (Okabe & Ito hues) on white: the palette to project.
    # Walls are near-black blocks, so cell boundaries survive a bad projector,
    # and the agent is a yellow disc with a dark outline so it never washes out.
    "high_contrast": {
        "wall": (40, 40, 40),
        "floor": (255, 255, 255),
        "grid": (185, 185, 185),
        "goal": (0, 158, 115),
        "pellet": (70, 70, 70),
        "agent": (240, 228, 66),
        "agent_outline": (40, 40, 40),
        "ghost": (86, 180, 233),
        "ghost_random": (0, 114, 178),
        "ghost_chaser": (213, 94, 0),
        "ghost_proximity": (204, 121, 167),
        "ghost_eye": (255, 255, 255),
        "heat_low": (247, 251, 255),
        "heat_high": (8, 48, 107),
    },
}

#: Which palette entry paints which ghost personality.
GHOST_COLOUR_KEYS: dict[str, str] = {
    "random": "ghost_random",
    "chaser": "ghost_chaser",
    "proximity": "ghost_proximity",
}

OVERLAY_KEYS = (
    "grid",
    "heatmap",
    "heatmap_alpha",
    "heatmap_range",
    "arrows",
    "arrow_colour",
)


def _resolve_palette(palette: str | Mapping[str, Colour]) -> Palette:
    """Look up a named palette, or validate a hand-written one."""
    if isinstance(palette, str):
        try:
            resolved = dict(PALETTES[palette])
        except KeyError:
            raise ValueError(
                f"unknown palette {palette!r}; available: {sorted(PALETTES)}"
            ) from None
    elif isinstance(palette, Mapping):
        resolved = dict(palette)
    else:
        raise TypeError(f"palette must be a name or a dict of RGB triples, got {type(palette)}")

    missing = [key for key in PALETTE_KEYS if key not in resolved]
    if missing:
        raise ValueError(f"palette is missing the entries {missing}")
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Shape masks
# ─────────────────────────────────────────────────────────────────────────────
#
# Each helper returns a read-only ``(k, k)`` boolean mask saying which pixels of
# one cell belong to the shape.  They are cached, because a frame draws the same
# ghost shape once per ghost and the same pellet dot dozens of times.


def _freeze(mask: np.ndarray) -> np.ndarray:
    """Make a cached mask read-only, so a caller cannot corrupt the cache."""
    mask.setflags(write=False)
    return mask


@functools.lru_cache(maxsize=256)
def _disc_mask(size: int, radius_fraction: float) -> np.ndarray:
    """The one circular-mask helper: pixels of a ``size`` x ``size`` cell inside a disc.

    The disc is centred on the cell and has radius ``radius_fraction * size``
    (at least one pixel, so that a dot never vanishes at small ``cell_pixels``).
    Pixel centres, not corners, are tested, which keeps the shape symmetric.
    """
    centres = np.arange(size, dtype=np.float64) + 0.5
    centre = size / 2.0
    dr = centres[:, None] - centre
    dc = centres[None, :] - centre
    radius = max(1.0, radius_fraction * size)
    return _freeze(dr * dr + dc * dc <= radius * radius)


@functools.lru_cache(maxsize=64)
def _ghost_body_mask(size: int) -> np.ndarray:
    """A ghost body: an inset block with a rounded top.

    The inset leaves a one-pixel gutter of floor around the ghost, so two ghosts
    in adjacent cells never bleed into one continuous blob.
    """
    inset = max(0, size // 8)
    body = np.zeros((size, size), dtype=bool)
    body[inset : size - inset, inset : size - inset] = True

    # Round the two top corners off using the same disc helper, applied to the
    # top square of the body.  Below the dome the body stays rectangular.
    span = size - 2 * inset
    if span >= 4:
        dome_radius = span / 2.0
        dome = _disc_mask(span, 0.5)
        dome_rows = int(np.ceil(dome_radius))
        body[inset : inset + dome_rows, inset : size - inset] = dome[:dome_rows]
    return _freeze(body)


@functools.lru_cache(maxsize=64)
def _ghost_eyes_mask(size: int) -> np.ndarray:
    """Two eyes for a ghost: small squares, one pixel each at ``cell_pixels=8``."""
    eyes = np.zeros((size, size), dtype=bool)
    eye = max(1, round(0.18 * size))
    row = max(0, round(0.30 * size))
    left = max(0, round(0.24 * size))
    right = size - left - eye
    if right <= left:  # the cell is too small for two separate eyes
        return _freeze(eyes)
    eyes[row : row + eye, left : left + eye] = True
    eyes[row : row + eye, right : right + eye] = True
    return _freeze(eyes)


@functools.lru_cache(maxsize=64)
def _arrow_masks(size: int) -> tuple[np.ndarray, ...]:
    """One arrow mask per action, in ``Action`` order: UP, RIGHT, DOWN, LEFT.

    The UP arrow is drawn from a triangular head plus a stem, and the other
    three are rotations of it, so all four are pixel-for-pixel consistent.
    """
    rows = (np.arange(size, dtype=np.float64) + 0.5) / size
    cols = (np.arange(size, dtype=np.float64) + 0.5) / size
    r = rows[:, None]
    c = np.abs(cols[None, :] - 0.5)

    head = (r >= 0.18) & (r <= 0.58) & (c <= 0.55 * (r - 0.10))
    stem = (r >= 0.50) & (r <= 0.84) & (c <= max(0.06, 0.6 / size))
    up = head | stem
    # np.rot90 turns anticlockwise, so UP -> LEFT -> DOWN -> RIGHT.
    return (
        _freeze(up),
        _freeze(np.rot90(up, 3).copy()),
        _freeze(np.rot90(up, 2).copy()),
        _freeze(np.rot90(up, 1).copy()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────


def _check_cell(cell: Any, height: int, width: int, what: str) -> Cell:
    """Validate a ``(row, col)`` pair, with an error message a student can act on."""
    row, col = (int(cell[0]), int(cell[1]))
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"{what} at {(row, col)} lies outside the {height}x{width} grid")
    return row, col


def _paste(image: np.ndarray, cell: Cell, mask: np.ndarray, colour: Colour, k: int) -> None:
    """Paint ``colour`` into the pixels of one cell selected by a boolean mask."""
    row, col = cell
    block = image[row * k : (row + 1) * k, col * k : (col + 1) * k]
    block[mask] = colour


def _cell_values(
    values: Any,
    height: int,
    width: int,
    what: str,
    dtype: type = np.float64,
    empty: Any = np.nan,
) -> np.ndarray:
    """Normalize a per-cell overlay to an ``(H, W)`` array.

    Accepted forms, in order of how often they are handy in a lab:
    a mapping ``{(row, col): value}``, an ``(H, W)`` array, or a flat array of
    length ``H * W`` in row-major order.  Cells with no value get ``empty``.
    """
    if isinstance(values, Mapping):
        grid = np.full((height, width), empty, dtype=dtype)
        for cell, value in values.items():
            row, col = _check_cell(cell, height, width, f"{what} cell")
            grid[row, col] = value
        return grid

    array = np.asarray(values, dtype=dtype)
    if array.shape == (height, width):
        return array
    if array.shape == (height * width,):
        return array.reshape(height, width)
    raise ValueError(
        f"{what} must be a dict keyed by (row, col), an array of shape "
        f"{(height, width)}, or a flat array of length {height * width}; got shape {array.shape}"
    )


def require_optional(module: str, extra: str) -> Any:
    """Import an optional dependency, or explain how to install it.

    Shared by :mod:`pragman.render`, :mod:`pragman.viz` and :mod:`pragman.interactive`, so that
    every missing extra produces the same one-line fix.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        package = module.split(".")[0]
        raise ImportError(
            f"{package!r} is required for this feature but is not installed.\n"
            f"Install it with:  pip install 'pragman[{extra}]'"
        ) from exc


def ghost_names_from_config(cfg: PragmanConfig) -> tuple[str, ...]:
    """The personality names of a config's ghosts, ready to pass to :func:`frame`.

    ``frame(state, cfg.layout, ghost_names=ghost_names_from_config(cfg))`` is how
    the environment and the ``"pixels"`` observation both colour their ghosts, and
    passing the *same* arguments in both places is what keeps them byte-identical.
    """
    return tuple(str(policy.name) for policy in cfg.ghost_policies)


def _ghost_colours(
    ghost_names: Sequence[str] | None, n_ghosts: int, palette: Palette
) -> list[Colour]:
    """One colour per ghost, chosen by *personality* rather than by index.

    With no names (or an unrecognized name) every ghost is painted the palette's
    neutral ``"ghost"`` colour, which is also what a homogeneous config looks
    like: all ghosts alike, because they all behave alike.
    """
    if ghost_names is None:
        return [palette["ghost"]] * n_ghosts
    names = list(ghost_names)
    if len(names) != n_ghosts:
        raise ValueError(
            f"ghost_names has {len(names)} entries but the state has {n_ghosts} ghosts"
        )
    return [palette[GHOST_COLOUR_KEYS.get(name, "ghost")] for name in names]


# ─────────────────────────────────────────────────────────────────────────────
# The frame renderer
# ─────────────────────────────────────────────────────────────────────────────


def frame(
    state: PragmanState,
    layout: Layout,
    cell_pixels: int = 8,
    palette: str | Mapping[str, Colour] = "classic",
    overlays: Mapping[str, Any] | None = None,
    ghost_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw one Prag-man state as an RGB image.

    Parameters
    ----------
    state:
        Any object with ``.agent``, ``.ghosts`` and ``.pellets`` -- normally a
        :class:`~pragman.state.PragmanState`.
    layout:
        The static world: walls, goal cells, size.  A ``Layout`` is taken rather
        than a config on purpose: ``cfg.cell_pixels`` plus a
        separate scale argument would be two knobs for one thing, and the
        byte-identity between the ``"pixels"`` observation and ``rgb_array``
        rendering only holds while there is exactly one.
    cell_pixels:
        Side length ``k`` of one cell in pixels.  8 is the observation default;
        24 to 40 is a comfortable size for slides.  Overlay arrows need ``k`` of
        about 12 or more to be legible.
    palette:
        ``"classic"``, ``"high_contrast"``, or your own dict of RGB triples
        (see :data:`PALETTES` for the keys).
    overlays:
        Optional dict of teaching overlays, all off by default:

        ``"grid"``
            ``True`` (or an RGB triple) to draw one-pixel cell boundaries.
        ``"heatmap"``
            Per-cell numbers -- a dict ``{(row, col): value}``, an ``(H, W)``
            array, or a flat length ``H*W`` array -- tinting the floor from
            ``heat_low`` to ``heat_high``.  ``NaN`` means "no value here", and
            walls are never tinted.
        ``"heatmap_alpha"``
            Blending strength, default 0.85.
        ``"heatmap_range"``
            ``(low, high)`` for the colour scale; by default the finite values
            are min-max scaled.
        ``"arrows"``
            Per-cell action indices (``UP=0, RIGHT=1, DOWN=2, LEFT=3``) in the
            same three shapes as ``"heatmap"``; negative values and ``NaN`` mean
            "no arrow".  Handy for drawing a greedy policy.
        ``"arrow_colour"``
            One RGB triple for every arrow.  By default each arrow is drawn in
            black or white, whichever contrasts with the cell beneath it, so
            arrows stay legible on top of a heatmap.

        There is deliberately **no text overlay**: see the module docstring, and
        use :mod:`pragman.viz` when you want numbers on the picture.
    ghost_names:
        Personality names (``"random"``, ``"chaser"``, ``"proximity"``) aligned
        with ``state.ghosts``, usually from :func:`ghost_names_from_config`.
        ``None`` paints every ghost the same neutral colour.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(layout.height * k, layout.width * k, 3)``.

    Notes
    -----
    The function is **pure and deterministic**: it reads nothing but its
    arguments and returns a freshly allocated array, which is what allows the
    ``"pixels"`` observation to be exactly this picture.
    """
    k = int(cell_pixels)
    if k < 1:
        raise ValueError(f"cell_pixels must be at least 1, got {cell_pixels}")
    colours = _resolve_palette(palette)

    walls = np.asarray(layout.walls, dtype=bool)
    height, width = int(layout.height), int(layout.width)
    if walls.shape != (height, width):
        raise ValueError(f"layout.walls has shape {walls.shape}, expected {(height, width)}")

    overlays = dict(overlays or {})
    unknown = sorted(set(overlays) - set(OVERLAY_KEYS))
    if unknown:
        raise ValueError(f"unknown overlay(s) {unknown}; available: {list(OVERLAY_KEYS)}")

    # 1. Paint the static board at *cell* resolution -- one pixel per cell.
    cells = np.empty((height, width, 3), dtype=np.uint8)
    cells[:] = colours["floor"]
    cells[walls] = colours["wall"]
    for cell in getattr(layout, "goal_cells", ()) or ():
        row, col = _check_cell(cell, height, width, "goal cell")
        cells[row, col] = colours["goal"]

    if overlays.get("heatmap") is not None:
        _apply_heatmap(cells, overlays, walls, colours, height, width)

    # 2. Blow the board up to pixel resolution.  This is the only "scaling" step,
    #    and it is why cell boundaries stay perfectly crisp.
    image = np.repeat(np.repeat(cells, k, axis=0), k, axis=1)

    # 3. Grid lines, under the sprites so they never cut the agent in half.
    grid = overlays.get("grid")
    if grid is not None and grid is not False:
        grid_colour = colours["grid"] if grid is True else tuple(grid)
        image[::k, :] = grid_colour
        image[:, ::k] = grid_colour
        image[-1, :] = grid_colour
        image[:, -1] = grid_colour

    # 4. Pellets: small centred dots.
    pellet_mask = _disc_mask(k, 0.16)
    for cell in state.pellets:
        _paste(image, _check_cell(cell, height, width, "pellet"), pellet_mask, colours["pellet"], k)

    # 5. Policy arrows, if asked for.
    if overlays.get("arrows") is not None:
        _apply_arrows(image, overlays, cells, walls, height, width, k)

    # 6. Ghosts: rounded blocks with two eyes, coloured by personality.
    ghosts = tuple(state.ghosts)
    body, eyes = _ghost_body_mask(k), _ghost_eyes_mask(k)
    for cell, colour in zip(ghosts, _ghost_colours(ghost_names, len(ghosts), colours), strict=True):
        position = _check_cell(cell, height, width, "ghost")
        _paste(image, position, body, colour, k)
        _paste(image, position, eyes, colours["ghost_eye"], k)

    # 7. The agent last, so it is never hidden by anything it is standing on.
    agent = _check_cell(state.agent, height, width, "agent")
    outer = _disc_mask(k, 0.40)
    inner = _disc_mask(k, max(0.0, 0.40 - max(1.0, k / 12.0) / k))
    _paste(image, agent, outer, colours["agent_outline"], k)
    _paste(image, agent, inner, colours["agent"], k)

    return image


def egocentric_crop(
    image: np.ndarray,
    agent: Cell,
    view_radius: int,
    cell_pixels: int = 8,
    palette: str | Mapping[str, Colour] = "classic",
) -> np.ndarray:
    """Cut a ``(2r+1)`` x ``(2r+1)`` cell window out of a frame, centred on the agent.

    This is the pixel-space half of the partial-observability option:
    ``view_radius=r`` shows the agent only its immediate surroundings.
    Everything beyond the edge of the board is padded with the palette's
    **wall** colour, which is what the channel encoding does too, and the window
    is *not* rotated -- the agent has no heading.

    Parameters
    ----------
    image:
        A full-board frame from :func:`frame`, drawn with the same
        ``cell_pixels`` and ``palette`` passed here.
    agent:
        The agent's ``(row, col)``; the window is centred on this cell.
    view_radius:
        Half-width ``r`` of the window, in cells.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``((2r+1)*k, (2r+1)*k, 3)``.
    """
    k = int(cell_pixels)
    radius = int(view_radius)
    if radius < 0:
        raise ValueError(f"view_radius must be non-negative, got {view_radius}")
    colours = _resolve_palette(palette)

    side = (2 * radius + 1) * k
    window = np.empty((side, side, 3), dtype=np.uint8)
    window[:] = colours["wall"]

    height, width = image.shape[0] // k, image.shape[1] // k
    row, col = int(agent[0]), int(agent[1])
    # The window spans cells [row - r, row + r]; clip it to the board and copy
    # the overlap.  Whatever is left keeps its wall padding.
    top, left = max(0, row - radius), max(0, col - radius)
    bottom, right = min(height, row + radius + 1), min(width, col + radius + 1)
    if bottom <= top or right <= left:
        return window
    into_row, into_col = (top - (row - radius)) * k, (left - (col - radius)) * k
    patch = image[top * k : bottom * k, left * k : right * k]
    window[into_row : into_row + patch.shape[0], into_col : into_col + patch.shape[1]] = patch
    return window


def _apply_heatmap(
    cells: np.ndarray,
    overlays: Mapping[str, Any],
    walls: np.ndarray,
    colours: Palette,
    height: int,
    width: int,
) -> None:
    """Tint the non-wall cells of ``cells`` (an ``(H, W, 3)`` array) in place."""
    values = _cell_values(overlays["heatmap"], height, width, "heatmap")
    paint = np.isfinite(values) & ~walls
    if not paint.any():
        return

    low, high = overlays.get("heatmap_range", (None, None))
    low = float(np.min(values[paint])) if low is None else float(low)
    high = float(np.max(values[paint])) if high is None else float(high)
    spread = high - low
    scaled = np.zeros_like(values) if spread <= 0 else (values - low) / spread
    scaled = np.clip(np.nan_to_num(scaled), 0.0, 1.0)[..., None]

    ramp = np.array(colours["heat_low"], dtype=np.float64) * (1.0 - scaled) + np.array(
        colours["heat_high"], dtype=np.float64
    ) * scaled
    alpha = float(overlays.get("heatmap_alpha", 0.85))
    blended = alpha * ramp + (1.0 - alpha) * cells.astype(np.float64)
    cells[paint] = np.clip(np.rint(blended[paint]), 0, 255).astype(np.uint8)


def _contrasting(background: np.ndarray) -> Colour:
    """Black or white, whichever will be legible on top of ``background``.

    Uses the usual relative-luminance weights; a light cell gets a dark arrow and
    a dark cell a light one, so arrows stay visible over any heatmap.
    """
    red, green, blue = (float(channel) for channel in background)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (20, 20, 20) if luminance > 140.0 else (245, 245, 245)


def _apply_arrows(
    image: np.ndarray,
    overlays: Mapping[str, Any],
    cells: np.ndarray,
    walls: np.ndarray,
    height: int,
    width: int,
    k: int,
) -> None:
    """Stamp one arrow into every cell that names an action."""
    values = _cell_values(overlays["arrows"], height, width, "arrows")
    override = overlays.get("arrow_colour")
    masks = _arrow_masks(k)
    for row in range(height):
        for col in range(width):
            value = values[row, col]
            if walls[row, col] or not np.isfinite(value) or value < 0:
                continue
            action = int(value)
            if not 0 <= action < 4:
                raise ValueError(
                    f"arrow at {(row, col)} is {action}; actions are 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT"
                )
            colour = tuple(override) if override is not None else _contrasting(cells[row, col])
            _paste(image, (row, col), masks[action], colour, k)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal rendering
# ─────────────────────────────────────────────────────────────────────────────

_RESET = "\x1b[0m"


def _sgr(foreground: Colour, background: Colour) -> str:
    """A 24-bit-colour escape sequence.  Understood by every modern terminal."""
    fr, fg, fb = foreground
    br, bg, bb = background
    return f"\x1b[38;2;{fr};{fg};{fb};48;2;{br};{bg};{bb}m"


def ansi(
    state: PragmanState,
    layout: Layout,
    palette: str | Mapping[str, Colour] = "classic",
    colour: bool = True,
    ghost_names: Sequence[str] | None = None,
) -> str:
    """Render a state as a coloured terminal string (``render_mode="ansi"``).

    The glyphs are exactly those of the ASCII *position* form -- ``#`` wall,
    ``.`` floor, ``o`` pellet, ``X`` goal, ``G`` ghost, ``P`` agent, with the
    same precedence ``P > G > o > X > .`` -- so ``ansi(state, layout,
    colour=False)`` is exactly ``state.ascii(layout)``.  One character per cell,
    and exactly ``layout.height`` lines with no trailing newline.

    Set ``colour=False`` for plain ASCII (useful in a log file or a doctest).
    """
    colours = _resolve_palette(palette)
    walls = np.asarray(layout.walls, dtype=bool)
    height, width = int(layout.height), int(layout.width)

    glyphs = np.full((height, width), ".", dtype="<U1")
    keys = np.full((height, width), "floor", dtype="<U16")
    glyphs[walls], keys[walls] = "#", "wall"

    for cell in getattr(layout, "goal_cells", ()) or ():
        row, col = _check_cell(cell, height, width, "goal cell")
        glyphs[row, col], keys[row, col] = "X", "goal"
    for cell in state.pellets:
        row, col = _check_cell(cell, height, width, "pellet")
        glyphs[row, col], keys[row, col] = "o", "pellet"

    ghosts = tuple(state.ghosts)
    ghost_keys = _ghost_colour_keys(ghost_names, len(ghosts))
    for cell, key in zip(ghosts, ghost_keys, strict=True):
        row, col = _check_cell(cell, height, width, "ghost")
        glyphs[row, col], keys[row, col] = "G", key

    row, col = _check_cell(state.agent, height, width, "agent")
    glyphs[row, col], keys[row, col] = "P", "agent"

    lines = []
    for r in range(height):
        if not colour:
            lines.append("".join(glyphs[r]))
            continue
        pieces = []
        for c in range(width):
            key = str(keys[r, c])
            wall = key == "wall"
            background = colours["wall"] if wall else colours["floor"]
            if wall:
                foreground = colours["floor"]
            elif key == "floor":
                foreground = colours["grid"]
            else:
                foreground = colours[key]
            pieces.append(f"{_sgr(foreground, background)}{glyphs[r, c]}")
        lines.append("".join(pieces) + _RESET)
    return "\n".join(lines)


def _ghost_colour_keys(ghost_names: Sequence[str] | None, n_ghosts: int) -> list[str]:
    """Palette keys (not colours) for each ghost -- the ANSI renderer's counterpart."""
    if ghost_names is None:
        return ["ghost"] * n_ghosts
    names = list(ghost_names)
    if len(names) != n_ghosts:
        raise ValueError(
            f"ghost_names has {len(names)} entries but the state has {n_ghosts} ghosts"
        )
    return [GHOST_COLOUR_KEYS.get(name, "ghost") for name in names]


# ─────────────────────────────────────────────────────────────────────────────
# GIF export (optional extra: pragman[viz])
# ─────────────────────────────────────────────────────────────────────────────


def save_gif(frames: Iterable[np.ndarray], path: str | os.PathLike[str], fps: int = 8) -> str:
    """Write a list of frames to an animated GIF, for slides and lab reports.

    Needs imageio (``pip install 'pragman[viz]'``).  ``frames`` is anything
    iterable of ``uint8`` ``(H, W, 3)`` arrays -- typically
    ``[frame(s, layout) for s in episode.states]``.  A cell size of 16 to 24
    pixels makes a GIF that is readable in a slide without being huge.

    Returns the path, so that ``print(save_gif(frames, "run.gif"))`` is one line.
    """
    stack = [np.asarray(f) for f in frames]
    if not stack:
        raise ValueError("save_gif needs at least one frame")
    shapes = {f.shape for f in stack}
    if len(shapes) != 1:
        raise ValueError(f"all frames must have the same shape, got {sorted(shapes)}")
    if any(f.dtype != np.uint8 for f in stack):
        raise ValueError("frames must be uint8 arrays, as returned by frame()")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    imageio = require_optional("imageio.v3", "viz")
    movie = np.stack(stack)
    duration_ms = 1000.0 / float(fps)  # imageio v3 times frames in milliseconds
    try:
        imageio.imwrite(path, movie, extension=".gif", duration=duration_ms, loop=0)
    except TypeError:  # pragma: no cover - a writer that ignores these keywords
        imageio.imwrite(path, movie, extension=".gif")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# The pygame window behind render_mode="human" (optional extra: pragman[play])
# ─────────────────────────────────────────────────────────────────────────────


class PygameWindow:
    """A minimal pygame window that shows frames and blocks to a frame rate.

    This is what ``render_mode="human"`` uses.  It owns no game logic at all: you
    hand it arrays from :func:`frame` and it puts them on the screen.

    >>> window = PygameWindow(fps=8)                      # doctest: +SKIP
    >>> window.show(frame(state, layout, cell_pixels=16)) # doctest: +SKIP
    >>> window.close()                                    # doctest: +SKIP

    Frames are integer-upscaled so that a 88x88 pixel board is still visible from
    the back of a lecture hall; the factor is chosen on the first frame so that
    the longest side is at least ``min_pixels``.  Needs pygame
    (``pip install 'pragman[play]'``).
    """

    def __init__(
        self,
        fps: int = 8,
        scale: int | None = None,
        min_pixels: int = 480,
        title: str = "Prag-man",
    ) -> None:
        self.pygame = require_optional("pygame", "play")
        self.fps = int(fps)
        self.scale = None if scale is None else max(1, int(scale))
        self.min_pixels = int(min_pixels)
        self.title = title
        #: Set to True once the user closes the window or presses Escape.
        self.closed = False
        self._screen: Any = None
        self._clock: Any = None
        self._pending: list[Any] = []

    # -- lifecycle ----------------------------------------------------------
    def _ensure_screen(self, height: int, width: int) -> None:
        if self._screen is not None:
            return
        pygame = self.pygame
        pygame.init()
        pygame.display.init()
        if self.scale is None:
            self.scale = max(1, -(-self.min_pixels // max(height, width)))
        self._screen = pygame.display.set_mode((width * self.scale, height * self.scale))
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()

    def close(self) -> None:
        """Shut the window down.  Safe to call twice."""
        if self._screen is not None:
            self.pygame.display.quit()
            self.pygame.quit()
            self._screen = None
        self.closed = True

    def __enter__(self) -> PygameWindow:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- drawing ------------------------------------------------------------
    def show(self, image: np.ndarray, tick: bool = True) -> None:
        """Display one frame, then sleep until the next tick at ``fps``.

        Once the window has been closed this quietly does nothing, so closing it
        mid-training stops the display rather than crashing the run.
        """
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) uint8 frame, got shape {image.shape}")
        if self.closed and self._screen is None:
            return
        height, width = image.shape[:2]
        self._ensure_screen(height, width)
        assert self.scale is not None  # set by _ensure_screen

        # pygame surfaces are indexed (x, y), numpy images (row, col): hence the swap.
        surface = self.pygame.surfarray.make_surface(np.ascontiguousarray(image.swapaxes(0, 1)))
        if self.scale != 1:
            surface = self.pygame.transform.scale(
                surface, (width * self.scale, height * self.scale)
            )
        self._screen.blit(surface, (0, 0))
        self.pygame.display.flip()
        self._drain_events()
        if tick:
            self._clock.tick(self.fps)

    # -- events -------------------------------------------------------------
    def _drain_events(self) -> None:
        if self._screen is None:
            return
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.closed = True
            self._pending.append(event)
        del self._pending[:-256]  # never grow without bound

    def events(self) -> list[Any]:
        """Return (and forget) the pygame events seen since the last call."""
        self._drain_events()
        events, self._pending = self._pending, []
        return events
