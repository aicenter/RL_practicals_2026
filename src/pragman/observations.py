"""What the agent actually sees: the four observation encodings.

:func:`observe` is **public and pure**. That matters more than it looks: Week
12's inverse-RL lab needs features phi(s) over ``model.states`` -- states it
never visited in an environment -- and Week 11 re-encodes recorded
:class:`~pragman.state.Episode` objects long after the environment that
produced them was closed. Both call this function directly, and there is a test
that what it returns for a state equals what ``env.step`` returned when the
environment was in that state.

The four encodings, in the order the course meets them:

===============  ====================================  ==================
``observation``  space                                 first used in
===============  ====================================  ==================
``"index"``      ``Discrete(S)``                       Week 4 (tabular)
``"coords"``     ``Box(int32, (2 + 2*n_ghosts,))``     Week 6 (linear FA)
``"channels"``   ``Box(uint8, (H, W, C))``             Week 7 (conv nets)
``"pixels"``     ``Box(uint8, (H*k, W*k, 3))``         Week 7 (from pixels)
===============  ====================================  ==================

**Both array encodings are channel-last** ``(H, W, C)``. Binary planes are
closer to per-cell features than to an image, features-last is the direction
the numerical ecosystem has settled on (PyTorch being the historical outlier),
and -- the decisive reason -- ``channels`` and ``pixels`` then differ *only* in
how a cell is drawn, so a lab can swap one for the other with a single config
change and watch what it does to learning. The price is one
``x = x.permute(0, 3, 1, 2)`` before the first ``nn.Conv2d``, which the example
networks write as the visible first line of ``forward``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from .errors import PragmanError
from .render import egocentric_crop, frame, ghost_names_from_config

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .config import PragmanConfig
    from .state import PragmanState

__all__ = [
    "observe",
    "observation_space",
    "channel_names",
    "pixel_frame",
    "MAX_INDEX_STATES",
]

#: How many states the ``"index"`` encoding is willing to enumerate before it
#: gives up with :class:`~pragman.errors.StateSpaceTooLarge`. The same cap as
#: :func:`pragman.model.build_tabular_model` uses, for the same reason: a
#: tabular encoding you cannot build a table for is no use to anybody.
MAX_INDEX_STATES = 1500


# ──────────────────────────────────────────────────────────────────────────────
# The public entry points
# ──────────────────────────────────────────────────────────────────────────────


def observe(state: PragmanState, cfg: PragmanConfig) -> np.ndarray | int:
    """Encode *state* the way *cfg* asks for.

    Pure: it reads nothing but its two arguments, so it can be applied to a
    state that came out of a tabular model, out of a saved episode, or out of a
    live environment, and gives the same answer for all three.

    Args:
        state: the situation to encode.
        cfg: the world, whose ``observation``, ``view_radius``, ``cell_pixels``
            and ``palette`` fields decide the encoding.

    Returns:
        An ``int`` for ``observation="index"``, otherwise a numpy array
        matching ``observation_space(cfg)``.

    Raises:
        StateSpaceTooLarge: under ``"index"``, if the reachable state space is
            bigger than :data:`MAX_INDEX_STATES`.
        ValueError: under ``"index"``, if *state* is not reachable in this
            world and therefore has no index -- usually a hand-written state
            that does not match the config it was injected into.
    """
    encoding = cfg.observation
    if encoding == "index":
        return _index(state, cfg)
    if encoding == "coords":
        return _coords(state, cfg)
    if encoding == "channels":
        return _channels(state, cfg)
    if encoding == "pixels":
        return _pixels(state, cfg)
    # Unreachable: PragmanConfig validates the field. Kept so that adding an
    # encoding to the config without adding it here fails loudly.
    raise PragmanError(f"no encoder for observation={encoding!r}")


def observation_space(cfg: PragmanConfig) -> spaces.Space:
    """The Gymnasium space that :func:`observe` fills for this config.

    ``C`` varies with the config (see :func:`channel_names`), so skeleton code
    should read it from ``env.observation_space.shape[-1]`` rather than
    hard-coding a number.
    """
    encoding = cfg.observation
    if encoding == "index":
        states, _ = _enumeration(cfg)
        return spaces.Discrete(len(states))

    if encoding == "coords":
        height, width = cfg.layout.shape
        # Per-element bounds, so that the space says "this is a row" and "this
        # is a column" rather than just "some number on the board".
        high = np.array([height - 1, width - 1] * (1 + cfg.n_ghosts), dtype=np.int32)
        return spaces.Box(low=np.zeros_like(high), high=high, dtype=np.int32)

    height, width = _view_shape(cfg)
    if encoding == "channels":
        n_channels = len(channel_names(cfg))
        return spaces.Box(low=0, high=1, shape=(height, width, n_channels), dtype=np.uint8)

    k = cfg.cell_pixels
    return spaces.Box(low=0, high=255, shape=(height * k, width * k, 3), dtype=np.uint8)


def channel_names(cfg: PragmanConfig) -> tuple[str, ...]:
    """The name of every plane of the ``"channels"`` encoding, in order.

    The fixed order is ``walls``, ``agent``, the ghost plane(s), ``pellets``,
    and ``goal`` -- the last only when the layout actually has goal cells,
    which is why ``C`` is a property of the config rather than a constant.

    Ghosts get **one plane per distinct personality**: a homogeneous config has
    a single ``ghosts`` plane (all its ghosts behave alike, so telling them
    apart would not help), while ``ghost_policy="mix"`` produces
    ``ghosts_random``, ``ghosts_chaser``, ``ghosts_proximity`` in the order the
    personalities first appear among ``cfg.ghost_policies`` -- that is, in the
    ghosts' row-major spawn order, the same enumeration order used everywhere
    else. Without this, ghosts that behave differently would look identical to
    the agent, quietly turning ``mix`` into a partial-observability problem
    instead of the harder-dynamics problem it is meant to be.

    A config with no ghosts keeps an all-zero ``ghosts`` plane, so that the
    four basic planes are always present and a network written for one world
    still accepts observations from a ghost-free one.
    """
    goal = ("goal",) if cfg.layout.goal_cells else ()
    return ("walls", "agent") + _ghost_planes(cfg) + ("pellets",) + goal


def pixel_frame(state: PragmanState, cfg: PragmanConfig) -> np.ndarray:
    """Draw the whole board exactly as this config wants it drawn.

    This is *the* function that turns a state into pixels: the ``"pixels"``
    observation crops its output, and ``PragmanEnv.render("rgb_array")``
    returns it unchanged. Having one call site for the arguments
    (``cell_pixels``, ``palette``, ghost colours) is what makes the two
    byte-identical -- the picture you learn from is the picture you project.
    """
    return frame(
        state,
        cfg.layout,
        cell_pixels=cfg.cell_pixels,
        palette=cfg.palette,
        ghost_names=ghost_names_from_config(cfg),
    )


# ──────────────────────────────────────────────────────────────────────────────
# index: one integer per reachable state (Weeks 4-6)
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=8)
def _enumeration(cfg: PragmanConfig) -> tuple[tuple, dict]:
    """Enumerate the reachable states once per config, and remember them.

    Imported lazily because :mod:`pragman.model` imports this module for its
    own feature work; keeping the edge one-directional at import time means the
    order in which a student imports things never matters.
    """
    from .model import enumerate_states

    states, index = enumerate_states(cfg, max_states=MAX_INDEX_STATES)
    return tuple(states), index


def _index(state: PragmanState, cfg: PragmanConfig) -> int:
    states, index = _enumeration(cfg)
    try:
        return int(index[state])
    except KeyError:
        raise ValueError(
            f"state {state} is not reachable in this world, so it has no tabular index. "
            f"The enumeration found {len(states)} reachable states; a hand-written state "
            "usually misses because of the number of ghosts, a pellet that should already "
            "have been eaten, or a cell that is a wall"
        ) from None


# ──────────────────────────────────────────────────────────────────────────────
# coords: the positions, and nothing else (Week 6)
# ──────────────────────────────────────────────────────────────────────────────


def _coords(state: PragmanState, cfg: PragmanConfig) -> np.ndarray:
    """Agent row and column, then each ghost's, as ``int32``.

    Deliberately **not Markov** once there are pellets on the board: two boards
    with the same positions but different pellets look identical here. That is
    a discussion point rather than a bug -- it is the cheapest way to show a
    class what "the observation is not the state" costs you.
    """
    values = [state.agent[0], state.agent[1]]
    for ghost in state.ghosts[: cfg.n_ghosts]:
        values.extend((ghost[0], ghost[1]))
    return np.array(values, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# channels: binary planes (Week 7 onwards)
# ──────────────────────────────────────────────────────────────────────────────


def _ghost_planes(cfg: PragmanConfig) -> tuple[str, ...]:
    """The ghost plane names: one ``ghosts`` plane, or one per personality."""
    distinct = list(dict.fromkeys(cfg.ghost_policy_names))
    if len(distinct) <= 1:
        return ("ghosts",)
    return tuple(f"ghosts_{name}" for name in distinct)


def _ghost_plane_per_ghost(cfg: PragmanConfig) -> tuple[str, ...]:
    """Which plane each ghost is drawn on, index-aligned with ``state.ghosts``."""
    if len(set(cfg.ghost_policy_names)) <= 1:
        return ("ghosts",) * cfg.n_ghosts
    return tuple(f"ghosts_{name}" for name in cfg.ghost_policy_names)


def _channels(state: PragmanState, cfg: PragmanConfig) -> np.ndarray:
    """Binary planes of shape ``(H, W, C)``, or ``(2r+1, 2r+1, C)`` when cropped."""
    layout = cfg.layout
    names = channel_names(cfg)
    plane_of = {name: position for position, name in enumerate(names)}

    planes = np.zeros((layout.height, layout.width, len(names)), dtype=np.uint8)
    planes[:, :, plane_of["walls"]] = layout.walls.astype(np.uint8)
    planes[state.agent[0], state.agent[1], plane_of["agent"]] = 1
    for ghost, plane in zip(state.ghosts, _ghost_plane_per_ghost(cfg), strict=True):
        planes[ghost[0], ghost[1], plane_of[plane]] = 1
    pellet_plane = plane_of["pellets"]
    for row, col in state.pellets:
        planes[row, col, pellet_plane] = 1
    if "goal" in plane_of:
        goal_plane = plane_of["goal"]
        for row, col in layout.goal_cells:
            planes[row, col, goal_plane] = 1

    if cfg.view_radius is None:
        return planes
    return _crop_channels(planes, state.agent, cfg.view_radius, plane_of["walls"])


def _crop_channels(
    planes: np.ndarray, agent: tuple[int, int], view_radius: int, wall_plane: int
) -> np.ndarray:
    """Cut a ``(2r+1, 2r+1)`` window out of *planes*, centred on the agent.

    Everything beyond the edge of the board is padded with **wall**, which is
    the same thing :func:`pragman.render.egocentric_crop` does in pixel space
    and the same thing :meth:`pragman.layout.Layout.is_wall` says about
    out-of-bounds cells. The window is *not* rotated: the agent has no heading,
    so "up" always means north, which is far easier to reason about (and to
    draw on a slide) than MiniGrid's rotating frame.
    """
    height, width, n_channels = planes.shape
    side = 2 * view_radius + 1
    window = np.zeros((side, side, n_channels), dtype=np.uint8)
    window[:, :, wall_plane] = 1

    row, col = agent
    top, left = max(0, row - view_radius), max(0, col - view_radius)
    bottom, right = min(height, row + view_radius + 1), min(width, col + view_radius + 1)
    if bottom <= top or right <= left:  # the whole window is off the board
        return window
    into_row, into_col = top - (row - view_radius), left - (col - view_radius)
    patch = planes[top:bottom, left:right]
    window[into_row : into_row + patch.shape[0], into_col : into_col + patch.shape[1]] = patch
    return window


# ──────────────────────────────────────────────────────────────────────────────
# pixels: the picture itself (Week 7)
# ──────────────────────────────────────────────────────────────────────────────


def _pixels(state: PragmanState, cfg: PragmanConfig) -> np.ndarray:
    """The rendered board, cropped to the agent's window if there is one."""
    image = pixel_frame(state, cfg)
    if cfg.view_radius is None:
        return image
    return egocentric_crop(
        image,
        state.agent,
        cfg.view_radius,
        cell_pixels=cfg.cell_pixels,
        palette=cfg.palette,
    )


def _view_shape(cfg: PragmanConfig) -> tuple[int, int]:
    """The size of the observed window in *cells*: the board, or ``(2r+1)`` square."""
    if cfg.view_radius is None:
        return cfg.layout.shape
    side = 2 * cfg.view_radius + 1
    return (side, side)
