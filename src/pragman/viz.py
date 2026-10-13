"""Matplotlib figures for the lecture notes.

:mod:`pragman.render` draws the *game*; this module draws the *analysis*.  The
split is deliberate: the numpy renderer has no text at all,
because drawing digits in pure numpy would mean shipping a bitmap font, so every
figure that wants numbers on it -- a value function, a visit-count table -- is
built here with matplotlib instead.

Three figures, one for each thing a lab wants to look at:

===========================  ==========================================
:func:`plot_values`          a value function or any per-cell number
:func:`plot_policy`          arrows, deterministic or stochastic
:func:`plot_state_visits`    where the agent has actually been
===========================  ==========================================

matplotlib is an optional extra: ``pip install 'pragman[viz]'``.  It is imported
lazily, so importing this module without it costs nothing until you plot.

Where do the numbers come from?
-------------------------------
Every function takes ``source`` first, which may be either

* a live environment (anything with a ``.layout`` attribute -- a ``PragmanEnv``),
  or
* a :class:`~pragman.layout.Layout` on its own,

and then the numbers, in whichever of these shapes you happen to hold:

``(H, W)`` array
    One number per grid cell.  Nothing to resolve; walls are ignored.
mapping
    ``{(row, col): value}``, or ``{state: value}`` -- the shape a tabular
    ``V`` or ``defaultdict`` from Weeks 4 to 6 has.
flat array of length ``S``
    One number per *state*, indexed like ``env.model.states``.  A state is more
    than an agent position (it also carries pellets and ghosts), so this form
    needs an explicit **state-index mapping**: pass ``states=model.states``, the
    same list whose order indexes ``V``.  Several states can share an agent
    cell, and ``reduce=`` says what to do about that ("mean" by default).
    If ``source`` is an env and ``states`` is omitted, ``env.model.states`` is
    used when its length matches, which is only cheap for the small enumerable
    configurations.
flat array of length ``H * W``
    Row-major per-cell values; accepted as a convenience.

Example
-------
>>> V, pi = my_value_iteration(env.model, gamma=0.95)         # doctest: +SKIP
>>> ax = plot_values(env, V, states=env.model.states)         # doctest: +SKIP
>>> plot_policy(env, pi, states=env.model.states, ax=ax)      # doctest: +SKIP
>>> ax.figure.savefig("week04_value_iteration.pdf")           # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from pragman.render import PALETTES, require_optional

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from matplotlib.axes import Axes

    from pragman.config import PragmanConfig
    from pragman.env import PragmanEnv
    from pragman.layout import Layout
    from pragman.state import PragmanState

    #: What the first argument of every figure may be: a live environment, a
    #: config, or a bare layout.  All three carry (or are) the map.
    Source = PragmanEnv | PragmanConfig | Layout

__all__ = ["plot_values", "plot_policy", "plot_state_visits", "to_cell_grid", "layout_of"]

Cell = tuple[int, int]

#: Where an action points on a matplotlib picture, as ``(dx, dy)``.  ``y`` grows
#: downwards, because row 0 of the grid is drawn at the top.
ARROW_DELTAS: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),  # UP
    (1.0, 0.0),  # RIGHT
    (0.0, 1.0),  # DOWN
    (-1.0, 0.0),  # LEFT
)

_WALL = tuple(c / 255 for c in PALETTES["high_contrast"]["wall"])
_FLOOR = tuple(c / 255 for c in PALETTES["high_contrast"]["floor"])
_GRID = tuple(c / 255 for c in PALETTES["high_contrast"]["grid"])
_GOAL = tuple(c / 255 for c in PALETTES["high_contrast"]["goal"])


# ─────────────────────────────────────────────────────────────────────────────
# Turning whatever the student is holding into an (H, W) grid
# ─────────────────────────────────────────────────────────────────────────────


def layout_of(source: Source) -> Layout:
    """The layout behind ``source``, which may be an env, a config, or a layout."""
    layout = getattr(source, "layout", None)
    if layout is not None:
        return layout
    if hasattr(source, "walls") and hasattr(source, "height"):
        return source
    raise TypeError(
        "expected an environment, a config, or a Layout as the first argument, "
        f"got {type(source).__name__}"
    )


def _reduce(values: list[float], how: str) -> float:
    """Combine the values of all states that share one agent cell."""
    array = np.asarray(values, dtype=np.float64)
    if how == "mean":
        return float(array.mean())
    if how == "sum":
        return float(array.sum())
    if how == "max":
        return float(array.max())
    if how == "min":
        return float(array.min())
    if how == "first":
        return float(array[0])
    if how == "mode":  # the most common value; what a policy wants
        unique, counts = np.unique(array, return_counts=True)
        return float(unique[int(np.argmax(counts))])
    raise ValueError(
        f"unknown reduce={how!r}; use 'mean', 'sum', 'max', 'min', 'first' or 'mode'"
    )


def to_cell_grid(
    source: Source,
    values: Any,
    states: Sequence[PragmanState] | None = None,
    reduce: str = "mean",
) -> np.ndarray:
    """Resolve per-state or per-cell numbers into an ``(H, W)`` float array.

    Cells with no value -- and every wall -- come back as ``NaN``, which is what
    the plotting functions mask out.  See the module docstring for the accepted
    shapes of ``values``; ``reduce`` says how to combine several states that
    share one agent cell ("mean", "sum", "max", "min", "first", "mode").

    This is exposed because it is occasionally useful on its own: it is how you
    turn a tabular ``V`` into something you can ``print`` as a table.
    """
    layout = layout_of(source)
    height, width = int(layout.height), int(layout.width)
    walls = np.asarray(layout.walls, dtype=bool)
    grid = np.full((height, width), np.nan, dtype=np.float64)

    if isinstance(values, Mapping):
        buckets: dict[Cell, list[float]] = {}
        for key, value in values.items():
            cell = key if _is_cell(key) else tuple(key.agent)
            buckets.setdefault((int(cell[0]), int(cell[1])), []).append(float(value))
        for (row, col), collected in buckets.items():
            grid[row, col] = _reduce(collected, reduce)
        return np.where(walls, np.nan, grid)

    array = np.asarray(values, dtype=np.float64)
    if array.shape == (height, width):
        return np.where(walls, np.nan, array)

    if array.ndim != 1:
        raise ValueError(
            f"expected an ({height}, {width}) array, a flat array, or a mapping; "
            f"got shape {array.shape}"
        )

    if states is None:
        states = _states_of(source, len(array), height * width)
    if states is not None:
        if len(states) != len(array):
            raise ValueError(
                f"values has {len(array)} entries but states has {len(states)}; "
                "they must be the same list order as model.states"
            )
        buckets = {}
        for state, value in zip(states, array, strict=True):
            cell = tuple(int(x) for x in state.agent)
            buckets.setdefault(cell, []).append(float(value))
        for (row, col), collected in buckets.items():
            grid[row, col] = _reduce(collected, reduce)
        return np.where(walls, np.nan, grid)

    if array.shape == (height * width,):
        return np.where(walls, np.nan, array.reshape(height, width))

    raise ValueError(
        f"cannot map {len(array)} numbers onto a {height}x{width} grid: pass "
        "states=model.states so that each value can be attributed to a cell"
    )


def _is_cell(key: Any) -> bool:
    """True for a ``(row, col)`` pair, false for a state object."""
    return (
        isinstance(key, tuple)
        and len(key) == 2
        and all(isinstance(part, (int, np.integer)) for part in key)
    )


def _states_of(source: Source, n_values: int, n_cells: int) -> Sequence[PragmanState] | None:
    """Fall back on ``env.model.states`` when the caller did not pass a state list.

    A flat array is read as *per state* whenever a matching state list can be
    found, and as *per cell* (row-major) only when one cannot.  That order
    matters: on ``empty_5x5`` there are exactly as many states as cells, but
    ``model.states`` is in breadth-first order, not row-major, so guessing
    "per cell" there would quietly draw the right numbers in the wrong squares.
    """
    try:
        model = getattr(source, "model", None)
    except Exception:  # noqa: BLE001 - a model that will not build is simply not usable here
        model = None
    states = getattr(model, "states", None) if model is not None else None
    if states is not None and len(states) == n_values:
        return states
    if n_values == n_cells:
        return None  # unambiguous enough: one value per cell, row-major
    raise ValueError(
        f"got {n_values} values for a grid of {n_cells} cells and no way to relate "
        "them: pass states=model.states (the list that indexes your value array)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The figures
# ─────────────────────────────────────────────────────────────────────────────


def _background(ax: Axes, layout: Layout, mark_goals: bool = True) -> None:
    """Draw walls, cell boundaries, and the goal cells, and fix the axes."""
    height, width = int(layout.height), int(layout.width)
    walls = np.asarray(layout.walls, dtype=bool)

    picture = np.empty((height, width, 3), dtype=np.float64)
    picture[:] = _FLOOR
    picture[walls] = _WALL
    ax.imshow(picture, extent=(0, width, height, 0), interpolation="nearest", zorder=0)

    if mark_goals:
        for cell in getattr(layout, "goal_cells", ()) or ():
            row, col = int(cell[0]), int(cell[1])
            ax.add_patch(
                _rectangle(col, row, facecolor="none", edgecolor=_GOAL, linewidth=2.5, zorder=3)
            )

    ax.set_xticks(np.arange(width + 1))
    ax.set_yticks(np.arange(height + 1))
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _rectangle(col: int, row: int, **kwargs: Any) -> Any:
    from matplotlib.patches import Rectangle

    return Rectangle((col, row), 1.0, 1.0, **kwargs)


def _new_axes(ax: Axes | None, layout: Layout, title: str | None) -> Axes:
    plt = require_optional("matplotlib.pyplot", "viz")
    if ax is None:
        side = 0.55
        _, ax = plt.subplots(
            figsize=(max(3.0, side * layout.width), max(3.0, side * layout.height))
        )
    if title:
        ax.set_title(title)
    return ax


def _text_colour(rgba: Sequence[float]) -> str:
    """Black or white lettering, whichever will be readable on that patch."""
    red, green, blue = rgba[0], rgba[1], rgba[2]
    return "black" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.55 else "white"


def plot_values(
    source: Source,
    values: Any,
    states: Sequence[PragmanState] | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = "{:.2f}",
    reduce: str = "mean",
    title: str | None = None,
    colourbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    """Draw a value function (or any per-cell number) as an annotated heatmap.

    This is the Week 4 figure: ``V*`` on the maze, with the number written in
    every cell.  ``values`` may be an ``(H, W)`` array, a mapping, or a flat
    per-state array together with ``states`` -- see the module docstring.

    Parameters worth knowing about: ``annotate=False`` drops the numbers (use it
    on a big maze, where they turn to soup), ``fmt`` is an ordinary format
    string, and ``reduce`` decides what to show when several states share a cell.

    Returns the :class:`~matplotlib.axes.Axes`, so ``ax.figure.savefig(...)``
    is the next line.
    """
    layout = layout_of(source)
    grid = to_cell_grid(source, values, states=states, reduce=reduce)
    plt = require_optional("matplotlib.pyplot", "viz")
    ax = _new_axes(ax, layout, title)
    _background(ax, layout)

    colours = plt.get_cmap(cmap)
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(
        masked,
        cmap=colours,
        extent=(0, layout.width, layout.height, 0),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    if colourbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    if annotate:
        normalize = image.norm
        for (row, col), value in np.ndenumerate(grid):
            if not np.isfinite(value):
                continue
            ax.text(
                col + 0.5,
                row + 0.5,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=8,
                color=_text_colour(colours(normalize(value))),
                zorder=4,
            )
    return ax


def plot_policy(
    source: Source,
    policy: Any,
    states: Sequence[PragmanState] | None = None,
    ax: Axes | None = None,
    values: Any = None,
    cmap: str = "viridis",
    reduce: str = "mode",
    title: str | None = None,
    colour: str = "black",
    show_probabilities: bool | None = None,
) -> Axes:
    """Draw a policy as one arrow per cell, optionally over a value heatmap.

    ``policy`` may be

    * deterministic: a flat array of actions (with ``states=``), an ``(H, W)``
      array of actions, or a mapping to actions -- one full-length arrow per cell;
    * stochastic: a ``(S, 4)`` or ``(H, W, 4)`` array, or a mapping to
      four probabilities -- four arrows per cell whose length and opacity follow
      the probability.  This is the picture to show in Week 8 when a policy
      gradient method is still undecided about where to go.

    Pass ``values=`` (anything :func:`plot_values` accepts) to lay the arrows over
    a value heatmap, which is the "values plus greedy policy" figure of Week 4.
    """
    layout = layout_of(source)
    ax = _new_axes(ax, layout, title)

    if values is not None:
        plot_values(
            source,
            values,
            states=states,
            ax=ax,
            cmap=cmap,
            annotate=False,
            colourbar=False,
        )
    else:
        _background(ax, layout)

    distributions = _policy_distributions(source, policy, states, reduce)
    if show_probabilities is None:
        show_probabilities = bool(
            np.any((distributions > 1e-9) & (distributions < 1 - 1e-9))
        )

    for (row, col), probabilities in _iter_cells(distributions):
        if show_probabilities:
            for action, probability in enumerate(probabilities):
                if probability > 0.02:
                    _arrow(ax, row, col, action, colour, probability)
        else:
            _arrow(ax, row, col, int(np.argmax(probabilities)), colour, 1.0)
    return ax


def _iter_cells(distributions: np.ndarray) -> Any:
    """Yield ``((row, col), probabilities)`` for every cell that has a policy."""
    height, width = distributions.shape[:2]
    for row in range(height):
        for col in range(width):
            probabilities = distributions[row, col]
            if np.isfinite(probabilities).all() and probabilities.sum() > 0:
                yield (row, col), probabilities


def _policy_distributions(
    source: Source, policy: Any, states: Sequence[PragmanState] | None, reduce: str
) -> np.ndarray:
    """Normalize any policy form to an ``(H, W, 4)`` array of probabilities."""
    layout = layout_of(source)
    height, width = int(layout.height), int(layout.width)

    if isinstance(policy, Mapping):
        sample = next(iter(policy.values()), 0)
        if np.ndim(sample) == 1:  # cell -> four probabilities
            grid = np.zeros((height, width, 4))
            for key, probabilities in policy.items():
                cell = key if _is_cell(key) else tuple(key.agent)
                grid[int(cell[0]), int(cell[1])] = probabilities
            return grid
        return _one_hot(to_cell_grid(source, policy, states=states, reduce=reduce))

    array = np.asarray(policy, dtype=np.float64)
    if array.ndim == 3 and array.shape == (height, width, 4):
        return array
    if array.ndim == 2 and array.shape[1] == 4:  # (S, 4): a stochastic tabular policy
        grid = np.zeros((height, width, 4))
        counts = np.zeros((height, width))
        if states is None:
            states = _states_of(source, len(array), height * width)
        if states is None:  # one row per cell, row-major
            return array.reshape(height, width, 4)
        if len(states) != len(array):
            raise ValueError(
                "a stochastic policy of shape (S, 4) needs states=model.states so that "
                "each row can be attributed to a cell"
            )
        for state, probabilities in zip(states, array, strict=True):
            row, col = (int(x) for x in state.agent)
            grid[row, col] += probabilities
            counts[row, col] += 1
        busy = counts > 0
        grid[busy] /= counts[busy][:, None]
        return grid
    return _one_hot(to_cell_grid(source, array, states=states, reduce=reduce))


def _one_hot(actions: np.ndarray) -> np.ndarray:
    """Turn an ``(H, W)`` array of action indices into ``(H, W, 4)`` indicators."""
    height, width = actions.shape
    grid = np.zeros((height, width, 4))
    for row in range(height):
        for col in range(width):
            value = actions[row, col]
            if not np.isfinite(value) or value < 0:
                continue
            action = int(round(float(value)))
            if not 0 <= action < 4:
                raise ValueError(
                    f"policy at {(row, col)} is {action}; actions are 0=UP, 1=RIGHT, "
                    "2=DOWN, 3=LEFT"
                )
            grid[row, col, action] = 1.0
    return grid


def _arrow(ax: Axes, row: int, col: int, action: int, colour: str, weight: float) -> None:
    """One arrow from the centre of a cell, its length set by ``weight``."""
    dx, dy = ARROW_DELTAS[action]
    length = 0.16 + 0.24 * float(weight)
    centre_x, centre_y = col + 0.5, row + 0.5
    ax.annotate(
        "",
        xy=(centre_x + length * dx, centre_y + length * dy),
        xytext=(centre_x, centre_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": colour,
            "linewidth": 1.4,
            "alpha": float(np.clip(0.25 + 0.75 * weight, 0.0, 1.0)),
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=5,
        annotation_clip=False,
    )


def plot_state_visits(
    source: Source,
    counts: Any,
    states: Sequence[PragmanState] | None = None,
    ax: Axes | None = None,
    cmap: str = "magma",
    log: bool = False,
    annotate: bool = True,
    fmt: str = "{:.0f}",
    title: str | None = None,
    colourbar: bool = True,
) -> Axes:
    """Draw how often each cell was visited -- the exploration picture.

    Counts over *states* are summed per cell (``reduce="sum"``), which is what
    "how often was the agent here" means.  ``log=True`` plots
    ``log10(1 + count)`` instead, because visit counts on a maze are usually
    dominated by the few cells near the start and everything else disappears.
    """
    grid = to_cell_grid(source, counts, states=states, reduce="sum")
    if log:
        grid = np.log10(1.0 + grid)
        fmt = "{:.1f}"
    return plot_values(
        source,
        grid,
        ax=ax,
        cmap=cmap,
        annotate=annotate,
        fmt=fmt,
        title=title,
        colourbar=colourbar,
    )
