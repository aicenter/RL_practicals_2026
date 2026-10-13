"""Prag-man: a small, readable Pac-Man-like gridworld for teaching RL.

::

    import pragman

    env = pragman.make("classic_11x11", render_mode="human")
    observation, info = env.reset(seed=0)
    while True:
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

One environment carries the course from value iteration in Week 4 to the last
practical in Week 13; :mod:`pragman.configs` lists the ten ready-made worlds and
which week each belongs to. Three layers are available at any time:

* the **environment** -- :func:`make`, :class:`PragmanEnv`: sample transitions;
* the **pure core** -- :func:`transition`, :func:`successors`: the same
  dynamics as plain functions, with no hidden state in the way;
* the **white-box model** -- ``env.model``, :func:`build_tabular_model`: the
  exact ``(P, R, rho_0)`` arrays, for the worlds small enough to enumerate.

Importing this package registers the Gymnasium ids as a side effect, so
``gym.make("Pragman/Classic11x11-v0")`` works too (Gymnasium 1.0 removed the
entry-point plugin mechanism, so the import has to come first).

The heavier and optional pieces -- :mod:`pragman.model`, :mod:`pragman.viz`
(matplotlib) and :mod:`pragman.interactive` (pygame) -- are imported on first use, so
``import pragman`` stays fast and needs nothing but numpy and gymnasium.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import configs, dynamics, errors, ghosts
from .config import PragmanConfig
from .dynamics import apply, slip_distribution, successors, transition
from .env import PragmanEnv, make, make_vec
from .errors import (
    ConfigError,
    LayoutError,
    PragmanError,
    StateInjectionDisabled,
    StateSpaceTooLarge,
)
from .ghosts import ChaserGhost, GhostPolicy, ProximityChaser, RandomGhost
from .layout import NAMED_LAYOUTS, Layout
from .observations import channel_names, observation_space, observe
from .registration import ENV_IDS, register_all
from .render import PALETTES, ansi, frame, save_gif
from .state import ACTIONS, Action, Cell, Episode, PragmanState, load_episodes, save_episodes
from .wrappers import ReplaceReward

__version__ = "0.1.0"

__all__ = [
    # the two lines every lab starts with
    "make",
    "make_vec",
    "PragmanEnv",
    "PragmanConfig",
    # the world
    "Layout",
    "NAMED_LAYOUTS",
    "PragmanState",
    "Cell",
    "Action",
    "ACTIONS",
    "Episode",
    "save_episodes",
    "load_episodes",
    "configs",
    "ENV_IDS",
    # the pure core
    "dynamics",
    "transition",
    "successors",
    "apply",
    "slip_distribution",
    # ghosts
    "ghosts",
    "GhostPolicy",
    "RandomGhost",
    "ChaserGhost",
    "ProximityChaser",
    # observations
    "observe",
    "observation_space",
    "channel_names",
    # the white-box model (imported on first use)
    "model",
    "build_tabular_model",
    "TabularModel",
    "enumerate_states",
    # rendering and figures
    "frame",
    "ansi",
    "save_gif",
    "PALETTES",
    "render",
    "viz",
    "plot_values",
    "plot_policy",
    "plot_state_visits",
    # playing and recording
    "play",
    "record_episode",
    # wrappers and errors
    "ReplaceReward",
    "errors",
    "PragmanError",
    "ConfigError",
    "LayoutError",
    "StateSpaceTooLarge",
    "StateInjectionDisabled",
    "to_cell_grid",
    "layout_of",
    "__version__",
]

# Gymnasium ids, as an import side effect. See registration.py.
register_all()


# ──────────────────────────────────────────────────────────────────────────────
# Lazily imported names (PEP 562)
#
# model.py pulls in the whole dynamic-programming machinery, viz.py needs
# matplotlib and play.py needs pygame -- the last two are optional extras.
# Importing them on first use keeps ``import pragman`` cheap, and keeps a
# missing extra from breaking an import that never needed it.
# ──────────────────────────────────────────────────────────────────────────────

_LAZY_MODULES = ("model", "viz", "interactive", "render")
_LAZY_NAMES = {
    "build_tabular_model": "model",
    "TabularModel": "model",
    "enumerate_states": "model",
    "to_cell_grid": "viz",
    "layout_of": "viz",
    "plot_values": "viz",
    "plot_policy": "viz",
    "plot_state_visits": "viz",
    "play": "interactive",
    "record_episode": "interactive",
}


def __getattr__(name: str) -> Any:
    """Import ``pragman.model``, ``pragman.viz`` and ``pragman.interactive`` on demand.

    The keyboard-play module is called ``interactive`` rather than ``play`` on
    purpose: a submodule named ``play`` would shadow the ``play()`` function
    the moment anything did ``import pragman.play``, and ``pragman.play(env)``
    -- the call the documentation shows -- would raise ``TypeError``.
    """
    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    module_name = _LAZY_NAMES.get(name)
    if module_name is not None:
        return getattr(importlib.import_module(f".{module_name}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
