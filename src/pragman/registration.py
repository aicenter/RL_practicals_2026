"""Gymnasium ids, so that ``gym.make("Pragman/Classic11x11-v0")`` works.

Registration happens as an import side effect of :mod:`pragman` -- Gymnasium's
entry-point plugin mechanism was removed in 1.0 -- so ``import pragman`` must
come before ``gym.make("Pragman/...")``. In lab skeletons we use
:func:`pragman.make` instead, which takes the factory name directly and returns
the unwrapped environment; the ids are here so that the Gymnasium-conformance
demonstration has something conventional to point at.

``max_episode_steps=None`` on every id, deliberately: Prag-man truncates itself
from ``cfg.max_steps``, and a ``TimeLimit`` wrapper on top would either
double-count the clock or silently override it.
"""

from __future__ import annotations

import gymnasium as gym

from . import configs
from .config import PragmanConfig
from .env import PragmanEnv

__all__ = ["ENV_IDS", "env_id", "register_all", "make_registered"]

#: Version suffix shared by every id. Bump it if the dynamics ever change in a
#: way that would invalidate published results.
VERSION = "v0"


def _id_for(name: str) -> str:
    """``"classic_11x11"`` -> ``"Pragman/Classic11x11-v0"``."""
    camel = "".join(part.capitalize() for part in name.split("_"))
    return f"Pragman/{camel}-{VERSION}"


#: Every factory name mapped to its Gymnasium id.
ENV_IDS: dict[str, str] = {name: _id_for(name) for name in configs.names()}


def env_id(name: str) -> str:
    """The Gymnasium id of a world, by factory name.

    Raises:
        ConfigError: listing the available worlds, if the name is unknown.
    """
    configs.by_name(name)  # raises a helpful ConfigError for a typo
    return ENV_IDS[name]


def make_registered(name: str, render_mode: str | None = None, **overrides) -> PragmanEnv:
    """The entry point every id points at: build a factory's world.

    Kept separate from :func:`pragman.make` because Gymnasium stores the
    factory name in the spec's ``kwargs`` and passes it back to us here.
    """
    cfg: PragmanConfig = configs.by_name(name)(**overrides)
    return PragmanEnv(cfg, render_mode=render_mode)


def register_all() -> None:
    """Register all ten ids with Gymnasium. Calling it twice is harmless."""
    for name, identifier in ENV_IDS.items():
        if identifier in gym.registry:
            continue
        gym.register(
            id=identifier,
            entry_point="pragman.registration:make_registered",
            kwargs={"name": name},
            # Prag-man owns its own clock; see the module docstring.
            max_episode_steps=None,
        )
