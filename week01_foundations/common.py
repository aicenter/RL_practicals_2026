"""Shared helper for the Week 1 scripts: build an environment from a command-line spec.

An environment spec is either

- a registered Gymnasium id, e.g. `CartPole-v1`, or
- `module:ClassName`, e.g. `week01_foundations.roomba:RoombaEnv` (or `my_env:MyEnv` for a file
  in this directory).

Extra constructor arguments can be passed as a JSON string, e.g. `{"reward": "cleanliness"}`.
"""

import importlib
import json
import sys
from pathlib import Path

import gymnasium as gym

# Make both `week01_foundations.roomba` and plain `roomba` importable, regardless of whether the
# script was started from the repository root or from this directory.
_HERE = Path(__file__).resolve().parent
for path in (_HERE, _HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def make_env(spec: str, env_kwargs: str = "{}", max_episode_steps: int | None = None) -> gym.Env:
    kwargs = json.loads(env_kwargs)
    if ":" in spec:
        module_name, class_name = spec.split(":")
        env_class = getattr(importlib.import_module(module_name), class_name)
        env = env_class(**kwargs)
    else:
        env = gym.make(spec, **kwargs)

    if max_episode_steps is not None:
        # A safety net for environments that never truncate on their own.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    # Turn any observation (Discrete, Dict, ...) into one flat vector, so that one network
    # architecture serves every environment. A Discrete observation becomes a one-hot vector.
    env = gym.wrappers.FlattenObservation(env)
    return env


def run_episodes(env: gym.Env, choose_action, episodes: int, seed: int = 0) -> list[dict]:
    """Run `episodes` episodes with `choose_action(obs) -> action` and return their statistics."""
    stats = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_return, episode_length = 0.0, 0
        while True:
            obs, reward, terminated, truncated, info = env.step(choose_action(obs))
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                break
        stats.append(
            {"return": episode_return, "length": episode_length, "terminated": bool(terminated), "truncated": bool(truncated)}
        )
    return stats
