"""Teaching wrappers. Currently one.

:class:`ReplaceReward` swaps the environment's reward for one you supply -- a
reward model learned from preferences, a reward recovered by inverse RL, or a
shaped reward. The environment never changes, only the number it pays.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from .state import Action, PragmanState

__all__ = ["ReplaceReward", "RewardFunction"]

#: What :class:`ReplaceReward` calls: ``(state, action, next_state, info) -> float``.
RewardFunction = Callable[[PragmanState, "int | Action", PragmanState, dict[str, Any]], float]


class ReplaceReward(gym.Wrapper):
    """Pay ``fn(state, action, next_state, info)`` instead of the true reward.

    The wrapped environment is untouched -- same dynamics, same terminations --
    so anything the agent learns to do differently under the new reward is a
    statement about the reward, not about the world.

    The environment's own reward is not thrown away: it is put into
    ``info["true_reward"]`` before ``fn`` is called, so a shaping function can
    add to it, and a training run can log the true return alongside the one the
    agent is optimizing.

    ::

        env = pragman.ReplaceReward(pragman.make("single_ghost_11x11"), my_reward)
    """

    def __init__(self, env: gym.Env, fn: RewardFunction) -> None:
        super().__init__(env)
        self.fn = fn
        self._state: PragmanState | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._state = info["state"]  # Prag-man always reports the true state
        return observation, info

    def step(self, action: int | Action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        state = self._state
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._state, info["true_reward"] = info["state"], float(reward)
        return (
            observation,
            float(self.fn(state, action, info["state"], info)),
            terminated,
            truncated,
            info,
        )
