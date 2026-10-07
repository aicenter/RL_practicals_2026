"""Handwritten digits (MNIST) as a contextual bandit (Week 3).

Every step shows the agent one image of a handwritten digit (the context x_t),
the agent names a digit (the action A_t), and the reward R_t says only whether
it was right:

    problem 1 ("digits"):           10 actions, R_t = 1 if A_t is the digit, else 0.
    problem 2 ("digits or pass"):   11 actions: the 10 digits as above, plus action 10,
                                    "pass", which always gives R_t = PASS_REWARD = 0.5.

The label itself is never revealed to the agent, which is the difference from
supervised learning. One episode is one whole learning run of `horizon` steps
(truncated, never terminated); each `reset(seed=...)` shuffles the 60 000
training images into a new order, so two agents run with the same seed see the
same images in the same order ("common random numbers").

Observations are dictionaries with two versions of the same context:
    obs["pca"]     51 numbers: the first 50 principal components of the image
                   (scaled so that their mean squared length is 1), and a constant 1;
    obs["pixels"]  784 numbers: the pixel intensities in [0, 1].

Evaluation (not for agents!): `env.unwrapped.label` is the true digit of the
current image, `info["label"]` that of the image just answered. The regret of a
step is 1 - R_t: an oracle that knows the label always gets reward 1.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlcourse import mnist

PASS_REWARD = 0.5
N_ACTIONS = {1: 10, 2: 11}


def reward(problem: int, actions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rewards of `actions` for images with true digits `labels` (works on arrays)."""
    actions, labels = np.asarray(actions), np.asarray(labels)
    r = (actions == labels).astype(float)
    if problem == 2:
        r[actions == 10] = PASS_REWARD
    return r


class DigitBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, problem: int = 1, horizon: int = 60_000):
        if problem not in N_ACTIONS:
            raise ValueError(f"problem must be 1 or 2, not {problem}")
        self.problem, self.horizon = problem, horizon
        self.pca, _ = mnist.features("pca")
        self.pixels, _ = mnist.features("pixels")
        self.labels = mnist.load()["y_train"].astype(np.int64)
        self.action_space = spaces.Discrete(N_ACTIONS[problem])
        self.observation_space = spaces.Dict({
            "pca": spaces.Box(-np.inf, np.inf, shape=(self.pca.shape[1],), dtype=np.float32),
            "pixels": spaces.Box(0.0, 1.0, shape=(784,), dtype=np.float32)})
        self.order = np.empty(0, dtype=np.int64)
        self.t = 0

    def _obs(self) -> dict:
        i = self.order[min(self.t, self.horizon - 1)]
        return {"pca": self.pca[i], "pixels": self.pixels[i]}

    @property
    def label(self) -> int:
        """True digit of the current image (for evaluation only)."""
        return int(self.labels[self.order[self.t]])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        n = len(self.labels)              # more than one pass if horizon > number of images
        self.order = np.concatenate([self.np_random.permutation(n)
                                     for _ in range(math.ceil(self.horizon / n))])[:self.horizon]
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        label = self.label
        r = float(reward(self.problem, [int(action)], [label])[0])
        self.t += 1
        return self._obs(), r, False, self.t >= self.horizon, {"label": label}
