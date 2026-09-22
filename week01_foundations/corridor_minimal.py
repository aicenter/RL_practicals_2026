"""The corridor in as few lines as possible. Compare with corridor.py."""

import random

import gymnasium as gym
from gymnasium import spaces


class MinimalCorridor(gym.Env):
    def __init__(self, n_cells=8, slip=0.1):
        self.n_cells, self.slip = n_cells, slip
        self.observation_space = spaces.Discrete(n_cells)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.pos = 0
        return self.pos, {}

    def step(self, action):
        direction = 1 if action == 1 else -1
        if random.random() < self.slip:
            direction = -direction
        self.pos = min(max(self.pos + direction, 0), self.n_cells - 1)
        terminated = self.pos == self.n_cells - 1
        return self.pos, -1.0, terminated, False, {}
