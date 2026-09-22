import random
from typing import Tuple

import gymnasium as gym
from gymnasium import spaces


class MinimalThermostat(gym.Env):
    def __init__(self, target=21.0, outside=10.0):

        # Define gymnasium spaces: the observation is a single real number
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(1,))
        self.action_space = spaces.Discrete(2)  # 0 = heater off, 1 = heater on

        self.target, self.outside = target, outside

    def reset(self) -> Tuple[float, dict]:
        # Start at some random room temperature
        self.temperature = random.uniform(15.0, 25.0)
        return self.temperature, {}

    def step(self, action) -> Tuple[float, float, bool, bool, dict]:

        # Heat leaks out, the heater adds some back, and something random happens
        self.temperature += 0.2 * (self.outside - self.temperature) + 3.0 * action + random.gauss(0, 0.3)

        reward = -abs(self.temperature - self.target)

        return self.temperature, reward, False, False, {}
