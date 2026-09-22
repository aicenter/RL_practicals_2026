"""Thermostat: a continuous state is not scary.

A room loses heat to the outside; a heater adds some back. Each step the agent switches the
heater ON or OFF and is rewarded for keeping the temperature close to a target. The state is a
single floating-point number and the physics is two lines.

Why this environment?

- The state is *continuous*. Nothing else changes: `reset` and `step` look exactly like in a
  tabular environment, only the observation space is a `Box` instead of a `Discrete`.
- The task never ends by itself. There is no "goal reached", so the episode can only be
  `truncated` by a time limit, never `terminated`. (Should reaching the target temperature end
  the episode? Think about what the agent would learn if it did.)
- A *non-Markov* trap. The heater has a lag: switching it on now only warms the room from the
  next step onwards. If the agent sees only the temperature, it cannot tell whether the heater is
  already on, so two identical observations may call for different actions. Try
  `ThermostatEnv(observe_heater=False)` and see whether you can control it well by hand.

Play it from the terminal:

    uv run python week01_foundations/thermostat.py                 # observation = (temperature, heater)
    uv run python week01_foundations/thermostat.py --hide-heater   # observation = temperature only
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OFF, ON = 0, 1


class ThermostatEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        target: float = 21.0,
        outside: float = 10.0,
        observe_heater: bool = True,
        max_steps: int = 50,
    ):
        self.target = target
        self.outside = outside
        self.observe_heater = observe_heater
        self.max_steps = max_steps

        # Physics constants: fraction of the temperature difference lost per step, degrees the
        # heater adds per step, and the size of random disturbances (doors, sun, ...).
        self.leak = 0.2
        self.heat = 3.0
        self.noise = 0.3

        # A continuous observation: a Box of shape (2,) or (1,), depending on what we reveal.
        if observe_heater:
            self.observation_space = spaces.Box(
                low=np.array([-50.0, 0.0], dtype=np.float32),
                high=np.array([50.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.temperature = 0.0
        self.heater_on = False
        self.t = 0

    def _observation(self) -> np.ndarray:
        # Build a fresh array every time. Handing out a reference to internal state is a classic
        # bug: the agent stores the observation, the environment then mutates it in place, and
        # the stored "past" observation quietly changes.
        if self.observe_heater:
            return np.array([self.temperature, float(self.heater_on)], dtype=np.float32)
        return np.array([self.temperature], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temperature = float(self.np_random.uniform(15.0, 25.0))
        self.heater_on = False
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action!r}"

        # The physics: heat leaks out, the heater (as set by the *previous* action) adds heat,
        # and something random happens.
        self.temperature += (
            self.leak * (self.outside - self.temperature)
            + self.heat * self.heater_on
            + self.np_random.normal(0.0, self.noise)
        )
        # The lag: this action only affects the next transition.
        self.heater_on = action == ON
        self.t += 1

        reward = -abs(self.temperature - self.target)
        terminated = False  # a thermostat's job is never "done"
        truncated = self.t >= self.max_steps
        return self._observation(), reward, terminated, truncated, {}

    def render(self):
        # A bar from 5 to 35 degrees, with the target marked '|' and the temperature '*'.
        lo, hi, width = 5.0, 35.0, 40
        bar = [" "] * width
        bar[int((self.target - lo) / (hi - lo) * (width - 1))] = "|"
        bar[int(np.clip((self.temperature - lo) / (hi - lo), 0, 1) * (width - 1))] = "*"
        heater = "ON " if self.heater_on else "off"
        return f"t={self.t:2d}  T={self.temperature:5.1f} C  heater {heater}  [{''.join(bar)}]"


def play(env: ThermostatEnv) -> None:
    keys = {"0": OFF, "1": ON}
    obs, info = env.reset()
    print(env.render())
    total = 0.0
    while True:
        key = input("action (0 = heater off, 1 = heater on, q = quit): ").strip().lower()
        if key == "q":
            return
        if key not in keys:
            continue
        obs, reward, terminated, truncated, info = env.step(keys[key])
        total += reward
        print(env.render())
        print(
            f"      obs={np.round(obs, 1)}  reward={reward:+.1f}  terminated={terminated}  "
            f"truncated={truncated}  return so far={total:.1f}"
        )
        if terminated or truncated:
            print("Episode over.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hide-heater", action="store_true", help="observe only the temperature")
    args = parser.parse_args()
    play(ThermostatEnv(observe_heater=not args.hide_heater))
