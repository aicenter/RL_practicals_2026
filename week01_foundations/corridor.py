"""Corridor: the smallest environment worth looking at.

A one-dimensional corridor of `n_cells` cells. The agent starts at the left end and wants to
reach the goal at the right end. At every step it chooses LEFT or RIGHT, but the floor is
slippery: with probability `slip` it moves the other way. Every step costs 1, so the agent
wants to get to the goal fast.

Why this environment?

- It is *tabular*: a handful of states and two actions, so the whole dynamics fit in one table
  `P[s, a, s']` = probability of landing in `s'` after taking `a` in `s`. Dynamic programming
  (Weeks 3-4) works directly with such a table. Gymnasium's own FrozenLake stores its dynamics
  the same way (`env.unwrapped.P`).
- It shows the two different ways an episode can end:
    * `terminated`: the goal is reached, the task is over;
    * `truncated`: we ran out of time, the task is *not* over, we just stop watching.

Play it from the terminal:

    uv run python week01_foundations/corridor.py

or interactively:

    >>> from corridor import CorridorEnv
    >>> env = CorridorEnv()
    >>> env.reset(seed=0)
    >>> env.step(1)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LEFT, RIGHT = 0, 1


class CorridorEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, n_cells: int = 8, slip: float = 0.1, max_steps: int = 30):
        self.n_cells = n_cells
        self.slip = slip
        self.max_steps = max_steps
        self.goal = n_cells - 1

        # What the agent sees and what it can do. The observation is simply the cell index.
        self.observation_space = spaces.Discrete(n_cells)
        self.action_space = spaces.Discrete(2)

        # The whole MDP in one table: P[s, a, s'] = probability of s' after taking a in s.
        self.P = np.zeros((n_cells, 2, n_cells))
        for s in range(n_cells):
            for a in (LEFT, RIGHT):
                intended = s - 1 if a == LEFT else s + 1
                slipped = s + 1 if a == LEFT else s - 1
                # Walking into the end wall keeps us where we are (hence the clip).
                self.P[s, a, np.clip(intended, 0, n_cells - 1)] += 1 - slip
                self.P[s, a, np.clip(slipped, 0, n_cells - 1)] += slip
        # The goal is absorbing: once there, we stay. (Irrelevant for play, since we terminate,
        # but it makes P a proper transition matrix for dynamic programming later.)
        self.P[self.goal] = 0.0
        self.P[self.goal, :, self.goal] = 1.0

        self.pos = 0
        self.t = 0

    def reset(self, seed=None, options=None):
        # This seeds self.np_random. Always use self.np_random, never the global `random` or
        # `np.random`, otherwise `reset(seed=...)` does nothing and your runs are not
        # reproducible.
        super().reset(seed=seed)
        self.pos = 0
        self.t = 0
        return self.pos, {}

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action!r}"

        # Sample the next state from the table. That is all the dynamics there are.
        self.pos = int(self.np_random.choice(self.n_cells, p=self.P[self.pos, action]))
        self.t += 1

        reward = -1.0  # every step costs one; for DP we could also store this as a table R[s, a, s']
        terminated = self.pos == self.goal  # the task is done
        truncated = self.t >= self.max_steps  # the task is not done, we just stop the episode
        # (Gymnasium's TimeLimit wrapper does exactly this truncation for you; we do it by hand
        # here so that you can see it.)
        return self.pos, reward, terminated, truncated, {}

    def render(self):
        cells = ["." for _ in range(self.n_cells)]
        cells[self.goal] = "G"
        cells[self.pos] = "A"
        return f"t={self.t:2d}  [ {' '.join(cells)} ]"


def play(env: CorridorEnv) -> None:
    keys = {"a": LEFT, "d": RIGHT}
    obs, info = env.reset()
    print(env.render())
    total = 0.0
    while True:
        key = input("action (a = left, d = right, q = quit): ").strip().lower()
        if key == "q":
            return
        if key not in keys:
            continue
        obs, reward, terminated, truncated, info = env.step(keys[key])
        total += reward
        print(env.render())
        print(
            f"      obs={obs}  reward={reward:+.1f}  terminated={terminated}  "
            f"truncated={truncated}  return so far={total:.1f}"
        )
        if terminated or truncated:
            print("Episode over.")
            return


if __name__ == "__main__":
    play(CorridorEnv())
