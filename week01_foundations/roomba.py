"""Roomba: a small grid world with a battery, a dock, and a flowerpot.

A robot vacuum cleaner drives around a room. Driving onto a dirty cell cleans it. Every move
drains the battery; standing on the dock recharges it. Somewhere in the room there is a wobbly
flowerpot: bumping into it spills fresh soil onto the cells around it.

Why this environment?

- It is about as complex as the worlds in the envelope exercise, so it is a template for your
  own environment.
- Both kinds of episode end appear: `terminated` when the battery dies or the room is clean,
  `truncated` when time runs out.
- *Reward hacking*. With `reward="dirt_removed"` the robot is paid per cell cleaned. What is the
  best policy? (Hint: the flowerpot is an infinite source of reward.) With
  `reward="cleanliness"` it is instead penalised for every dirty cell in the room at every step.
  Same room, same robot, very different optimal behaviour: reward design *is* the task design.
- *Partial observability*. With `view="full"` the robot sees the whole room. With
  `view="local"` it only sees the 3x3 patch around itself, which is closer to a real robot
  and makes the observation non-Markov: it no longer knows where the remaining dirt is.

Play it from the terminal:

    uv run python week01_foundations/roomba.py
    uv run python week01_foundations/roomba.py --reward cleanliness --view local
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Actions.
UP, RIGHT, DOWN, LEFT, STAY = 0, 1, 2, 3, 4
MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1), STAY: (0, 0)}

# Cell codes in the observation.
CLEAN, DIRT, DOCK, POT, ROBOT, WALL = 0, 1, 2, 3, 4, 5
SYMBOLS = {CLEAN: ".", DIRT: "*", DOCK: "D", POT: "P", ROBOT: "R", WALL: "#"}

# The room. D = dock (the robot starts here), * = dirt, P = flowerpot, . = clean floor.
DEFAULT_LAYOUT = """
D..*..
.*..P.
..*...
*....*
"""


class RoombaEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        layout: str = DEFAULT_LAYOUT,
        reward: str = "dirt_removed",  # or "cleanliness"
        view: str = "full",  # or "local"
        battery_capacity: int = 25,
        max_steps: int = 100,
    ):
        assert reward in ("dirt_removed", "cleanliness")
        assert view in ("full", "local")
        self.reward_mode = reward
        self.view = view
        self.battery_capacity = battery_capacity
        self.max_steps = max_steps

        # Parse the layout into a fixed part (dock, pot, initial dirt) that reset() copies.
        rows = [row for row in layout.strip().splitlines()]
        self.height, self.width = len(rows), len(rows[0])
        self.dock = self.pot = None
        self.initial_dirt = np.zeros((self.height, self.width), dtype=bool)
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "D":
                    self.dock = (r, c)
                elif ch == "P":
                    self.pot = (r, c)
                elif ch == "*":
                    self.initial_dirt[r, c] = True
        assert self.dock is not None and self.pot is not None, "layout needs a dock D and a pot P"

        # The observation is a dictionary: a grid of cell codes plus the battery level.
        grid_shape = (self.height, self.width) if view == "full" else (3, 3)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=WALL, shape=grid_shape, dtype=np.int8),
                "battery": spaces.Discrete(battery_capacity + 1),
            }
        )
        self.action_space = spaces.Discrete(5)

        # The changing part of the state.
        self.robot = self.dock
        self.dirty = self.initial_dirt.copy()
        self.battery = battery_capacity
        self.t = 0

    # ---- state -> observation -----------------------------------------------------------

    def _full_grid(self) -> np.ndarray:
        grid = np.full((self.height, self.width), CLEAN, dtype=np.int8)
        grid[self.dirty] = DIRT
        grid[self.dock] = DOCK
        grid[self.pot] = POT
        grid[self.robot] = ROBOT
        return grid

    def _observation(self) -> dict:
        grid = self._full_grid()
        if self.view == "local":
            # Pad the room with walls, then cut out the 3x3 patch around the robot.
            padded = np.pad(grid, 1, constant_values=WALL)
            r, c = self.robot
            grid = padded[r : r + 3, c : c + 3]
        return {"grid": grid, "battery": self.battery}

    # ---- the Gymnasium API ---------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot = self.dock
        self.dirty = self.initial_dirt.copy()
        self.battery = self.battery_capacity
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action!r}"

        # 1. Move (or bump into something).
        dr, dc = MOVES[action]
        target = (self.robot[0] + dr, self.robot[1] + dc)
        if not (0 <= target[0] < self.height and 0 <= target[1] < self.width):
            pass  # walked into a wall: stay put
        elif target == self.pot:
            self._spill_soil()  # bumped the pot: stay put, soil everywhere
        else:
            self.robot = target

        # 2. Clean the cell we are standing on.
        cleaned = int(self.dirty[self.robot])
        self.dirty[self.robot] = False

        # 3. Battery: charging on the dock, draining otherwise.
        if action == STAY and self.robot == self.dock:
            self.battery = min(self.battery_capacity, self.battery + 5)
        else:
            self.battery -= 1
        self.t += 1

        # 4. Reward. This one choice decides what the "optimal" robot does.
        if self.reward_mode == "dirt_removed":
            reward = float(cleaned)
        else:
            reward = -float(self.dirty.sum())

        terminated = self.battery <= 0 or not self.dirty.any()
        truncated = self.t >= self.max_steps
        return self._observation(), reward, terminated, truncated, {"cleaned": cleaned}

    def _spill_soil(self) -> None:
        r, c = self.pot
        for dr, dc in MOVES.values():
            if (dr, dc) == (0, 0):
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                self.dirty[nr, nc] = True

    def render(self):
        header = (
            f"t={self.t:3d}  battery {self.battery:2d}/{self.battery_capacity}  "
            f"dirty cells {int(self.dirty.sum())}"
        )
        rows = ["".join(SYMBOLS[code] for code in row) for row in self._full_grid()]
        return "\n".join([header, *rows])


def grid_to_str(grid: np.ndarray) -> str:
    return "\n".join("".join(SYMBOLS[code] for code in row) for row in grid)


def play(env: RoombaEnv) -> None:
    keys = {"w": UP, "d": RIGHT, "s": DOWN, "a": LEFT, "x": STAY}
    print("Legend: R robot, D dock, P flowerpot, * dirt, . clean floor\n")
    obs, info = env.reset()
    print(env.render())
    total = 0.0
    while True:
        key = input("action (w/a/s/d = move, x = stay/charge, q = quit): ").strip().lower()
        if key == "q":
            return
        if key not in keys:
            continue
        obs, reward, terminated, truncated, info = env.step(keys[key])
        total += reward
        print(env.render())
        if env.view == "local":
            print("the robot sees:\n" + grid_to_str(obs["grid"]))
        print(
            f"      reward={reward:+.1f}  terminated={terminated}  truncated={truncated}  "
            f"return so far={total:.1f}"
        )
        if terminated or truncated:
            print("Episode over.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["dirt_removed", "cleanliness"], default="dirt_removed")
    parser.add_argument("--view", choices=["full", "local"], default="full")
    args = parser.parse_args()
    play(RoombaEnv(reward=args.reward, view=args.view))
