# Week 1: Foundations

Goal: build intuition for the basic building blocks of RL (agent, environment,
observation, action, reward, and episode) and see how an environment is
usually exposed in Python.

## 0. Setup

Follow the [main README](../README.md) and make sure
`uv run python check_setup.py` passes.

## 1. Play with three environments

This directory contains three small environments written against the
[Gymnasium](https://gymnasium.farama.org/) API. Every environment has the same
three methods: `__init__` declares what the agent sees and what it can do,
`reset` starts a new episode, and `step` applies one action and returns the
observation, reward, `terminated`, `truncated`, and an info dictionary.

| File | World | What it illustrates |
| --- | --- | --- |
| `corridor.py` | a slippery one-dimensional corridor | a tabular MDP with an explicit transition table; termination vs. truncation |
| `thermostat.py` | a room with a laggy heater | a continuous state; an episode that never terminates; a non-Markov observation |
| `roomba.py` | a robot vacuum with a battery and a wobbly flowerpot | reward design, and how the reward can be hacked; partial observability |

Each file also has a bare-minimum sibling (`corridor_minimal.py`,
`thermostat_minimal.py`) that works from the console but skips the
corner cases. Compare the two versions and think about what each extra line
buys you.

**Play from the terminal.** Each full environment can be driven with the
keyboard:

```bash
uv run python week01_foundations/corridor.py
uv run python week01_foundations/thermostat.py --hide-heater
uv run python week01_foundations/roomba.py --reward cleanliness --view local
```

**Play from a Python console.** Start `uv run ipython` in the repository root
and call the API yourself:

```python
from week01_foundations.corridor import CorridorEnv
from week01_foundations.thermostat import ThermostatEnv
from week01_foundations.roomba import RoombaEnv

env = CorridorEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)
print(env.render())
```

Things to try:

- Reset the corridor twice with the same seed and take the same actions.
  Do you get the same trajectory? Now try the same with
  `week01_foundations.corridor_minimal.MinimalCorridor`.
- Keep stepping after `terminated` is `True`. What happens in each version?
- In the thermostat, compare `ThermostatEnv()` with
  `ThermostatEnv(observe_heater=False)`. Can you hold the temperature at the
  target by hand in both cases? What does the observation lack in the second?
- In the Roomba with the default reward, find the policy that collects the
  most reward. Is it the policy the designer wanted? Switch to
  `RoombaEnv(reward="cleanliness")` and check again.
- Run Gymnasium's environment checker on each environment and read what it
  complains about:

  ```python
  from gymnasium.utils.env_checker import check_env
  check_env(env, skip_render_check=True)
  ```

## 2. Homework: implement your own environment

In the practical, your group turned an informal scenario into an MDP on paper.
Now write it in Python:

1. Create a new file, e.g. `week01_foundations/my_env.py`, with a class that
   subclasses `gymnasium.Env` and implements `__init__`, `reset`, and `step`.
   Start from `corridor_minimal.py` if you like. A `render` method that prints
   the state as text will make your life much easier.
2. Keep it small. A handful of states, a few actions, and one source of
   randomness are enough. You can always add the rest of your design later.
3. Play a few episodes by hand from the console, and make sure
   `check_env(env)` passes.
4. In a short comment at the top of the file, write down what you left out of
   your paper design and why, and whether your observation is Markov.

The [Gymnasium guide to custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
covers the API in more detail. Put your solution in a new file rather than
editing the provided ones, so that `git pull` does not conflict with your work.
