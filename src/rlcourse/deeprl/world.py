"""The world of the deep-RL practicals (Weeks 7-9): one PragMan maze, so that results compare.

    from rlcourse.deeprl.world import make_env

    env = make_env()
    observation, info = env.reset(seed=0)      # observation: (8, 8, 4) uint8 planes

It is PragMan's `pellets_8x8`: 29 pellets and one ghost that moves at random, at most 100 steps
an episode. Too many states for a table (2**29 pellet patterns alone), but small enough that a
deep Q-network learns it on a laptop CPU in a few minutes.

The observation is four binary planes, channel-last: walls, agent, ghosts, pellets
(`env.channel_names`). There are four actions: up, right, down, left.

Reading the returns: a pellet is worth 0.1, eating the last one another 1.0, being caught -1.0,
and every step that does not end the episode costs 0.01. So clearing the board pays 2.9 + 1.0
minus the steps it took, about 3.6 at best, and an agent that ends at return 1.0 has eaten
around twenty pellets, not "a few".
"""

from __future__ import annotations

from typing import Any

import pragman

#: The PragMan factory the deep-RL weeks use (see pragman.configs).
WORLD = "pellets_8x8"

#: Reference returns of the world, for orientation (mean over 1000 episodes).
RANDOM_POLICY_RETURN = -0.40


def make_env(**overrides: Any) -> pragman.PragmanEnv:
    """A fresh copy of the course world. Keyword arguments override PragmanConfig fields,
    e.g. `make_env(ghost_policy="chaser")` for a harder version (not for leaderboard runs)."""
    return pragman.make(WORLD, **overrides)
