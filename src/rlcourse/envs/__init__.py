"""Course environments, registered with Gymnasium on import.

Usage:
    import gymnasium as gym
    import rlcourse.envs  # noqa: F401  (registers the environments)

    env = gym.make("rlcourse/Bandit1-v0")
"""

import gymnasium as gym

# Week 2: multi-armed bandits, one environment per problem class (1-5).
# The episode length is the `horizon` argument (default 1000); the environment
# truncates by itself, so no TimeLimit wrapper is needed.
for _problem in range(1, 6):
    gym.register(
        id=f"rlcourse/Bandit{_problem}-v0",
        entry_point="rlcourse.envs.bandits:BanditEnv",
        kwargs={"problem": _problem},
    )

# TODO: register the Pacman environment once it exists, e.g.:
# gym.register(
#     id="rlcourse/Pacman-v0",
#     entry_point="rlcourse.envs.pacman:PacmanEnv",
#     max_episode_steps=500,
# )
