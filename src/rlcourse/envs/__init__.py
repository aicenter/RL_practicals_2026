"""Course environments, registered with Gymnasium on import.

Usage:
    import gymnasium as gym
    import rlcourse.envs  # noqa: F401  (registers the environments)

    env = gym.make("rlcourse/Pacman-v0", render_mode="human")
"""

import gymnasium as gym

# TODO: register the Pacman environment once it exists, e.g.:
# gym.register(
#     id="rlcourse/Pacman-v0",
#     entry_point="rlcourse.envs.pacman:PacmanEnv",
#     max_episode_steps=500,
# )
