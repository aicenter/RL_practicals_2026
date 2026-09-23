"""How well does a random policy do? A number to compare learning agents against.

    uv run python week01_foundations/random_baseline.py --env week01_foundations.corridor:CorridorEnv
    uv run python week01_foundations/random_baseline.py --env week01_foundations.roomba:RoombaEnv --env-kwargs '{"reward": "cleanliness"}'
    uv run python week01_foundations/random_baseline.py --env CartPole-v1
"""

from dataclasses import dataclass

import numpy as np
import tyro

from common import make_env, run_episodes


@dataclass
class Args:
    env: str = "week01_foundations.corridor:CorridorEnv"
    """environment spec: a Gymnasium id or `module:ClassName`"""
    env_kwargs: str = "{}"
    """extra constructor arguments as a JSON string"""
    max_episode_steps: int | None = None
    """time limit, for environments that never truncate on their own"""
    episodes: int = 100
    """how many episodes to run"""
    seed: int = 0


def main(args: Args) -> None:
    env = make_env(args.env, args.env_kwargs, args.max_episode_steps)
    env.action_space.seed(args.seed)

    stats = run_episodes(env, lambda obs: env.action_space.sample(), args.episodes, args.seed)
    returns = np.array([s["return"] for s in stats])
    lengths = np.array([s["length"] for s in stats])
    n_terminated = sum(s["terminated"] for s in stats)
    n_truncated = sum(s["truncated"] and not s["terminated"] for s in stats)

    print(f"Random policy on {args.env} {args.env_kwargs if args.env_kwargs != '{}' else ''}")
    print(f"  episodes:        {args.episodes}")
    print(f"  return:          {returns.mean():8.2f}  (std {returns.std():.2f}, min {returns.min():.1f}, max {returns.max():.1f})")
    print(f"  episode length:  {lengths.mean():8.2f}")
    print(f"  ended by:        {n_terminated} terminated, {n_truncated} truncated")


if __name__ == "__main__":
    main(tyro.cli(Args))
