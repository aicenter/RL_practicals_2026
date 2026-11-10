"""Evaluating a policy on fixed episodes, so that numbers from different runs are comparable.

    from rlcourse.deeprl.evaluation import evaluate

    result = evaluate(agent.greedy_action, value_fn=agent.start_value, gamma=0.99)
    print(result.mean_return, result.death_rate)

Every call plays the same episodes (the same environment seeds), so two policies are compared on
the same ghost dice, and a difference between them is a difference between the policies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from rlcourse.deeprl.world import make_env

#: The environment seeds of evaluation episode k are EVAL_SEED + k. Training never uses these.
EVAL_SEED = 1_000_000

Policy = Callable[[np.ndarray], int]
"""A policy, as evaluation sees it: an observation in, an action out."""

ValueFunction = Callable[[np.ndarray], float]
"""An estimate of the value of an observation, e.g. max_a Q(s, a)."""


@dataclass(frozen=True)
class Evaluation:
    """What a policy achieved over a set of evaluation episodes (all numbers are means)."""

    episodes: int
    mean_return: float
    """Undiscounted return per episode: the number we rank by."""
    stderr_return: float
    """Standard error of mean_return over the episodes."""
    pellets: float
    """Pellets eaten per episode."""
    death_rate: float
    """Fraction of episodes in which a ghost caught the agent."""
    clear_rate: float
    """Fraction of episodes in which the agent ate every pellet."""
    steps: float
    """Episode length."""
    predicted_start_value: float | None = None
    """The agent's own estimate of the value of the start (if value_fn was given) ..."""
    discounted_start_return: float | None = None
    """... and the discounted return that actually followed it, for comparison."""

    @property
    def overestimation(self) -> float | None:
        """How much the agent thinks the start is worth, minus what it actually got."""
        if self.predicted_start_value is None or self.discounted_start_return is None:
            return None
        return self.predicted_start_value - self.discounted_start_return

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    policy: Policy,
    n_episodes: int = 50,
    *,
    value_fn: ValueFunction | None = None,
    gamma: float = 0.99,
    seed: int = EVAL_SEED,
) -> Evaluation:
    """Play n_episodes episodes with `policy` and summarize them.

    Args:
        policy: called with each observation; returns the action. To evaluate a greedy policy,
            pass the greedy one -- evaluation adds no exploration of its own.
        n_episodes: how many episodes; episode k uses environment seed `seed + k`.
        value_fn: optional; if given, it is called on the first observation of every episode,
            and the result is reported next to the discounted return that actually followed.
        gamma: the discount for that comparison (it does not affect anything else).
        seed: the first environment seed.
    """
    env = make_env()
    returns, discounted, pellets, deaths, clears, lengths, predicted = [], [], [], [], [], [], []
    for k in range(n_episodes):
        observation, info = env.reset(seed=seed + k)
        if value_fn is not None:
            predicted.append(float(value_fn(observation)))
        pellets_at_start = info["pellets_left"]
        total, total_discounted, t = 0.0, 0.0, 0
        while True:
            observation, reward, terminated, truncated, info = env.step(policy(observation))
            total += float(reward)
            total_discounted += gamma**t * float(reward)
            t += 1
            if terminated or truncated:
                break
        returns.append(total)
        discounted.append(total_discounted)
        pellets.append(pellets_at_start - info["pellets_left"])
        deaths.append(bool(info["death"]))
        clears.append(info["pellets_left"] == 0)
        lengths.append(t)

    return Evaluation(
        episodes=n_episodes,
        mean_return=float(np.mean(returns)),
        stderr_return=float(np.std(returns, ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0,
        pellets=float(np.mean(pellets)),
        death_rate=float(np.mean(deaths)),
        clear_rate=float(np.mean(clears)),
        steps=float(np.mean(lengths)),
        predicted_start_value=float(np.mean(predicted)) if predicted else None,
        discounted_start_return=float(np.mean(discounted)) if predicted else None,
    )
