"""Bandit agents -- the file you will edit this week.

Every agent follows the same three-method interface, used by bandit_tools.py:

    agent = EpsilonGreedy(epsilon=0.1)       # hyperparameters go to __init__
    agent.reset(n_actions, horizon, rng)     # called at the start of every run
    a = agent.select_action()                # choose A_t
    agent.update(a, reward)                  # learn from R_t

Notation follows the Week 2 lecture: Q_t(a) is our estimate of the action value
q(a), N_t(a) the number of times action a was selected, t the number of steps
taken so far. The base class keeps Q and N up to date with the sample average
(the incremental update from the lecture), so for the four algorithms below you
only need to write `select_action` (and, where needed, extend `reset`).

Rules of the game:
- Use `self.rng` for all randomness, so that runs are reproducible.
- An agent sees only its own actions and rewards. Peeking into the environment
  (e.g. `env.unwrapped.means`) is cheating.
- `np.argmax` breaks ties by always picking the *lowest* index -- use
  `argmax_random` below instead.
"""

from __future__ import annotations

import math

import numpy as np


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest entry of `values`, ties broken uniformly at random."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


class BanditAgent:
    """Base class: bookkeeping shared by all agents."""

    def reset(self, n_actions: int, horizon: int, rng: np.random.Generator) -> None:
        """Forget everything and get ready for a new run of `horizon` steps."""
        self.n_actions = n_actions
        self.horizon = horizon      # the total number of steps T of this run
        self.rng = rng
        self.Q = np.zeros(n_actions)   # Q_t(a): estimated value of each action
        self.N = np.zeros(n_actions)   # N_t(a): how often each action was selected
        self.t = 0                     # number of steps taken so far

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        """Sample-average update: Q(a) <- Q(a) + (R - Q(a)) / N(a)."""
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                           if k not in {"n_actions", "horizon", "rng", "Q", "N", "t"}
                           and not k.startswith("_"))
        return f"{type(self).__name__}({params})"


class RandomAgent(BanditAgent):
    """Pulls arms uniformly at random. Any algorithm should beat this."""

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_actions))


# -----------------------------------------------------------------------------
# Example: the greedy algorithm. It works -- but not very well. Why?
# -----------------------------------------------------------------------------

class Greedy(BanditAgent):
    """Always pull the action with the highest estimate Q_t(a).

    All estimates start at Q_0(a) = 0, so which arms get tried at all depends
    on how the first few rewards compare to 0.
    """

    def select_action(self) -> int:
        return argmax_random(self.Q, self.rng)


# -----------------------------------------------------------------------------
# Your algorithms. Complete the two your group was assigned (see README.md).
# -----------------------------------------------------------------------------

class EpsilonGreedy(BanditAgent):
    """With probability epsilon pick a uniformly random action, otherwise act greedily."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select_action(self) -> int:
        # TODO
        raise NotImplementedError


class ExploreThenCommit(BanditAgent):
    """Play every action m times, then commit to the one with the highest Q(a)."""

    def __init__(self, m: int = 10):
        self.m = m

    def select_action(self) -> int:
        # TODO
        raise NotImplementedError


class Boltzmann(BanditAgent):
    """Sample actions from the softmax policy pi_t(a) proportional to exp(Q_t(a) / tau)."""

    def __init__(self, tau: float = 0.1):
        self.tau = tau

    def select_action(self) -> int:
        # TODO
        raise NotImplementedError


class UCB(BanditAgent):
    """Pick argmax_a  Q_t(a) + c * sqrt(log(t) / N_t(a))."""

    def __init__(self, c: float = 1.0):
        self.c = c

    def select_action(self) -> int:
        # TODO
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Final showdown (Task 5): one setting per algorithm, used on ALL five classes.
# Fill in the algorithms you implemented; leave the others out.
# -----------------------------------------------------------------------------

def final_agents() -> dict[str, BanditAgent]:
    return {
        # "EpsilonGreedy": EpsilonGreedy(epsilon=...),
        # "UCB": UCB(c=...),
    }
