"""Multi-armed bandits as Gymnasium environments (Week 2).

    Spoiler warning: the docstrings below describe what is inside each problem
    class. Play the bandits by hand (week02_bandits/play.py) *before* reading on.

A bandit is the simplest possible MDP: a single state, so the observation is
always 0, and every action ("arm") yields a random reward from its own
distribution. One episode is one whole learning run of `horizon` pulls: the
episode is truncated after `horizon` steps and never terminates.

Each environment is a *problem class*, i.e. a distribution over bandit
instances rather than one fixed bandit: every `env.reset(seed=...)` draws a new
instance (new arm means, new noise levels, new arm order). Tune your algorithm
for the class, not for one lucky instance.

Evaluation (not for agents!): after `reset`, `env.unwrapped.means` holds the
true arm means q(a) and `env.unwrapped.gaps()` the gaps
Delta_a = v_* - q(a), which the course tools use to compute the regret.
An agent that looks at these is cheating.

Implementation note: all rewards of an episode are drawn at `reset` into a
table R[a, k] = reward of the k-th pull of arm a. The table depends only on the
seed, so two agents run with the same seed face exactly the same luck
("common random numbers"), which makes comparisons much less noisy.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _bernoulli_table(p, values, rng, T):
    """Reward table for arms paying `values[a]` with probability `p[a]`, else 0."""
    return values[:, None] * (rng.random((len(p), T)) < p[:, None])


def sample_class1(rng, K=30):
    """Class 1 -- Bernoulli arms, and many of them (K = 30).

    Rewards are 0 or 1. Most arms have means between 0.1 and 0.5; one arm is
    clearly better (0.6-0.8) and one is a little worse than the best. With
    T = 1000 pulls you can afford only ~33 pulls per arm, so exploring every
    arm thoroughly is expensive.
    """
    means = rng.uniform(0.1, 0.5, size=K)
    best = 0.5 + rng.uniform(0.1, 0.3)
    means[0] = best
    means[1] = best - rng.uniform(0.05, 0.15)
    means = rng.permutation(means)
    return means, lambda r, T: _bernoulli_table(means, np.ones(K), r, T)


def sample_class2(rng, K=10):
    """Class 2 -- Gaussian arms with similar noise and a near-tie at the top.

    Standard deviations 0.8-1.2. The best two arms are 0.05-0.3 apart; all the
    others are 0.5-1.5 below the best. The overall level is random (the best
    mean is anywhere in [-1, 1]), so do not assume rewards are positive.
    """
    top = rng.uniform(-1.0, 1.0)
    means = top - rng.uniform(0.5, 1.5, size=K)
    means[0] = top
    means[1] = top - rng.uniform(0.05, 0.3)
    sd = rng.uniform(0.8, 1.2, size=K)
    perm = rng.permutation(K)
    means, sd = means[perm], sd[perm]
    return means, lambda r, T: means[:, None] + sd[:, None] * r.standard_normal((K, T))


def sample_class3(rng, K=6):
    """Class 3 -- Gaussian arms with wildly different noise levels.

    Means spread over an interval of width 1 (at a random level in [-1, 2]);
    standard deviations between 0.1 and 3, log-uniformly. Sometimes the
    noisiest arm is the best one, sometimes it is not.
    """
    offset = rng.uniform(-1.0, 1.0)
    means = offset + rng.uniform(0.0, 1.0, size=K)
    sd = np.exp(rng.uniform(np.log(0.1), np.log(3.0), size=K))
    return means, lambda r, T: means[:, None] + sd[:, None] * r.standard_normal((K, T))


def sample_class4(rng, K=6):
    """Class 4 -- heavy tails: every arm has Student-t noise.

    Means in [0, 1], noise 0.5 * t_nu with nu between 1.1 and 1.6. The mean
    exists but the variance is infinite: most rewards are unremarkable, but
    every now and then an arm produces a reward of +-50 (or worse) that drags
    its sample mean far off for a long time.
    """
    means = rng.uniform(0.0, 1.0, size=K)
    nu = rng.uniform(1.1, 1.6, size=K)
    scale = 0.5
    return means, lambda r, T: np.stack(
        [means[a] + scale * r.standard_t(nu[a], T) for a in range(K)])


def sample_class5(rng, K=5):
    """Class 5 -- lotteries: steady arms plus one or two rare jackpots.

    The steady arms pay 1 with probability 0.2-0.8 (else 0). A jackpot arm pays
    5-30 with a small probability, chosen so that its mean is within +-0.25 of
    the best steady arm: sometimes the jackpot is the best arm, sometimes it is
    a trap.
    """
    n_jack = rng.integers(1, 3)
    n_steady = K - n_jack
    p = rng.uniform(0.2, 0.8, size=n_steady)
    v = np.exp(rng.uniform(np.log(5), np.log(30), size=n_jack))
    jmeans = np.clip(p.max() + rng.uniform(-0.25, 0.25, size=n_jack), 0.05, None)
    values = np.concatenate([np.ones(n_steady), v])
    probs = np.concatenate([p, jmeans / v])
    perm = rng.permutation(K)
    values, probs = values[perm], probs[perm]
    means = values * probs
    return means, lambda r, T: _bernoulli_table(probs, values, r, T)


PROBLEM_CLASSES = {
    1: (sample_class1, 30),
    2: (sample_class2, 10),
    3: (sample_class3, 6),
    4: (sample_class4, 6),
    5: (sample_class5, 5),
}


class BanditEnv(gym.Env):
    """A K-armed bandit drawn from problem class `problem` (1-5).

    observation: always 0 (there is only one state)
    action:      the arm to pull, 0 .. K-1
    reward:      a random reward of the pulled arm
    """

    metadata = {"render_modes": []}

    def __init__(self, problem: int = 1, horizon: int = 1000):
        self._sampler, K = PROBLEM_CLASSES[problem]
        self.problem = problem
        self.horizon = horizon
        self.action_space = spaces.Discrete(K)
        self.observation_space = spaces.Discrete(1)
        self.means = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.means, draw = self._sampler(self.np_random, K=self.action_space.n)
        self._table = draw(self.np_random, self.horizon)
        self._pulls = np.zeros(self.action_space.n, dtype=np.int64)
        self._t = 0
        return 0, {}

    def step(self, action):
        if self._t >= self.horizon:
            raise RuntimeError("Episode is over: call reset() first.")
        a = int(action)
        reward = float(self._table[a, self._pulls[a]])
        self._pulls[a] += 1
        self._t += 1
        return 0, reward, False, self._t >= self.horizon, {}

    # --- for evaluation only; agents must not use these ---------------------
    def gaps(self):
        """Delta_a = v_* - q(a) for every arm of the current instance."""
        return self.means.max() - self.means
