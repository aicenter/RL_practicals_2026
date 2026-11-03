"""Off-policy evaluation by importance sampling -- the file the other half of your group edits.

We want v_pi, but the episodes came from somebody else: a behaviour policy b. We cannot run pi.
The transition probabilities are the same under both policies, so they cancel, and the
probability ratio of the tail of an episode from step t depends only on the two policies:

    rho_{t:T-1} = prod_{k=t}^{T-1} pi(A_k | S_k) / b(A_k | S_k).

Then E_b[ rho_{t:T-1} G_t | S_t = s ] = v_pi(s), as long as b(a | s) > 0 wherever
pi(a | s) > 0 (coverage). Two estimators, over the first visits to s in all episodes so far:

    ordinary:  V(s) = sum_i rho_i G_i / n               unbiased; variance can be enormous
    weighted:  V(s) = sum_i rho_i G_i / sum_i rho_i     biased, but bounded by the largest |G_i|

The weighted one has an incremental form with C(s), the sum of all weights so far:

    C(s) <- C(s) + rho,    V(s) <- V(s) + (rho / C(s)) * (G - V(s)).

A dataset of n episodes with very unequal weights is worth far fewer than n episodes. The
*effective sample size* puts a number on this:

    ESS = (sum_i w_i)^2 / sum_i w_i^2,

which is n if all the weights are equal and 1 if one weight dominates.

In code:
- `target` and `behaviour` are (n_states, n_actions) arrays of probabilities pi(a | s), as in
  `world.policy`: `pi = load("maze", tired=0.1).policy`.
- ep.states is S_0, ..., S_T, and ep.actions is A_0, ..., A_{T-1}.

Fill in the places marked todo(...). Check your work with

    uv run python week06_mc/mc_tools.py
"""

from __future__ import annotations

import numpy as np

from mc_tools import Episode


def todo(what: str):
    raise NotImplementedError(what)


def tail_ratios(episode: Episode, target: np.ndarray, behaviour: np.ndarray) -> np.ndarray:
    """rho_{t:T-1} for every t = 0, ..., T-1, as an array of length T. So element 0 is the
    ratio of the whole episode, and element T-1 the ratio of its last action alone.

    Hint: as with returns, go backwards. One loop."""
    return todo("rho_{t:T-1} for every t, in one backward loop")


def effective_sample_size(weights: np.ndarray) -> float:
    """(sum of the weights)^2 / (sum of their squares); 0 if all the weights are 0."""
    return todo("the effective sample size of these weights")


class OffPolicyEstimator:
    """What the two estimators share: the first visits to each state, with their G and rho.

    Interface as in prediction: `update(episode)` once per episode, and `V` is the estimate."""

    def __init__(self, n_states: int, gamma: float, target: np.ndarray, behaviour: np.ndarray):
        self.gamma = gamma
        self.target, self.behaviour = target, behaviour
        self.V = np.zeros(n_states)
        self.N = np.zeros(n_states, dtype=np.int64)    # first visits to each state so far
        self.C = np.zeros(n_states)                    # the sum of their weights

    def update(self, episode: Episode) -> None:
        T = len(episode)
        discounts = self.gamma ** np.arange(T)
        G = np.cumsum((episode.rewards * discounts)[::-1])[::-1] / discounts
        rho = tail_ratios(episode, self.target, self.behaviour)
        seen = set()
        for t, s in enumerate(episode.states[:-1]):
            if s not in seen:
                seen.add(s)
                self.learn(s, rho[t], G[t])

    def learn(self, s: int, rho: float, G: float) -> None:
        raise NotImplementedError


class OrdinaryIS(OffPolicyEstimator):
    """V(s) is the plain average of rho_i * G_i, over all first visits to s."""

    def learn(self, s, rho, G):
        todo("increase N[s], then keep V[s] the running average of rho * G")


class WeightedIS(OffPolicyEstimator):
    """V(s) is the average of the G_i, weighted by rho_i."""

    def learn(self, s, rho, G):
        todo("increase C[s] by rho, then the incremental update; careful: what if C[s] is 0?")
