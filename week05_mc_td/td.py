"""Temporal-difference prediction -- the file one half of your group edits this week.

We estimate V_pi, the value function of a fixed policy pi, from episodes of pi alone:
no model. TD learning does not wait for the end of the episode: after each step it moves
V(S_t) towards a target that uses its *own current estimate* of what comes next
(bootstrapping). TD(0), from the Week 5 lecture:

    V(S_t) <- V(S_t) + alpha_n * (R_{t+1} + gamma V(S_{t+1}) - V(S_t)).

n-step TD looks n rewards ahead before it bootstraps:

    G_{t:t+n} = R_{t+1} + gamma R_{t+2} + ... + gamma^{n-1} R_{t+n} + gamma^n V(S_{t+n}),
    V(S_t) <- V(S_t) + alpha_n * (G_{t:t+n} - V(S_t)).

n = 1 is TD(0); n = infinity (never bootstrapping) is every-visit Monte Carlo.

In code (see the top of pred_tools.py):
- V is an array of shape (n_states,); self.N[s] counts the updates of V[s] so far.
- An episode has ep.states (S_0, ..., S_T), ep.rewards (rewards[t] is the reward of the
  step from S_t to S_{t+1}, i.e. R_{t+1}), and ep.terminated.
- self.alpha is the step-size schedule: self.alpha(n) is the step size of the n-th update
  of a state (n = 1, 2, ...). A number passed as alpha= means a constant step size.
- The end of an episode. If ep.terminated, S_T is terminal and its value is 0 by
  definition: use 0, not V[S_T]. If not, the episode was only cut off by the time limit,
  S_T is an ordinary state, and the right thing is to bootstrap from V[S_T] as usual.

We process the steps of an episode in time order, t = 0, 1, ..., T-1, exactly as an
agent learning online would; the episode is given whole only for convenience.

Fill in the places marked todo(...). Check your work with

    uv run python week05_mc_td/pred_tools.py
"""

from __future__ import annotations

import numpy as np

from pred_tools import Episode, as_schedule


def todo(what: str):
    raise NotImplementedError(what)


class TD0:
    """TD(0) prediction of V_pi."""

    def __init__(self, n_states: int, gamma: float, alpha=0.1, V0: float = 0.0):
        self.gamma = gamma
        self.alpha = as_schedule(alpha)
        self.V = np.full(n_states, float(V0))
        self.N = np.zeros(n_states, dtype=np.int64)

    def update(self, episode: Episode) -> None:
        """For t = 0, ..., T-1: increase N[S_t] by one, then
        V(S_t) += alpha(N[S_t]) * (R_{t+1} + gamma V(S_{t+1}) - V(S_t)),
        where V(S_T) counts as 0 if the episode terminated (see the top of this file)."""
        s, r = episode.states, episode.rewards
        todo("one TD(0) update per step, in time order")


class NStepTD:
    """n-step TD prediction of V_pi."""

    def __init__(self, n_states: int, gamma: float, n: int = 4, alpha=0.1, V0: float = 0.0):
        self.gamma = gamma
        self.n = n
        self.alpha = as_schedule(alpha)
        self.V = np.full(n_states, float(V0))
        self.N = np.zeros(n_states, dtype=np.int64)

    def update(self, episode: Episode) -> None:
        """For tau = 0, ..., T-1, in this order: compute the n-step return G_{tau:tau+n}
        (with the current V), increase N[S_tau] by one, and move V(S_tau) towards it.

        Near the end of the episode there are fewer than n rewards left: if tau + n >= T,
        add up the rewards until T, and then bootstrap from V(S_T) only if the episode did
        not terminate. (Updating V(S_tau) at time tau + n - 1, as an online agent would,
        gives exactly the same result: updates are still made in the order tau = 0, 1, ...)"""
        s, r, g, n = episode.states, episode.rewards, self.gamma, self.n
        todo("for each tau: the n-step return (careful near the end), then the update")
