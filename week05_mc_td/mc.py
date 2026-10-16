"""Monte Carlo prediction -- the file one half of your group edits this week.

We estimate V_pi, the value function of a fixed policy pi, from episodes of pi alone:
no model. Monte Carlo waits until the end of an episode and then moves each visited
V(S_t) towards the return that actually followed:

    V(S_t) <- V(S_t) + alpha_n * (G_t - V(S_t)),
    G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ... (until the episode ends).

This is the incremental-mean update of the Week 5 lecture, with G_t as the noisy sample:
with alpha_n = 1/n (n = how many times V(S_t) has been updated, including this time),
V(s) is exactly the average of the returns seen from s.

In code (see the top of pred_tools.py):
- V is an array of shape (n_states,); self.N[s] counts the updates of V[s] so far.
- An episode has ep.states (S_0, ..., S_T), ep.rewards (rewards[t] is the reward of the
  step from S_t to S_{t+1}), and ep.terminated.
- self.alpha is the step-size schedule: self.alpha(n) is the step size of the n-th update
  of a state (n = 1, 2, ...). A number passed as alpha= means a constant step size.

Fill in the places marked todo(...). Check your work with

    uv run python week05_mc_td/pred_tools.py
"""

from __future__ import annotations

import numpy as np

from pred_tools import Episode, as_schedule, one_over_n


def todo(what: str):
    raise NotImplementedError(what)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """The returns G_0, ..., G_{T-1} of an episode with these T rewards:

        G_t = rewards[t] + gamma * rewards[t+1] + gamma^2 * rewards[t+2] + ...

    Hint: go backwards. G_{T-1} = rewards[T-1], and G_t = rewards[t] + gamma * G_{t+1}.
    One loop, no powers of gamma needed."""
    return todo("G_t for every t, in one backward loop")


class MC:
    """Monte Carlo prediction of V_pi, every-visit (default) or first-visit."""

    def __init__(self, n_states: int, gamma: float, alpha=one_over_n(), first_visit: bool = False,
                 V0: float = 0.0):
        self.gamma = gamma
        self.alpha = as_schedule(alpha)        # default: the running average, alpha_n = 1/n
        self.first_visit = first_visit
        self.V = np.full(n_states, float(V0))
        self.N = np.zeros(n_states, dtype=np.int64)

    def update(self, episode: Episode) -> None:
        """Learn from one finished episode.

        For t = 0, 1, ..., T-1, *in this order*: increase N[S_t] by one, then move V(S_t)
        towards G_t with step size self.alpha(N[S_t]).
        - Every-visit: every visit to a state is an update.
        - First-visit (self.first_visit): only the first visit to each state in this
          episode is an update; later visits to the same state are skipped.
        If the episode was cut off by the time limit (episode.terminated is False), Monte
        Carlo has nothing better to do than to use the return it has -- so no special case."""
        todo("the returns, then one update per (first) visit, in time order")
