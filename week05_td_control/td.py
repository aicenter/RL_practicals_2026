"""Temporal-difference prediction -- the file one half of your group edits this week.

We estimate V_pi, the value function of a fixed policy pi, from episodes of pi alone:
no model. TD(0) does not wait for the end of the episode: after each step it moves V(S_t)
towards a target that uses its *own current estimate* of what comes next (bootstrapping):

    V(S_t) <- V(S_t) + alpha_n * (R_{t+1} + gamma V(S_{t+1}) - V(S_t)).

This is the Robbins-Monro iteration of the lecture with f(V) = T^pi V - V: the increment is
an unbiased sample of the Bellman backup minus the current value.

Batch TD(0) answers a different question: what does TD(0) compute from a *fixed* batch of
episodes? Sweep over all transitions of the batch, add up the increments without applying
them, apply the sums at the end of the sweep, and repeat until V stops changing. The lecture
(Proposition 5.2) says the result is the value function of the empirical model -- the
Week 4 direct solve on the model estimated from the data. You will verify that.

In code (see the top of td_tools.py):
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

    uv run python week05_td_control/td_tools.py
"""

from __future__ import annotations

import numpy as np

from td_tools import Episode, as_schedule


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


class BatchTD0:
    """Batch TD(0): repeated sweeps over a fixed batch of episodes, increments accumulated
    within a sweep and applied at its end, until V stops changing."""

    def __init__(self, n_states: int, gamma: float, alpha: float | None = None, V0: float = 0.0):
        self.gamma = gamma
        self.alpha = alpha          # None: set in fit() to 1 / max_s N(s), the largest step the
        self.V = np.full(n_states, float(V0))   # lecture's proposition allows
        self.sweeps = 0

    def sweep(self, episodes: list[Episode], alpha: float) -> float:
        """One sweep: for every transition (S_t, R_{t+1}, S_{t+1}) of every episode, add
        alpha * (R_{t+1} + gamma V(S_{t+1}) - V(S_t)) to an accumulator for S_t, *without*
        changing V (use the V from the start of the sweep throughout). Then add the
        accumulated increments to V. Return the largest absolute change of any V(s).
        The end of an episode is handled as in TD0."""
        todo("accumulate the TD(0) increments over the whole batch, then apply them at once")

    def fit(self, episodes: list[Episode], tol: float = 1e-10, max_sweeps: int = 200_000):
        """Sweep until the largest change in a sweep is below tol (or max_sweeps). Returns V."""
        if self.alpha is None:
            counts = np.bincount(np.concatenate([ep.states[:-1] for ep in episodes]),
                                 minlength=len(self.V))
            self.alpha = 1.0 / counts.max()
        for _ in range(max_sweeps):
            self.sweeps += 1
            if self.sweep(episodes, self.alpha) < tol:
                break
        return self.V
