"""Monte Carlo prediction and control -- the file one half of your group edits this week.

Monte Carlo learns from the return that actually followed, and never bootstraps. It waits for
the end of the episode, then moves each visited estimate towards G_t:

    V(S_t) <- V(S_t) + alpha_n * (G_t - V(S_t)),                     prediction
    Q(S_t, A_t) <- Q(S_t, A_t) + alpha_n * (G_t - Q(S_t, A_t)),      control
    G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ... (until the episode ends).

With alpha_n = 1/n (n = how many times this entry has been updated, including this time), the
estimate is exactly the average of the returns seen so far.

Control is generalized policy iteration: act epsilon-greedily with respect to Q (the tools do
the acting), and after each episode update Q from the returns of that episode.

In code:
- An episode has ep.states (S_0, ..., S_T), ep.actions (A_0, ..., A_{T-1}), ep.rewards
  (rewards[t] is the reward of the step from S_t to S_{t+1}, i.e. R_{t+1}), and ep.terminated.
- self.alpha is the step-size schedule: self.alpha(n) is the step size of the n-th update of
  an entry (n = 1, 2, ...). A number passed as alpha= means a constant step size.
- If an episode was cut off by the time limit, Monte Carlo has nothing better to do than to
  use the return it has: no special case. (Whether that is a good idea is group A3's question.)

Fill in the places marked todo(...). Check your work with

    uv run python week06_mc/mc_tools.py
"""

from __future__ import annotations

import numpy as np

from mc_tools import Episode, as_schedule, one_over_n


def todo(what: str):
    raise NotImplementedError(what)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """The returns G_0, ..., G_{T-1} of an episode with these T rewards:

        G_t = rewards[t] + gamma * rewards[t+1] + gamma^2 * rewards[t+2] + ...

    Hint: loop and backward recursion?"""

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
          episode is an update; later visits to the same state are skipped."""
        todo("the returns, then one update per (first) visit, in time order")


class MCControl:
    """On-policy Monte Carlo control, first-visit (default) or every-visit.

    It works with `run_control` like the agents of Week 5: `update` is called after every
    step. But Monte Carlo cannot learn from one step, so `update` only remembers it, and the
    learning happens in `end_episode`, which `run_control` calls when the episode is over
    (whether it terminated or was cut off by the time limit)."""

    def __init__(self, n_states: int, n_actions: int, gamma: float, alpha=one_over_n(),
                 first_visit: bool = True, Q0: float = 0.0):
        self.gamma = gamma
        self.alpha = as_schedule(alpha)
        self.first_visit = first_visit
        self.Q = np.full((n_states, n_actions), float(Q0))
        self.N = np.zeros((n_states, n_actions), dtype=np.int64)
        self.n_actions = n_actions
        self._steps: list[tuple[int, int, float]] = []

    def update(self, s: int, a: int, r: float, s2: int, a2: int, done: bool, eps: float) -> None:
        self._steps.append((s, a, r))          # nothing to learn yet

    def learn_from(self, episode: Episode, eps: float = 0.0) -> None:
        """Learn from a logged episode, as `learn_from` of the Week 5 agents."""
        self.learn(episode.states[:-1], episode.actions, episode.rewards)

    def end_episode(self) -> None:
        if self._steps:
            states, actions, rewards = zip(*self._steps)
            self._steps = []
            self.learn(states, actions, np.array(rewards))

    def learn(self, states, actions, rewards: np.ndarray) -> None:
        """Learn from one finished episode: states[t], actions[t], and rewards[t] = R_{t+1}.

        For t = 0, 1, ..., T-1, *in this order*: increase N[S_t, A_t] by one, then move
        Q(S_t, A_t) towards G_t with step size self.alpha(N[S_t, A_t]). First-visit: only
        the first visit to each *pair* (s, a) in this episode is an update."""
        todo("the returns, then one update per (first) visit of a pair, in time order")
