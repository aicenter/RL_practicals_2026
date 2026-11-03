"""n-step SARSA -- for group A2.

SARSA bootstraps after one step, Monte Carlo after none at all. n-step SARSA is the dial between
them. From S_t, A_t it uses n rewards, and then the estimate of the pair it has reached:

    G_{t:t+n} = R_{t+1} + gamma R_{t+2} + ... + gamma^{n-1} R_{t+n} + gamma^n Q(S_{t+n}, A_{t+n}).

If the episode terminates within these n steps, there is nothing to bootstrap from: use the
rewards there are. If it is cut off by the time limit, bootstrap from the last pair as usual.

The agent learns about the pair of step t only at step t + n, so it keeps the last n steps in
a queue. The queue is done for you; you write what happens to its oldest step. n = 1 must be
exactly SARSA, and a huge n exactly every-visit Monte Carlo control with the same constant step
size -- the checker tests both.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from mc_tools import as_schedule


def todo(what: str):
    raise NotImplementedError(what)


class NStepSARSA:
    def __init__(self, n_states: int, n_actions: int, gamma: float, n: int = 4, alpha=0.1,
                 Q0: float = 0.0):
        self.gamma, self.n = gamma, n
        self.alpha = as_schedule(alpha)
        self.Q = np.full((n_states, n_actions), float(Q0))
        self.N = np.zeros((n_states, n_actions), dtype=np.int64)
        self.n_actions = n_actions
        self.queue: deque = deque()            # at most n entries (s, a, r, s2, a2, done)

    def update(self, s: int, a: int, r: float, s2: int, a2: int, done: bool, eps: float) -> None:
        self.queue.append((s, a, r, s2, a2, done))
        if len(self.queue) == self.n:
            self.update_oldest()
        if done:
            self.end_episode()

    def end_episode(self) -> None:
        while self.queue:                      # fewer than n steps are left for these
            self.update_oldest()

    def update_oldest(self) -> None:
        """Take the oldest step out of the queue (self.queue.popleft() gives its s, a, r, ...;
        do this after computing the target, or remember its reward). With all the steps
        in the queue, compute the n-step target: the discounted rewards in the queue, plus,
        unless the newest step in the queue (self.queue[-1]) has done = True, the discounted
        Q of its (s2, a2). Then increase N[s, a] and move Q[s, a] towards the target with step
        size self.alpha(N[s, a])."""
        todo("the n-step target of the oldest step in the queue, and its update")
