"""TD control -- the file the other half of your group edits this week.

Control: learn to act well, from one transition at a time, with no model. All three
algorithms of the lecture keep a table Q(s, a) and, after the transition
(S_t, A_t, R_{t+1}, S_{t+1}), move Q(S_t, A_t) towards a target

    Y_t = R_{t+1} + gamma * (something about S_{t+1}),
    Q(S_t, A_t) <- Q(S_t, A_t) + alpha_n * (Y_t - Q(S_t, A_t)).

They differ only in the "something", i.e. in the *target policy* whose value of S_{t+1}
they plug in (lecture, "one update, three target policies"):

    SARSA            Q(S_{t+1}, A_{t+1})                      the action actually taken next
    Expected SARSA   sum_a' pi(a' | S_{t+1}) Q(S_{t+1}, a')   averaged over the behaviour policy
    Q-learning       max_a' Q(S_{t+1}, a')                    the greedy action

The *behaviour policy* -- the one that chooses the actions and so decides which (s, a) get
updated -- is epsilon-greedy with respect to Q in all three; the tools do the acting
(`run_control` in td_tools.py), your classes only learn.

In code:
- self.Q is an (n_states, n_actions) array; self.N[s, a] counts the updates of Q[s, a].
- update(s, a, r, s2, a2, done, eps) receives one transition: state s, action a, reward r,
  next state s2, the next action a2 the behaviour policy has *already chosen* for s2, whether
  s2 is terminal (then the target is r alone: a terminal state is worth 0), and the current
  exploration probability eps of the behaviour policy (Expected SARSA needs it to write down
  the epsilon-greedy policy at s2; the others ignore it).
- The epsilon-greedy policy w.r.t. Q at state s2 gives every action probability
  eps / n_actions and the greedy action 1 - eps + eps / n_actions. If several actions tie
  for the maximum, split the greedy mass equally between them.
- self.alpha(n) is the step size of the n-th update of an entry (as in td.py).

Fill in the places marked todo(...). Check your work with

    uv run python week05_td_control/td_tools.py
"""

from __future__ import annotations

import numpy as np

from td_tools import Episode, as_schedule


def todo(what: str):
    raise NotImplementedError(what)


class TDControl:
    """Common machinery: the table, the step size, the update. Subclasses define target()."""

    def __init__(self, n_states: int, n_actions: int, gamma: float, alpha=0.1, Q0: float = 0.0):
        self.gamma = gamma
        self.alpha = as_schedule(alpha)
        self.Q = np.full((n_states, n_actions), float(Q0))
        self.N = np.zeros((n_states, n_actions), dtype=np.int64)
        self.n_actions = n_actions

    def target(self, r: float, s2: int, a2: int, done: bool, eps: float) -> float:
        raise NotImplementedError

    def update(self, s: int, a: int, r: float, s2: int, a2: int, done: bool, eps: float) -> None:
        self.N[s, a] += 1
        self.Q[s, a] += self.alpha(self.N[s, a]) * (self.target(r, s2, a2, done, eps) - self.Q[s, a])

    def learn_from(self, episode: Episode, eps: float = 0.0) -> None:
        """Learn from a logged episode of *someone else's* behaviour (off-policy data), in time
        order. For SARSA, a2 is the action the logged behaviour took next. If the episode was
        cut off by the time limit, its last transition is skipped (no next action exists)."""
        s, a, r = episode.states, episode.actions, episode.rewards
        T = len(r)
        for t in range(T):
            last = t == T - 1
            if last and not episode.terminated:
                break
            a2 = a[t + 1] if not last else 0
            self.update(s[t], a[t], r[t], s[t + 1], a2, last and episode.terminated, eps)


class SARSA(TDControl):
    """On-policy: the target uses the action the behaviour policy actually takes next."""

    def target(self, r, s2, a2, done, eps):
        return todo("SARSA target: r, or r + gamma * Q(s2, a2)")


class ExpectedSARSA(TDControl):
    """The target averages Q(s2, .) over the epsilon-greedy behaviour policy at s2."""

    def target(self, r, s2, a2, done, eps):
        return todo("Expected SARSA target: r + gamma * sum_a' pi_eps(a' | s2) Q(s2, a')")


class QLearning(TDControl):
    """Off-policy: the target evaluates the greedy policy, whatever the behaviour does next."""

    def target(self, r, s2, a2, done, eps):
        return todo("Q-learning target: r + gamma * max_a' Q(s2, a')")


# --- optional (group B2): Double Q-learning -----------------------------------------------------

class DoubleQLearning(TDControl):
    """Two tables. Each update picks one of them at random (or `which` = 0 / 1 when given):
    the chosen table selects the greedy action at s2, the *other* table evaluates it, and
    the chosen table is updated towards r + gamma * Q_other(s2, argmax_a' Q_chosen(s2, a')).
    The behaviour policy acts on self.Q = (Q1 + Q2) / 2."""

    def __init__(self, n_states: int, n_actions: int, gamma: float, alpha=0.1, Q0: float = 0.0,
                 seed: int = 0):
        super().__init__(n_states, n_actions, gamma, alpha, Q0)
        self.Q1 = np.full((n_states, n_actions), float(Q0))
        self.Q2 = np.full((n_states, n_actions), float(Q0))
        self.N1 = np.zeros((n_states, n_actions), dtype=np.int64)
        self.N2 = np.zeros((n_states, n_actions), dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def update(self, s, a, r, s2, a2, done, eps, which: int | None = None):
        if which is None:
            which = int(self.rng.integers(2))
        todo("Double Q-learning: update Q1 or Q2 (which), selecting with it and evaluating "
             "with the other table; then self.Q[s, a] = (Q1[s, a] + Q2[s, a]) / 2")
