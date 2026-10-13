"""Dynamic programming -- the file you edit this week.

Notation follows the Week 4 lecture: P(s' | s, a) and R(s, a) are the model, gamma the
discount, V a value function, Q an action-value function, pi a policy, and
T^pi, T^* the Bellman operators.

In code:
- V is an array of shape (S,): V[s] is the value of state s; Q has shape (S, A).
- A deterministic policy is an integer array of shape (S,): policy[s] is the action in s.
  A stochastic policy is an array of shape (S, A): policy[s, a] = pi(a | s).
- The model (see the top of dp_tools.py):
      model.R                          (S, A) expected rewards R(s, a)
      model.expect(V)                  (S, A) array: sum_s' P(s'|s,a) V(s')       S*A backups
      model.expect(V, actions=policy)  (S,)  the same for a = policy[s] only        S backups
      model.expect(V, states)          (n, A) for the given states only            n*A backups
      model.P                          (S, A, S) array P(s'|s,a); only for policy_evaluation_exact
- Terminal states are absorbing with reward 0, so they need no special treatment:
  V(terminal) = 0 comes out of the equations by itself, as long as you start from 0 there.

Fill in the places marked todo(...). Check your work with

    uv run python week04_dynamic_programming/dp_tools.py
"""

from __future__ import annotations

import numpy as np

EPS = 1e-3      # solve() must return a policy with V_pi(s) >= V*(s) - EPS in every state


def todo(what: str):
    raise NotImplementedError(what)


def greedy(Q: np.ndarray) -> np.ndarray:
    """The greedy deterministic policy: policy[s] = argmax_a Q(s, a)."""
    return Q.argmax(axis=1)


def as_matrix(policy: np.ndarray, n_actions: int = 4) -> np.ndarray:
    """A deterministic policy (array of actions) as an (S, A) array of probabilities."""
    return np.eye(n_actions)[policy]


# --- Task 1 -------------------------------------------------------------------------------

def bellman_q(model, V: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s, a) = R(s, a) + gamma * sum_s' P(s'|s,a) V(s'), for all s and a: an (S, A) array.

    This one-step lookahead is the heart of everything in this file:
    (T^* V)(s) = max_a Q(s, a), and (T^pi V)(s) = sum_a pi(a|s) Q(s, a)."""
    return todo("bellman_q: one line, using model.R and model.expect")


def policy_evaluation(model, policy: np.ndarray, gamma: float, tol: float = 1e-8,
                      V0: np.ndarray | None = None, max_sweeps: int = 1_000_000):
    """Iterative policy evaluation: V_{k+1} = T^pi V_k, from V_0 = V0 (default 0),
    until max_s |V_{k+1}(s) - V_k(s)| < tol.

    Returns (V, number of sweeps)."""
    S = model.n_states
    V = np.zeros(S) if V0 is None else np.array(V0, dtype=np.float64)
    policy = np.asarray(policy)
    deterministic = policy.ndim == 1
    for sweep in range(1, max_sweeps + 1):
        if deterministic:
            # (T^pi V)(s) = R(s, pi(s)) + gamma * E[V(S') | s, pi(s)]. Only S backups, not S*A.
            V_new = todo("T^pi V for a deterministic policy (model.R[np.arange(S), policy], ...)")
        else:
            V_new = todo("T^pi V for a stochastic policy")
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            return V, sweep
    return V, max_sweeps


def policy_evaluation_exact(model, policy: np.ndarray, gamma: float) -> np.ndarray:
    """V_pi exactly, by solving the Bellman expectation equation V = R_pi + gamma P_pi V as
    a linear system (np.linalg.solve). Uses model.P, so it does not count backups.

    P_pi[s, s'] = sum_a pi(a|s) P(s'|s, a) and R_pi[s] = sum_a pi(a|s) R(s, a)."""
    S = model.n_states
    pm = as_matrix(policy, model.n_actions) if np.ndim(policy) == 1 else policy
    P_pi = np.einsum("sa,sat->st", pm, model.P)
    R_pi = (pm * model.R).sum(axis=1)
    return todo("solve (I - gamma P_pi) V = R_pi")


def value_iteration(model, gamma: float, eps: float = EPS, V0: np.ndarray | None = None):
    """Value iteration: V_{k+1} = T^* V_k, from V_0 = V0 (default 0).

    Stopping rule: stop as soon as delta_k = max_s |V_{k+1}(s) - V_k(s)| < eps (1 - gamma) / (2 gamma).
    Then the greedy policy is eps-optimal: V_pi(s) >= V*(s) - eps in every state s.

    Returns (V, greedy policy, deltas), where deltas = [delta_0, delta_1, ...], one per sweep."""
    V = np.zeros(model.n_states) if V0 is None else np.array(V0, dtype=np.float64)
    threshold = eps * (1 - gamma) / (2 * gamma)
    deltas = []
    while True:
        todo("one sweep: Q, V_new, delta; append delta to deltas; stop (return) or continue")


def policy_iteration(model, gamma: float, policy0: np.ndarray | None = None):
    """Policy iteration with exact evaluation: evaluate pi_k exactly, then improve greedily,
    until the policy no longer changes. Starts from policy0 (default: action 0 everywhere).

    Careful with ties: change the action in s only if another action is *strictly* better
    (say by more than 1e-12), otherwise two equally good actions can swap forever.

    Returns (V, policy, number of iterations)."""
    policy = np.zeros(model.n_states, dtype=np.int64) if policy0 is None else policy0.copy()
    iterations = 0
    while True:
        iterations += 1
        todo("evaluate, improve, and return (V, policy, iterations) once the policy is stable")


# --- Task 3: the final -----------------------------------------------------------------------

def solve(model, gamma: float, eps: float = EPS) -> np.ndarray:
    """Your best solver. Return a deterministic policy (integer array of shape (S,)) with
    V_pi(s) >= V*(s) - eps in every state, using as few backups as you can.

    Rules: access the model only through model.R, model.expect, model.n_states,
    model.n_actions, model.terminal, and model.rho0 (model.P is switched off here).
    Do not hard-code anything about particular worlds: it is evaluated on worlds you
    have not seen, with gamma between 0.9 and 0.999.

    The starting point is plain value iteration. Can you do better?"""
    V, policy, deltas = value_iteration(model, gamma, eps)
    return policy
