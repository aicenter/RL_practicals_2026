"""Week 5 tools: worlds, episodes, true values, errors, learning curves, control runs, checks.

No need to edit this file. The files you edit are `td.py` (TD(0) prediction, batch TD(0)) and
`control.py` (SARSA, Expected SARSA, Q-learning). Typical use, from a script of your own in
this directory:

    from td_tools import *
    from td import TD0, BatchTD0
    from control import SARSA, ExpectedSARSA, QLearning

    world = load("maze")                       # a world, with the policy we evaluate
    episodes = world.episodes(100, seed=0)     # 100 episodes of that policy
    est = TD0(world.n_states, world.gamma, alpha=0.1)
    for ep in episodes:
        est.update(ep)                         # learns from one episode at a time
    print(world.error(est.V))                  # how far est.V is from the true V_pi

    curves = learning_curves({"TD(0), 0.1": partial(TD0, alpha=0.1)}, world, 100, runs=20)
    plot_curves(curves); plt.show()

    # control: the agent acts epsilon-greedily in the world and learns online
    res = run_control({"SARSA": SARSA, "Q-learning": QLearning}, load("ghost_lair"),
                      n_episodes=500, runs=10, epsilon=0.1)
    plot_control(res); plt.show()

Run this file itself to check your implementation (Task 1):

    uv run python week05_td_control/td_tools.py

EPISODES. `world.episodes(n, seed)` returns a list of `Episode`s. An episode of length T is

    ep.states      (T+1,) int array: S_0, S_1, ..., S_T
    ep.actions     (T,)   int array: A_0, ..., A_{T-1}
    ep.rewards     (T,)   float array: rewards[t] is the reward of the step from S_t to S_{t+1}
                          (R_{t+1} in the lecture notes and in Sutton & Barto)
    ep.terminated  bool:  True if S_T is terminal (its value is 0 by definition);
                          False if the episode was cut off by the time limit (S_T is an
                          ordinary state, whose value is not 0)

The same seed always gives the same episodes, so two estimators given the same seed
learn from exactly the same data: their difference is the algorithm, not the luck.

WORLDS. `load(name)` returns a `World`: the Prag-man world written out as arrays (as in
Week 4), together with the policy pi whose values we estimate. We know the model, so we
know the true V_pi and the optimal q_* -- but the learners never look at them: they see
only episodes (prediction) or one transition at a time (control, `world.step`).
"""

from __future__ import annotations

import argparse
import functools
import traceback
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import pragman
from pragman.dynamics import successors
from pragman.model import build_tabular_model
from pragman.state import ACTIONS
from pragman.viz import plot_values

__all__ = ["load", "WORLDS", "World", "Episode", "constant", "one_over_n", "as_schedule",
           "learning_curves", "plot_curves", "final_error", "visits", "returns_from_start",
           "pretend_terminated", "plot_world", "partial", "plt", "np", "pragman",
           "ACTION_NAMES", "epsilon_greedy", "greedy_policy", "optimal_values", "gap",
           "run_control", "plot_control", "glie", "empirical_model", "certainty_equivalence",
           "two_state_batch", "plot_policy"]

ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


# --- episodes and step sizes ----------------------------------------------------------------

@dataclass(frozen=True)
class Episode:
    """One episode: S_0, A_0, R_1, S_1, ..., S_T. See the top of this file."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool

    def __len__(self) -> int:
        return len(self.rewards)


def constant(alpha: float) -> Callable[[int], float]:
    """The constant step size alpha_n = alpha."""
    if not 0 < alpha <= 1:
        raise ValueError(f"a constant step size should be in (0, 1], got {alpha}")
    f = lambda n: alpha                                        # noqa: E731
    f.__name__ = f"constant({alpha})"
    return f


def one_over_n(beta: float = 1.0, c: float = 1.0) -> Callable[[int], float]:
    """alpha_n = min(1, c / n**beta), where n is the number of updates of this state so far
    (including the current one, so n = 1 on the first update). beta = 1, c = 1 is the
    running average: V(s) is then exactly the mean of the targets seen in s."""
    f = lambda n: min(1.0, c / n ** beta)                      # noqa: E731
    f.__name__ = f"one_over_n(beta={beta}, c={c})"
    return f


def as_schedule(alpha) -> Callable[[int], float]:
    """A step-size schedule: a number means a constant step size; a function n -> alpha_n
    (such as one_over_n()) is used as it is."""
    return alpha if callable(alpha) else constant(float(alpha))


# --- the worlds -------------------------------------------------------------------------------

# Two routes to the goal: short, past the ghost's lair, or long, around it (as in Week 4).
GHOST_LAIR = """
#########
#P.....X#
#.##.##.#
#.#...#.#
#.#.G.#.#
#.#####.#
#.......#
#########
"""



def _gridworld(**overrides):
    return pragman.PragmanConfig(**{"layout": "XPPP\nPPPP\nPPPP\nPPPX", "observation": "index",
                                    "living_reward": -1.0, "goal_reward": -1.0, **overrides})


def _ghost_lair(**overrides):
    return pragman.PragmanConfig(**{"layout": GHOST_LAIR, "n_ghosts": 1,
                                    "ghost_policy": "proximity", "slip_prob": 0.1,
                                    "observation": "index", **overrides})


def _corridor(length: int = 20, **overrides):
    """A corridor of `length` cells: the agent starts at one end, the goal is at the other."""
    return pragman.PragmanConfig(**{"layout": "P" + "." * (length - 2) + "X", "observation": "index",
                                    "living_reward": 0.0, "slip_prob": 0.0, **overrides})


# name: (config factory or None, gamma, default policy, default max_steps, description)
WORLDS = {
    "random_walk": (None, 1.0, "random", 1000,
                    "Sutton & Barto's random walk (Example 6.2): states A-E, start in C, "
                    "+1 for leaving on the right"),
    "gridworld":   (_gridworld, 1.0, "random", 1000,
                    "the 4x4 gridworld of Week 4, random policy, reward -1 per step"),
    "maze":        (pragman.configs.maze_8x8, 0.99, "tired", 1000,
                    "the 8x8 maze of Week 4 (10% slip), a tired policy"),
    "corridor":    (_corridor, 0.99, "tired", 1000,
                    "a corridor of 20 cells (length=...), reward only at the goal, a tired policy"),
    "ghost_lair":  (_ghost_lair, 0.99, "tired", 1000,
                    "the ghost's lair of Week 4 (841 states), a tired policy"),
    "maxbias":     (None, 1.0, "random", 1000,
                    "Sutton & Barto's Example 6.7: from A, RIGHT ends with 0; LEFT goes to B, "
                    "where every one of 10 actions ends with a reward N(-0.1, 1)"),
}

# The policies, by name. "tired": the optimal action with probability 1 - TIRED, otherwise
# a uniformly random one.
TIRED = 0.3
RISKY_GAMMA = 0.95       # "risky" = tired version of the optimal policy for this discount


class World:
    """A world with a policy: everything needed to sample episodes and to know the truth.

    Attributes you may use:
        name, gamma
        n_states        how many states the estimators see (V has this length)
        max_steps       the time limit of an episode
        episodes(n, seed)   n episodes of the policy (a list of Episode)
        V_true          (n_states,) the true values V_pi, where they are well defined
        error(V)        the error of an estimate V (see its docstring)
        mu              (n_true_states,) the on-policy distribution: how often each state
                        is visited, as a fraction of all visits
        start           the start state (if there is only one)
        describe(s)     a picture of state s, as text
    The arrays of the model (as in Week 4): P (S, A, S), R (S, A), rho0 (S,), terminal (S,),
    and policy (S, A), the probabilities pi(a | s).
    """

    def __init__(self, name, P, R, R_next, rho0, terminal, gamma, policy, max_steps,
                 cfg=None, states=None, obs=None, description="", reward_noise=None):
        self.name, self.cfg, self.states = name, cfg, states
        self._noise = None if reward_noise is None else np.asarray(reward_noise, dtype=float)
        self.description = description
        self.P, self.R, self.rho0, self.terminal = P, R, rho0, terminal
        self._R_next = R_next                   # (S, A, K) reward of each successor
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.n_true_states, self.n_actions = R.shape
        self.policy = policy
        self.obs = np.arange(self.n_true_states) if obs is None else np.asarray(obs)
        self.n_states = int(self.obs.max()) + 1
        self._prepare()

    # --- internals
    def _prepare(self):
        S, A = self.n_true_states, self.n_actions
        K = int((self.P > 0).sum(axis=2).max())
        succ = np.zeros((S, A, K), dtype=np.int64)
        cum = np.ones((S, A, K))
        for s in range(S):
            for a in range(A):
                nz = np.flatnonzero(self.P[s, a] > 0)
                succ[s, a, :len(nz)] = nz
                succ[s, a, len(nz):] = nz[-1]
                cum[s, a, :len(nz)] = np.cumsum(self.P[s, a, nz])
                cum[s, a, len(nz) - 1:] = 1.0
        self._succ, self._cum = succ, cum
        self._cum_pi = np.cumsum(self.policy, axis=1)
        self._cum_pi[:, -1] = 1.0
        self._cum_rho0 = np.cumsum(self.rho0)
        self._cum_rho0[-1] = 1.0
        # the truth: V_pi on the true states, by a linear solve over the non-terminal ones
        P_pi = np.einsum("sa,sat->st", self.policy, self.P)
        R_pi = (self.policy * self.R).sum(1)
        live = ~self.terminal
        V = np.zeros(S)
        V[live] = np.linalg.solve(np.eye(live.sum()) - self.gamma * P_pi[np.ix_(live, live)],
                                  R_pi[live])
        self.V_true_states = V
        # the on-policy distribution: expected visits from rho0, normalized
        visits_ = np.zeros(S)
        visits_[live] = np.linalg.solve(np.eye(live.sum()) - P_pi[np.ix_(live, live)].T,
                                        self.rho0[live])
        self.mu = visits_ / visits_.sum()
        self.expected_length = float(visits_.sum())
        # the truth as the estimators see it: for each observed state, the average of V_pi
        # over the true states behind it, weighted by how often they are visited
        w = np.bincount(self.obs, weights=self.mu, minlength=self.n_states)
        vw = np.bincount(self.obs, weights=self.mu * V, minlength=self.n_states)
        self.V_true = np.where(w > 0, vw / np.maximum(w, 1e-300), 0.0)
        starts = np.flatnonzero(self.rho0)
        self.start = int(self.obs[starts[0]]) if len(starts) == 1 else None

    def _step(self, s, rng):
        a = (rng.random(len(s))[:, None] < self._cum_pi[s]).argmax(1)
        k = (rng.random(len(s))[:, None] < self._cum[s, a]).argmax(1)
        r = self._R_next[s, a, k]
        if self._noise is not None:
            r = r + self._noise[s, a] * rng.normal(size=len(s))
        return a, self._succ[s, a, k], r

    def step(self, s: int, a: int, rng) -> tuple[int, float, bool]:
        """One sampled transition for control: (next true state, reward, terminated).

        `s` is a *true* state index (as in `world.states`), not an observation; the agent's
        observation of a true state is `world.obs[s]` (identical unless observe="position")."""
        k = int((rng.random() < self._cum[s, a]).argmax())
        s2, r = int(self._succ[s, a, k]), float(self._R_next[s, a, k])
        if self._noise is not None and self._noise[s, a] > 0:
            r += float(self._noise[s, a]) * float(rng.normal())
        return s2, r, bool(self.terminal[s2])

    def reset(self, rng, start: str = "fixed") -> int:
        """A start state: from rho0 ("fixed"), or uniformly from all non-terminal states
        ("random", exploring starts)."""
        if start == "random":
            live = np.flatnonzero(~self.terminal)
            return int(rng.choice(live))
        return int((rng.random() < self._cum_rho0).argmax())

    # --- the public interface
    def episodes(self, n: int, seed: int = 0) -> list[Episode]:
        """n episodes of the policy, from the start distribution. Same seed, same episodes."""
        return _sample(self, n, seed)

    def error(self, V, weights: str = "on-policy") -> float:
        """How far the estimate V is from the truth: the root-mean-square error over states,

            sqrt( sum_s mu(s) * (V(s) - V_pi(s))**2 ),

        weighted by mu, the on-policy distribution (how often the policy visits s): errors
        in states the policy visits often count more. weights="uniform" weighs every
        non-terminal state the same instead. (If the estimators see less than the full state,
        the error is measured against V_pi of the full state: see `load(..., observe=...)`.)"""
        V = np.asarray(V, dtype=float)
        if V.shape != (self.n_states,):
            raise ValueError(f"V must have shape ({self.n_states},), got {V.shape}")
        diff2 = (V[self.obs] - self.V_true_states) ** 2
        if weights == "on-policy":
            w = self.mu
        elif weights == "uniform":
            w = (~self.terminal) / (~self.terminal).sum()
        else:
            raise ValueError("weights must be 'on-policy' or 'uniform'")
        return float(np.sqrt((w * diff2).sum()))

    def outcomes(self) -> dict:
        """How episodes end under the policy (no time limit): the probability of reaching
        the goal and of being caught by a ghost."""
        if self.states is None:
            return {}
        P_pi = np.einsum("sa,sat->st", self.policy, self.P)
        live = ~self.terminal
        visits_ = np.zeros(self.n_true_states)
        visits_[live] = np.linalg.solve(np.eye(live.sum()) - P_pi[np.ix_(live, live)].T,
                                        self.rho0[live])
        final = visits_ @ P_pi * self.terminal                  # where the episodes end
        caught = np.array([st.terminated and st.agent in st.ghosts for st in self.states])
        return {"goal": float(final[self.terminal & ~caught].sum()),
                "caught": float(final[caught].sum())}

    def describe(self, s: int) -> str:
        """A picture of (true) state s, as text."""
        if self.states is None:
            return f"state {s}"
        return self.states[s].ascii(self.cfg.layout)

    def __repr__(self):
        return (f"World({self.name!r}: {self.n_states} states, gamma={self.gamma}, "
                f"expected episode length {self.expected_length:.1f})")


def _sample(world: World, n: int, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    s = (rng.random(n)[:, None] < world._cum_rho0).argmax(1)
    S_, A_, R_ = [s], [], []
    done = world.terminal[s].copy()
    length = np.full(n, world.max_steps)
    for t in range(world.max_steps):
        a, s2, r = world._step(s, rng)
        s2 = np.where(done, s, s2)
        r = np.where(done, 0.0, r)
        A_.append(a), R_.append(r), S_.append(s2)
        newly = ~done & world.terminal[s2]
        length[newly] = t + 1
        done |= newly
        s = s2
        if done.all():
            break
    S_, A_, R_ = np.array(S_).T, np.array(A_).T, np.array(R_).T
    obs = world.obs
    return [Episode(obs[S_[i, :length[i] + 1]], A_[i, :length[i]], R_[i, :length[i]],
                    bool(world.terminal[S_[i, length[i]]])) for i in range(n)]


@functools.lru_cache(maxsize=16)
def _model(cfg):
    tm = build_tabular_model(cfg)
    S, A = tm.R.shape
    K = int((tm.P > 0).sum(axis=2).max())
    R_next = np.zeros((S, A, K))
    for s, st in enumerate(tm.states):
        for a, action in enumerate(ACTIONS):
            nz = list(np.flatnonzero(tm.P[s, a] > 0))
            rew = {}
            for p, nxt, r, _ in successors(cfg, st, action):
                if p > 0:
                    rew[tm.index[nxt]] = r
            for k, j in enumerate(nz):
                R_next[s, a, k] = rew[j]
            R_next[s, a, len(nz):] = rew[nz[-1]]
    return tm, R_next


def _random_walk_arrays(size=5, left_reward=0.0):
    # states 0..size-1 (A, B, C, ... for size 5), then the left end and the right end
    # (both terminal); actions LEFT, RIGHT; the start is in the middle
    S, A = size + 2, 2
    L, Rt = size, size + 1
    P = np.zeros((S, A, S))
    R_next = np.zeros((S, A, 1))
    for s in range(size):
        left, right = (s - 1 if s > 0 else L), (s + 1 if s < size - 1 else Rt)
        P[s, 0, left] = P[s, 1, right] = 1.0
        R_next[s, 0, 0] = left_reward if left == L else 0.0
        R_next[s, 1, 0] = 1.0 if right == Rt else 0.0
    P[L, :, L] = P[Rt, :, Rt] = 1.0
    R = R_next[:, :, 0]
    rho0 = np.zeros(S)
    rho0[size // 2] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[L:] = True
    return P, R, R_next, rho0, terminal


def _maxbias_arrays(n_arms=10, mean=-0.1, std=1.0):
    # states: 0 = A (start), 1 = B, 2 = terminal. Actions 0..n_arms-1. In A, action 1 is
    # LEFT (to B, reward 0) and every other action is RIGHT (terminal, reward 0); in B every
    # action ends the episode with a reward N(mean, std^2) (the noise is added when sampling).
    S, A = 3, n_arms
    P = np.zeros((S, A, S))
    R_next = np.zeros((S, A, 1))
    P[0, :, 2] = 1.0
    P[0, 1, :] = 0.0
    P[0, 1, 1] = 1.0
    P[1, :, 2] = 1.0
    R_next[1, :, 0] = mean
    P[2, :, 2] = 1.0
    R = R_next[:, :, 0].copy()
    noise = np.zeros((S, A))
    noise[1, :] = std
    rho0 = np.array([1.0, 0.0, 0.0])
    terminal = np.array([False, False, True])
    return P, R, R_next, rho0, terminal, noise


def _optimal_policy(P, R, terminal, gamma):
    """An optimal deterministic policy (policy iteration on the true model)."""
    S = R.shape[0]
    gamma = min(gamma, 0.9999)              # with gamma = 1, a policy that never ends has no value
    V = np.zeros(S)                         # start from the greedy policy of value iteration,
    for _ in range(100_000):                # which always terminates
        V_new = (R + gamma * P @ V).max(1)
        if np.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    pi = (R + gamma * P @ V).argmax(1)
    live = ~terminal
    for _ in range(1000):
        P_pi = P[np.arange(S), pi]
        R_pi = R[np.arange(S), pi]
        V = np.zeros(S)
        V[live] = np.linalg.solve(np.eye(live.sum()) - gamma * P_pi[np.ix_(live, live)], R_pi[live])
        Q = R + gamma * P @ V
        better = Q.max(1) > Q[np.arange(S), pi] + 1e-12
        if not better.any():
            return pi
        pi = np.where(better, Q.argmax(1), pi)
    raise RuntimeError("policy iteration did not converge")


def _policy_matrix(spec, P, R, terminal, gamma, A, tired=TIRED):
    S = R.shape[0]
    if isinstance(spec, np.ndarray) or isinstance(spec, list):
        spec = np.asarray(spec)
        if spec.shape == (S,):
            return np.eye(A)[spec]
        if spec.shape == (S, A):
            if not np.allclose(spec.sum(1), 1):
                raise ValueError("each row of a stochastic policy must sum to 1")
            return spec.astype(float)
        raise ValueError(f"a policy must have shape ({S},) or ({S}, {A})")
    uniform = np.full((S, A), 1.0 / A)
    if spec == "random":
        return uniform
    if spec == "optimal":
        return np.eye(A)[_optimal_policy(P, R, terminal, gamma)]
    if spec == "tired":
        return (1 - tired) * np.eye(A)[_optimal_policy(P, R, terminal, gamma)] + tired * uniform
    if spec == "risky":
        return (1 - tired) * np.eye(A)[_optimal_policy(P, R, terminal, RISKY_GAMMA)] \
            + tired * uniform
    raise ValueError(f"unknown policy {spec!r}: 'random', 'optimal', 'tired', 'risky', "
                     "or an array")


def load(world: str = "maze", policy=None, gamma: float | None = None,
         max_steps: int | None = None, observe: str = "state", tired: float = TIRED,
         **overrides) -> World:
    """A world, with the policy whose values we want to estimate.

    world:     "random_walk", "gridworld", "maze", "corridor", or "ghost_lair" (see WORLDS),
               or a PragmanConfig of your own (then pass gamma too).
    policy:    "random", "optimal", "tired" (the optimal action with probability 1 - tired,
               otherwise a random one), "risky" (the tired version of the policy that is
               optimal for gamma = 0.95: in ghost_lair, the short route past the ghost),
               or an array: (S,) actions or (S, A) probabilities. Default: the world's own.
    gamma:     the discount (default: the world's own).
    max_steps: the time limit of an episode (default 1000).
    observe:   "state" (default): the estimators see the full state. "position": they see
               only the agent's cell, not the ghost or the pellets (one value per cell).
    tired:     how often the "tired" and "risky" policies act at random (default 0.3).
    overrides: any PragmanConfig field, e.g. load("maze", slip_prob=0.3); for the corridor
               also its length, e.g. load("corridor", length=40). For the random
               walk: size (default 5) and left_reward (default 0), e.g.
               load("random_walk", size=19, left_reward=-1) for Sutton & Barto's Figure 7.2.
    """
    if isinstance(world, str):
        if world not in WORLDS:
            raise ValueError(f"unknown world {world!r}; choose from {', '.join(WORLDS)}")
        factory, g0, p0, m0, desc = WORLDS[world]
        name = world
    else:
        factory, g0, p0, m0, desc, name = None, None, "tired", 1000, "your own world", "custom"
    gamma = g0 if gamma is None else gamma
    if gamma is None:
        raise ValueError("pass gamma=... for a world of your own")
    policy = p0 if policy is None else policy
    max_steps = m0 if max_steps is None else max_steps
    noise = None
    if name == "random_walk":
        if set(overrides) - {"size", "left_reward"} or observe != "state":
            raise ValueError("random_walk takes only size=... (odd) and left_reward=...")
        P, R, R_next, rho0, terminal = _random_walk_arrays(**overrides)
        cfg = states = obs = None
    elif name == "maxbias":
        if set(overrides) - {"n_arms", "mean", "std"} or observe != "state":
            raise ValueError("maxbias takes only n_arms=..., mean=... and std=...")
        P, R, R_next, rho0, terminal, noise = _maxbias_arrays(**overrides)
        cfg = states = obs = None
    else:
        cfg = factory(**overrides) if factory is not None else (
            replace(world, **overrides) if overrides else world)
        tm, R_next = _model(cfg)
        P, R, rho0, terminal, states = tm.P, tm.R, tm.rho0, tm.terminal, tm.states
        if observe == "state":
            obs = None
        elif observe == "position":
            cells = sorted({st.agent for st in states})
            where = {c: i for i, c in enumerate(cells)}
            obs = np.array([where[st.agent] for st in states])
        else:
            raise ValueError("observe must be 'state' or 'position'")
    pm = _policy_matrix(policy, P, R, terminal, gamma, R.shape[1], tired)
    w = World(name, P, R, R_next, rho0, terminal, gamma, pm, max_steps, cfg=cfg, states=states,
              obs=obs, description=desc, reward_noise=noise)
    w.observe = observe
    return w


# --- learning curves --------------------------------------------------------------------------

def learning_curves(estimators: dict, world: World, n_episodes: int = 100, runs: int = 20,
                    seed: int = 0, weights: str = "on-policy") -> dict[str, np.ndarray]:
    """The error of each estimator after 0, 1, ..., n_episodes episodes, in several runs.

    estimators: {name: factory}, where factory(n_states, gamma) makes a fresh estimator,
                e.g. {"MC": MC, "TD, alpha=0.1": partial(TD0, alpha=0.1)}.
    Run r uses world.episodes(n_episodes, seed=seed + r) for *every* estimator, so they
    are compared on the same data.
    Returns {name: array of shape (runs, n_episodes + 1)}."""
    out = {name: np.zeros((runs, n_episodes + 1)) for name in estimators}
    for r in range(runs):
        eps = world.episodes(n_episodes, seed=seed + r)
        for name, factory in estimators.items():
            est = factory(world.n_states, world.gamma)
            out[name][r, 0] = world.error(est.V, weights)
            for i, ep in enumerate(eps):
                est.update(ep)
                out[name][r, i + 1] = world.error(est.V, weights)
    return out


def final_error(curves: np.ndarray, last: int = 1) -> tuple[float, float]:
    """(mean, standard error) over runs of the error averaged over the last `last` points of
    the curves; e.g. last=n_episodes + 1 averages over the whole curve."""
    x = curves[:, -last:].mean(axis=1)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def plot_curves(curves: dict, ax=None, log: bool = False, title: str | None = None):
    """Mean error against episodes, with a band of +- one standard error over runs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        m = c.mean(0)
        se = c.std(0, ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0 * m
        x = np.arange(len(m))
        ax.plot(x, m, label=name)
        ax.fill_between(x, m - se, m + se, alpha=0.2)
    ax.set_xlabel("episodes")
    ax.set_ylabel("RMS error of V")
    if log:
        ax.set_yscale("log")
    ax.legend()
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


# --- small helpers ----------------------------------------------------------------------------

def visits(episodes: list[Episode], n_states: int) -> np.ndarray:
    """How often each state was visited (S_0, ..., S_{T-1}; the last state does not count)."""
    return np.bincount(np.concatenate([ep.states[:-1] for ep in episodes]),
                       minlength=n_states)


def returns_from_start(episodes: list[Episode], gamma: float) -> np.ndarray:
    """G_0 of each episode: the discounted return from its first state."""
    return np.array([np.sum(ep.rewards * gamma ** np.arange(len(ep))) for ep in episodes])


def pretend_terminated(episodes: list[Episode]) -> list[Episode]:
    """The same episodes, but each one marked as terminated -- as a careless implementation
    would treat an episode cut off by the time limit."""
    return [replace(ep, terminated=True) for ep in episodes]


def _slice(world, like):
    ref = world.states[like]
    return [i for i, st in enumerate(world.states)
            if not st.terminated and st.ghosts == ref.ghosts and st.pellets == ref.pellets]


def plot_world(world: World, V, like: int | None = None, ax=None, title: str | None = None,
               fmt: str = "{:.2f}", cmap: str = "viridis"):
    """Values (or any number per state, e.g. errors or visit counts) on the map.

    V has one entry per state the estimators see (world.n_states). With a ghost, a state
    is more than the agent's cell, so we show one *slice*: the states that look like true
    state `like` (default: the start state) except for where the agent is."""
    V = np.asarray(V, dtype=float)
    if world.cfg is None:                               # the random walk: a plain bar
        k = world.n_states - 2
        if ax is None:
            _, ax = plt.subplots(figsize=(max(6, 0.6 * k), 1.6))
        ax.imshow(V[None, :k], cmap=cmap, aspect="auto")
        for i, v in enumerate(V[:k]):
            label = "ABCDE"[i] + "\n" if k == 5 else ""
            ax.text(i, 0, label + fmt.format(v), ha="center", va="center", color="w")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title or world.name)
        return ax
    if like is None:
        like = int(np.argmax(world.rho0))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    if world.observe == "position":
        cells = sorted({st.agent for st in world.states})
        vals = {c: V[i] for i, c in enumerate(cells)}
        plot_values(world.cfg, vals, ax=ax, title=title or world.name, fmt=fmt, cmap=cmap)
        return ax
    idx = _slice(world, like)
    states = [world.states[i] for i in idx]
    if title is None:
        title = world.name + ("" if not world.states[like].ghosts else
                              f"  (ghost at {world.states[like].ghosts[0]})")
    plot_values(world.cfg, V[world.obs[idx]], states=states, ax=ax, title=title, fmt=fmt,
                cmap=cmap)
    for g in world.states[like].ghosts:
        ax.text(g[1] + 0.5, g[0] + 0.5, "G", ha="center", va="center", fontsize=16, color="red",
                weight="bold")
    return ax



# --- control: acting in the world ---------------------------------------------------------------

def epsilon_greedy(q: np.ndarray, eps: float, rng) -> int:
    """An epsilon-greedy action for one row q = Q[s, :]: with probability eps a uniformly random
    action, otherwise the greedy one (ties broken at random)."""
    if rng.random() < eps:
        return int(rng.integers(len(q)))
    best = np.flatnonzero(q == q.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """The greedy deterministic policy of an action-value table: policy[s] = argmax_a Q[s, a]."""
    return np.asarray(Q).argmax(axis=1)


def _gamma_eff(world):
    return min(world.gamma, 0.9999)       # with gamma = 1 a policy that never ends has no value


@functools.lru_cache(maxsize=32)
def _optimal(world_id):
    world = _REGISTRY[world_id]
    P, R, g = world.P, world.R, _gamma_eff(world)
    V = np.zeros(world.n_true_states)
    for _ in range(200_000):
        V_new = (R + g * P @ V).max(1)
        if np.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    V[world.terminal] = 0.0
    return V, R + g * P @ V


_REGISTRY: dict = {}


def _register(world):
    _REGISTRY[id(world)] = world
    return id(world)


def optimal_values(world: World) -> tuple[np.ndarray, np.ndarray]:
    """(v_*, q_*) of the world, over the true states, by value iteration on the model."""
    V, Q = _optimal(_register(world))
    return V.copy(), Q.copy()


def policy_values(world: World, policy) -> np.ndarray:
    """V_pi over the true states for a deterministic (S,) or stochastic (S, A) policy."""
    pm = _policy_matrix(np.asarray(policy), world.P, world.R, world.terminal, world.gamma,
                        world.n_actions)
    P_pi = np.einsum("sa,sat->st", pm, world.P)
    R_pi = (pm * world.R).sum(1)
    live = ~world.terminal
    V = np.zeros(world.n_true_states)
    V[live] = np.linalg.solve(np.eye(live.sum()) - _gamma_eff(world) * P_pi[np.ix_(live, live)],
                              R_pi[live])
    return V


def gap(world: World, policy, where: str = "start") -> float:
    """How far a policy is from optimal: v_*(s) - v_pi(s) at the start state (where="start"),
    or in the worst state (where="max"). `policy` is an (S,) array of actions or an (S, A)
    array of probabilities, over the true states; for a Q table use greedy_policy(Q)."""
    V_star, _ = optimal_values(world)
    d = V_star - policy_values(world, policy)
    if where == "max":
        return float(d.max())
    return float((world.rho0 * d).sum())


def glie(eps0: float = 0.1, tau: float = 100.0) -> Callable[[int], float]:
    """The exploration schedule eps_k = eps0 * tau / (tau + k), k = 0, 1, ... the episode index.
    It is GLIE: sum_k eps_k = infinity and eps_k -> 0."""
    f = lambda k: eps0 * tau / (tau + k)                       # noqa: E731
    f.__name__ = f"glie(eps0={eps0}, tau={tau})"
    return f


def run_control(agents: dict, world: World, n_episodes: int = 500, runs: int = 10,
                epsilon=0.1, seed: int = 0, start: str = "fixed", gap_every: int = 1,
                alpha=None) -> dict:
    """Let each agent act epsilon-greedily in the world and learn online, for n_episodes
    episodes, in several runs (seeds). Within a run all agents see the same sequence of start
    states and random numbers, so they are compared on equal luck.

    agents:   {name: factory}, factory(n_states, n_actions, gamma) -> agent with a table .Q and
              a method .update(s, a, r, s2, a2, done, eps) (see control.py). `alpha=...`, if
              given, is passed on to every factory.
    epsilon:  a number (fixed) or a function of the episode index k (e.g. glie()).
    start:    "fixed" (from rho0) or "random" (exploring starts, uniform over states).
    Returns {name: {"return": (runs, n_episodes) the online return of every episode,
                    "gap": (runs, n_episodes) v_* - v_greedy at the start, after every episode
                           (every gap_every episodes; NaN in between),
                    "first_action": (runs, n_episodes) the first action of every episode,
                    "Q": the final table of the last run}}.
    The online return is undiscounted (the sum of rewards), so that it is comparable
    across worlds with different gamma; it is the quantity the agent actually earns."""
    eps_of = epsilon if callable(epsilon) else (lambda k: epsilon)
    S, A, g = world.n_states, world.n_actions, world.gamma
    out = {name: {"return": np.zeros((runs, n_episodes)), "gap": np.full((runs, n_episodes), np.nan),
                  "first_action": np.zeros((runs, n_episodes), dtype=int), "Q": None}
           for name in agents}
    for r in range(runs):
        made = {}
        for name, factory in agents.items():
            made[name] = factory(S, A, g) if alpha is None else factory(S, A, g, alpha=alpha)
        for name, agent in made.items():
            rng = np.random.default_rng(seed + r)         # same stream for every agent
            for k in range(n_episodes):
                eps = eps_of(k)
                s = world.reset(rng, start)
                a = epsilon_greedy(agent.Q[world.obs[s]], eps, rng)
                out[name]["first_action"][r, k] = a
                G = 0.0
                for t in range(world.max_steps):
                    s2, rew, done = world.step(s, a, rng)
                    G += rew
                    a2 = epsilon_greedy(agent.Q[world.obs[s2]], eps, rng)
                    agent.update(world.obs[s], a, rew, world.obs[s2], a2, done, eps)
                    s, a = s2, a2
                    if done:
                        break
                out[name]["return"][r, k] = G
                if (k + 1) % gap_every == 0 or k == n_episodes - 1:
                    out[name]["gap"][r, k] = gap(world, greedy_policy(agent.Q)[world.obs]
                                                 if world.n_states != world.n_true_states
                                                 else greedy_policy(agent.Q))
            out[name]["Q"] = agent.Q
    return out


def _smooth(x, k):
    if k <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0, axis=-1), axis=-1)
    out = np.empty_like(x)
    for i in range(x.shape[-1]):
        lo = max(0, i - k + 1)
        out[..., i] = (c[..., i + 1] - c[..., lo]) / (i + 1 - lo)
    return out


def plot_control(results: dict, what: str = "return", ax=None, smooth: int = 20,
                 title: str | None = None, log: bool = False):
    """Mean over runs (band: +- one standard error) of the online return per episode
    (what="return", smoothed over `smooth` episodes) or of the gap of the greedy policy
    (what="gap")."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        y = res[what]
        if what == "gap":
            cols = ~np.isnan(y[0])
            x, y = np.flatnonzero(cols) + 1, y[:, cols]
        else:
            x, y = np.arange(1, y.shape[1] + 1), _smooth(y, smooth)
        m = y.mean(0)
        se = y.std(0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0 * m
        ax.plot(x, m, label=name)
        ax.fill_between(x, m - se, m + se, alpha=0.2)
    ax.set_xlabel("episodes")
    ax.set_ylabel({"return": f"online return (moving average, {smooth})",
                   "gap": "v_*(start) - v_greedy(start)"}[what])
    if log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


def plot_policy(world: World, Q: np.ndarray, like: int | None = None, ax=None,
                title: str | None = None):
    """The greedy policy of Q (arrows) over the greedy values V(s) = max_a Q(s, a), on the map."""
    V = np.asarray(Q).max(axis=1)
    ax = plot_world(world, V, like=like, ax=ax, title=title, fmt="")
    if world.cfg is None:
        return ax
    if like is None:
        like = int(np.argmax(world.rho0))
    idx = _slice(world, like)
    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}     # UP, RIGHT, DOWN, LEFT
    pol = greedy_policy(Q)
    for i in idx:
        st = world.states[i]
        dx, dy = arrows[int(pol[world.obs[i]])]
        ax.arrow(st.agent[1] + 0.5 - dx / 2, st.agent[0] + 0.5 - dy / 2, dx, dy,
                 head_width=0.15, length_includes_head=True, color="w")
    return ax


# --- batch TD(0) and the empirical model -------------------------------------------------------

def empirical_model(episodes: list[Episode], n_states: int):
    """The maximum-likelihood Markov reward process of the policy from a batch of episodes:
    (R_hat (S,), P_hat (S, S), N (S,) visit counts), where termination is a transition to a
    virtual absorbing state with index n_states (so P_hat has n_states + 1 columns)."""
    S = n_states
    N = np.zeros(S)
    Rsum = np.zeros(S)
    C = np.zeros((S, S + 1))
    for ep in episodes:
        s, r = ep.states, ep.rewards
        T = len(r)
        for t in range(T):
            N[s[t]] += 1
            Rsum[s[t]] += r[t]
            nxt = S if (t == T - 1 and ep.terminated) else s[t + 1]
            C[s[t], nxt] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        R_hat = np.where(N > 0, Rsum / N, 0.0)
        P_hat = np.where(N[:, None] > 0, C / np.maximum(N[:, None], 1), 0.0)
    return R_hat, P_hat, N


def certainty_equivalence(episodes: list[Episode], n_states: int, gamma: float) -> np.ndarray:
    """The value function of the empirical model: the solution of V = R_hat + gamma P_hat V
    (the Week 4 direct solve on the model estimated from the batch; the virtual terminal
    state has value 0). States never visited get 0."""
    R_hat, P_hat, N = empirical_model(episodes, n_states)
    S = n_states
    seen = N > 0
    A = np.eye(seen.sum()) - gamma * P_hat[np.ix_(seen, np.flatnonzero(seen))]
    V = np.zeros(S)
    V[seen] = np.linalg.solve(A, R_hat[seen])
    return V


def two_state_batch() -> list[Episode]:
    """The eight episodes of the lecture (Sutton & Barto, Example 6.4): states A = 0, B = 1,
    terminal = 2. Once A -0-> B -0-> end, six times B -1-> end, once B -0-> end."""
    A, B, T = 0, 1, 2
    eps = [Episode(np.array([A, B, T]), np.array([0, 0]), np.array([0.0, 0.0]), True)]
    eps += [Episode(np.array([B, T]), np.array([0]), np.array([1.0]), True) for _ in range(6)]
    eps += [Episode(np.array([B, T]), np.array([0]), np.array([0.0]), True)]
    return eps


# --- Task 1: checking your implementation -----------------------------------------------------

# A hand-made episode: S_0..S_3 = 0, 1, 0, 2, rewards 1, 2, 3; state 2 is terminal.
_HAND = Episode(np.array([0, 1, 0, 2]), np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]), True)
_RW_TRUE = np.array([1, 2, 3, 4, 5]) / 6
_FILES = ("td.py", "control.py")


def _ok(cond, msg_ok, msg_bad):
    print(("  OK    " + msg_ok) if cond else ("  FAIL  " + msg_bad))
    return bool(cond)


def _todo_message(e: NotImplementedError, fallback: str) -> str:
    names, tb = [], e.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        if Path(code.co_filename).name in _FILES and code.co_name != "todo":
            names.append(code.co_qualname)
        tb = tb.tb_next
    where, msg = (names[-1] if names else fallback), str(e)
    return msg if msg.startswith(where) else f"{where}: {msg}" if msg else where


def _todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), True
    except NotImplementedError as e:
        print(f"  TODO  {_todo_message(e, getattr(fn, '__qualname__', str(fn)))}")
        return None, False


def _run_part(title: str, part: Callable) -> None:
    print(title)
    try:
        part()
    except Exception as e:                                      # noqa: BLE001
        print(f"  FAIL  {type(e).__name__}: {e}")
        frames = [f for f in traceback.extract_tb(e.__traceback__)
                  if Path(f.filename).name in _FILES]
        if frames:
            f = frames[-1]
            print(f"        in {Path(f.filename).name}, line {f.lineno}, in {f.name}:  "
                  f"{(f.line or '').strip()}")
    print()


def _learn(est, episodes):
    for ep in episodes:
        est.update(ep)
    return est.V


def check() -> None:
    import td
    import control
    print("Checking td.py and control.py (Task 1). Each part is checked once it is implemented.\n")
    rw = load("random_walk")

    def part1():
        V, done = _todo(lambda: _learn(td.TD0(3, 1.0, alpha=0.5), [_HAND]))
        if not done:
            return
        _ok(np.allclose(V, [1.75, 1.25, 0.0]),
            "TD(0), alpha = 0.5, on 0 -1-> 1 -2-> 0 -3-> end: V = [1.75, 1.25, 0]",
            f"TD(0) with alpha = 0.5 on 0 -1-> 1 -2-> 0 -3-> end: expected [1.75, 1.25, 0], "
            f"got {V}")
        V = _learn(td.TD0(3, 1.0, alpha=0.5, V0=1.0), [_HAND])
        _ok(np.allclose(V, [2.25, 2.25, 1.0]),
            "... with V0 = 1: the terminal state counts as 0 (V = [2.25, 2.25, 1])",
            f"with V0 = 1: expected [2.25, 2.25, 1] (the value of a terminal state is 0, "
            f"whatever V says), got {V}")
        V = _learn(td.TD0(3, 1.0, alpha=0.5, V0=1.0), [replace(_HAND, terminated=False)])
        _ok(np.allclose(V, [2.75, 2.25, 1.0]),
            "... and if the episode was cut off, we bootstrap from V(S_T): V = [2.75, 2.25, 1]",
            f"a truncated episode (terminated=False), V0 = 1: expected [2.75, 2.25, 1] "
            f"(bootstrap from V(S_T)), got {V}")
        V = _learn(td.TD0(rw.n_states, 1.0, alpha=one_over_n(0.7)), rw.episodes(3000, seed=0))
        _ok(np.abs(V[:5] - _RW_TRUE).max() < 0.05,
            f"random walk, 3000 episodes: V = {np.round(V[:5], 3)} (true: 1/6, ..., 5/6)",
            f"random walk, 3000 episodes: V = {np.round(V[:5], 3)}, "
            f"expected about {np.round(_RW_TRUE, 3)}")

    def part2():
        eps = two_state_batch()
        est = td.BatchTD0(3, 1.0, alpha=0.1)
        V, done = _todo(lambda: est.fit(eps))
        if not done:
            return
        _ok(np.allclose(V[:2], [0.75, 0.75], atol=1e-6),
            "the two-state example of the lecture: V(A) = V(B) = 3/4",
            f"two-state example: expected V(A) = V(B) = 0.75, got {np.round(V[:2], 4)}")
        eps = rw.episodes(40, seed=5)
        V = td.BatchTD0(rw.n_states, 1.0).fit(eps)
        Vce = certainty_equivalence(eps, rw.n_states, 1.0)
        _ok(np.allclose(V, Vce, atol=1e-6),
            "random walk, 40 episodes: batch TD(0) = the value of the empirical model "
            "(Proposition 5.2)",
            f"batch TD(0) {np.round(V[:5], 4)} should equal the certainty-equivalence "
            f"estimate {np.round(Vce[:5], 4)}")
        cmax = visits(eps, rw.n_states).max()
        V2 = td.BatchTD0(rw.n_states, 1.0, alpha=0.3 / cmax).fit(eps)
        _ok(np.allclose(V2, Vce, atol=1e-5),
            "... and the fixed point does not depend on the step size",
            "the fixed point should not depend on alpha (is your stopping tolerance tight?)")

    def _one(cls, **kw):               # a 2-state, 2-action table to check targets by hand
        ag = cls(2, 2, 0.5, alpha=0.5)
        ag.Q[:] = 1.0
        ag.Q[1] = [1.0, 3.0]
        ag.update(0, 0, 2.0, 1, 0, False, kw.get("eps", 0.5))
        return ag.Q[0, 0]

    def part3():
        q, done = _todo(lambda: _one(control.SARSA))
        if not done:
            return
        _ok(np.isclose(q, 1.75), "SARSA: target 2 + 0.5 * Q(s', a') with a' = 0: Q(s,a) -> 1.75",
            f"SARSA: Q(s,a) = 1, r = 2, gamma = 0.5, Q(s',.) = [1, 3], a' = 0, alpha = 0.5: "
            f"expected 1 + 0.5 * (2 + 0.5 * 1 - 1) = 1.75, got {q}")
        ag = control.SARSA(2, 2, 0.5, alpha=0.5)
        ag.Q[:] = 1.0
        ag.update(0, 0, 2.0, 1, 0, True, 0.1)
        _ok(np.isclose(ag.Q[0, 0], 1.5), "... terminal transition: the target is just r (Q -> 1.5)",
            f"after a terminal transition the target is r alone: expected 1.5, got {ag.Q[0, 0]}")

    def part4():
        q, done = _todo(lambda: _one(control.ExpectedSARSA, eps=0.5))
        if not done:
            return
        _ok(np.isclose(q, 2.125),
            "Expected SARSA, eps = 0.5: target 2 + 0.5 * (0.25 * 1 + 0.75 * 3): Q(s,a) -> 2.125",
            f"Expected SARSA with eps = 0.5 over 2 actions: the eps-greedy policy at s' is "
            f"[0.25, 0.75], expected target 3.25 and Q = 2.125, got {q}")
        q0 = _one(control.ExpectedSARSA, eps=0.0)
        _ok(np.isclose(q0, 2.25), "... with eps = 0 it is the Q-learning target (Q -> 2.25)",
            f"with eps = 0 Expected SARSA should equal Q-learning: expected 2.25, got {q0}")

    def part5():
        q, done = _todo(lambda: _one(control.QLearning))
        if not done:
            return
        _ok(np.isclose(q, 2.25), "Q-learning: target 2 + 0.5 * max Q(s',.) = 3.5: Q(s,a) -> 2.25",
            f"Q-learning: expected 1 + 0.5 * (3.5 - 1) = 2.25, got {q}")
        world = load("gridworld")
        res = run_control({"Q": control.QLearning}, world, n_episodes=300, runs=1, epsilon=0.2,
                          alpha=0.2, gap_every=300)
        g = res["Q"]["gap"][0, -1]
        _ok(g < 0.05, f"gridworld, 300 episodes of eps-greedy Q-learning: the greedy policy is "
            f"optimal (gap {g:.3f})",
            f"gridworld, 300 episodes: the greedy policy should be (nearly) optimal, gap = {g:.3f}")

    def part6():
        if not hasattr(control, "DoubleQLearning"):
            print("  (no DoubleQLearning class: optional, for group B2)")
            return
        ag = control.DoubleQLearning(2, 2, 0.5, alpha=0.5)
        ag.Q1[:] = 1.0
        ag.Q1[1] = [1.0, 3.0]
        ag.Q2[:] = 1.0
        ag.Q2[1] = [5.0, 0.0]
        _, done = _todo(lambda: ag.update(0, 0, 2.0, 1, 0, False, 0.1, which=0))
        if not done:
            return
        _ok(np.isclose(ag.Q1[0, 0], 1.5),
            "Double Q-learning: Q1 selects a' = 1, Q2 evaluates it as 0: target 2, Q1(s,a) -> 1.5",
            f"Q1 selects argmax Q1(s',.) = 1, Q2(s',1) = 0, target 2 + 0.5 * 0 = 2, "
            f"expected 1.5, got {ag.Q1[0, 0]}")

    _run_part("1. TD0", part1)
    _run_part("2. BatchTD0", part2)
    _run_part("3. SARSA", part3)
    _run_part("4. ExpectedSARSA", part4)
    _run_part("5. QLearning", part5)
    _run_part("6. DoubleQLearning (optional)", part6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check td.py and control.py.")
    ap.parse_args()
    check()
