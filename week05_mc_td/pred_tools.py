"""Week 5 tools: the worlds, sampling episodes, the true values, errors, curves, and checks.

No need to edit this file. The files you edit are `mc.py` (Monte Carlo) and `td.py`
(temporal-difference learning). Typical use, from a script of your own in this directory:

    from pred_tools import *
    from mc import MC
    from td import TD0, NStepTD

    world = load("maze")                       # a world, with the policy we evaluate
    episodes = world.episodes(100, seed=0)     # 100 episodes of that policy
    est = MC(world.n_states, world.gamma)      # an estimator ...
    for ep in episodes:
        est.update(ep)                         # ... learns from one episode at a time
    print(world.error(est.V))                  # how far est.V is from the true V_pi

    curves = learning_curves({"MC": MC, "TD(0), alpha=0.1": partial(TD0, alpha=0.1)},
                             world, n_episodes=100, runs=20)
    plot_curves(curves)
    plt.show()

Run this file itself to check your implementation (Task 1):

    uv run python week05_mc_td/pred_tools.py

EPISODES. `world.episodes(n, seed)` returns a list of `Episode`s. An episode of length T is

    ep.states      (T+1,) int array: S_0, S_1, ..., S_T
    ep.actions     (T,)   int array: A_0, ..., A_{T-1}
    ep.rewards     (T,)   float array: rewards[t] is the reward of the step from S_t to S_{t+1}
                          (R_{t+1} in Sutton & Barto; R_t in the Week 5 notes)
    ep.terminated  bool:  True if S_T is terminal (its value is 0 by definition);
                          False if the episode was cut off by the time limit (S_T is an
                          ordinary state, whose value is not 0)

The same seed always gives the same episodes, so two estimators given the same seed
learn from exactly the same data: their difference is the algorithm, not the luck.

WORLDS. `load(name)` returns a `World`: the Prag-man world written out as arrays (as in
Week 4), together with the policy pi whose values we estimate. We know the model, so we
know the true V_pi -- but the estimators never look at it: they see only episodes.
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
           "ACTION_NAMES"]

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
                 cfg=None, states=None, obs=None, description=""):
        self.name, self.cfg, self.states = name, cfg, states
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
        return a, self._succ[s, a, k], self._R_next[s, a, k]

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
    if name == "random_walk":
        if set(overrides) - {"size", "left_reward"} or observe != "state":
            raise ValueError("random_walk takes only size=... (odd) and left_reward=...")
        P, R, R_next, rho0, terminal = _random_walk_arrays(**overrides)
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
              obs=obs, description=desc)
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


# --- Task 1: checking your implementation -----------------------------------------------------

# A hand-made episode: S_0..S_3 = 0, 1, 0, 2, rewards 1, 2, 3; state 2 is terminal.
_HAND = Episode(np.array([0, 1, 0, 2]), np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]), True)
_RW_TRUE = np.array([1, 2, 3, 4, 5]) / 6


def _ok(cond, msg_ok, msg_bad):
    print(("  OK    " + msg_ok) if cond else ("  FAIL  " + msg_bad))
    return bool(cond)


def _todo_message(e: NotImplementedError, fallback: str) -> str:
    names, tb = [], e.__traceback__
    while tb is not None:                       # e.g. "MC.update", "discounted_returns"
        code = tb.tb_frame.f_code
        if Path(code.co_filename).name in ("mc.py", "td.py") and code.co_name != "todo":
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
                  if Path(f.filename).name in ("mc.py", "td.py")]
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
    import mc
    import td
    print("Checking mc.py and td.py (Task 1). Each part is checked once it is implemented.\n")
    rw = load("random_walk")

    def part1():
        G, done = _todo(mc.discounted_returns, np.array([1.0, 0.0, 2.0]), 0.5)
        if not done:
            return
        _ok(np.allclose(G, [1.5, 1.0, 2.0]),
            "G_t for rewards 1, 0, 2 and gamma = 0.5: [1.5, 1.0, 2.0]",
            f"rewards 1, 0, 2, gamma = 0.5: expected G = [1.5, 1.0, 2.0], got {G}")
        r = np.random.default_rng(1).normal(size=50)
        G = mc.discounted_returns(r, 0.9)
        brute = np.array([np.sum(r[t:] * 0.9 ** np.arange(50 - t)) for t in range(50)])
        _ok(np.allclose(G, brute), "50 random rewards, gamma = 0.9: matches the definition",
            "50 random rewards, gamma = 0.9: does not match G_t = sum_k gamma^k R_{t+k+1}")

    def part2():
        est, done = _todo(lambda: _learn(mc.MC(3, 1.0, alpha=one_over_n()), [_HAND]))
        if not done:
            return
        _ok(np.allclose(est[:2], [4.5, 5.0]),
            "every-visit MC on the episode 0 -1-> 1 -2-> 0 -3-> end: V = [4.5, 5.0]",
            f"every-visit MC on 0 -1-> 1 -2-> 0 -3-> end (gamma = 1, running average): "
            f"expected V[:2] = [4.5, 5.0], got {est[:2]}")
        V = _learn(mc.MC(3, 1.0, alpha=one_over_n(), first_visit=True), [_HAND])
        _ok(np.allclose(V[:2], [6.0, 5.0]), "first-visit MC on the same episode: V = [6.0, 5.0]",
            f"first-visit MC: expected V[:2] = [6.0, 5.0], got {V[:2]}")
        V = _learn(mc.MC(3, 1.0, alpha=0.5), [_HAND])
        _ok(np.allclose(V[:2], [3.0, 2.5]),
            "every-visit MC, alpha = 0.5, updating in time order: V = [3.0, 2.5]",
            f"every-visit MC with alpha = 0.5: expected V[:2] = [3.0, 2.5], got {V[:2]} "
            "(update in time order, t = 0, 1, ..., T-1)")
        V = _learn(mc.MC(3, 1.0, alpha=one_over_n(), V0=7.0), [_HAND])
        _ok(np.allclose(V[:2], [4.5, 5.0]), "with the running average, V0 is forgotten",
            f"with V0 = 7 and the running average, expected V[:2] = [4.5, 5.0], got {V[:2]}")
        V = _learn(mc.MC(rw.n_states, 1.0), rw.episodes(3000, seed=0))
        _ok(np.abs(V[:5] - _RW_TRUE).max() < 0.03,
            f"random walk, 3000 episodes: V = {np.round(V[:5], 3)} (true: 1/6, ..., 5/6)",
            f"random walk, 3000 episodes: V = {np.round(V[:5], 3)}, "
            f"expected about {np.round(_RW_TRUE, 3)}")

    def part3():
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

    def part4():
        V, done = _todo(lambda: _learn(td.NStepTD(3, 1.0, n=2, alpha=0.5), [_HAND]))
        if not done:
            return
        _ok(np.allclose(V, [2.25, 2.5, 0.0]),
            "2-step TD, alpha = 0.5, on 0 -1-> 1 -2-> 0 -3-> end: V = [2.25, 2.5, 0]",
            f"2-step TD with alpha = 0.5: expected [2.25, 2.5, 0], got {V}")
        V = _learn(td.NStepTD(3, 1.0, n=2, alpha=0.5, V0=1.0), [replace(_HAND, terminated=False)])
        _ok(np.allclose(V, [3.25, 3.5, 1.0]),
            "... on the same episode cut off by the time limit, V0 = 1: V = [3.25, 3.5, 1]",
            f"2-step TD, truncated episode (terminated=False), V0 = 1: expected "
            f"[3.25, 3.5, 1] (bootstrap from V(S_T) whenever tau + n >= T), got {V}")
        maze = load("maze")
        eps = maze.episodes(30, seed=3)
        assert all(ep.terminated for ep in eps)
        V1 = _learn(td.NStepTD(maze.n_states, maze.gamma, n=1, alpha=0.3, V0=0.2), eps)
        try:
            V0 = _learn(td.TD0(maze.n_states, maze.gamma, alpha=0.3, V0=0.2), eps)
            _ok(np.allclose(V1, V0), "n = 1 is TD(0) (30 episodes on the maze)",
                "n = 1 should give exactly what TD0 gives")
        except NotImplementedError:
            print("  TODO  (n = 1 vs TD0: implement TD0 first)")
        Vinf = _learn(td.NStepTD(maze.n_states, maze.gamma, n=10**6, alpha=0.3, V0=0.2), eps)
        try:
            Vmc = _learn(mc.MC(maze.n_states, maze.gamma, alpha=0.3, V0=0.2), eps)
            _ok(np.allclose(Vinf, Vmc), "n = infinity is every-visit MC with the same alpha "
                "(30 episodes on the maze)",
                "n = 10**6 should give exactly what every-visit MC with the same constant "
                "alpha gives (if one of them is right, which one is wrong?)")
        except NotImplementedError:
            print("  TODO  (n = infinity vs MC: needs mc.MC)")

    _run_part("1. discounted_returns", part1)
    _run_part("2. MC (every-visit and first-visit)", part2)
    _run_part("3. TD0", part3)
    _run_part("4. NStepTD", part4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check mc.py and td.py.")
    ap.parse_args()
    check()
