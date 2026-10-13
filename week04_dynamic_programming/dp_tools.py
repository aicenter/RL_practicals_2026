"""Week 4 tools: the worlds, the model object, checks, plots, and the final evaluation.

No need to edit this file. The file you edit is `dp.py`. Typical use, from a
script of your own in this directory:

    from dp_tools import *
    import dp

    model, gamma = load("ghost_lair")        # the model of one world, and its discount
    V, policy, deltas = dp.value_iteration(model, gamma)
    print(model.backups, "backups;  gap to optimal:", gap(model, gamma, policy))
    plot_world(model, V, policy)
    plt.show()

Run this file itself to check your implementation (Task 1), or with --final for
the final evaluation (Task 3):

    uv run python week04_dynamic_programming/dp_tools.py
    uv run python week04_dynamic_programming/dp_tools.py --final

THE MODEL. `load` returns a `DPModel`: the Prag-man world written out as the
arrays of the lecture, for the states reachable from the start.

    model.n_states, model.n_actions      S and A (A = 4: UP, RIGHT, DOWN, LEFT)
    model.R                              (S, A) array, the expected reward R(s, a)
    model.P                              (S, A, S) array, P(s' | s, a)
    model.expect(V)                      (S, A) array: sum_s' P(s' | s, a) V(s')
    model.expect(V, states, actions)     the same for chosen (s, a) pairs only
    model.terminal                       (S,) bool; terminal states are absorbing, reward 0
    model.rho0                           (S,) the distribution of the start state
    model.states                         the Prag-man states, in index order
    model.backups                        how many backups this model has computed

COUNTING BACKUPS. One *backup* is one expectation sum_s' P(s'|s,a) V(s') for one
pair (s, a): reading the model once. `model.expect` counts them: a full call
costs S * A, `model.expect(V, actions=policy)` costs S (one action per state),
`model.expect(V, [s])` costs A. Reading `model.P` directly is not counted -- and
in the final evaluation (and when we check `solve`) it is not available at all.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import pragman
from pragman.model import build_tabular_model
from pragman.viz import plot_values, plot_policy
from rlcourse import leaderboard

WEEK_DIR = Path(__file__).parent
FINAL_CACHE = WEEK_DIR / "results" / "final_evaluation.json"
# The file you edit this week. The final evaluation uploads it to the course
# leaderboard, as a record of your work (we may look at it and give feedback).
SUBMITTED_FILES = ("dp.py",)

EPS = 1e-3            # a policy is "eps-optimal" if V_pi(s) >= V*(s) - EPS in EVERY state s
FAIL_RATIO = 10.0     # in the final score, a world where the policy is not eps-optimal counts as 10x VI
MEMORY_LIMIT = 3000   # states; P is dense, S * 4 * S * 8 bytes (3000 states: 290 MB)

ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_ARROWS = "↑→↓←"

__all__ = ["load", "pellet_maze", "WORLDS", "DPModel", "EPS", "optimal", "gap", "measure",
           "outcomes", "plot_world", "plt", "np", "pragman", "ACTION_NAMES"]


# --- the worlds -------------------------------------------------------------------------

# The ghost's lair: a short route along the top, past the lair, and a long detour along
# the bottom. The ghost sleeps (wanders) until you come within 4 steps; then it chases.
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

# The 8x8 maze of `maze_8x8`, with its goal replaced by pellets. The episode ends
# when the last pellet is eaten. Pellets are numbered in the order pellet_maze() adds them.
PELLET_CELLS = ((7, 0), (0, 7), (4, 6), (2, 3), (6, 4), (5, 0), (0, 4))


def pellet_maze(n_pellets: int = 4, **overrides) -> pragman.PragmanConfig:
    """maze_8x8 without its goal, but with `n_pellets` pellets: about 52 * 2**n states."""
    if not 0 <= n_pellets <= len(PELLET_CELLS):
        raise ValueError(f"n_pellets must be between 0 and {len(PELLET_CELLS)}")
    rows = [list(r) for r in pragman.NAMED_LAYOUTS["maze_8x8"].split("\n")]
    rows[7][7] = "."                                     # no goal: eat the pellets
    for r, c in PELLET_CELLS[:n_pellets]:
        rows[r][c] = "o"
    if n_pellets == 0:
        rows[7][7] = "X"                                 # ... unless there are none
    return pragman.PragmanConfig(**{"layout": "\n".join("".join(r) for r in rows),
                                    "observation": "index", "slip_prob": 0.1, **overrides})


def _gridworld(**overrides):
    """The 4x4 gridworld of the Week 4 lecture (Sutton & Barto, Example 4.1): reward -1 on
    every step, two terminal corners, and a start in every other cell."""
    return pragman.PragmanConfig(**{"layout": "XPPP\nPPPP\nPPPP\nPPPX", "observation": "index",
                                    "living_reward": -1.0, "goal_reward": -1.0, **overrides})


def _ghost_lair(**overrides):
    return pragman.PragmanConfig(**{"layout": GHOST_LAIR, "n_ghosts": 1,
                                    "ghost_policy": "proximity", "slip_prob": 0.1,
                                    "observation": "index", **overrides})


# name: (config factory, discount, description)
WORLDS: dict[str, tuple[Callable, float, str]] = {
    "gridworld":  (_gridworld, 1.0, "the 4x4 gridworld of the lecture, 16 states"),
    "maze":       (pragman.configs.maze_8x8, 0.99, "maze_8x8: 52 states, 10% slip"),
    "ghost_lair": (_ghost_lair, 0.99, "two routes and a sleeping ghost: 841 states"),
    "pellets":    (pellet_maze, 0.999, "the maze with 4 pellets: 752 states"),
}

# The final evaluation uses these worlds (name, factory, overrides, gamma).
_FINAL_WORLDS = (
    ("maze", pragman.configs.maze_8x8, {}, 0.99),
    ("ghost_lair", _ghost_lair, {}, 0.99),
    ("pellets", pellet_maze, {}, 0.999),
    ("ice_maze", pragman.configs.maze_8x8, {"slip_prob": 0.4}, 0.999),
    ("dry_maze", pragman.configs.maze_8x8, {"slip_prob": 0.0}, 0.99),
    ("haunted", pragman.configs.haunted_5x5, {"ghost_policy": "chaser", "slip_prob": 0.1}, 0.9),
    ("impatient_lair", _ghost_lair, {}, 0.95),
)


# --- the model ----------------------------------------------------------------------------

class DPModel:
    """A Prag-man world as the arrays P, R of the lecture, with a backup counter.

    Build one with `load(...)`. Attributes are described at the top of this file."""

    def __init__(self, cfg: pragman.PragmanConfig, max_states: int = 1500,
                 name: str = "custom", hide_P: bool = False):
        if max_states > MEMORY_LIMIT:
            raise ValueError(f"max_states > {MEMORY_LIMIT}: the dense P would need "
                             f"{max_states**2 * 32 / 1e9:.1f} GB or more")
        tm = _build(cfg, max_states)
        self.name, self.cfg = name, cfg
        self.states, self.index = tm.states, tm.index
        self.n_states, self.n_actions = tm.R.shape
        self.R = tm.R.copy()
        self.R.flags.writeable = False
        self.terminal = tm.terminal.copy()
        self.rho0 = tm.rho0.copy()
        self._P = tm.P                     # shared with the cache below: do not modify
        self._hide_P = hide_P
        self.backups = 0

    @property
    def P(self) -> np.ndarray:
        """P[s, a, s'] = P(s' | s, a). Not counted as backups; not available in the final."""
        if self._hide_P:
            raise AttributeError("model.P is not available here: solve() may read the model "
                                 "only through model.expect (and model.R), which counts backups.")
        return self._P

    def expect(self, V: np.ndarray, states=None, actions=None) -> np.ndarray:
        """E[V(S') | S=s, A=a] = sum_s' P(s'|s,a) V(s').

        expect(V)                  -> (S, A) array, for all states and actions;  costs S*A backups
        expect(V, states)          -> (n, A) array, for the given n states;       costs n*A
        expect(V, actions=pi)      -> (S,) array, for the action pi[s] in each s;  costs S
        expect(V, states, actions) -> (n,) array, for the pairs (states[i], actions[i]); costs n
        """
        V = np.asarray(V, dtype=np.float64)
        if V.shape != (self.n_states,):
            raise ValueError(f"V must have shape ({self.n_states},), got {V.shape}")
        if states is None:
            states = np.arange(self.n_states) if actions is not None else None
        else:
            states = np.atleast_1d(np.asarray(states, dtype=np.int64))
        if actions is None:
            if states is None:
                self.backups += self.n_states * self.n_actions
                return self._P @ V
            self.backups += len(states) * self.n_actions
            return self._P[states] @ V
        actions = np.broadcast_to(np.asarray(actions, dtype=np.int64), states.shape)
        self.backups += len(states)
        return np.einsum("ij,j->i", self._P[states, actions], V)

    def describe(self, s: int) -> str:
        """The board of state s, as text."""
        return self.states[s].ascii(self.cfg.layout)

    def __repr__(self) -> str:
        return f"DPModel({self.name!r}, n_states={self.n_states}, backups={self.backups})"


@functools.lru_cache(maxsize=32)
def _build(cfg, max_states):
    return build_tabular_model(cfg, max_states=max_states)


def load(world="maze", gamma: float | None = None, max_states: int = 1500,
         **overrides) -> tuple[DPModel, float]:
    """A fresh model (backup counter at zero) and its discount.

    world: a name from WORLDS ("gridworld", "maze", "ghost_lair", "pellets"), or a
           PragmanConfig of your own (e.g. pellet_maze(5); then pass gamma too).
    gamma: overrides the world's discount.
    overrides: any PragmanConfig field, e.g. load("maze", slip_prob=0.3).
    """
    if isinstance(world, str):
        if world not in WORLDS:
            raise ValueError(f"unknown world {world!r}; choose from {', '.join(WORLDS)}")
        factory, default_gamma, _ = WORLDS[world]
        cfg, name = factory(**overrides), world
    else:
        cfg, default_gamma, name = world, None, "custom"
        if overrides:
            from dataclasses import replace
            cfg = replace(cfg, **overrides)
    gamma = default_gamma if gamma is None else gamma
    if gamma is None:
        raise ValueError("pass gamma=... for a world of your own")
    return DPModel(cfg, max_states=max_states, name=name), float(gamma)


# --- ground truth (none of this counts backups) -------------------------------------------
# These use the model directly. They give you the true answers to compare with; they are
# not a model of how you should write your own algorithms.

def _policy_matrix(model, policy):
    policy = np.asarray(policy)
    if policy.shape == (model.n_states,):
        if not np.issubdtype(policy.dtype, np.integer):
            raise ValueError("a deterministic policy must be an integer array of actions")
        if policy.min() < 0 or policy.max() >= model.n_actions:
            raise ValueError("policy contains an action outside 0..3")
        return np.eye(model.n_actions)[policy]
    if policy.shape == (model.n_states, model.n_actions):
        return policy
    raise ValueError(f"a policy must have shape ({model.n_states},) or "
                     f"({model.n_states}, {model.n_actions}), got {policy.shape}")


def _values(model, gamma, policy):
    pm = _policy_matrix(model, policy)
    P_pi = np.einsum("sa,sat->st", pm, model._P)
    R_pi = (pm * model.R).sum(1)
    return np.linalg.solve(np.eye(model.n_states) - gamma * P_pi, R_pi)


_OPTIMAL: dict = {}


def optimal(model: DPModel, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """The optimal values V* and an optimal policy (computed once per world and gamma)."""
    key = (model.cfg, model.n_states, gamma)
    if key not in _OPTIMAL:
        if gamma >= 1.0:
            raise ValueError("optimal() needs gamma < 1")
        S = model.n_states
        pi = np.zeros(S, dtype=np.int64)
        for _ in range(1000):
            V = _values(model, gamma, pi)
            Q = model.R + gamma * (model._P @ V)
            better = Q.max(1) > Q[np.arange(S), pi] + 1e-12
            if not better.any():
                break
            pi = np.where(better, Q.argmax(1), pi)
        residual = np.abs(Q.max(1) - V).max()
        assert residual < 1e-8, f"reference solver did not converge (residual {residual:.1e})"
        _OPTIMAL[key] = (V, pi)
    V, pi = _OPTIMAL[key]
    return V.copy(), pi.copy()


def gap(model: DPModel, gamma: float, policy) -> float:
    """max_s (V*(s) - V_pi(s)): how far the policy is from optimal in its worst state.

    `policy` is an integer array of actions (deterministic) or an (S, A) array of
    probabilities. The policy is eps-optimal if gap <= EPS."""
    V_star, _ = optimal(model, gamma)
    return float((V_star - _values(model, gamma, policy)).max())


def measure(solver: Callable, world="maze", gamma: float | None = None,
            eps: float = EPS, hide_P: bool = True, **overrides) -> dict:
    """Run `solver(model, gamma, eps) -> policy` on a fresh model and report its cost.

    world: a name, a PragmanConfig (with gamma=...), or a DPModel (its counter is reset).
    Returns a dict with backups, gap, ok (gap <= eps), and seconds."""
    if isinstance(world, DPModel):
        model = world
        if gamma is None:
            raise ValueError("pass gamma=... together with a model")
    else:
        model, gamma = load(world, gamma, **overrides)
    model.backups, model._hide_P = 0, hide_P
    t = time.perf_counter()
    try:
        policy = solver(model, gamma, eps)
    finally:
        model._hide_P = False
    seconds = time.perf_counter() - t
    if isinstance(policy, tuple):
        raise TypeError("the solver must return just the policy (an integer array), not a tuple")
    g = gap(model, gamma, np.asarray(policy))
    return {"backups": model.backups, "gap": g, "ok": g <= eps, "seconds": seconds}


def _classify(model):
    kinds = np.full(model.n_states, "", dtype=object)
    for i, st in enumerate(model.states):
        if not st.terminated:
            continue
        if st.agent in st.ghosts:
            kinds[i] = "caught"
        elif st.agent in model.cfg.layout.goal_cells:
            kinds[i] = "goal"
        else:
            kinds[i] = "cleared"
    return kinds


def outcomes(model: DPModel, policy, start=None) -> dict:
    """How episodes end under `policy`, from the start distribution (or from state `start`):
    the probability of reaching the goal, of eating the last pellet, of being caught by a
    ghost, and of never ending. No discounting: this is about what *happens*."""
    pm = _policy_matrix(model, policy)
    P_pi = np.einsum("sa,sat->st", pm, model._P)
    M = P_pi.copy()
    for _ in range(16):                     # M = P_pi ** 65536: long enough to settle
        M = M @ M
    d = model.rho0 if start is None else np.eye(model.n_states)[start]
    final = d @ M
    kinds = _classify(model)
    out = {k: float(final[kinds == k].sum()) for k in ("goal", "cleared", "caught")}
    out["never"] = float(final[kinds == ""].sum())
    return {k: v for k, v in out.items() if v > 1e-9 or k == "caught"}


# --- plotting -------------------------------------------------------------------------------

def _slice(model, like):
    """Indices of the states that differ from `like` only in the agent's cell."""
    ref = model.states[like]
    return [i for i, st in enumerate(model.states)
            if not st.terminated and st.ghosts == ref.ghosts and st.pellets == ref.pellets]


def plot_world(model: DPModel, V=None, policy=None, like: int | None = None, ax=None,
               title: str | None = None, fmt: str = "{:.2f}"):
    """Values and/or a deterministic policy on the map.

    With ghosts or pellets, a state is more than the agent's cell, so we show one *slice*:
    the states that look like state `like` (default: the start state) except for where the
    agent is -- e.g. the ghost on its spawn cell and all pellets still there."""
    if like is None:
        like = int(np.argmax(model.rho0))
    idx = _slice(model, like)
    states = [model.states[i] for i in idx]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    if title is None:
        title = model.name + ("" if not model.states[like].ghosts else
                              f"  (ghost at {model.states[like].ghosts[0]})")
    vals = None if V is None else np.asarray(V)[idx]
    if policy is not None:
        plot_policy(model.cfg, np.asarray(policy)[idx], states=states, ax=ax, title=title,
                    values=None if vals is None else dict(zip(states, vals)))
    elif vals is not None:
        plot_values(model.cfg, vals, states=states, ax=ax, title=title, fmt=fmt)
    for g in model.states[like].ghosts:
        ax.text(g[1] + 0.5, g[0] + 0.5, "G", ha="center", va="center", fontsize=16, color="red",
                weight="bold")
    return ax


# --- Task 1: checking your implementation -------------------------------------------------

# Sutton & Barto, Figure 4.1: values of the uniformly random policy on the 4x4 gridworld.
_SB_VALUES = np.array([[0, -14, -20, -22], [-14, -18, -20, -20],
                       [-20, -20, -18, -14], [-22, -20, -14, 0]], dtype=float)


def _ok(cond, msg_ok, msg_bad):
    print(("  OK    " + msg_ok) if cond else ("  FAIL  " + msg_bad))
    return bool(cond)


def _todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), True
    except NotImplementedError:
        print(f"  TODO  {fn.__name__} is not implemented yet")
        return None, False


def check() -> None:
    import dp
    rng = np.random.default_rng(0)
    print("Checking dp.py (Task 1). Each part is checked once its function is implemented.\n")

    print("1. bellman_q")
    model, gamma = load("maze")
    V = rng.normal(size=model.n_states)
    Q, done = _todo(dp.bellman_q, model, V, gamma)
    if done:
        _ok(np.shape(Q) == (model.n_states, model.n_actions) and
            np.allclose(Q, model.R + gamma * model._P @ V),
            "Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s')",
            "wrong values: Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s'), shape (S, A)")

    print("\n2. policy_evaluation (iterative) on the lecture's 4x4 gridworld, random policy, gamma = 1")
    model, gamma = load("gridworld")
    uniform = np.full((model.n_states, model.n_actions), 0.25)
    out, done = _todo(dp.policy_evaluation, model, uniform, gamma, tol=1e-6)
    if done:
        V, sweeps = out
        grid = np.zeros((4, 4))
        for i, st in enumerate(model.states):
            if not st.terminated:
                grid[st.agent] = V[i]
        _ok(np.allclose(grid, _SB_VALUES, atol=0.01),
            f"reproduces Sutton & Barto's Figure 4.1 ({sweeps} sweeps):\n{np.round(grid, 1)}",
            f"expected\n{_SB_VALUES}\ngot\n{np.round(grid, 2)}")
        m2, g2 = load("maze")
        pi = rng.integers(0, 4, m2.n_states)
        out2, done2 = _todo(dp.policy_evaluation, m2, pi, g2, tol=1e-10)
        if done2:
            _ok(np.allclose(out2[0], _values(m2, g2, pi), atol=1e-6),
                "a deterministic policy (array of actions) on the maze also works",
                "wrong values for a deterministic policy given as an array of actions")

    print("\n3. policy_evaluation_exact")
    model, gamma = load("maze")
    pi = rng.integers(0, 4, model.n_states)
    V, done = _todo(dp.policy_evaluation_exact, model, pi, gamma)
    if done:
        _ok(np.allclose(V, _values(model, gamma, pi), atol=1e-8), "matches V_pi",
            "does not match V_pi: solve (I - gamma P_pi) V = R_pi")

    print("\n4. value_iteration")
    for name in ("maze", "ghost_lair"):
        model, gamma = load(name)
        out, done = _todo(dp.value_iteration, model, gamma, EPS)
        if not done:
            break
        V, policy, deltas = out
        g = gap(model, gamma, policy)
        _ok(g <= EPS, f"{name}: eps-optimal policy after {len(deltas)} sweeps, "
            f"{model.backups} backups (gap {g:.1e})",
            f"{name}: policy is not eps-optimal (gap {g:.2e} > {EPS})")

    print("\n5. policy_iteration")
    for name in ("maze", "ghost_lair"):
        model, gamma = load(name)
        out, done = _todo(dp.policy_iteration, model, gamma)
        if not done:
            break
        V, policy, iterations = out
        g = gap(model, gamma, policy)
        _ok(g <= 1e-8, f"{name}: optimal policy after {iterations} iterations",
            f"{name}: policy is not optimal (gap {g:.2e})")

    print("\n6. solve (Task 3), with model.P hidden, against plain value iteration")
    for name in ("maze", "ghost_lair", "pellets"):
        model, gamma = load(name)
        ref = _reference_backups(model, gamma)
        try:
            r = measure(dp.solve, model, gamma)
        except NotImplementedError:
            print("  TODO  solve")
            break
        _ok(r["ok"], f"{name}: eps-optimal, {r['backups']} backups = "
            f"{r['backups'] / ref:.2f} x plain VI ({r['seconds']:.1f} s)",
            f"{name}: NOT eps-optimal (gap {r['gap']:.2e}); {r['backups']} backups")


# --- Task 3: the final evaluation -----------------------------------------------------------

def _reference_backups(model, gamma, eps=EPS):
    """Backups used by plain value iteration from V = 0 with the lecture's stopping rule."""
    V = np.zeros(model.n_states)
    sweeps = 0
    while True:
        V_new = (model.R + gamma * model._P @ V).max(1)
        sweeps += 1
        if np.abs(V_new - V).max() < eps * (1 - gamma) / (2 * gamma):
            return sweeps * model.n_states * model.n_actions
        V = V_new


def _geomean(xs):
    return float(np.exp(np.mean(np.log(xs))))


def final_evaluation() -> None:
    import dp
    history = json.loads(FINAL_CACHE.read_text()) if FINAL_CACHE.exists() else []
    if history:
        last = history[-1]
        print(f"You already ran the final evaluation on {last['time']} ({len(history)} run(s) so far). "
              "Its results were:\n")
        print(last["report"])
        print("By default you are NOT supposed to run it again: the point is to see how choices made\n"
              "*before* seeing these results hold up. Better results from a re-run will not affect\n"
              "your grade in any way.")
        try:
            answer = input('Press Enter to abort. Type "OVERRIDE" to proceed: ')
        except EOFError:
            answer = ""
        if answer.strip() != "OVERRIDE":
            print("Aborted.")
            return

    print(f"Final evaluation of dp.solve on {len(_FINAL_WORLDS)} worlds -- the three you know and "
          f"{len(_FINAL_WORLDS) - 3} new ones.\n")
    results = {}
    for name, factory, overrides, gamma in _FINAL_WORLDS:
        print(f"  {name} ...", flush=True)
        model = DPModel(factory(**overrides), name=name)
        ref = _reference_backups(model, gamma)
        try:
            r = measure(dp.solve, model, gamma)
            error = None
        except Exception as e:                                # a crash fails this world only
            r, error = {"backups": None, "gap": None, "ok": False, "seconds": None}, \
                f"{type(e).__name__}: {e}"
        ratio = r["backups"] / ref if r["ok"] else FAIL_RATIO
        results[name] = {**r, "gamma": gamma, "n_states": model.n_states, "vi_backups": ref,
                         "ratio": ratio, "error": error}
    score = _geomean([r["ratio"] for r in results.values()])
    n_ok = sum(r["ok"] for r in results.values())

    lines = [f"{'world':16s}{'S':>6s}{'gamma':>7s}{'VI backups':>12s}{'yours':>11s}"
             f"{'/ VI':>7s}{'gap':>10s}{'time':>7s}"]
    for name, r in results.items():
        if r["error"]:
            lines.append(f"{name:16s}{r['n_states']:6d}{r['gamma']:7.3f}{r['vi_backups']:12d}"
                         f"   CRASHED: {r['error'][:60]}")
            continue
        flag = "" if r["ok"] else "  NOT eps-optimal"
        lines.append(f"{name:16s}{r['n_states']:6d}{r['gamma']:7.3f}{r['vi_backups']:12d}"
                     f"{r['backups']:11d}{r['ratio']:7.2f}{r['gap']:10.1e}{r['seconds']:6.1f}s{flag}")
    lines += ["", f"Score: {score:.3f}  (geometric mean of 'yours / VI'; lower is better; "
                  f"{n_ok}/{len(results)} worlds eps-optimal).",
              f"A world where the policy is not eps-optimal (eps = {EPS}) counts as {FAIL_RATIO:.0f}x VI."]
    report = "\n".join(lines)
    print("\n" + report)
    print("\nWas it what you expected? On which world did you gain the most against VI, and why\n"
          "there? Did any of the new worlds catch you out?")

    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "eps": EPS,
             "results": results, "score": score, "n_ok": n_ok, "report": report,
             "override": bool(history)}
    history.append(entry)
    FINAL_CACHE.parent.mkdir(exist_ok=True)
    FINAL_CACHE.write_text(json.dumps(history, indent=1))
    print()
    leaderboard.submit(WEEK_DIR, "final", {k: v for k, v in entry.items() if k != "report"},
                       files=[WEEK_DIR / f for f in SUBMITTED_FILES])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check dp.py (default), or run the final evaluation.")
    ap.add_argument("--final", action="store_true", help="final evaluation of dp.solve (Task 3)")
    args = ap.parse_args()
    if args.final:
        final_evaluation()
    else:
        check()
