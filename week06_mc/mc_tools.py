"""Week 6 tools: everything from Week 5, plus what Monte Carlo needs, and the checker.

No need to edit this file. The files you edit are `mc.py` (Monte Carlo prediction and control),
`importance.py` (off-policy evaluation), and, for group A2, `nstep.py`. Your `td.py` and
`control.py` from Week 5 come along (see the README). Typical use, from a script of your own in
this directory -- import mc_tools first, it makes the Week 5 tools available:

    from mc_tools import *
    from mc import MC, MCControl
    from control import SARSA, QLearning

    res = run_control({"MC": MCControl, "SARSA": partial(SARSA, alpha=0.3)}, load("maze"),
                      n_episodes=500, runs=5, epsilon=0.1)
    plot_control(res, "gap"); plt.show()

Run this file itself to check your implementation (Task 1):

    uv run python week06_mc/mc_tools.py

New this week (the rest is as in Week 5):

    run_control(...)      as in Week 5, but it tells each agent when an episode is over (a
                          Monte Carlo agent needs to know), it also records how many steps
                          every episode took, and start="exploring" gives exploring starts
    plot_steps(res)       episode length against episodes
    start_value(world)    the true value of the world's policy at the start
    q_true(world)         the true q_pi of the world's policy, over the true states
    q_error(world, Q)     how far Q is from q_pi, weighted by how often pi uses each pair
    count_loops(world, Q) how many states the greedy policy of Q sends round a loop of two
"""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_WEEK5 = Path(__file__).resolve().parent.parent / "week05_td_control"
if str(_WEEK5) not in sys.path:
    sys.path.append(str(_WEEK5))       # for td_tools; your own copies here come first

import td_tools
from td_tools import *
from td_tools import _run_part, _smooth, _todo, _ok

__all__ = [*td_tools.__all__, "plot_steps", "start_value", "q_true", "q_error", "count_loops"]
td_tools._FILES = ("mc.py", "importance.py", "nstep.py", "td.py", "control.py")   # for messages


def start_value(world: World) -> float:
    """The true value of the world's policy, v_pi at the start of an episode."""
    return float(world.rho0 @ world.V_true_states)


def q_true(world: World) -> np.ndarray:
    """The true action values q_pi(s, a) of the world's policy, over the true states."""
    return world.R + world.gamma * world.P @ world.V_true_states


def q_error(world: World, Q: np.ndarray) -> float:
    """How far a table Q is from q_pi of the world's policy: the root-mean-square error over
    (s, a), weighted by how often the policy visits s and takes a there."""
    w = world.mu[:, None] * world.policy
    return float(np.sqrt((w * (np.asarray(Q)[world.obs] - q_true(world)) ** 2).sum() / w.sum()))


def count_loops(world: World, Q: np.ndarray) -> int:
    """How many states the greedy policy of Q sends round a loop of two: if nothing slips,
    it walks from s to a neighbour, and from there straight back."""
    n = world.n_true_states
    pick = greedy_policy(Q)[world.obs]
    nxt = world.P[np.arange(n), pick].argmax(axis=1)
    return int(((nxt[nxt] == np.arange(n)) & (nxt != np.arange(n)) & ~world.terminal).sum())


def run_control(agents: dict, world: World, n_episodes: int = 500, runs: int = 10,
                epsilon=0.1, seed: int = 0, start: str = "fixed", gap_every: int = 50,
                alpha=None) -> dict:
    """Let each agent act epsilon-greedily in the world and learn online, for n_episodes
    episodes, in several runs (seeds). As in Week 5, except:

    - if an agent has a method end_episode(), it is called when an episode is over, whether
      it terminated or hit the time limit;
    - start="exploring" gives exploring starts: a random state and a random first action;
      start="random" is a random state only, and "fixed" (default) the world's own start;
    - the result also has "steps": (runs, n_episodes), the length of every episode;
    - gap_every is 50 by default (computing the gap solves a linear system).

    Returns {name: {"return", "gap", "steps", "first_action", "Q"}}; "Q" and the agent
    ("agent") are those of the last run."""
    eps_of = epsilon if callable(epsilon) else (lambda k: epsilon)
    S, A, g = world.n_states, world.n_actions, world.gamma
    obs = world.obs
    out = {name: {"return": np.zeros((runs, n_episodes)),
                  "gap": np.full((runs, n_episodes), np.nan),
                  "steps": np.zeros((runs, n_episodes), dtype=int),
                  "first_action": np.zeros((runs, n_episodes), dtype=int),
                  "Q": None, "agent": None}
           for name in agents}
    for r in range(runs):
        for name, factory in agents.items():
            agent = factory(S, A, g) if alpha is None else factory(S, A, g, alpha=alpha)
            end_episode = getattr(agent, "end_episode", lambda: None)
            rng = np.random.default_rng(seed + r)
            for k in range(n_episodes):
                eps = eps_of(k)
                s = world.reset(rng, "random" if start == "exploring" else start)
                if start == "exploring":
                    a = int(rng.integers(A))
                else:
                    a = epsilon_greedy(agent.Q[obs[s]], eps, rng)
                out[name]["first_action"][r, k] = a
                G, t = 0.0, 0
                for t in range(world.max_steps):
                    s2, rew, done = world.step(s, a, rng)
                    G += rew
                    a2 = epsilon_greedy(agent.Q[obs[s2]], eps, rng)
                    agent.update(obs[s], a, rew, obs[s2], a2, done, eps)
                    s, a = s2, a2
                    if done:
                        break
                end_episode()
                out[name]["return"][r, k] = G
                out[name]["steps"][r, k] = t + 1
                if (k + 1) % gap_every == 0 or k == n_episodes - 1:
                    out[name]["gap"][r, k] = gap(world, greedy_policy(agent.Q)[obs])
            out[name]["Q"], out[name]["agent"] = agent.Q, agent
    return out


def plot_steps(results: dict, ax=None, smooth: int = 20, title: str | None = None):
    """Mean over runs of the length of each episode (smoothed), on a log scale."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        y = _smooth(res["steps"].astype(float), smooth).mean(0)
        ax.plot(np.arange(1, len(y) + 1), y, label=name)
    ax.set_xlabel("episodes")
    ax.set_ylabel(f"steps per episode (moving average, {smooth})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


# --- the checker ------------------------------------------------------------------------------

_HAND = Episode(np.array([0, 1, 0, 2]), np.array([0, 0, 1]),
                np.array([1.0, 2.0, 3.0]), True)
_RW_TRUE = np.array([1, 2, 3, 4, 5]) / 6
_PI = np.array([[0.5, 0.5], [1.0, 0.0], [0.5, 0.5]])
_B = np.array([[0.25, 0.75], [0.5, 0.5], [0.5, 0.5]])


def _feed(agent, steps, done=True):
    """Play a hand-made episode [(s, a, r), ...] to an agent."""
    for i, (s, a, r) in enumerate(steps):
        last = i == len(steps) - 1
        n = steps[i + 1] if not last else (2, 0, 0.0)
        agent.update(s, a, r, n[0], n[1], last and done, 0.1)
    agent.end_episode()


def check() -> None:
    import control
    import importance
    import mc
    import nstep
    import td
    print("Checking mc.py and importance.py (Task 1), and nstep.py (group A2).\n")
    rw = load("random_walk")

    def part0():
        here = Path(__file__).resolve().parent
        for module in (td, control):
            file = Path(module.__file__).resolve()
            print(f"  {'OK   ' if file.parent == here else 'NOTE '} {file.name}: using {file}"
                  + ("" if file.parent == here else ": copy it here (Task 0)"))
        ag = control.SARSA(2, 2, 0.5, alpha=0.5)
        ag.Q[:] = 1.0
        _, done = _todo(lambda: ag.update(0, 0, 2.0, 1, 0, False, 0.1))
        if done:
            _ok(np.isclose(ag.Q[0, 0], 1.75), "SARSA still does what it did in Week 5",
                f"SARSA: expected 1.75, got {ag.Q[0, 0]}")
        est = td.TD0(3, 1.0, alpha=0.5)
        _, done = _todo(lambda: est.update(_HAND))
        if done:
            _ok(np.allclose(est.V, [1.75, 1.25, 0.0]), "TD(0) too",
                f"TD(0): expected [1.75, 1.25, 0], got {est.V}")

    def part1():
        G, done = _todo(lambda: mc.discounted_returns(np.array([1.0, 2.0, 3.0]), 0.5))
        if done:
            _ok(np.allclose(G, [2.75, 3.5, 3.0]), "rewards 1, 2, 3, gamma 0.5: G = [2.75, 3.5, 3]",
                f"rewards 1, 2, 3, gamma 0.5: expected [2.75, 3.5, 3], got {G}")

    def part2():
        est = mc.MC(3, 1.0, alpha=0.5)
        _, done = _todo(lambda: est.update(_HAND))
        if not done:
            return
        _ok(np.allclose(est.V, [3.0, 2.5, 0.0]),
            "every-visit, alpha = 0.5, on 0 -1-> 1 -2-> 0 -3-> end: V = [3, 2.5, 0]",
            f"every-visit, alpha = 0.5: expected [3, 2.5, 0], got {est.V} "
            f"([3.75, 2.5, 0] means you went backwards in time)")
        est = mc.MC(3, 1.0, first_visit=True)
        est.update(_HAND)
        _ok(np.allclose(est.V, [6.0, 5.0, 0.0]), "first-visit, 1/n: V = [6, 5, 0]",
            f"first-visit: only G_0 = 6 counts for state 0; expected [6, 5, 0], got {est.V}")
        est = mc.MC(rw.n_states, 1.0)
        for ep in rw.episodes(3000, seed=0):
            est.update(ep)
        _ok(np.abs(est.V[:5] - _RW_TRUE).max() < 0.05,
            f"random walk, 3000 episodes: V = {np.round(est.V[:5], 3)} (true: 1/6, ..., 5/6)",
            f"random walk: V = {np.round(est.V[:5], 3)}, expected about {np.round(_RW_TRUE, 3)}")

    def part3():
        steps = [(0, 0, 1.0), (1, 1, 2.0), (0, 0, 3.0)]
        ag = mc.MCControl(3, 2, 1.0)
        _, done = _todo(lambda: _feed(ag, steps))
        if not done:
            return
        _ok(ag.Q[0, 0] == 6 and ag.Q[1, 1] == 5 and ag.N[0, 0] == 1,
            "first-visit, 1/n: Q(0, 0) = 6, Q(1, 1) = 5, and N(0, 0) = 1",
            f"first-visit: expected Q(0, 0) = 6, Q(1, 1) = 5, N(0, 0) = 1, got "
            f"{ag.Q[0, 0]}, {ag.Q[1, 1]}, {ag.N[0, 0]}")
        _feed(ag, [(0, 0, 2.0)])
        _ok(np.isclose(ag.Q[0, 0], 4.0),
            "... and a second episode with G = 2 gives the average, Q(0, 0) = 4",
            f"after a second episode with G = 2: expected the average 4, got {ag.Q[0, 0]}")
        ag = mc.MCControl(3, 2, 1.0, alpha=0.5, first_visit=False)
        _feed(ag, steps)
        _ok(np.isclose(ag.Q[0, 0], 3.0) and np.isclose(ag.Q[1, 1], 2.5),
            "every-visit, alpha = 0.5: Q(0, 0) = 3, Q(1, 1) = 2.5",
            f"every-visit, alpha = 0.5: expected Q(0, 0) = 3, Q(1, 1) = 2.5, got "
            f"{ag.Q[0, 0]}, {ag.Q[1, 1]}")
        res = run_control({"MC": mc.MCControl}, load("gridworld"), n_episodes=300, runs=1,
                          epsilon=0.2, gap_every=300)
        g = res["MC"]["gap"][0, -1]
        _ok(g < 0.5, f"gridworld, 300 episodes of eps-greedy MC control: gap {g:.3f}",
            f"gridworld, 300 episodes: the greedy policy should be nearly optimal, gap = {g:.3f}")

    def part4():
        rho, done = _todo(lambda: importance.tail_ratios(_HAND, _PI, _B))
        if not done:
            return
        _ok(np.allclose(rho, [8 / 3, 4 / 3, 2 / 3]),
            "per-step ratios 2, 2, 2/3: tail ratios [8/3, 4/3, 2/3]",
            f"per-step ratios are 2, 2, 2/3: expected [8/3, 4/3, 2/3], got {rho}")
        ep = replace(_HAND, actions=np.array([0, 1, 1]))
        rho = importance.tail_ratios(ep, _PI, _B)
        _ok(np.allclose(rho, [0, 0, 2 / 3]),
            "... and an action pi never takes zeroes everything before it: [0, 0, 2/3]",
            f"pi never takes action 1 in state 1: expected [0, 0, 2/3], got {rho}")

    def part5():
        ess, done = _todo(lambda: importance.effective_sample_size(np.ones(4)))
        if not done:
            return
        f = importance.effective_sample_size
        _ok(np.isclose(ess, 4) and np.isclose(f(np.array([5.0, 0, 0, 0])), 1)
            and np.isclose(f(np.array([3.0, 1.0])), 1.6),
            "ESS: equal weights give n, one weight gives 1, and [3, 1] gives 1.6",
            f"ESS: expected 4, 1, 1.6, got {ess}, {f(np.array([5.0, 0, 0, 0]))}, "
            f"{f(np.array([3.0, 1.0]))}")
        _ok(f(np.zeros(3)) == 0, "... and 0 if all the weights are 0",
            "all the weights are 0: the ESS should be 0, not nan")

    def part6():
        est = importance.OrdinaryIS(3, 1.0, _PI, _B)
        _, done = _todo(lambda: est.update(_HAND))
        if not done:
            return
        _ok(np.allclose(est.V, [16.0, 20 / 3, 0.0]),
            "ordinary IS, one episode: V = rho * G = [16, 20/3, 0], outside the range of "
            "the returns",
            f"ordinary IS on the hand-made episode: expected [16, 6.67, 0], got {est.V}")
        b, pi = load("maze", tired=0.3), load("maze", tired=0.1)
        est = importance.OrdinaryIS(b.n_states, b.gamma, pi.policy, b.policy)
        for ep in b.episodes(4000, seed=0):
            est.update(ep)
        v, t = est.V[b.start], start_value(pi)
        _ok(abs(v - t) < 0.08, f"maze: {v:.3f} from logs of the tired policy, the less tired "
            f"policy's true value is {t:.3f}", f"maze: expected about {t:.3f}, got {v:.3f}")

    def part7():
        est = importance.WeightedIS(3, 1.0, _PI, _B)
        _, done = _todo(lambda: est.update(_HAND))
        if not done:
            return
        _ok(np.allclose(est.V, [6.0, 5.0, 0.0]),
            "weighted IS, one episode: V = G = [6, 5, 0]",
            f"weighted IS after one episode: V = G, expected [6, 5, 0], got {est.V}")
        est = importance.WeightedIS(3, 1.0, _PI, _B)
        est.update(replace(_HAND, states=np.array([1, 0, 0, 2]), actions=np.array([1, 0, 0])))
        _ok(np.isfinite(est.V).all() and est.V[1] == 0,
            "... and a weight of 0 on the first visit leaves V alone (no division by 0)",
            f"a first visit with rho = 0 and C = 0 must leave V as it is, got {est.V}")
        b = load("maze", tired=0.3)
        eps = b.episodes(300, seed=0)
        est = importance.WeightedIS(b.n_states, b.gamma, b.policy, b.policy)
        for ep in eps:
            est.update(ep)
        G = returns_from_start(eps, b.gamma).mean()
        _ok(np.isclose(est.V[b.start], G), "with pi = b it is the plain average of the returns",
            f"with pi = b, V(start) should be the average return {G:.4f}, got {est.V[b.start]:.4f}")
        pi = load("maze", tired=0.1)
        est = importance.WeightedIS(b.n_states, b.gamma, pi.policy, b.policy)
        for ep in b.episodes(4000, seed=0):
            est.update(ep)
        v, t = est.V[b.start], start_value(pi)
        _ok(abs(v - t) < 0.02, f"maze: {v:.3f}, the true value is {t:.3f}",
            f"maze: expected about {t:.3f}, got {v:.3f}")

    def part8():
        rng = np.random.default_rng(0)
        steps = [(int(rng.integers(3)), int(rng.integers(2)), float(rng.normal())) for _ in range(9)]
        ag = nstep.NStepSARSA(3, 2, 0.9, n=1, alpha=0.5)
        _, done = _todo(lambda: _feed(ag, steps))
        if not done:
            return
        ref = control.SARSA(3, 2, 0.9, alpha=0.5)
        _feed(_Ended(ref), steps)
        _ok(np.allclose(ag.Q, ref.Q), "n = 1 is exactly SARSA",
            f"n = 1 should be SARSA: {ag.Q.ravel()} against {ref.Q.ravel()}")
        ag = nstep.NStepSARSA(3, 2, 0.9, n=10**6, alpha=0.5)
        _feed(ag, steps)
        ref = mc.MCControl(3, 2, 0.9, alpha=0.5, first_visit=False)
        _feed(ref, steps)
        _ok(np.allclose(ag.Q, ref.Q), "a huge n is exactly every-visit MC control",
            f"a huge n should be every-visit MC: {ag.Q.ravel()} against {ref.Q.ravel()}")
        ag = nstep.NStepSARSA(3, 2, 1.0, n=10**6, alpha=1.0, Q0=1.0)
        _feed(ag, [(0, 0, 1.0), (1, 0, 1.0)], done=False)
        _ok(np.isclose(ag.Q[0, 0], 3.0),
            "cut off by the time limit: it bootstraps from the last pair (Q(0, 0) = 3)",
            f"episode cut off, rewards 1, 1, Q0 = 1, gamma 1, alpha 1: expected 3, got {ag.Q[0, 0]}")

    _run_part("0. Your Week 5 code", part0)
    _run_part("1. discounted_returns", part1)
    _run_part("2. MC", part2)
    _run_part("3. MCControl", part3)
    _run_part("4. tail_ratios", part4)
    _run_part("5. effective_sample_size", part5)
    _run_part("6. OrdinaryIS", part6)
    _run_part("7. WeightedIS", part7)
    _run_part("8. NStepSARSA (group A2)", part8)


class _Ended:
    """A Week 5 agent with an end_episode, so that _feed can play to it."""

    def __init__(self, agent):
        self.agent = agent
        self.update = agent.update

    def end_episode(self):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check mc.py, importance.py, and nstep.py.")
    ap.parse_args()
    sys.modules.setdefault("mc_tools", sys.modules["__main__"])
    check()
