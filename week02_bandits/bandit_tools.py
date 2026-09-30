"""Tools for running, evaluating, tuning and plotting bandit agents (Week 2).

You should not need to edit this file. Typical use, from the repository root:

    uv run python week02_bandits/bandit_tools.py     # sanity-check your agents

and then either write your experiments in a script of your own inside
week02_bandits/ (run it with `uv run python week02_bandits/my_script.py`), or
work interactively:

    uv run ipython
    >>> %cd week02_bandits
    >>> from bandit_tools import *
    >>> from agents import *
    >>> r = evaluate(EpsilonGreedy(0.1), problem=2)
    >>> r
    >>> tuned = tune(lambda eps: EpsilonGreedy(eps), [0.01, 0.03, 0.1, 0.3], problems=[2, 4])
    >>> plot_tuning(tuned, xlabel="epsilon")

We measure agents by the total regret after T steps (see the Week 2 lecture),

    L_T = sum_t Delta_{A_t} = sum_a N_T(a) Delta_a,

averaged over many runs, each on a freshly drawn instance of the problem class.
This is the *pseudo*-regret: it uses the true gaps Delta_a rather than the
rewards actually received, which removes most of the reward noise.

Seeds: tune on TUNE_SEEDS. When you have picked your hyperparameters, check them
ONCE on TEST_SEEDS -- the best of many tries on the same seeds is optimistically
biased (the "winner's curse").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import rlcourse.envs  # noqa: F401  (registers the bandit environments)

TUNE_SEEDS = range(0, 200)
TEST_SEEDS = range(10_000, 11_000)
HORIZON = 1000
PROBLEMS = (1, 2, 3, 4, 5)


def make_env(problem: int, horizon: int = HORIZON) -> gym.Env:
    return gym.make(f"rlcourse/Bandit{problem}-v0", horizon=horizon)


def run_episode(env: gym.Env, agent, seed: int) -> dict:
    """One learning run: reset env and agent, then `horizon` pulls."""
    env.reset(seed=seed)
    base = env.unwrapped
    agent.reset(base.action_space.n, base.horizon, np.random.default_rng([seed, 2026]))
    actions = np.empty(base.horizon, dtype=np.int64)
    rewards = np.empty(base.horizon)
    truncated, t = False, 0
    while not truncated:
        a = agent.select_action()
        _, r, _, truncated, _ = env.step(a)
        agent.update(a, r)
        actions[t], rewards[t] = a, r
        t += 1
    return {"actions": actions, "rewards": rewards,
            "regret": np.cumsum(base.gaps()[actions])}   # L_t for t = 1..T


@dataclass
class Result:
    """Regret statistics of one agent on one problem class."""
    label: str
    problem: int
    final: np.ndarray                  # L_T of every run
    curve: np.ndarray                  # mean L_t over runs, t = 1..T
    seconds: float
    seeds: list = field(repr=False)

    @property
    def mean(self): return float(self.final.mean())
    @property
    def se(self): return float(self.final.std(ddof=1) / np.sqrt(len(self.final)))
    @property
    def median(self): return float(np.median(self.final))
    @property
    def p90(self): return float(np.percentile(self.final, 90))

    def __repr__(self):
        return (f"{self.label} on problem {self.problem}: regret {self.mean:.1f} ± {self.se:.1f} "
                f"(median {self.median:.1f}, 90th percentile {self.p90:.1f}; "
                f"{len(self.final)} runs, {self.seconds:.1f} s)")


def evaluate(agent, problem: int, seeds: Iterable[int] = TUNE_SEEDS,
             horizon: int = HORIZON, label: str | None = None) -> Result:
    """Run `agent` once per seed on problem class `problem`."""
    env = make_env(problem, horizon)
    seeds = list(seeds)
    start = time.perf_counter()
    runs = [run_episode(env, agent, s)["regret"] for s in seeds]
    return Result(label or repr(agent), problem, np.array([r[-1] for r in runs]),
                  np.mean(runs, axis=0), time.perf_counter() - start, seeds)


def paired_difference(a: Result, b: Result) -> tuple[float, float]:
    """Mean and standard error of (regret of a) - (regret of b), run by run.

    Both results must use the same seeds. Because both agents then face the same
    instances and the same luck, this is much more precise than comparing the
    two means with their separate standard errors."""
    assert a.seeds == b.seeds and a.problem == b.problem, "need the same problem and seeds"
    d = a.final - b.final
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def tune(make_agent: Callable, values: Sequence, problems: Sequence[int] = PROBLEMS,
         seeds: Iterable[int] = TUNE_SEEDS, horizon: int = HORIZON) -> dict:
    """Evaluate `make_agent(v)` for every v in `values` on every problem.

    Returns {problem: [Result for each value]} and prints a table as it goes.
    Rough cost: 1-3 s per value and problem with 200 seeds and T = 1000."""
    seeds = list(seeds)
    out = {}
    for p in problems:
        print(f"problem {p}:")
        out[p] = []
        for v in values:
            r = evaluate(make_agent(v), p, seeds, horizon)
            r.value = v
            out[p].append(r)
            print(f"   {v!s:>10}: {r.mean:8.1f} ± {r.se:5.1f}   ({r.seconds:.1f} s)", flush=True)
        best = min(out[p], key=lambda r: r.mean)
        print(f"   best: {best.value} -> {best.mean:.1f} ± {best.se:.1f}")
    return out


def compare(agents: dict, problems: Sequence[int] = PROBLEMS,
            seeds: Iterable[int] = TUNE_SEEDS, horizon: int = HORIZON) -> dict:
    """Evaluate several agents {label: agent} on several problems; print a table."""
    seeds = list(seeds)
    res = {p: {name: evaluate(ag, p, seeds, horizon, label=name) for name, ag in agents.items()}
           for p in problems}
    width = max(len(n) for n in agents) + 2
    print(" " * width + "".join(f"{'problem ' + str(p):>18}" for p in problems))
    for name in agents:
        print(f"{name:<{width}}" + "".join(
            f"{res[p][name].mean:11.1f} ± {res[p][name].se:4.1f}" for p in problems))
    return res


# --- plotting -----------------------------------------------------------------

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]


def plot_tuning(tuned: dict, xlabel: str = "hyperparameter", logx: bool = True, ax=None):
    """Mean regret (± 1 SE) against the hyperparameter, one line per problem class."""
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    for i, (p, results) in enumerate(tuned.items()):
        x = [r.value for r in results]
        m = np.array([r.mean for r in results])
        se = np.array([r.se for r in results])
        ax.plot(x, m, "-o", ms=4, color=COLOURS[i % 8], label=f"problem {p}")
        ax.fill_between(x, m - se, m + se, color=COLOURS[i % 8], alpha=0.2, lw=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"mean regret $L_T$ (T = {len(results[0].curve)})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    return ax


def plot_regret(results: Sequence[Result], logx: bool = False, ax=None):
    """Mean regret curves L_t of several results (same problem, ideally)."""
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    for i, r in enumerate(results):
        ax.plot(np.arange(1, len(r.curve) + 1), r.curve, color=COLOURS[i % 8], lw=2,
                label=f"{r.label} (problem {r.problem})")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel("mean regret $L_t$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    return ax


def plot_histogram(results: Sequence[Result], bins: int = 40, ax=None):
    """Distribution of the final regret L_T over runs: the mean is not the whole story."""
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    for i, r in enumerate(results):
        ax.hist(r.final, bins=bins, alpha=0.5, color=COLOURS[i % 8], label=r.label)
    ax.set_xlabel("final regret $L_T$")
    ax.set_ylabel("number of runs")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show(block=False)
    return ax


# --- sanity check ---------------------------------------------------------------

def check_agents() -> None:
    """Run each agent in agents.py briefly and check it behaves like an agent."""
    import agents as A
    candidates = {"Greedy": A.Greedy(), "EpsilonGreedy": A.EpsilonGreedy(),
                  "ExploreThenCommit": A.ExploreThenCommit(), "Boltzmann": A.Boltzmann(),
                  "UCB": A.UCB()}
    random_regret = evaluate(A.RandomAgent(), 2, seeds=range(30)).mean
    for name, agent in candidates.items():
        try:
            r = evaluate(agent, 2, seeds=range(30))
        except NotImplementedError:
            print(f"[ -- ] {name}: not implemented yet")
            continue
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            continue
        verdict = "OK" if r.mean < 0.5 * random_regret else "??"
        print(f"[ {verdict} ] {name}: regret {r.mean:.1f} on problem 2 "
              f"(random agent: {random_regret:.1f})")
    print("'??' means: runs, but hardly better than random -- worth a second look.")


if __name__ == "__main__":
    check_agents()
