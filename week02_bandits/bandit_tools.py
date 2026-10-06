"""Tools for running, evaluating, tuning and plotting bandit agents (Week 2).

You should not need to edit this file. Typical use, from the repository root:

    uv run python week02_bandits/bandit_tools.py     # sanity-check your agents
    uv run python week02_bandits/bandit_tools.py --final --classes 1 2   # Task 5, run once

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

import argparse
import datetime
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import rlcourse.envs  # noqa: F401  (registers the bandit environments)

TUNE_SEEDS = range(0, 200)
TEST_SEEDS = range(10_000, 11_000)
HORIZON = 1000
PROBLEMS = (1, 2, 3, 4, 5)
FINAL_SEEDS = range(20_000, 20_500)
# Mean regret on FINAL_SEEDS of each basic algorithm with its hyperparameter tuned
# separately for each problem class (by the instructors).
TUNED_REGRET = {
    "EpsilonGreedy":     {1: 89.7, 2: 118.5, 3: 70.5, 4: 141.6, 5: 85.5},
    "ExploreThenCommit": {1: 182.8, 2: 176.9, 3: 117.9, 4: 222.6, 5: 80.8},
    "Boltzmann":         {1: 84.7, 2: 95.3, 3: 66.8, 4: 130.6, 5: 100.9},
    "UCB":               {1: 67.2, 2: 67.9, 3: 57.5, 4: 132.5, 5: 79.0},
}
REFERENCE_REGRET = TUNED_REGRET["UCB"]      # the yardstick for the overall score
FINAL_CACHE = Path(__file__).parent / "results" / "final_evaluation.json"


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


# --- final evaluation (Task 5) -------------------------------------------------------

def score(agent, problems: Sequence[int] = PROBLEMS, seeds: Iterable[int] = FINAL_SEEDS) -> dict:
    """Regret on each problem, its ratio to REFERENCE_REGRET, and the geometric mean of the ratios.

    The geometric mean treats a 2x improvement on any class the same, whatever that
    class's regret scale; 1.0 means 'as good as a UCB tuned separately for each class'."""
    seeds = list(seeds)
    regret = {p: evaluate(agent, p, seeds).mean for p in problems}
    ratio = {p: regret[p] / REFERENCE_REGRET[p] for p in problems}
    return {"regret": regret, "ratio": ratio,
            "score": float(np.exp(np.mean(np.log(list(ratio.values())))))}


def _geomean(xs):
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def final_evaluation(our_problems: Sequence[int]) -> None:
    import agents as A
    final = A.final_agents()
    if not final:
        print("final_agents() in agents.py is still empty: fill in your final settings first.")
        return

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

    unseen = [p for p in PROBLEMS if p not in our_problems]
    print(f"Final evaluation of {', '.join(final)}.\n"
          f"Each agent now runs on ALL FIVE problem classes -- including the {len(unseen)} you have not\n"
          f"seen ({', '.join(map(str, unseen))}) -- with {len(FINAL_SEEDS)} new runs per class. "
          f"This takes a minute or two.\n")
    results = {}
    for name, agent in final.items():
        print(f"  {name} ...", flush=True)
        results[name] = score(agent)

    def table(title, ratios, note):
        out = [title, ""]
        out.append(f"{'':20s}" + "".join(
            f"{'class ' + str(p) + ('*' if p in our_problems else ''):>10s}" for p in PROBLEMS)
            + f"{'yours*':>9s}{'unseen':>9s}{'ALL':>8s}")
        for name, rat in ratios.items():
            if rat is None:
                out.append(f"{name:20s}  (not one of the four basic algorithms)")
                continue
            out.append(f"{name:20s}" + "".join(f"{rat[p]:10.2f}" for p in PROBLEMS)
                       + f"{_geomean([rat[p] for p in our_problems]):9.2f}"
                       + f"{_geomean([rat[p] for p in unseen]):9.2f}"
                       + f"{_geomean(list(rat.values())):8.2f}")
        return out + [note, ""]

    lines = [f"{'Mean regret':20s}" + "".join(f"{'class ' + str(p):>10s}" for p in PROBLEMS)]
    for name, r in results.items():
        lines.append(f"{name:20s}" + "".join(f"{r['regret'][p]:10.1f}" for p in PROBLEMS))
    lines.append("")
    lines += table("1) OVERALL SCORE: regret / regret of a UCB tuned separately for each class",
                   {n: r["ratio"] for n, r in results.items()},
                   "   Lower is better; 1.00 = as good as that UCB. 'ALL' is the overall score.")

    def basic(agent):                       # the basic algorithm this agent is (derived from)
        return next((c.__name__ for c in type(agent).__mro__ if c.__name__ in TUNED_REGRET), None)

    transfer = {}
    for name, agent in final.items():
        b = basic(agent)
        transfer[name] = (None if b is None else
                          {p: results[name]["regret"][p] / TUNED_REGRET[b][p] for p in PROBLEMS})
    lines += table("2) TRANSFER: regret / regret of the same algorithm tuned separately for each class",
                   transfer,
                   "   How much your single setting loses against the best setting for each class.\n"
                   "   'yours*' and 'unseen' are geometric means over your classes and the other ones.")
    report = "\n".join(lines)
    print("\n" + report)
    print("Compare 'yours*' with 'unseen' in table 2. Settings tuned on a few problems pick up their\n"
          "particular quirks (here: reward scale, number of arms, noise) and can transfer worse than\n"
          "expected. The same happens, at a much larger scale, when AI systems are optimized against\n"
          "a benchmark or a reward model and then used in situations they were not tuned for.\n"
          "Was your prediction right? What would you change, knowing this?")

    history.append({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "our_problems": list(our_problems),
                    "agents": {n: repr(a) for n, a in final.items()},
                    "results": results, "report": report})
    FINAL_CACHE.parent.mkdir(exist_ok=True)
    FINAL_CACHE.write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check your agents (default), or run the final evaluation.")
    ap.add_argument("--final", action="store_true", help="final evaluation of final_agents() (Task 5)")
    ap.add_argument("--classes", type=int, nargs="+", choices=PROBLEMS, default=[],
                    help="the problem classes your group tuned on, e.g. --classes 1 2")
    args = ap.parse_args()
    if args.final:
        if len(args.classes) != 2:
            ap.error("please give your group's two problem classes, e.g. --final --classes 1 2")
        final_evaluation(args.classes)
    else:
        check_agents()
