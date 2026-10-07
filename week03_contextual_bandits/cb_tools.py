"""Tools for running, evaluating, tuning and plotting contextual bandit agents (Week 3).

You should not need to edit this file. Typical use, from the repository root:

    uv run python week03_contextual_bandits/cb_tools.py           # sanity-check your agents
    uv run python week03_contextual_bandits/cb_tools.py --final   # Task 5, run once

and then either write your experiments in a script of your own inside
week03_contextual_bandits/ (run it with `uv run python week03_contextual_bandits/my_script.py`),
or work interactively:

    uv run ipython
    >>> %cd week03_contextual_bandits
    >>> from cb_tools import *
    >>> from agents import *
    >>> r = evaluate(LinUCB(beta=0.3), problem=1)
    >>> r
    >>> tuned = tune(lambda beta: LinUCB(beta), [0.1, 0.3, 1, 3], problems=[1, 2])
    >>> plot_tuning(tuned, xlabel="beta")

Two measures of an agent (see the README):

- Regret L_T = sum_t (1 - R_t): the reward lost during learning, against an oracle
  that knows every label and so always gets reward 1. On problem 1 this is simply
  the number of wrong guesses; a pass (problem 2) costs 0.5.
- Test reward: after the run, the agent's greedy policy (`predict`, no exploration)
  is applied to the 10 000 test images, which it has never seen. On problem 1 this
  is its test accuracy.

Every run sees the training images in a different order (the seed). Two agents run
with the same seeds see the same images in the same order, so compare them with
`paired_difference`.
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
import torch

import rlcourse.envs  # noqa: F401  (registers the environments)
from rlcourse import leaderboard, mnist
from rlcourse.envs.digits import reward as problem_reward

torch.set_num_threads(1)         # our networks are tiny: one thread is fastest

HORIZON = 60_000                 # one pass over the training images
PROBLEMS = (1, 2)
TUNE_SEEDS = range(0, 2)
FINAL_SEEDS = range(1000, 1003)
CURVE_EVERY = 100                # the regret curve is stored every CURVE_EVERY steps
WEEK_DIR = Path(__file__).parent
FINAL_CACHE = WEEK_DIR / "results" / "final_evaluation.json"
# The files you edit this week. The final evaluation uploads them to the course
# leaderboard, as a record of your work (we may look at them and give feedback).
SUBMITTED_FILES = ("agents.py",)

_envs: dict = {}


def make_env(problem: int, horizon: int = HORIZON) -> gym.Env:
    key = (problem, horizon)
    if key not in _envs:                  # creating one loads the data, so we reuse them
        _envs[key] = gym.make(f"rlcourse/Digits{problem}-v0", horizon=horizon)
    return _envs[key]


def test_data(features: str) -> tuple[np.ndarray, np.ndarray]:
    """Test contexts of the given kind and their true labels."""
    return mnist.features(features)[1], mnist.load()["y_test"].astype(np.int64)


def test_reward(agent, problem: int) -> float:
    """Mean reward of the agent's greedy policy on the 10 000 test images."""
    X, y = test_data(agent.features)
    return float(problem_reward(problem, agent.predict(X), y).mean())


def run_episode(env: gym.Env, agent, seed: int, agent_seed: int | None = None,
                with_test: bool = True) -> dict:
    """One learning run: reset env and agent, then `horizon` steps; then the test reward."""
    obs, _ = env.reset(seed=seed)
    base = env.unwrapped
    rng = np.random.default_rng([seed if agent_seed is None else agent_seed, 2026])
    feat = agent.features
    agent.reset(base.action_space.n, obs[feat].shape[0], base.horizon, rng)
    full = getattr(agent, "full_information", False)
    actions = np.empty(base.horizon, dtype=np.int64)
    rewards = np.empty(base.horizon)
    for t in range(base.horizon):
        x = obs[feat]
        a = int(agent.act(x))
        obs, r, _, _, info = env.step(a)
        if full:
            agent.update(x, a, r, label=info["label"])
        else:
            agent.update(x, a, r)
        actions[t], rewards[t] = a, r
    return {"actions": actions, "rewards": rewards, "regret": np.cumsum(1.0 - rewards),
            "test_reward": test_reward(agent, base.problem) if with_test else None}


@dataclass
class Result:
    """Statistics of one agent on one problem, over several runs (seeds)."""
    label: str
    problem: int
    final: np.ndarray                 # regret L_T of every run
    test: np.ndarray                  # test reward of every run
    late: np.ndarray                  # regret per step over the last 10% of each run
    curve: np.ndarray                 # mean L_t over runs, at t = CURVE_EVERY, 2*CURVE_EVERY, ...
    pass_rate: float                  # fraction of passes (action 10) during the runs
    seconds: float
    seeds: list = field(repr=False)

    @property
    def mean(self): return float(self.final.mean())
    @property
    def se(self): return float(self.final.std(ddof=1) / np.sqrt(len(self.final))) if len(self.final) > 1 else float("nan")
    @property
    def test_mean(self): return float(self.test.mean())

    def __repr__(self):
        passes = f", passes {self.pass_rate:.0%}" if self.problem == 2 else ""
        return (f"{self.label} on problem {self.problem}: regret {self.mean:.0f} ± {self.se:.0f} "
                f"(regret per step at the end {self.late.mean():.3f}{passes}); "
                f"test reward {self.test_mean:.3f}  [{len(self.final)} runs, {self.seconds:.0f} s]")


def evaluate(agent, problem: int, seeds: Iterable[int] = TUNE_SEEDS,
             horizon: int = HORIZON, label: str | None = None) -> Result:
    """Run `agent` once per seed on `problem` (1 or 2)."""
    env = make_env(problem, horizon)
    seeds = list(seeds)
    start = time.perf_counter()
    runs = [run_episode(env, agent, s) for s in seeds]
    tail = max(1, horizon // 10)
    return Result(label or repr(agent), problem,
                  final=np.array([r["regret"][-1] for r in runs]),
                  test=np.array([r["test_reward"] for r in runs]),
                  late=np.array([(r["regret"][-1] - r["regret"][-tail - 1]) / tail
                                 if tail < horizon else r["regret"][-1] / horizon for r in runs]),
                  curve=np.mean([r["regret"][CURVE_EVERY - 1::CURVE_EVERY] for r in runs], axis=0),
                  pass_rate=float(np.mean([np.mean(r["actions"] == 10) for r in runs])),
                  seconds=time.perf_counter() - start, seeds=seeds)


def paired_difference(a: Result, b: Result) -> tuple[float, float]:
    """Mean and standard error of (regret of a) - (regret of b), run by run.

    Both results must use the same seeds: the two agents then see the same images
    in the same order, which makes the comparison much more precise."""
    assert a.seeds == b.seeds and a.problem == b.problem, "need the same problem and seeds"
    d = a.final - b.final
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


def tune(make_agent: Callable, values: Sequence, problems: Sequence[int] = PROBLEMS,
         seeds: Iterable[int] = TUNE_SEEDS, horizon: int = HORIZON) -> dict:
    """Evaluate `make_agent(v)` for every v in `values` on every problem.

    Returns {problem: [Result for each value]} and prints a table as it goes.
    Rough cost per value, problem and seed (T = 60 000): 2-5 s for the ridge
    agents on "pca", 5-15 s for the networks. For a first coarse sweep, pass
    e.g. horizon=20_000 -- but check the winner at the full horizon."""
    seeds = list(seeds)
    out = {}
    for p in problems:
        print(f"problem {p}:            regret    regret/step at end   test reward")
        out[p] = []
        for v in values:
            r = evaluate(make_agent(v), p, seeds, horizon)
            r.value = v
            out[p].append(r)
            print(f"   {v!s:>10}: {r.mean:8.0f} ± {r.se:4.0f}      {r.late.mean():.3f}"
                  f"            {r.test_mean:.3f}   ({r.seconds:.0f} s)", flush=True)
        best = min(out[p], key=lambda r: r.mean)
        print(f"   lowest regret: {best.value} -> {best.mean:.0f} ± {best.se:.0f}")
    return out


def compare(agents: dict, problems: Sequence[int] = PROBLEMS,
            seeds: Iterable[int] = TUNE_SEEDS, horizon: int = HORIZON) -> dict:
    """Evaluate several agents {label: agent} on several problems; print a table."""
    seeds = list(seeds)
    res = {p: {name: evaluate(ag, p, seeds, horizon, label=name) for name, ag in agents.items()}
           for p in problems}
    width = max(len(n) for n in agents) + 2
    print(" " * width + "".join(f"{'problem ' + str(p) + ': regret':>22}{'test':>8}" for p in problems))
    for name in agents:
        print(f"{name:<{width}}" + "".join(
            f"{res[p][name].mean:15.0f} ± {res[p][name].se:4.0f}{res[p][name].test_mean:8.3f}"
            for p in problems))
    return res


# --- plotting -----------------------------------------------------------------

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]


def _finish(ax, legend=True):
    if legend:
        ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    return ax


def plot_tuning(tuned: dict, xlabel: str = "hyperparameter", metric: str = "regret",
                logx: bool = True, ax=None):
    """Regret (metric="regret") or test reward (metric="test") against the hyperparameter,
    one line per problem."""
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    for i, (p, results) in enumerate(tuned.items()):
        x = [r.value for r in results]
        vals = [r.final if metric == "regret" else r.test for r in results]
        m = np.array([v.mean() for v in vals])
        se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals])
        ax.plot(x, m, "-o", ms=4, color=COLOURS[i % 8], label=f"problem {p}")
        ax.fill_between(x, m - se, m + se, color=COLOURS[i % 8], alpha=0.2, lw=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("regret $L_T$" if metric == "regret" else "test reward")
    return _finish(ax)


def plot_regret(results: Sequence[Result], ax=None):
    """Mean cumulative regret L_t of several results."""
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    for i, r in enumerate(results):
        t = CURVE_EVERY * np.arange(1, len(r.curve) + 1)
        ax.plot(t, r.curve, color=COLOURS[i % 8], lw=2, label=f"{r.label} (problem {r.problem})")
    ax.set_xlabel("step t")
    ax.set_ylabel("regret $L_t$")
    return _finish(ax)


def plot_learning_curve(results: Sequence[Result], window: int = 1000, ax=None):
    """Regret per step (on problem 1: the error rate), averaged over `window` steps."""
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    k = max(1, window // CURVE_EVERY)
    for i, r in enumerate(results):
        c = np.concatenate([[0.0], r.curve])
        rate = (c[k:] - c[:-k]) / (k * CURVE_EVERY)
        t = CURVE_EVERY * np.arange(k, len(c))
        ax.plot(t, rate, color=COLOURS[i % 8], lw=2, label=f"{r.label} (problem {r.problem})")
    ax.set_xlabel("step t")
    ax.set_ylabel(f"regret per step (mean over {k * CURVE_EVERY} steps)")
    ax.set_ylim(bottom=0)
    return _finish(ax)


# --- sanity check ---------------------------------------------------------------

def check_agents(horizon: int = 5000) -> None:
    """Run each agent in agents.py briefly and check that it learns something."""
    import agents as A
    candidates = {"GreedyRidge": A.GreedyRidge(), "LinUCB": A.LinUCB(), "LinearTS": A.LinearTS(),
                  "SupervisedNet": A.SupervisedNet(), "Reinforce": A.Reinforce()}
    rnd = 0.9 * horizon
    print(f"Each agent: one run of T = {horizon} steps on problem 1 "
          f"(a random agent has regret about {rnd:.0f}).")
    for name, agent in candidates.items():
        try:
            r = evaluate(agent, 1, seeds=[0], horizon=horizon)
        except NotImplementedError:
            print(f"[ -- ] {name}: not implemented yet")
            continue
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            continue
        verdict = "OK" if r.mean < 0.85 * rnd and r.test_mean > 0.3 else "??"
        print(f"[ {verdict} ] {name}: regret {r.mean:.0f}, test reward {r.test_mean:.3f} "
              f"({r.seconds:.1f} s)")
    print("'??' means: runs, but hardly better than random -- worth a second look.")


# --- final evaluation (Task 5) -------------------------------------------------------

def reference_agents() -> dict:
    """The references in the final table: a supervised learner and greedy ridge regression."""
    import agents as A
    return {"SupervisedNet (ref)": A.SupervisedNet(), "GreedyRidge (ref)": A.GreedyRidge()}


def summarize(agent, seeds: Iterable[int] = FINAL_SEEDS) -> dict:
    res = {p: evaluate(agent, p, seeds) for p in PROBLEMS}
    return {"regret": {p: r.mean for p, r in res.items()},
            "test": {p: r.test_mean for p, r in res.items()},
            "pass_rate": {p: r.pass_rate for p, r in res.items()},
            "full_information": bool(getattr(agent, "full_information", False))}


def _geomean(xs):
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def final_evaluation() -> None:
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

    refs = reference_agents()
    print(f"Final evaluation of {', '.join(final)} on both problems, {len(FINAL_SEEDS)} new runs "
          f"each (T = {HORIZON}),\nplus the references {', '.join(refs)}. This takes a few minutes.\n")
    results = {}
    for name, agent in {**final, **refs}.items():
        print(f"  {name} ...", flush=True)
        results[name] = summarize(agent)
    sup = results["SupervisedNet (ref)"]["regret"]
    for name, r in results.items():
        r["ratio"] = {p: r["regret"][p] / sup[p] for p in PROBLEMS}
        r["score"] = _geomean(list(r["ratio"].values()))

    lines = [f"{'':20s}{'regret (problem 1, 2)':>24s}{'/ supervised':>15s}{'score':>8s}"
             f"{'test reward (1, 2)':>21s}{'passes':>8s}"]
    for name, r in results.items():
        lines.append(f"{name:20s}{r['regret'][1]:12.0f}{r['regret'][2]:10.0f}"
                     f"{r['ratio'][1]:9.2f}{r['ratio'][2]:6.2f}{r['score']:8.2f}"
                     f"{r['test'][1]:11.3f}{r['test'][2]:8.3f}{r['pass_rate'][2]:8.0%}")
    lines += ["",
              "Score: regret relative to SupervisedNet, a learner that is TOLD THE LABEL after every",
              "step, geometric mean over the two problems (lower is better). It measures the price",
              "of bandit feedback. 'passes': how often the agent passed while learning problem 2."]
    report = "\n".join(lines)
    print("\n" + report)
    if any(r["full_information"] for n, r in results.items() if n in final):
        print("\nNote: agents with full_information = True see the labels: they are not bandit\n"
              "algorithms, and the leaderboard lists them separately.")
    print("\nWas your prediction right? Where does your agent lose most against the supervised\n"
          "learner: early, or all the way through? And on problem 2: did it learn to answer,\n"
          "or did it settle for passing?")

    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
             "horizon": HORIZON, "seeds": list(FINAL_SEEDS),
             "agents": {n: repr(a) for n, a in final.items()},
             "results": results, "report": report, "override": bool(history)}
    history.append(entry)
    FINAL_CACHE.parent.mkdir(exist_ok=True)
    FINAL_CACHE.write_text(json.dumps(history, indent=1))
    print()
    leaderboard.submit(WEEK_DIR, "final", {k: v for k, v in entry.items() if k != "report"},
                       files=[WEEK_DIR / f for f in SUBMITTED_FILES])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check your agents (default), or run the final evaluation.")
    ap.add_argument("--final", action="store_true", help="final evaluation of final_agents() (Task 5)")
    args = ap.parse_args()
    if args.final:
        final_evaluation()
    else:
        check_agents()
