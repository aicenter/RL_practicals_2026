"""Running an experiment: several seeds in parallel, progress, results on disk, the leaderboard.

Your training function stays a plain loop. It reports to a `Monitor`, which keeps the
bookkeeping (episode returns, periodic evaluation, progress lines) out of the way:

    from rlcourse.deeprl.experiment import Monitor, RunResult, run_experiment

    def train(config: Config, seed: int) -> RunResult:
        monitor = Monitor(config.total_steps, seed)
        for step in range(1, config.total_steps + 1):
            ...                                         # act, learn
            monitor.record_step(reward, terminated or truncated)
            monitor.log(loss=loss)                      # anything you want averaged and kept
            monitor.evaluate_if_due(step, agent.greedy_action)
        return monitor.finish(agent.greedy_action)

    run_experiment(train, Config(), seeds=(0, 1, 2), results_dir=Path("week07_dqn/results"))

`run_experiment` runs one process per seed (each with one CPU thread), so three seeds take about
as long as one on a laptop with four cores. With `workers=1` everything runs in the current
process, which is what you want in a debugger.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlcourse import leaderboard
from rlcourse.deeprl.evaluation import Evaluation, Policy, ValueFunction, evaluate

# --- one training run ----------------------------------------------------------------------------


@dataclass
class RunResult:
    """Everything a single training run (one seed) produced."""

    seed: int
    steps: list[int]
    """Environment steps at which the policy was evaluated ..."""
    curve: list[Evaluation]
    """... and what the evaluations found."""
    behaviour_return: list[float]
    """Mean return of the training episodes (exploration included) since the previous evaluation."""
    diagnostics: dict[str, list[float]]
    """Whatever was passed to Monitor.log, averaged between evaluations."""
    final: Evaluation
    """The final evaluation, on more episodes: the result of the run."""
    seconds: float

    @property
    def area_under_curve(self) -> float:
        """Mean evaluation return over the whole run: rewards learning fast, not only learning well."""
        return float(np.mean([e.mean_return for e in self.curve])) if self.curve else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["area_under_curve"] = self.area_under_curve
        return d


class Monitor:
    """Bookkeeping for one training run: episode returns, periodic evaluation, progress lines."""

    def __init__(
        self,
        total_steps: int,
        seed: int,
        *,
        eval_every: int = 10_000,
        eval_episodes: int = 50,
        final_episodes: int = 100,
        gamma: float = 0.99,
        verbose: bool = True,
    ):
        self.total_steps = total_steps
        self.seed = seed
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.final_episodes = final_episodes
        self.gamma = gamma
        self.verbose = verbose
        self.started = time.perf_counter()
        self._episode_return = 0.0
        self._finished_returns: list[float] = []
        self._scalars: dict[str, list[float]] = defaultdict(list)
        self._recent_returns: deque[float] = deque(maxlen=100)
        self.steps: list[int] = []
        self.curve: list[Evaluation] = []
        self.behaviour_return: list[float] = []
        self.diagnostics: dict[str, list[float]] = defaultdict(list)

    def record_step(self, reward: float, episode_over: bool) -> None:
        """Call once per environment step, with the reward and whether the episode just ended."""
        self._episode_return += float(reward)
        if episode_over:
            self._finished_returns.append(self._episode_return)
            self._recent_returns.append(self._episode_return)
            self._episode_return = 0.0

    def log(self, **scalars: float) -> None:
        """Record numbers (e.g. loss=..., epsilon=...); they are averaged between evaluations."""
        for name, value in scalars.items():
            self._scalars[name].append(float(value))

    def evaluate_if_due(self, step: int, policy: Policy, value_fn: ValueFunction | None = None) -> None:
        """Every `eval_every` steps: evaluate `policy` on fixed episodes and print a progress line."""
        if step % self.eval_every != 0:
            return
        result = evaluate(policy, self.eval_episodes, value_fn=value_fn, gamma=self.gamma)
        self.steps.append(step)
        self.curve.append(result)
        self.behaviour_return.append(
            float(np.mean(self._finished_returns)) if self._finished_returns else float("nan")
        )
        self._finished_returns.clear()
        logged = {name: float(np.mean(values)) for name, values in self._scalars.items() if values}
        for name, value in logged.items():
            self.diagnostics[name].append(value)
        self._scalars.clear()
        if self.verbose:
            elapsed = time.perf_counter() - self.started
            line = (
                f"[seed {self.seed}] step {step:>7,} | return {result.mean_return:5.2f} "
                f"(pellets {result.pellets:4.1f}, deaths {result.death_rate:4.0%})"
            )
            if result.overestimation is not None:
                line += (
                    f" | V(s0) {result.predicted_start_value:5.2f} vs G0 {result.discounted_start_return:5.2f}"
                )
            line += "".join(f" | {name} {value:.3g}" for name, value in logged.items())
            line += f" | {step / elapsed:,.0f} steps/s"
            print(line, flush=True)

    def finish(self, policy: Policy, value_fn: ValueFunction | None = None) -> RunResult:
        """The final evaluation, on `final_episodes` episodes. Returns the result of the run."""
        final = evaluate(policy, self.final_episodes, value_fn=value_fn, gamma=self.gamma)
        seconds = time.perf_counter() - self.started
        if self.verbose:
            print(
                f"[seed {self.seed}] done in {seconds:.0f} s: return {final.mean_return:.2f} "
                f"± {final.stderr_return:.2f} over {final.episodes} episodes "
                f"(pellets {final.pellets:.1f}, deaths {final.death_rate:.0%}, "
                f"cleared {final.clear_rate:.0%})",
                flush=True,
            )
        return RunResult(
            seed=self.seed,
            steps=self.steps,
            curve=self.curve,
            behaviour_return=self.behaviour_return,
            diagnostics=dict(self.diagnostics),
            final=final,
            seconds=seconds,
        )


# --- several seeds ------------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """The runs of one configuration with several seeds, and their summary."""

    name: str
    config: dict[str, Any]
    changes: dict[str, Any]
    """The config fields that differ from their defaults: what makes this run not vanilla."""
    runs: list[RunResult] = field(default_factory=list)

    @property
    def final_returns(self) -> list[float]:
        return [run.final.mean_return for run in self.runs]

    def summary(self) -> dict[str, Any]:
        finals = np.array(self.final_returns)
        aucs = np.array([run.area_under_curve for run in self.runs])

        def stderr(x: np.ndarray) -> float:
            return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

        return {
            "name": self.name,
            "changes": self.changes,
            "seeds": [run.seed for run in self.runs],
            "total_steps": self.config.get("total_steps"),
            "final_return": float(finals.mean()),
            "final_return_stderr": stderr(finals),
            "final_returns": finals.tolist(),
            "area_under_curve": float(aucs.mean()),
            "area_under_curve_stderr": stderr(aucs),
            "pellets": float(np.mean([run.final.pellets for run in self.runs])),
            "death_rate": float(np.mean([run.final.death_rate for run in self.runs])),
            "clear_rate": float(np.mean([run.final.clear_rate for run in self.runs])),
            "seconds": float(np.mean([run.seconds for run in self.runs])),
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config,
            "changes": self.changes,
            "summary": self.summary(),
            "runs": [run.to_dict() for run in self.runs],
        }


def config_changes(config: Any) -> dict[str, Any]:
    """The fields of a config dataclass whose values differ from the defaults."""
    defaults = type(config)()
    return {f.name: getattr(config, f.name) for f in fields(config)
            if getattr(config, f.name) != getattr(defaults, f.name)}


def default_name(changes: dict[str, Any]) -> str:
    """'vanilla', or e.g. 'double' or 'n_step=3+lr=0.001', from the changed config fields."""
    if not changes:
        return "vanilla"
    parts = [name if value is True else f"{name}={value}" for name, value in changes.items()]
    return "+".join(parts)


def _run_one[ConfigT](train: Callable[[ConfigT, int], RunResult], config: ConfigT, seed: int) -> RunResult:
    """One seed. Runs in a worker process (or in this one, with workers=1)."""
    torch.set_num_threads(1)  # one core per seed: the seeds run side by side
    torch.manual_seed(seed)
    return train(config, seed)


def run_experiment[ConfigT](
    train: Callable[[ConfigT, int], RunResult],
    config: ConfigT,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    workers: int | None = None,
    name: str | None = None,
    results_dir: Path | None = None,
    submit: bool = False,
    week_dir: Path | None = None,
    nickname: str | None = None,
    files: Sequence[Path] = (),
    leaderboard_kind: str = "train",
) -> ExperimentResult:
    """Train with every seed, summarize, save to `results_dir/<name>.json`, optionally submit.

    Args:
        train: `train(config, seed) -> RunResult`. With workers > 1 it runs in other processes,
            so it must be importable: a top-level function, called from under
            `if __name__ == "__main__":` in your script.
        config: a dataclass instance; fields that differ from the defaults name the experiment.
        seeds: one training run per seed.
        workers: processes to use; default: one per seed, at most (number of CPUs - 1).
        name: the name of the results file; default: derived from the config changes.
        results_dir: where to save `<name>.json`; None: don't save.
        submit: send the summary (and `files`) to the course leaderboard for `week_dir`.
    """
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    changes = config_changes(config)
    name = name or default_name(changes)
    experiment = ExperimentResult(name=name, config=asdict(config), changes=changes)
    if workers is None:
        workers = max(1, min(len(seeds), (os.cpu_count() or 2) - 1))
    print(f"Experiment '{name}': {len(seeds)} seed(s) {list(seeds)}, {workers} process(es).")
    print(f"Changes from the defaults: {changes or 'none'}", flush=True)

    if workers == 1:
        experiment.runs = [_run_one(train, config, seed) for seed in seeds]
    else:
        # "spawn" on every platform (it is the only option on Windows and the default on macOS),
        # so that the script behaves the same everywhere.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_run_one, train, config, seed) for seed in seeds]
            experiment.runs = [future.result() for future in futures]

    summary = experiment.summary()
    print(
        f"\n'{name}', mean ± standard error over {len(seeds)} seeds:\n"
        f"  area under the curve {summary['area_under_curve']:.2f} ± {summary['area_under_curve_stderr']:.2f}"
        f"  (mean evaluation return over training: the leaderboard's main number)\n"
        f"  final return         {summary['final_return']:.2f} ± {summary['final_return_stderr']:.2f}"
        f"  (the policy at the end, {experiment.runs[0].final.episodes} episodes per seed)\n"
        f"  {summary['seconds']:.0f} s per run."
    )

    if results_dir is not None:
        path = save_experiment(experiment, Path(results_dir))
        print(f"Saved {path}. Compare runs with: uv run python -m rlcourse.deeprl.plotting {path.parent}")
    if submit:
        if week_dir is None:
            raise ValueError("submit=True needs week_dir")
        payload = {**summary, "config": experiment.config}
        leaderboard.submit(week_dir, leaderboard_kind, payload, nickname=nickname, files=files)
    return experiment


# --- results on disk -------------------------------------------------------------------------


def save_experiment(experiment: ExperimentResult, results_dir: Path) -> Path:
    """Write the experiment to `results_dir/<name>.json` (overwriting a previous one)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_=.+" else "_" for c in experiment.name)
    path = results_dir / f"{safe}.json"
    path.write_text(json.dumps(experiment.to_dict(), indent=1))
    return path


def load_experiment(path: Path) -> dict:
    """Read back what `save_experiment` wrote, as plain dictionaries."""
    return json.loads(Path(path).read_text())
