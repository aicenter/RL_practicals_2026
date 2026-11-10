"""Learning curves of saved experiments, side by side.

From the command line (a directory means every .json in it):

    uv run python -m rlcourse.deeprl.plotting week07_dqn/results
    uv run python -m rlcourse.deeprl.plotting week07_dqn/results/vanilla.json week07_dqn/results/double.json
    uv run python -m rlcourse.deeprl.plotting week07_dqn/results --metric overestimation --save curves.png

or from Python:

    from rlcourse.deeprl.plotting import plot_experiments
    plot_experiments([Path("week07_dqn/results/vanilla.json")])
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rlcourse.deeprl.experiment import load_experiment

#: What can be plotted: a field of the periodic evaluations, or "behaviour_return", or the name
#: of anything the training script passed to Monitor.log (e.g. "loss").
EVALUATION_METRICS = ("mean_return", "pellets", "death_rate", "clear_rate", "steps",
                      "predicted_start_value", "discounted_start_return", "overestimation")


def _metric(run: dict, metric: str) -> np.ndarray:
    if metric == "overestimation":
        return np.array([np.nan if e["predicted_start_value"] is None
                         else e["predicted_start_value"] - e["discounted_start_return"]
                         for e in run["curve"]], dtype=float)
    if metric in EVALUATION_METRICS:
        return np.array([np.nan if e[metric] is None else e[metric] for e in run["curve"]], dtype=float)
    if metric == "behaviour_return":
        return np.array(run["behaviour_return"], dtype=float)
    if metric in run["diagnostics"]:
        values = np.array(run["diagnostics"][metric], dtype=float)
        return np.concatenate([np.full(len(run["steps"]) - len(values), np.nan), values])
    raise KeyError(f"no metric {metric!r} in this run; try one of {EVALUATION_METRICS} "
                   f"or {sorted(run['diagnostics'])}")


def expand(paths: Sequence[Path]) -> list[Path]:
    """Directories become the .json files inside them, sorted by name."""
    out: list[Path] = []
    for path in map(Path, paths):
        out.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    return out


def plot_experiments(paths: Sequence[Path], metric: str = "mean_return", ax=None, show_seeds: bool = True):
    """One line per experiment (mean over seeds), with the individual seeds faintly behind it."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for path in expand(paths):
        experiment = load_experiment(path)
        runs = experiment["runs"]
        length = min(len(run["steps"]) for run in runs)
        steps = np.array(runs[0]["steps"][:length])
        values = np.stack([_metric(run, metric)[:length] for run in runs])
        summary = experiment["summary"]
        label = f"{experiment['name']}  (final {summary['final_return']:.2f})"
        (line,) = ax.plot(steps, np.nanmean(values, axis=0), lw=2, label=label)
        if show_seeds and len(runs) > 1:
            for row in values:
                ax.plot(steps, row, lw=0.8, alpha=0.3, color=line.get_color())
    ax.set_xlabel("environment steps")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return ax


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves of saved experiments.")
    parser.add_argument("paths", nargs="+", type=Path, help=".json files or directories of them")
    parser.add_argument("--metric", default="mean_return",
                        help=f"one of {', '.join(EVALUATION_METRICS)}, behaviour_return, "
                             "or anything logged with Monitor.log (e.g. loss)")
    parser.add_argument("--save", type=Path, default=None, help="write the figure here instead of showing it")
    args = parser.parse_args()
    plot_experiments(args.paths, args.metric)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    _main()
