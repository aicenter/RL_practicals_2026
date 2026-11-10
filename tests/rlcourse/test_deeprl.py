"""Tests of the shared deep-RL tools (src/rlcourse/deeprl/)."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import torch

from rlcourse.deeprl.evaluation import evaluate
from rlcourse.deeprl.experiment import (
    Monitor,
    RunResult,
    config_changes,
    default_name,
    load_experiment,
    run_experiment,
)
from rlcourse.deeprl.networks import GridEncoder
from rlcourse.deeprl.world import make_env

matplotlib.use("Agg")


def test_world_shape_and_actions():
    env = make_env()
    observation, _ = env.reset(seed=0)
    assert observation.shape == (8, 8, 4) and observation.dtype == np.uint8
    assert env.action_space.n == 4


def test_evaluation_is_reproducible():
    def policy(observation: np.ndarray) -> int:
        return int(observation.sum()) % 4      # deterministic, but not constant

    first, second = evaluate(policy, 5), evaluate(policy, 5)
    assert first == second
    assert first.overestimation is None


def test_evaluation_reports_overestimation():
    result = evaluate(lambda o: 0, 3, value_fn=lambda o: 10.0)
    assert result.predicted_start_value == 10.0
    assert result.overestimation == 10.0 - result.discounted_start_return


def test_grid_encoder_shape():
    encoder = GridEncoder((8, 8, 4))
    features = encoder(torch.zeros(5, 8, 8, 4, dtype=torch.uint8))
    assert features.shape == (5, encoder.n_features)


@dataclass(frozen=True)
class _Config:
    total_steps: int = 40
    double: bool = False
    lr: float = 1e-3


def _train(config: _Config, seed: int) -> RunResult:
    monitor = Monitor(config.total_steps, seed, eval_every=20, eval_episodes=2, final_episodes=2,
                      verbose=False)
    env = make_env()
    env.reset(seed=seed)
    for step in range(1, config.total_steps + 1):
        _, reward, terminated, truncated, _ = env.step(step % 4)
        monitor.record_step(reward, terminated or truncated)
        if terminated or truncated:
            env.reset()
        monitor.log(loss=1.0 / step)
        monitor.evaluate_if_due(step, lambda o: 1)
    return monitor.finish(lambda o: 1)


def test_config_changes_and_names():
    assert config_changes(_Config()) == {}
    assert default_name(config_changes(_Config(double=True, lr=0.01))) == "double+lr=0.01"
    assert default_name({}) == "vanilla"


def test_run_experiment_saves_results(tmp_path):
    experiment = run_experiment(_train, _Config(double=True), seeds=(0, 1), workers=1,
                                results_dir=tmp_path)
    assert experiment.name == "double"
    saved = load_experiment(tmp_path / "double.json")
    assert saved["changes"] == {"double": True}
    assert len(saved["runs"]) == 2 and saved["runs"][0]["steps"] == [20, 40]
    assert len(saved["runs"][0]["diagnostics"]["loss"]) == 2

    from rlcourse.deeprl.plotting import plot_experiments

    plot_experiments([tmp_path])
    plot_experiments([tmp_path], metric="loss")
