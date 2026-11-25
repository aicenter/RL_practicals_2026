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
    assert default_name({}, "core") == "core"


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


def test_pg_tools_board_text_and_saving(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from rlcourse.deeprl.pg_tools import board_text, load_agent, save_agent

    observation, _ = make_env().reset(seed=0)
    text = board_text(observation).splitlines()
    assert len(text) == 8 and text[0] == "########" and "P" in text[1]

    @dataclass(frozen=True)
    class Config:
        gamma: float = 0.99

    def make_agent(config):
        return SimpleNamespace(config=config, policy_network=nn.Linear(3, 2))

    agent = make_agent(Config(gamma=0.5))
    assert save_agent(agent, agent.config, 0) is None            # train did not ask for saving
    monkeypatch.setenv("RLCOURSE_POLICY_DIR", str(tmp_path))
    monkeypatch.setenv("RLCOURSE_POLICY_NAME", "core")
    path = save_agent(agent, agent.config, 3)
    assert path == tmp_path / "core_seed3.pt"
    loaded = load_agent(path, make_agent, Config)
    assert loaded.config.gamma == 0.5
    assert torch.equal(loaded.policy_network.weight, agent.policy_network.weight)


def test_monitor_record_episode():
    monitor = Monitor(10, 0, eval_every=10, eval_episodes=1, final_episodes=1, verbose=False)
    monitor.record_episode(1.5)
    monitor.record_episode(-0.5)
    monitor.evaluate_if_due(10, lambda o: 0)
    assert monitor.behaviour_return == [0.5]


def test_ppo_tools_diagnostics():
    from rlcourse.deeprl.ppo_tools import explained_variance, policy_change

    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(explained_variance(target, target) - 1.0) < 1e-6
    assert abs(explained_variance(torch.full((4,), 2.5), target)) < 1e-6
    old = torch.log(torch.tensor([0.5, 0.5]))
    new = torch.log(torch.tensor([0.5, 0.75]))
    change = policy_change(new, old, clip_eps=0.2)
    assert change["clip_frac"] == 0.5 and abs(change["ratio_max"] - 1.5) < 1e-6
    assert abs(change["kl"] - (0.5 - np.log(1.5)) / 2) < 1e-6
    assert policy_change(old, old, 0.2)["kl"] == 0.0


def test_ppo_tools_hint_reward():
    from rlcourse.deeprl.ppo_tools import make_training_env, next_to_pellet

    plain, hinted = make_env(), make_training_env(hint_reward=0.05)
    assert make_training_env(0.0).__class__ is plain.__class__
    plain.reset(seed=0)
    _, info = hinted.reset(seed=0)
    for action in [1, 1, 2, 3, 0, 1, 2, 2]:
        _, r_plain, term, trunc, _ = plain.step(action)
        _, r_hinted, _, _, info = hinted.step(action)
        assert info["true_reward"] == r_plain
        assert abs(r_hinted - r_plain - 0.05 * next_to_pellet(info["state"])) < 1e-9
        if term or trunc:
            break
