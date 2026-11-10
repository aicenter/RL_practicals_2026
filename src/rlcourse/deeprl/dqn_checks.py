"""Checks for the Week 7 DQN skeleton (`uv run python week07_dqn/dqn.py check`).

The student script passes its own classes and functions in, so nothing here imports the
script: `run_checks(Config=Config, ReplayBuffer=ReplayBuffer, ...)`. Each check runs on its
own, small, hand-made example; an unfinished piece shows as TODO and does not stop the others.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

SHAPE = (8, 8, 4)          # the shape of a board in the course world
N_ACTIONS = 4


class CheckFailed(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _board(n_ones: int) -> np.ndarray:
    """A fake board with `n_ones` cells set to 1, so that a stub network can recognize it."""
    board = np.zeros(SHAPE, dtype=np.uint8)
    board.reshape(-1)[:n_ones] = 1
    return board


class _CountingNetwork(nn.Module):
    """Q(s, a) = (number of ones on the board) * WEIGHTS[a]: values you can work out by hand."""

    def __init__(self, weights: list[float]):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        counts = observations.float().flatten(1).sum(1, keepdim=True)
        return counts * self.weights


# --- the individual checks ------------------------------------------------------------------


def check_linear_schedule(linear_schedule: Callable) -> None:
    for step, want in [(0, 1.0), (50, 0.55), (100, 0.1), (250, 0.1)]:
        got = linear_schedule(step, 1.0, 0.1, 100)
        expect(abs(got - want) < 1e-9,
               f"linear_schedule({step}, start=1.0, end=0.1, duration=100) = {got}, expected {want}")


def check_replay_buffer(ReplayBuffer: type) -> None:
    buffer = ReplayBuffer(3, SHAPE)
    expect(len(buffer) == 0, "a new buffer should have length 0")
    for i in range(2):
        buffer.add(_board(i), i, float(i), _board(i + 10), i == 1)
    expect(len(buffer) == 2, f"after 2 additions len(buffer) = {len(buffer)}, expected 2")

    rng = np.random.default_rng(0)
    batch = buffer.sample(500, rng)
    expect(batch.observations.shape == (500, *SHAPE),
           f"observations have shape {tuple(batch.observations.shape)}, expected (500, *{SHAPE})")
    for name, dtype in [("observations", torch.uint8), ("actions", torch.int64),
                        ("rewards", torch.float32), ("next_observations", torch.uint8),
                        ("terminated", torch.float32)]:
        tensor = getattr(batch, name)
        expect(isinstance(tensor, torch.Tensor), f"batch.{name} should be a torch.Tensor")
        expect(tensor.dtype == dtype, f"batch.{name} has dtype {tensor.dtype}, expected {dtype}")
    expect(set(batch.actions.tolist()) == {0, 1},
           "with 2 stored transitions, a large sample should contain both, and only them "
           f"(got actions {sorted(set(batch.actions.tolist()))}); sample from the filled part only")
    # Each row must be one consistent transition.
    ones = batch.observations.flatten(1).sum(1)
    expect(torch.equal(ones, batch.actions) and torch.equal(batch.rewards, batch.actions.float()),
           "the fields of a sampled row do not belong to the same transition")
    expect(torch.equal(batch.next_observations.flatten(1).sum(1), batch.actions + 10),
           "next_observations do not match their transitions")
    expect(torch.equal(batch.terminated, (batch.actions == 1).float()),
           "terminated should be 1.0 for transitions stored with terminated=True, else 0.0")

    for i in range(2, 5):   # 5 additions to a buffer of capacity 3: the first two are gone
        buffer.add(_board(i), i, float(i), _board(i + 10), False)
    expect(len(buffer) == 3, f"a full buffer of capacity 3 has length {len(buffer)}, expected 3")
    kept = set(buffer.sample(500, rng).actions.tolist())
    expect(kept == {2, 3, 4},
           f"after adding transitions 0..4 to a buffer of capacity 3, it holds {sorted(kept)}; "
           "expected the three most recent, [2, 3, 4]")

    a = ReplayBuffer(10, SHAPE)
    b = ReplayBuffer(10, SHAPE)
    for buf in (a, b):
        for i in range(10):
            buf.add(_board(i), i % 4, 0.0, _board(i), False)
    same = torch.equal(a.sample(32, np.random.default_rng(1)).actions,
                       b.sample(32, np.random.default_rng(1)).actions)
    expect(same, "two samples with identically seeded rngs differ: use the rng you are given")


def check_act(DQNAgent: type, Config: type) -> None:
    agent = DQNAgent(SHAPE, N_ACTIONS, Config())
    agent.q_network = _CountingNetwork([0.0, 1.0, 3.0, 2.0])
    board = _board(5)
    rng = np.random.default_rng(0)
    greedy = [agent.act(board, 0.0, rng) for _ in range(20)]
    expect(all(a == 2 for a in greedy),
           f"with epsilon = 0, act should always pick the best action (2 here); got {set(greedy)}")
    expect(all(isinstance(a, int) for a in greedy), "act should return a Python int")
    explore = np.array([agent.act(board, 1.0, rng) for _ in range(4000)])
    counts = np.bincount(explore, minlength=N_ACTIONS)
    expect(counts.min() > 850,
           f"with epsilon = 1, actions should be uniform; counts over 4000 calls: {counts.tolist()}")
    half = np.array([agent.act(board, 0.5, rng) for _ in range(4000)])
    share = (half == 2).mean()
    expect(abs(share - 0.625) < 0.04,
           f"with epsilon = 0.5, the greedy action should be taken 1/2 + 1/2 * 1/4 = 62.5% of the "
           f"time (the random choice can pick it too); got {share:.1%}")


def _fixed_batch() -> SimpleNamespace:
    """Three transitions with boards of 1..6 ones; the second one terminated."""
    observations = torch.from_numpy(np.stack([_board(1), _board(2), _board(3)]))
    next_observations = torch.from_numpy(np.stack([_board(2), _board(4), _board(6)]))
    return SimpleNamespace(
        observations=observations,
        actions=torch.tensor([0, 1, 3]),
        rewards=torch.tensor([0.1, -1.0, 0.5]),
        next_observations=next_observations,
        terminated=torch.tensor([0.0, 1.0, 0.0]),
    )


def check_td_targets(DQNAgent: type, Config: type) -> None:
    config = replace(Config(), gamma=0.9)
    agent = DQNAgent(SHAPE, N_ACTIONS, config)
    agent.q_network = _CountingNetwork([5.0, 5.0, 5.0, 5.0])        # must NOT be used
    agent.target_network = _CountingNetwork([0.5, 1.0, -1.0, 0.25])  # max_a' Q = (ones) * 1.0
    batch = _fixed_batch()
    y = agent.td_targets(batch)
    expect(isinstance(y, torch.Tensor) and y.shape == (3,),
           f"td_targets should return a tensor of shape (3,) for 3 transitions, got {getattr(y, 'shape', y)}")
    want = torch.tensor([0.1 + 0.9 * 2.0, -1.0, 0.5 + 0.9 * 6.0])
    if torch.allclose(y, torch.tensor([0.1 + 0.9 * 10.0, -1.0, 0.5 + 0.9 * 30.0])):
        raise CheckFailed("the targets use the online network; they should use the target network")
    if torch.allclose(y, torch.tensor([0.1 + 0.9 * 2.0, -1.0 + 0.9 * 4.0, 0.5 + 0.9 * 6.0])):
        raise CheckFailed("the terminated transition (the second) bootstraps; there y = r")
    expect(torch.allclose(y, want, atol=1e-5),
           f"targets {[round(v, 4) for v in y.tolist()]}, expected {[round(v, 4) for v in want.tolist()]}")
    expect(not y.requires_grad, "the targets should carry no gradient: compute them under torch.no_grad()")


def check_update(DQNAgent: type, Config: type) -> None:
    torch.manual_seed(0)
    config = Config()
    agent = DQNAgent(SHAPE, N_ACTIONS, config)
    target_before = [p.detach().clone() for p in agent.target_network.parameters()]
    batch = _fixed_batch()
    with torch.no_grad():
        y = agent.td_targets(batch)
        before = agent.q_network(batch.observations)
    loss = agent.update(batch)
    expect(isinstance(loss, float), f"update should return the loss as a Python float, got {type(loss).__name__}")
    with torch.no_grad():
        after = agent.q_network(batch.observations)
    taken = batch.actions.unsqueeze(1)
    gap_before = (before.gather(1, taken).squeeze(1) - y).abs().sum()
    gap_after = (after.gather(1, taken).squeeze(1) - y).abs().sum()
    expect(not torch.equal(before, after), "update did not change the online network")
    expect(gap_after < gap_before,
           "after one update, Q(s, a) of the actions taken should be closer to the targets")
    expected_loss = float(torch.nn.functional.smooth_l1_loss(before.gather(1, taken).squeeze(1), y))
    expect(abs(loss - expected_loss) < 1e-4,
           f"update returned {loss:.5f}, but the Huber loss of this batch before the step is "
           f"{expected_loss:.5f}")
    for p, q in zip(agent.target_network.parameters(), target_before):
        expect(torch.equal(p, q), "update changed the target network; it should change only the online one")
    for _ in range(200):
        last = agent.update(batch)
    expect(last < 0.5 * loss, f"200 updates on the same batch should fit it: loss {loss:.4f} -> {last:.4f}")


def check_sync_target(DQNAgent: type, Config: type) -> None:
    agent = DQNAgent(SHAPE, N_ACTIONS, Config())
    with torch.no_grad():
        for p in agent.q_network.parameters():
            p.add_(1.0)
    agent.sync_target()
    for p, q in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        expect(torch.equal(p, q), "after sync_target, the target network should equal the online one")
    with torch.no_grad():
        next(agent.q_network.parameters()).add_(1.0)
    expect(not torch.equal(next(agent.q_network.parameters()), next(agent.target_network.parameters())),
           "after sync_target, the two networks share parameters; the target must be a copy")


def check_train(train: Callable, Config: type) -> None:
    config = replace(Config(), total_steps=1_500, learning_starts=200, eval_every=500,
                     target_update_every=300, epsilon_decay_steps=1_000)
    started = time.perf_counter()
    result = train(config, 0)
    seconds = time.perf_counter() - started
    expect(result.steps == [500, 1000, 1500], f"evaluations at steps {result.steps}, expected [500, 1000, 1500]")
    losses = result.diagnostics.get("loss", [])
    expect(len(losses) == 3 and all(np.isfinite(losses)),
           f"expected a finite loss logged in each evaluation period, got {losses}")
    print(f"         (the short run took {seconds:.0f} s)")


# --- running them -----------------------------------------------------------------------------


def run_checks(*, Config: type, ReplayBuffer: type, DQNAgent: type, linear_schedule: Callable,
               train: Callable) -> bool:
    """Run every check and print OK / TODO / FAIL for each. Returns True if all passed."""
    checks: list[tuple[str, Callable[[], Any]]] = [
        ("linear_schedule", lambda: check_linear_schedule(linear_schedule)),
        ("ReplayBuffer.add and .sample", lambda: check_replay_buffer(ReplayBuffer)),
        ("DQNAgent.act", lambda: check_act(DQNAgent, Config)),
        ("DQNAgent.td_targets", lambda: check_td_targets(DQNAgent, Config)),
        ("DQNAgent.update", lambda: check_update(DQNAgent, Config)),
        ("DQNAgent.sync_target", lambda: check_sync_target(DQNAgent, Config)),
        ("train (a short run)", lambda: check_train(train, Config)),
    ]
    passed = 0
    torch.set_num_threads(1)
    for name, check in checks:
        try:
            check()
        except NotImplementedError as e:
            print(f"[TODO] {name}: {e}")
            continue
        except CheckFailed as e:
            print(f"[FAIL] {name}: {e}")
            continue
        except Exception:  # noqa: BLE001  (report any error in student code, then go on)
            print(f"[FAIL] {name} raised an exception:")
            traceback.print_exc(limit=-3)
            continue
        print(f"[ OK ] {name}")
        passed += 1
    print(f"\n{passed} of {len(checks)} checks passed.")
    if passed == len(checks):
        print("All done: train with `uv run python week07_dqn/dqn.py train`.")
    return passed == len(checks)
