"""Checks for the Week 8 policy-gradient skeleton (`uv run python week08_pg/pg.py check`).

The student script passes its own classes and functions in, so nothing here imports the
script: `run_checks(Config=Config, PGAgent=PGAgent, ...)`. Each check runs on its own, small,
hand-made example; an unfinished piece shows as TODO and does not stop the others.
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
import torch.nn.functional as F
from torch import nn

SHAPE = (8, 8, 4)          # the shape of a board in the course world
N_ACTIONS = 4


class CheckFailed(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _rounded(x) -> list[float]:
    return [round(float(v), 4) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


class _FixedLogits(nn.Module):
    """The same (trainable) logits for every board: a policy you can work out by hand."""

    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(observations.shape[0], -1)


def _episode(rewards: list[float], actions: list[int] | None = None, terminated: bool = True):
    """A hand-made episode with the fields of `Episode` (the boards are blank)."""
    n = len(rewards)
    return SimpleNamespace(
        observations=np.zeros((n, *SHAPE), dtype=np.uint8),
        actions=np.array(actions if actions is not None else [0] * n, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        terminated=terminated,
        last_observation=np.zeros(SHAPE, dtype=np.uint8),
    )


# --- the individual checks ------------------------------------------------------------------


def check_returns_to_go(returns_to_go: Callable) -> None:
    rewards = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    got = returns_to_go(rewards, 0.5)
    expect(isinstance(got, np.ndarray), f"returns_to_go should return a numpy array, got {type(got).__name__}")
    expect(got.shape == (3,), f"returns_to_go of 3 rewards has shape {got.shape}, expected (3,)")
    expect(np.array_equal(rewards, [1.0, 0.0, 2.0]), "returns_to_go changed the rewards it was given")
    want = [1.0 + 0.5 * 0.0 + 0.25 * 2.0, 0.0 + 0.5 * 2.0, 2.0]
    if np.allclose(got, [1.0, 1.0, 3.0]) or np.allclose(got, [1.0, 0.5, 2.5]):
        raise CheckFailed(f"returns_to_go([1, 0, 2], gamma=0.5) = {_rounded(got)}: that sums the rewards "
                          "*before* t; G_t sums the rewards from t to the end")
    expect(np.allclose(got, want, atol=1e-5),
           f"returns_to_go([1, 0, 2], gamma=0.5) = {_rounded(got)}, expected {want}")
    got = returns_to_go(np.array([-1.0, 0.5, 0.5, 3.0], dtype=np.float32), 1.0)
    expect(np.allclose(got, [3.0, 4.0, 3.5, 3.0]),
           f"returns_to_go([-1, 0.5, 0.5, 3], gamma=1) = {_rounded(got)}, expected [3.0, 4.0, 3.5, 3.0]")
    got = returns_to_go(np.array([5.0], dtype=np.float32), 0.9)
    expect(np.allclose(got, [5.0]), f"returns_to_go([5], gamma=0.9) = {_rounded(got)}, expected [5.0]")


def check_act(PGAgent: type, Config: type) -> None:
    agent = PGAgent(SHAPE, N_ACTIONS, Config())
    agent.policy_network = _FixedLogits(np.log([0.1, 0.2, 0.3, 0.4]).tolist())
    board = np.zeros(SHAPE, dtype=np.uint8)
    rng = np.random.default_rng(0)
    actions = [agent.act(board, rng) for _ in range(4000)]
    expect(all(isinstance(a, int) for a in actions), "act should return a Python int")
    share = np.bincount(actions, minlength=N_ACTIONS) / len(actions)
    if share[3] > 0.99:
        raise CheckFailed("act always takes the most likely action; it should *sample* from pi(. | s)")
    expect(np.allclose(share, [0.1, 0.2, 0.3, 0.4], atol=0.03),
           f"with pi = [0.1, 0.2, 0.3, 0.4], act chose the actions {np.round(share, 3).tolist()} of the time")
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    same = [agent.act(board, rng_a) for _ in range(30)] == [agent.act(board, rng_b) for _ in range(30)]
    expect(same,
           "two identically seeded rngs gave different actions: use the rng you are given")


def check_weights(PGAgent: type, Config: type) -> None:
    episodes = [_episode([0.0, 0.0, 1.0]), _episode([1.0, -1.0], terminated=False)]
    rtg = np.array([0.25, 0.5, 1.0, 0.5, -1.0])            # gamma = 0.5
    full = np.array([0.25, 0.25, 0.25, 0.5, 0.5])
    cases = [
        ("the core (return-to-go, batch mean)", dict(), rtg - rtg.mean()),
        ("--baseline none", dict(baseline="none"), rtg),
        ("--no-rtg --baseline none", dict(rtg=False, baseline="none"), full),
        ("--no-rtg", dict(rtg=False), full - full.mean()),
    ]
    for label, changes, want in cases:
        agent = PGAgent(SHAPE, N_ACTIONS, replace(Config(), gamma=0.5, **changes))
        w = agent.weights(episodes)
        expect(isinstance(w, torch.Tensor), f"weights should return a torch.Tensor, got {type(w).__name__}")
        expect(w.shape == (5,), f"weights of two episodes with 3 + 2 steps: shape {tuple(w.shape)}, expected (5,)")
        expect(w.dtype == torch.float32, f"weights should be float32, got {w.dtype}")
        expect(not w.requires_grad, "the weights should carry no gradient")
        got = w.numpy()
        if np.allclose(got, want, atol=1e-5):
            continue
        hint = ""
        if label.startswith("the core") and np.allclose(got, rtg):
            hint = " (that is G_t without the baseline)"
        elif label.startswith("the core") and np.allclose(
                got, np.concatenate([rtg[:3] - rtg[:3].mean(), rtg[3:] - rtg[3:].mean()])):
            hint = " (that subtracts each episode's own mean; the baseline is the mean over the whole batch)"
        elif np.allclose(got, rtg - rtg.mean()) and "no-rtg" in label:
            hint = " (that is return-to-go: with --no-rtg every step gets its episode's whole return R(tau))"
        elif "no-rtg" in label and (np.allclose(got, [1, 1, 1, 0, 0]) or np.allclose(got, [0.4, 0.4, 0.4, -0.6, -0.6])):
            hint = (" (that is the undiscounted sum of the rewards; R(tau) here is the discounted "
                    "return, which is G_0)")
        elif label == "--no-rtg" and np.allclose(got, full - 0.375):
            hint = (" (that subtracts the mean over *episodes*; we take the mean over all steps of the "
                    "batch, the same as with return-to-go)")
        raise CheckFailed(f"{label}, gamma = 0.5, episodes with rewards [0, 0, 1] and [1, -1]: "
                          f"weights {_rounded(got)}, expected {_rounded(want)}{hint}")


def check_update(PGAgent: type, Config: type) -> None:
    episodes = [_episode([0.0] * 3, actions=[2, 2, 0]), _episode([0.0] * 2, actions=[1, 3])]
    fixed_weights = torch.tensor([1.0, 0.5, -1.0, 0.0, -0.5])
    initial = [0.0, 0.5, 1.0, -0.5]
    actions = torch.tensor([2, 2, 0, 1, 3])

    def make(entropy_coef: float, lr: float = 1e-2):
        agent = PGAgent(SHAPE, N_ACTIONS, replace(Config(), entropy_coef=entropy_coef))
        agent.policy_network = _FixedLogits(initial)
        agent.optimizer = torch.optim.SGD(agent.policy_network.parameters(), lr=lr)
        agent.weights = lambda eps: fixed_weights.clone()      # the check chooses the weights
        return agent

    # 1. The returned entropy (before the step).
    agent = make(0.0)
    p = torch.softmax(torch.tensor(initial), 0)
    want_entropy = float(-(p * p.log()).sum())
    entropy = agent.update(episodes)
    expect(isinstance(entropy, float), f"update should return the entropy as a Python float, got {type(entropy).__name__}")
    expect(abs(entropy - want_entropy) < 1e-4,
           f"update returned {entropy:.4f}; the mean entropy of the policy before the step is {want_entropy:.4f}")

    # 2. The step itself: plain SGD, so the change of the logits is -lr * gradient of the loss.
    for coef in (0.0, 0.1):
        agent = make(coef)
        before = agent.policy_network.logits.detach().clone()
        agent.update(episodes)
        change = (agent.policy_network.logits.detach() - before).numpy()
        expect(np.abs(change).sum() > 0, "update did not change the policy network")
        def step_of(kind: str) -> np.ndarray:
            """The SGD step (lr = 0.01) of one candidate loss, from the logits before the update."""
            x = before.clone().requires_grad_(True)
            log_probs = F.log_softmax(x.expand(5, -1), dim=1)
            chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ent = -(log_probs.exp() * log_probs).sum(1).mean()
            weighted = chosen * fixed_weights
            loss = {"right": -weighted.mean() - coef * ent,
                    "sum": -weighted.sum() - coef * ent,
                    "per episode": -weighted.sum() / 2 - coef * ent,
                    "ascent": weighted.mean() + coef * ent,
                    "entropy sign": -weighted.mean() + coef * ent,
                    "no entropy": -weighted.mean()}[kind]
            (grad,) = torch.autograd.grad(loss, [x])
            return (-1e-2 * grad).numpy()

        steps = {k: step_of(k) for k in ("right", "sum", "per episode", "ascent", "entropy sign", "no entropy")}
        if np.allclose(change, steps["right"], atol=1e-6):
            continue
        if np.allclose(change, steps["ascent"], atol=1e-6):
            raise CheckFailed("update makes the weighted actions *less* likely: the optimizer minimizes, "
                              "so the loss is minus the weighted log-probabilities")
        if np.allclose(change, steps["sum"], atol=1e-6) or np.allclose(change, steps["per episode"], atol=1e-6):
            raise CheckFailed("update sums the weighted log-probabilities over the steps; take their *mean* "
                              "over all steps of the batch (so that the entropy bonus, a mean over the "
                              "steps too, keeps its weight)")
        if coef > 0 and np.allclose(change, steps["entropy sign"], atol=1e-6):
            raise CheckFailed("the entropy bonus has the wrong sign: it should make the policy *more* random, "
                              "so the loss *subtracts* entropy_coef * entropy")
        if coef > 0 and np.allclose(change, steps["no entropy"], atol=1e-6):
            raise CheckFailed("update ignores the entropy bonus (entropy_coef * entropy)")
        raise CheckFailed(f"with entropy_coef = {coef} and SGD at lr = 0.01, the logits moved by "
                          f"{_rounded(change)}; the loss in the docstring moves them by "
                          f"{_rounded(steps['right'])}")

    # 3. Gradient clipping: with huge weights, the step is at most lr * max_grad_norm.
    agent = make(0.0, lr=1.0)
    agent.weights = lambda eps: 1e4 * fixed_weights.clone()
    before = agent.policy_network.logits.detach().clone()
    agent.update(episodes)
    moved = float((agent.policy_network.logits.detach() - before).norm())
    limit = agent.config.max_grad_norm
    expect(moved <= limit * 1.001,
           f"with huge weights, one SGD step at lr = 1 moved the logits by {moved:.1f}; clipping the "
           f"gradient norm to max_grad_norm = {limit} should keep it at most {limit}")


def check_train(train: Callable, Config: type) -> None:
    config = replace(Config(), total_steps=6_000, eval_every=2_000)
    started = time.perf_counter()
    result = train(config, 0)
    seconds = time.perf_counter() - started
    expect(result.steps == [2000, 4000, 6000], f"evaluations at steps {result.steps}, expected [2000, 4000, 6000]")
    entropy = result.diagnostics.get("entropy", [])
    expect(len(entropy) == 3 and all(np.isfinite(entropy)),
           f"expected the entropy logged in each evaluation period, got {entropy}")
    expect(all(0.0 < e <= np.log(N_ACTIONS) + 1e-6 for e in entropy),
           f"the entropy of a policy over 4 actions lies between 0 and ln 4 = 1.386; got {_rounded(entropy)}")
    print(f"         (the short run took {seconds:.1f} s)")


# --- running them -----------------------------------------------------------------------------


def run_checks(*, Config: type, PGAgent: type, returns_to_go: Callable, train: Callable,
               script: str = "week08_pg/pg.py") -> bool:
    """Run every check and print OK / TODO / FAIL for each. Returns True if all passed."""
    checks: list[tuple[str, Callable[[], Any]]] = [
        ("returns_to_go", lambda: check_returns_to_go(returns_to_go)),
        ("PGAgent.act", lambda: check_act(PGAgent, Config)),
        ("PGAgent.weights", lambda: check_weights(PGAgent, Config)),
        ("PGAgent.update", lambda: check_update(PGAgent, Config)),
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
        print(f"All done: train with `uv run python {script} train`.")
    return passed == len(checks)
