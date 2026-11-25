"""Checks for the Week 9 actor-critic skeleton (`uv run python week09_ppo/ppo.py check`).

The student script passes its own pieces in, so nothing here imports the script:
`run_checks(Config=Config, gae=gae, ...)`. Each check runs on a small, hand-made example whose
answer was worked out in advance; an unfinished piece shows as TODO and does not stop the others.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch


class CheckFailed(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _t(rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def _rounded(x: torch.Tensor):
    return np.round(x.detach().double().numpy(), 4).tolist()


def _close(x: torch.Tensor, want, atol: float = 1e-4) -> bool:
    return np.allclose(x.detach().double().numpy(), np.asarray(want, dtype=np.float64), atol=atol)


# --- gae ----------------------------------------------------------------------------------------


def _call_gae(gae: Callable, rewards, values, last_values, terminated, truncated, final_values,
              gamma: float, lam: float):
    inputs = [_t(x) for x in (rewards, values, last_values, terminated, truncated, final_values)]
    copies = [x.clone() for x in inputs]
    out = gae(*inputs, gamma, lam)
    expect(isinstance(out, tuple) and len(out) == 2, "gae should return a pair (advantages, targets)")
    advantages, targets = out
    expect(isinstance(advantages, torch.Tensor) and isinstance(targets, torch.Tensor),
           "gae should return two tensors")
    shape = tuple(inputs[0].shape)
    expect(tuple(advantages.shape) == shape and tuple(targets.shape) == shape,
           f"advantages and targets should have the shape of the rewards, {shape}; got "
           f"{tuple(advantages.shape)} and {tuple(targets.shape)}")
    expect(all(torch.equal(a, b) for a, b in zip(inputs, copies)),
           "gae changed one of the tensors it was given (in place, e.g. with +=): work on a copy")
    return advantages, targets


def check_gae(gae: Callable) -> None:
    # One environment, three steps, no episode ends, gamma = 0.5.
    one = dict(rewards=[[1.0], [0.0], [2.0]], values=[[0.5], [1.0], [1.5]], last_values=[2.0],
               terminated=[[0.0]] * 3, truncated=[[0.0]] * 3, final_values=[[0.0]] * 3)
    advantages, targets = _call_gae(gae, **one, gamma=0.5, lam=0.0)
    if _close(advantages, [1.25, 0.5, 1.5]):
        raise CheckFailed("with lambda = 0, gae gave the advantages of lambda = 1: is lambda in the "
                          "recursion A_t = delta_t + gamma lambda A_{t+1}?")
    expect(_close(advantages, [[1.0], [-0.25], [1.5]]),
           f"lambda = 0: the advantages should be the one-step TD errors delta_t = r_t + gamma V(s_t+1) "
           f"- V(s_t) = [1.0, -0.25, 1.5]; got {_rounded(advantages.flatten())}. Remember that V(s_T), "
           "after the last step, is last_values")
    expect(_close(targets, [[1.5], [0.75], [3.0]]),
           f"the targets should be advantages + values = [1.5, 0.75, 3.0]; got {_rounded(targets.flatten())}")
    advantages, _ = _call_gae(gae, **one, gamma=0.5, lam=1.0)
    if _close(advantages, [[2.25], [1.25], [1.5]]):
        raise CheckFailed("lambda = 1, gamma = 0.5: you got [2.25, 1.25, 1.5], as if the recursion "
                          "used lambda instead of gamma * lambda")
    expect(_close(advantages, [[1.25], [0.5], [1.5]]),
           f"lambda = 1: the advantages should be the returns to the end of the segment, bootstrapped "
           f"from V(s_T), minus V(s_t): [1.25, 0.5, 1.5]; got {_rounded(advantages.flatten())}")
    advantages, _ = _call_gae(gae, **one, gamma=0.5, lam=0.5)
    if _close(advantages, [[1.25], [0.5], [1.5]]):
        raise CheckFailed("lambda = 0.5, gamma = 0.5: you got [1.25, 0.5, 1.5], as if the recursion "
                          "used lambda instead of gamma * lambda")
    expect(_close(advantages, [[1.03125], [0.125], [1.5]]),
           f"lambda = 0.5, gamma = 0.5: expected [1.03125, 0.125, 1.5]; got {_rounded(advantages.flatten())}")

    # Two environments, four steps, gamma = 0.9, lambda = 0.8. Environment 0 terminates at
    # step 1 (V of its last board, 0.7, must not be used); environment 1 is cut off by the time
    # limit at step 2 (V of its last board is 0.8). Rows t + 1 after an end belong to new episodes.
    two = dict(rewards=[[0.1, -0.01], [-1.0, 0.1], [0.1, -0.01], [-0.01, 0.1]],
               values=[[1.0, 2.0], [0.5, 1.5], [2.5, 1.0], [2.0, 3.0]], last_values=[1.5, 2.5],
               terminated=[[0, 0], [1, 0], [0, 0], [0, 0]], truncated=[[0, 0], [0, 0], [0, 1], [0, 0]],
               final_values=[[0, 0], [0.7, 0], [0, 0.8], [0, 0]])
    advantages, targets = _call_gae(gae, **two, gamma=0.9, lam=0.8)
    want = [[-1.53, -1.170336], [-1.5, -0.7088], [-1.0752, -0.29], [-0.66, -0.65]]
    if _close(advantages, want):
        expect(_close(targets, [[-0.53, 0.829664], [-1.0, 0.7912], [1.4248, 0.71], [1.34, 2.35]]),
               "the advantages are right, but the targets are not advantages + values")
        return
    hints = [
        ([[-1.53, -1.543584], [-1.5, -1.2272], [-1.0752, -1.01], [-0.66, -0.65]],
         "the time limit is treated as a termination: a truncated episode's last board still has "
         "a value, gamma * final_values (as in the lecture's A2C box)"),
        ([[-1.0764, -1.170336], [-0.87, -0.7088], [-1.0752, -0.29], [-0.66, -0.65]],
         "a *terminated* episode is bootstrapped from final_values: after being caught (or clearing "
         "the board) the future is worth nothing; final_values only matters where truncated"),
        ([[-0.467384, -0.386515], [-0.024144, 0.37984], [-1.0752, 1.222], [-0.66, -0.65]],
         "the episode ends are ignored: at d_t = 1, neither V(s_t+1) nor A_t+1 (both from the next "
         "episode) may be used"),
        ([[0.09, 0.229344], [0.75, 1.2352], [-1.0752, 2.41], [-0.66, -0.65]],
         "delta_t at the end of an episode uses V(s_t+1) of the *next* episode's first board: "
         "multiply it by (1 - d_t)"),
        ([[-2.087384, -1.412947], [-2.274144, -1.04576], [-1.0752, -0.758], [-0.66, -0.65]],
         "the recursion A_t = delta_t + gamma lambda A_t+1 runs on into the next episode: multiply "
         "A_t+1 by (1 - d_t)"),
        ([[-1.53, -0.143904], [-1.5, 0.7168], [-1.0752, 1.69], [-0.66, -0.65]],
         "the truncated episode is bootstrapped from values[t + 1], which is the first board of the "
         "*next* episode; use final_values (V of the last board before the reset)"),
    ]
    for wrong, hint in hints:
        if _close(advantages, wrong):
            raise CheckFailed(f"two environments with episode ends in the middle of the segment: {hint}")
    raise CheckFailed(f"two environments with episode ends in the middle of the segment: advantages "
                      f"{_rounded(advantages)}, expected {want}")


# --- ppo_loss ------------------------------------------------------------------------------------

#: A minibatch of five transitions with ratios rho = 2, 0.6, 1, 0.5, 1.5 (eps = 0.2):
#: A > 0 with rho above and below the clip range, A < 0 inside, below, and above it.
P_NEW, P_OLD = [0.5, 0.15, 0.4, 0.1, 0.3], [0.25, 0.25, 0.4, 0.2, 0.2]
ADVANTAGES = [1.0, 1.0, -1.0, -1.0, -1.0]
VALUES, TARGETS, ENTROPY = [0.5, 1.0, -0.5, 0.0, 2.0], [1.0, 0.5, 0.0, 0.5, 1.0], [1.2, 1.0, 0.8, 1.3, 0.5]


def _call_loss(ppo_loss: Callable, config: Any):
    log_probs = torch.tensor(np.log(P_NEW), dtype=torch.float32, requires_grad=True)
    old = torch.tensor(np.log(P_OLD), dtype=torch.float32, requires_grad=True)
    advantages = _t(ADVANTAGES).requires_grad_(True)
    values = _t(VALUES).requires_grad_(True)
    targets = _t(TARGETS).requires_grad_(True)
    entropy = _t(ENTROPY).requires_grad_(True)
    loss = ppo_loss(log_probs, old, advantages, values, targets, entropy, config)
    expect(isinstance(loss, torch.Tensor) and loss.dim() == 0,
           f"ppo_loss should return a scalar tensor, got {type(loss).__name__} "
           f"{tuple(getattr(loss, 'shape', ()))}")
    expect(loss.requires_grad, "the loss carries no gradient: did you compute it from .item() or "
                               "under torch.no_grad()?")
    loss.backward()
    grads = {name: (x.grad if x.grad is not None else torch.zeros_like(x))
             for name, x in [("log_probs", log_probs), ("old", old), ("advantages", advantages),
                             ("values", values), ("targets", targets), ("entropy", entropy)]}
    return float(loss.detach()), grads


def check_ppo_loss(ppo_loss: Callable, Config: type) -> None:
    config = replace(Config(), normalize_advantages=False, clip=True, clip_eps=0.2, value_coef=0.5,
                     entropy_coef=0.1, detach=True)
    loss, grads = _call_loss(ppo_loss, config)
    g = grads["log_probs"]
    want_g = [0.0, -0.12, 0.2, 0.0, 0.3]
    if _close(grads["values"], [-0.05, 0.05, -0.05, -0.05, 0.1]):
        raise CheckFailed("the value term is 0.5 * value_coef * mean (V - G)^2; the lecture's (and our) "
                          "loss has value_coef * mean (V - G)^2, without the 1/2")
    expect(_close(grads["values"], [-0.1, 0.1, -0.1, -0.1, 0.2]),
           f"the gradient of the loss with respect to the values is {_rounded(grads['values'])}; "
           "value_coef * mean (V - G)^2 gives [-0.1, 0.1, -0.1, -0.1, 0.2]")
    if _close(grads["entropy"], [0.02] * 5):
        raise CheckFailed("the entropy bonus has the wrong sign: the loss *subtracts* "
                          "entropy_coef * mean entropy, so that more randomness lowers the loss")
    expect(_close(grads["entropy"], [-0.02] * 5),
           f"the gradient with respect to the entropy is {_rounded(grads['entropy'])}; "
           "- entropy_coef * mean entropy gives -0.02 for each of the five")
    for name in ("old", "targets", "advantages"):
        expect(_close(grads[name], [0.0] * 5),
               f"gradient flows into the {'old log-probabilities' if name == 'old' else name}: they "
               "are constants (computed before the update); detach them")
    if not _close(g, want_g):
        hints = [([0.0, 0.12, -0.2, 0.0, -0.3], "the policy term has the wrong sign: the optimizer "
                  "minimizes, so the loss is *minus* the mean surrogate"),
                 ([-0.4, 0.0, 0.2, 0.1, 0.0], "that is the maximum of the two surrogates; PPO takes "
                  "the minimum (the pessimistic one)"),
                 ([0.0, 0.0, 0.2, 0.0, 0.0], "that uses only the clipped surrogate; PPO takes the "
                  "minimum of the clipped and the unclipped one"),
                 ([-0.4, -0.12, 0.2, 0.1, 0.3], "nothing is clipped, although config.clip is True"),
                 ([-0.2, -0.2, 0.0, 0.0, 0.0], "the ratio is the *exponential* of the difference of "
                  "the log-probabilities"),
                 ([0.1, 0.0, -0.2, -0.4, 0.0], "the ratio is upside down: rho = pi / pi_old, i.e. "
                  "exp(log_probs - old_log_probs)")]
        for wrong, hint in hints:
            if _close(g, wrong):
                raise CheckFailed(f"ratios rho = [2, 0.6, 1, 0.5, 1.5], advantages [1, 1, -1, -1, -1], "
                                  f"eps = 0.2: {hint}")
        raise CheckFailed(f"ratios rho = [2, 0.6, 1, 0.5, 1.5], advantages [1, 1, -1, -1, -1], eps = 0.2: "
                          f"the gradient with respect to log_probs is {_rounded(g)}, expected {want_g} "
                          "(zero exactly where the clip acts)")
    expect(abs(loss - 0.404) < 1e-4, f"the loss is {loss:.4f}, expected 0.404")

    _, grads = _call_loss(ppo_loss, replace(config, clip=False))
    expect(_close(grads["log_probs"], [-0.4, -0.12, 0.2, 0.1, 0.3]),
           f"with --no-clip, the policy term is the plain surrogate rho * A: the gradient with respect "
           f"to log_probs should be [-0.4, -0.12, 0.2, 0.1, 0.3]; got {_rounded(grads['log_probs'])}")

    _, grads = _call_loss(ppo_loss, replace(config, normalize_advantages=True))
    if _close(grads["log_probs"], want_g):
        raise CheckFailed("config.normalize_advantages is True, but the advantages were not standardized")
    if _close(grads["log_probs"], [0.0, -0.146969, 0.163299, 0.0, 0.244949]):
        raise CheckFailed("the advantages were divided by the *biased* standard deviation (numpy's "
                          "default); torch's .std() (unbiased) is what we use")
    expect(_close(grads["log_probs"], [0.0, -0.131453, 0.146059, 0.0, 0.219089]),
           f"with normalized advantages (A - mean) / (std + 1e-8), the gradient with respect to log_probs "
           f"should be [0, -0.1315, 0.1461, 0, 0.2191]; got {_rounded(grads['log_probs'])}")

    _, grads = _call_loss(ppo_loss, replace(config, detach=False))
    expect(not _close(grads["advantages"], [0.0] * 5),
           "with --no-detach (config.detach False), the advantages should *not* be detached, so that "
           "group B1 can see what happens")


# --- train ------------------------------------------------------------------------------------


def check_train(train: Callable, Config: type, A2CConfig: type) -> None:
    for label, config in [("PPO", replace(Config(), total_steps=4096, eval_every=2048)),
                          ("A2C", replace(A2CConfig(), total_steps=2000, eval_every=1000))]:
        started = time.perf_counter()
        result = train(config, 0)
        seconds = time.perf_counter() - started
        every = config.eval_every
        expect(result.steps == [every, 2 * every],
               f"{label}: evaluations at steps {result.steps}, expected {[every, 2 * every]}")
        for name in ("kl", "clip_frac", "entropy", "expl_var"):
            values = result.diagnostics.get(name, [])
            expect(len(values) == 2 and all(np.isfinite(values)),
                   f"{label}: expected the diagnostic {name!r} in each evaluation period, got {values}")
        expect(all(0 < e <= np.log(4) + 1e-4 for e in result.diagnostics["entropy"]),
               f"{label}: the entropy of a policy over 4 actions is between 0 and ln 4 = 1.386; "
               f"got {result.diagnostics['entropy']}")
        expect(all(k >= -1e-6 for k in result.diagnostics["kl"]),
               f"{label}: a KL divergence cannot be negative; got {result.diagnostics['kl']}")
        print(f"         ({label}: a run of {config.total_steps} steps took {seconds:.1f} s)")


# --- running them -----------------------------------------------------------------------------


def run_checks(*, Config: type, A2CConfig: type, gae: Callable, ppo_loss: Callable, train: Callable,
               script: str = "week09_ppo/ppo.py") -> bool:
    """Run every check and print OK / TODO / FAIL for each. Returns True if all passed."""
    checks: list[tuple[str, Callable[[], Any]]] = [
        ("gae", lambda: check_gae(gae)),
        ("ppo_loss", lambda: check_ppo_loss(ppo_loss, Config)),
        ("train (two short runs: PPO and A2C)", lambda: check_train(train, Config, A2CConfig)),
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
        print(f"All done: train with `uv run python {script} train` (PPO) and "
              f"`uv run python {script} train --preset a2c` (A2C).")
    return passed == len(checks)
