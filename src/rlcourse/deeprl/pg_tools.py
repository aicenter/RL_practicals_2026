"""Tools for the Week 8 policy-gradient script: saving and loading agents, watching a policy
play, and measuring how noisy a policy-gradient estimate is (`pg.py watch` and `pg.py variance`).

The student script passes its own pieces in (its agent, `run_episode`, `returns_to_go`), so
nothing here imports the script.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rlcourse.deeprl.experiment import config_changes, default_name
from rlcourse.deeprl.world import make_env

#: `train` sets these (in the main process; the worker processes inherit them), so that each
#: worker can save its trained agent; a bare call of `train` (e.g. from `check`) saves nothing.
POLICY_DIR_VAR = "RLCOURSE_POLICY_DIR"
POLICY_NAME_VAR = "RLCOURSE_POLICY_NAME"

ACTION_NAMES = ("up", "right", "down", "left")


# --- saving and loading -------------------------------------------------------------------------


def _modules(agent: Any) -> dict[str, nn.Module]:
    """Every network the agent holds as an attribute (policy_network, value_network, ...)."""
    return {name: value for name, value in vars(agent).items() if isinstance(value, nn.Module)}


def prepare_saving(directory: Path, name: str) -> None:
    """Called by `train` before the runs start: where (and under which name) to save agents."""
    os.environ[POLICY_DIR_VAR] = str(directory)
    os.environ[POLICY_NAME_VAR] = name


def save_agent(agent: Any, config: Any, seed: int) -> Path | None:
    """Save the agent's networks and config to <directory>/<name>_seed<seed>.pt, if `train`
    asked for it (see prepare_saving); otherwise do nothing and return None."""
    directory = os.environ.get(POLICY_DIR_VAR)
    if not directory:
        return None
    name = os.environ.get(POLICY_NAME_VAR) or default_name(config_changes(config))
    path = Path(directory) / f"{name}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config),
                "networks": {k: m.state_dict() for k, m in _modules(agent).items()}}, path)
    return path


def load_agent(path: Path, make_agent: Callable[[Any], Any], config_type: type) -> Any:
    """Rebuild a saved agent: `make_agent(config)` creates it, then the saved weights go in.
    Config fields that no longer exist are ignored; new ones take their defaults."""
    path = Path(path)
    if not path.exists():
        here = sorted(p.name for p in path.parent.glob("*.pt")) if path.parent.is_dir() else []
        raise SystemExit(f"No saved policy {path}." + (f" In {path.parent}: {', '.join(here)}" if here else
                         " Policies are saved by `train`, as results/policies/<name>_seed<k>.pt."))
    saved = torch.load(path, weights_only=False)
    known = {f.name for f in fields(config_type)}
    config = config_type(**{k: v for k, v in saved["config"].items() if k in known})
    agent = make_agent(config)
    modules = _modules(agent)
    for name, state in saved["networks"].items():
        if name in modules:
            modules[name].load_state_dict(state)
    return agent


# --- watching a policy ------------------------------------------------------------------------


def board_text(observation: np.ndarray) -> str:
    """A board as plain text: # wall, P agent, G ghost, X both, o pellet."""
    walls, agent, ghosts, pellets = (observation[..., k] for k in range(4))
    rows = []
    for i in range(observation.shape[0]):
        row = ""
        for j in range(observation.shape[1]):
            if agent[i, j] and ghosts[i, j]:
                row += "X"
            elif agent[i, j]:
                row += "P"
            elif ghosts[i, j]:
                row += "G"
            elif walls[i, j]:
                row += "#"
            elif pellets[i, j]:
                row += "o"
            else:
                row += " "
        rows.append(row)
    return "\n".join(rows)


def watch(agent: Any, *, episodes: int = 1, sampled: bool = False, delay: float = 0.15,
          quiet: bool = False, seed: int = 0, **world: Any) -> None:
    """Play `episodes` episodes with the agent's greedy (or sampled) policy and show them.

    `world` overrides the standard world (e.g. ghost_policy="chaser"); the episodes are
    reproducible (environment seeds 2_000_000 + k). With quiet=True only the summary is printed."""
    torch.set_num_threads(1)     # one core is plenty, and fast even while training runs
    env = make_env(**world)
    rng = np.random.default_rng(seed)
    totals, pellets, caught, cleared, lengths = [], [], [], [], []
    for k in range(episodes):
        observation, info = env.reset(seed=2_000_000 + k)
        start, total, t = info["pellets_left"], 0.0, 0
        while True:
            if sampled:
                action = agent.act(observation, rng)
            else:
                action = agent.greedy_action(observation)
            if not quiet:
                probs = agent.probabilities(observation)
                spread = "  ".join(f"{n} {p:.2f}" for n, p in zip(ACTION_NAMES, probs))
                print("\x1b[2J\x1b[H" if delay > 0 else "", end="")
                print(f"episode {k + 1}/{episodes}, step {t}, return so far {total:+.2f}\n"
                      f"{board_text(observation)}\npi: {spread}  ->  {ACTION_NAMES[action]}", flush=True)
                if delay > 0:
                    time.sleep(delay)
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            t += 1
            if terminated or truncated:
                break
        totals.append(total)
        pellets.append(start - info["pellets_left"])
        caught.append(bool(info["death"]))
        cleared.append(info["pellets_left"] == 0)
        lengths.append(t)
        if not quiet:
            ending = "caught" if info["death"] else ("cleared the board" if info["pellets_left"] == 0
                                                     else "out of time")
            print(f"{board_text(observation)}\n--> return {total:+.2f}, {pellets[-1]} pellets, "
                  f"{t} steps, {ending}\n")
            if delay > 0 and k + 1 < episodes:
                time.sleep(1.0)
    world_text = ", ".join(f"{k}={v}" for k, v in world.items()) or "the standard world"
    print(f"{episodes} episode(s) in {world_text}, {'sampled' if sampled else 'greedy'} actions: "
          f"return {np.mean(totals):.2f}, pellets {np.mean(pellets):.1f}, caught {np.mean(caught):.0%}, "
          f"cleared {np.mean(cleared):.0%}, steps {np.mean(lengths):.1f}")


# --- how noisy is a gradient estimate? --------------------------------------------------------


def expected_cosine(estimates: np.ndarray) -> tuple[float, float]:
    """How well one estimate points in the direction of the true gradient g, from M independent
    unbiased estimates x_1..x_M (rows), without knowing g: cos ~ |g| / sqrt(E|x|^2), where
    |g|^2 is estimated without bias by the mean of x_i . x_j over pairs i != j. (This is the
    root-mean-square cosine; 1 = no noise, 0 = all noise.) Returns it and a jackknife
    standard error."""
    m = len(estimates)
    total = estimates.sum(0)
    norms = (estimates ** 2).sum(1)

    def cosine(s: np.ndarray, sum_norms: float, k: int) -> float:
        g2 = (s @ s - sum_norms) / (k * (k - 1))
        return float(np.sqrt(max(g2, 0.0) / (sum_norms / k)))

    full = cosine(total, norms.sum(), m)
    leave_out = np.array([cosine(total - estimates[i], norms.sum() - norms[i], m - 1) for i in range(m)])
    stderr = float(np.sqrt((m - 1) / m * ((leave_out - leave_out.mean()) ** 2).sum()))
    return full, stderr


def gradient_variance(agent: Any, run_episode: Callable, returns_to_go: Callable, *,
                      batches: int = 40, episodes_per_batch: int = 8, seed: int = 0) -> None:
    """Freeze the agent's policy, collect `batches` batches of episodes, and compute on every
    batch the gradient estimate of several estimators *on the same episodes*. Prints, per
    estimator, how much the estimates vary from batch to batch, and how well a single batch's
    estimate points in the direction of the true gradient (see expected_cosine)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = make_env()
    env.reset(seed=3_000_000 + seed)
    rng = np.random.default_rng(seed)
    gamma = agent.config.gamma
    params = [p for p in agent.policy_network.parameters() if p.requires_grad]

    started = time.perf_counter()
    collected = [[run_episode(env, agent, rng) for _ in range(episodes_per_batch)]
                 for _ in range(batches)]
    changes = {k: v for k, v in config_changes(agent.config).items() if k not in ("total_steps", "eval_every")}
    yours = "yours: " + default_name(changes, "core")
    yours = yours if len(yours) <= 40 else yours[:37] + "..."
    estimators: dict[str, Callable[[list], np.ndarray]] = {
        "R(tau): the whole return": lambda eps: np.concatenate(
            [np.full(len(e.rewards), returns_to_go(e.rewards, gamma)[0]) for e in eps]),
        "G_t: return-to-go": lambda eps: np.concatenate([returns_to_go(e.rewards, gamma) for e in eps]),
        "G_t - batch mean": lambda eps: (lambda g: g - g.mean())(
            np.concatenate([returns_to_go(e.rewards, gamma) for e in eps])),
        yours: lambda eps: np.asarray(torch.as_tensor(agent.weights(eps)).detach().cpu(), dtype=np.float64),
    }
    grads: dict[str, list[np.ndarray]] = {name: [] for name in estimators}
    for eps in collected:
        observations = torch.from_numpy(np.concatenate([e.observations for e in eps]))
        actions = torch.from_numpy(np.concatenate([e.actions for e in eps]).astype(np.int64))
        log_probs = F.log_softmax(agent.policy_network(observations), dim=1)
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        for name, weights_of in estimators.items():
            w = torch.as_tensor(np.asarray(weights_of(eps), dtype=np.float32))
            g = torch.autograd.grad(-(chosen * w).mean(), params, retain_graph=True)
            grads[name].append(-torch.cat([x.flatten() for x in g]).double().numpy())
    seconds = time.perf_counter() - started

    G = {name: np.stack(v) for name, v in grads.items()}
    reference = G["G_t: return-to-go"].var(0, ddof=1).sum()
    returns = [float(np.sum(e.rewards)) for eps in collected for e in eps]
    lengths = [len(e.rewards) for eps in collected for e in eps]
    print(f"{batches} batches of {episodes_per_batch} episodes ({seconds:.0f} s); mean return "
          f"{np.mean(returns):.2f}, mean episode length {np.mean(lengths):.0f} steps.\n")
    print("The same episodes for every estimator. 'variance': the total variance of one batch's gradient\n"
          "estimate (summed over all parameters), and relative to that of plain return-to-go. 'cosine':\n"
          "between one batch's estimate and the true gradient (estimated from how much the batches agree\n"
          "with each other), ± standard error: 1 = always the right direction, 0 = all noise; '~0' means\n"
          "that the batches do not agree with each other measurably (more --batches would tell).\n")
    print(f"{'estimator':42s} {'variance':>9s} {'relative':>9s} {'cosine':>13s}")
    for name, g in G.items():
        var = g.var(0, ddof=1).sum()
        cos, se = expected_cosine(g)
        cos_text = f"{cos:7.2f} ± {se:.2f}" if cos > 0 else "  ~0 (noise)"
        print(f"{name:42s} {var:9.3g} {var / reference:9.2f} {cos_text:>13s}")
    if default_name(changes, "core") == "core":
        print("\n(For the core, 'yours' is the same estimator as 'G_t - batch mean'.)")
