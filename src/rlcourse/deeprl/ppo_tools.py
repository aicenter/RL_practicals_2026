"""Tools for the Week 9 actor-critic script: the diagnostics of an update, the training world
of group B2, and `score` (what a trained policy does, and how good its critic is).

The student script passes its own pieces in (its agents), so nothing here imports the script.
Saving, loading, and `watch` are the Week 8 ones (`rlcourse.deeprl.pg_tools`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import pragman
from rlcourse.deeprl.world import make_env

#: The episodes `score` plays (environment seeds SCORE_SEED + k; neither training nor evaluation
#: uses them).
SCORE_SEED = 4_000_000


# --- the diagnostics of an update -------------------------------------------------------------


def explained_variance(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(target - predicted) / Var(target): 1 = the predictions explain the targets
    perfectly, 0 = no better than predicting their mean, < 0 = worse than that."""
    return float(1 - (target - predicted).var() / (target.var() + 1e-8))


@torch.no_grad()
def policy_change(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, clip_eps: float) -> dict:
    """How far an update moved the policy, measured on the actions of the batch:

    - kl: the lecture's estimate of KL(pi_old || pi_new), the mean of (rho - 1) - log rho;
    - clip_frac: the fraction of the ratios rho outside [1 - eps, 1 + eps];
    - ratio_max: the largest ratio rho = pi_new(a|s) / pi_old(a|s) in the batch."""
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    return {"kl": float((ratio - 1 - log_ratio).mean()),
            "clip_frac": float(((ratio - 1).abs() > clip_eps).float().mean()),
            "ratio_max": float(ratio.max())}


# --- group B2: an extra reward in the training world ----------------------------------------------


def next_to_pellet(state: pragman.PragmanState) -> bool:
    """Is there a pellet on a cell next to the agent (up, right, down, or left)?"""
    row, col = state.agent
    return any((row + dr, col + dc) in state.pellets for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)))


def make_training_env(hint_reward: float = 0.0):
    """The world we *train* in: the course world, plus (if hint_reward is not 0) an extra
    reward of `hint_reward` for every step that ends next to a pellet, a hint that good
    positions are near food. Evaluation always uses the standard world, without the hint.
    The world's own reward stays available in info["true_reward"]."""
    env = make_env()
    if not hint_reward:
        return env

    def hinted(state, action, next_state, info) -> float:
        return info["true_reward"] + hint_reward * next_to_pellet(next_state)

    return pragman.ReplaceReward(env, hinted)


# --- score: what does a trained policy do? ---------------------------------------------------------


@torch.no_grad()
def _score_one(agent: Any, reference: Any | None, hint_reward: float, episodes: int, sampled: bool) -> dict:
    env = make_training_env()
    # What its critic learned to predict: returns with its own discount (and hint, if any).
    gamma = float(agent.config.gamma)
    trained_hint = float(getattr(agent.config, "hint_reward", 0.0))
    rng = np.random.default_rng(0)
    rows = []
    predicted, realised = [], []
    for k in range(episodes):
        observation, info = env.reset(seed=SCORE_SEED + k)
        at_start = info["pellets_left"]
        true, hinted, kls, values, rewards = 0.0, 0.0, [], [], []
        while True:
            logits, value = agent.network(torch.from_numpy(observation).unsqueeze(0))
            log_p = F.log_softmax(logits[0].double(), dim=0)
            p = log_p.exp().numpy()
            action = int(rng.choice(len(p), p=p / p.sum())) if sampled else int(p.argmax())
            if reference is not None:
                ref_logits, _ = reference.network(torch.from_numpy(observation).unsqueeze(0))
                log_q = F.log_softmax(ref_logits[0].double(), dim=0)
                kls.append(float((log_p.exp() * (log_p - log_q)).sum()))
            values.append(float(value[0]))
            observation, reward, terminated, truncated, info = env.step(action)
            near = next_to_pellet(info["state"])
            rewards.append(float(reward) + trained_hint * near)
            hinted += float(reward) + hint_reward * near
            true += float(reward)
            if terminated or truncated:
                break
        # The critic against what actually followed (in the rewards it was trained on, bootstrapped
        # at the time limit, as in training).
        following = 0.0 if terminated else float(agent.network(torch.from_numpy(observation).unsqueeze(0))[1][0])
        returns = []
        for reward in reversed(rewards):
            following = reward + gamma * following
            returns.append(following)
        predicted += values
        realised += returns[::-1]
        rows.append((true, hinted, np.mean(kls) if kls else np.nan, at_start - info["pellets_left"],
                     bool(info["death"]), len(rewards), values[0], returns[-1]))
    true, hinted, kl, pellets, deaths, steps, v0, g0 = map(np.array, zip(*rows))
    error = np.array(realised) - np.array(predicted)
    return {"return": true.mean(), "hinted": hinted.mean(), "kl_ref": np.nanmean(kl) if reference else np.nan,
            "pellets": pellets.mean(), "deaths": deaths.mean(), "steps": steps.mean(),
            "critic_ev": 1 - error.var() / (np.var(realised) + 1e-8), "v0": v0.mean(), "g0": g0.mean()}


def score(agents: Sequence[tuple[str, Any]], *, reference: Any | None = None, hint_reward: float = 0.0,
          episodes: int = 100, sampled: bool = True) -> list[dict]:
    """Play `episodes` episodes with each agent (sampled actions, as in training, unless
    sampled=False) and print one row per agent: the return (standard rewards), the return with
    the hint (if hint_reward), the KL divergence from the reference policy (if given; per step,
    averaged over the states the agent visits), what happened in the episodes, and the critic
    against the discounted returns that actually followed (explained variance; V(s0) vs G0;
    each with the agent's own discount; for an agent trained with a hint, the returns include
    its hint)."""
    torch.set_num_threads(1)
    rows = [{"name": name, **_score_one(agent, reference, hint_reward, episodes, sampled)}
            for name, agent in agents]
    width = max([len(r["name"]) for r in rows] + [6])
    header = f"{'policy':{width}s} {'return':>7s}"
    if hint_reward:
        header += f" {'hinted':>7s}"
    if reference is not None:
        header += f" {'KL ref':>7s}"
    header += f" {'pellets':>7s} {'caught':>6s} {'steps':>5s} {'critic EV':>9s} {'V(s0)':>6s} {'G0':>6s}"
    print(f"{episodes} episodes each, {'sampled' if sampled else 'greedy'} actions, standard world"
          + (f" (hinted: with the hint reward {hint_reward:g})" if hint_reward else "") + ".")
    print("critic EV: explained variance of V(s_t) for the discounted returns that actually followed;\n"
          "V(s0) vs G0: the critic's value of the start, and the discounted return that followed it\n"
          "(both with the policy's own gamma).\n")
    print(header)
    for r in rows:
        line = f"{r['name']:{width}s} {r['return']:7.2f}"
        if hint_reward:
            line += f" {r['hinted']:7.2f}"
        if reference is not None:
            line += f" {r['kl_ref']:7.3f}"
        line += (f" {r['pellets']:7.1f} {r['deaths']:6.0%} {r['steps']:5.0f} {r['critic_ev']:9.2f}"
                 f" {r['v0']:6.2f} {r['g0']:6.2f}")
        print(line)
    return rows
