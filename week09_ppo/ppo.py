"""Actor-critic on PragMan: PPO (and A2C, as a preset): the file you edit this week.

PPO from the lecture (Week 9): N environments side by side, a segment of T steps from each,
advantages by GAE, then K epochs of minibatch steps on the clipped surrogate. A2C is the same
code with one epoch, one minibatch, lambda = 1, and no clipping (`--preset a2c`).

    advantages (GAE):  delta_t = r_t + gamma (1 - d_t) V(s_{t+1}) - V(s_t)
                       A_t     = delta_t + gamma lambda (1 - d_t) A_{t+1}       (backwards in t)
    critic targets:    G_t     = A_t + V(s_t)
    ratio:             rho_t   = pi(a_t | s_t) / pi_old(a_t | s_t)
    loss:              - mean of min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t)
                       + c_V mean of (V(s_t) - G_t)^2  -  beta mean entropy of pi(. | s_t)

Run everything from the repository root:

    uv run python week09_ppo/ppo.py check                  # checks each piece as soon as you write it
    uv run python week09_ppo/ppo.py train                  # PPO, 3 seeds side by side (~3 min)
    uv run python week09_ppo/ppo.py train --preset a2c     # A2C
    uv run python week09_ppo/ppo.py train --help           # every hyperparameter, with its default
    uv run python week09_ppo/ppo.py train --workers 1 --seeds 0   # one seed, in this process
    uv run python week09_ppo/ppo.py train --submit         # ... and send the result to the leaderboard
    uv run python week09_ppo/ppo.py watch week09_ppo/results/policies/ppo_seed0.pt
    uv run python week09_ppo/ppo.py score week09_ppo/results/policies/*_seed0.pt

Results are saved in week09_ppo/results/<name>.json (compare them with
`uv run python -m rlcourse.deeprl.plotting week09_ppo/results`), and the trained agents in
week09_ppo/results/policies/<name>_seed<k>.pt.

Fill in the two places marked todo(...): `gae` and `ppo_loss`. The rest is given. The shared
tools are in src/rlcourse/deeprl/ (the same world, network body, evaluation, and experiment
runner as for DQN and REINFORCE).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import nn

from rlcourse.deeprl.experiment import Monitor, RunResult, config_changes, default_name, run_experiment
from rlcourse.deeprl.networks import GridEncoder
from rlcourse.deeprl.pg_tools import load_agent, prepare_saving, save_agent, watch
from rlcourse.deeprl.ppo_checks import run_checks
from rlcourse.deeprl.ppo_tools import explained_variance, make_training_env, policy_change, score
from rlcourse.deeprl.world import make_env

WEEK_DIR = Path(__file__).resolve().parent
#: Group B2's reference policy: this file's PPO, trained for 100 000 steps (seed 0).
REFERENCE_POLICY = WEEK_DIR / "reference_policy.pt"


def todo(what: str):
    raise NotImplementedError(what)


# --- hyperparameters --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """PPO hyperparameters. Every field is also a command-line option of `train`."""

    total_steps: int = 300_000
    """Environment steps of training (all environments together). Leaderboard runs use the default."""
    n_envs: int = 16
    """N: environments stepped side by side."""
    n_steps: int = 64
    """T: steps per environment in each segment. A batch has N * T transitions."""
    epochs: int = 4
    """K: passes over each batch."""
    minibatches: int = 4
    """Each epoch splits the batch into this many minibatches: one gradient step each."""
    learning_rate: float = 2e-3
    """Adam step size."""
    gamma: float = 0.99
    """Discount factor."""
    gae_lambda: float = 0.95
    """GAE's lambda: 0 = one-step TD errors, 1 = returns to the end of the segment."""
    clip: bool = True
    """Clip the ratio (PPO); with --no-clip, the plain surrogate rho * A."""
    clip_eps: float = 0.2
    """epsilon: ratios outside [1 - eps, 1 + eps] get no further incentive."""
    value_coef: float = 0.5
    """c_V: weight of the critic's squared error."""
    entropy_coef: float = 0.01
    """beta: weight of the entropy bonus (as in Week 8)."""
    max_grad_norm: float = 0.5
    """Gradients with a larger norm are scaled down to this norm."""
    normalize_advantages: bool = True
    """Standardize the advantages in each minibatch (zero mean, unit standard deviation)."""
    detach: bool = True
    """Treat the advantages as constants in the loss (--no-detach: group B1)."""
    eval_every: int = 30_000
    """Evaluate the greedy policy every this-many environment steps (for the learning curve)."""
    preset: Literal["ppo", "a2c"] = "ppo"
    """Which defaults: PPO's, or A2C's (`train --preset a2c`). Set it only on the command line."""

    # --- group B1: classic bugs (the defaults are correct) ------------------------------------
    bug: Literal["none", "truncation_as_termination", "bootstrap_from_reset", "no_episode_cut"] = "none"
    """B1: an episode cut off by the time limit counts as terminated; or its value is taken
    from the first board of the *next* episode; or GAE runs across the ends of episodes."""

    # --- group B2: a hint in the reward, and a reference policy --------------------------------
    hint_reward: float = 0.0
    """B2: extra training reward for every step that ends next to a pellet (evaluation never has it)."""
    from_reference: bool = False
    """B2: start from the reference policy (week09_ppo/reference_policy.pt) instead of from scratch."""
    reference_coef: float = 0.0
    """B2: weight of a penalty on KL(pi || pi_ref), the divergence from the reference policy."""


@dataclass(frozen=True)
class A2CConfig(Config):
    """A2C: the same algorithm with one gradient step per segment (`train --preset a2c`).

    With one epoch and one minibatch, rho = 1 at the only step, so clipping never acts; with
    lambda = 1, GAE gives A2C's returns to the end of the segment. Short segments and no
    advantage normalization are A2C's own choices (tuned for it, so that the comparison is fair)."""

    n_steps: int = 5
    epochs: int = 1
    minibatches: int = 1
    gae_lambda: float = 1.0
    clip: bool = False
    normalize_advantages: bool = False
    preset: Literal["ppo", "a2c"] = "a2c"


PRESETS: dict[str, type[Config]] = {"ppo": Config, "a2c": A2CConfig}


# --- the two pieces you write -----------------------------------------------------------------


def gae(rewards: torch.Tensor, values: torch.Tensor, last_values: torch.Tensor,
        terminated: torch.Tensor, truncated: torch.Tensor, final_values: torch.Tensor,
        gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over one segment; returns (advantages, targets).

    All tensors have shape (T, N): step t of the segment in environment j is row t, column j,
    except last_values, of shape (N,). In environment j, at step t:

    - rewards[t, j] = r_t, the reward of the step from s_t;
    - values[t, j] = V(s_t), the critic's value of the board the agent acted on;
    - last_values[j] = V(s_T), the value of the board after the last step of the segment;
    - terminated[t, j] = 1.0 if the episode ended at this step (caught, or the board cleared);
    - truncated[t, j] = 1.0 if it was cut off by the time limit at this step;
    - final_values[t, j] = where an episode ended at this step (either way), V of its last
      board, the one *before* the reset; 0 elsewhere.

    If an episode ended at step t, row t + 1 belongs to the *next* episode: neither its value
    nor its advantage may flow back into step t. What follows the end instead: if terminated,
    nothing (the future is worth 0); if truncated, gamma * V(last board), as in the lecture's A2C
    algorithm box (fold it into the reward, then cut the recursion as for a termination).

    Compute it backwards, t = T-1, ..., 0, as the recursion at the top of this file (eq.
    gae-recursion in the notes), with d_t = 1 where the episode ended at step t. Targets:
    advantages + values. Both are plain numbers in the loss (the values came from
    torch.no_grad(), so nothing here needs a gradient)."""
    return todo("the backward recursion over the segment: advantages and targets")


def ppo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor,
             values: torch.Tensor, targets: torch.Tensor, entropy: torch.Tensor,
             config: Config) -> torch.Tensor:
    """The PPO loss of one minibatch (the formula at the top of this file), a scalar tensor.

    All arguments are 1-D tensors over the minibatch's transitions:
    - log_probs: log pi(a_t | s_t) of the current network (with gradient);
    - old_log_probs: log pi_old(a_t | s_t), stored when the segment was collected;
    - advantages and targets: from `gae`;
    - values: V(s_t) of the current network (with gradient);
    - entropy: the entropy of pi(. | s_t) of the current network (with gradient).

    Steps:
    - The advantages, the targets, and the old log-probabilities are constants: detach them,
      the advantages only if config.detach (group B1's switch).
    - If config.normalize_advantages: standardize the advantages (mean 0, standard deviation 1;
      add 1e-8 to the standard deviation).
    - rho_t = exp(log_probs - old_log_probs); the surrogate rho_t A_t; if config.clip, the
      minimum of it and the clipped surrogate, with config.clip_eps.
    - The loss: minus the mean surrogate, plus config.value_coef times the mean squared error of
      the values, minus config.entropy_coef times the mean entropy."""
    return todo("the clipped surrogate, the value loss, and the entropy bonus")


# --- the network ------------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """The actor pi(. | s) and the critic V(s) with one shared body: a GridEncoder (as in Weeks
    7-8) reads the board, then a linear policy head (four logits) and a linear value head.

    The policy head starts with tiny weights, so that the first policy is close to uniform."""

    def __init__(self, observation_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.encoder = GridEncoder(observation_shape)
        self.policy_head = nn.Linear(self.encoder.n_features, n_actions)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        self.value_head = nn.Linear(self.encoder.n_features, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, H, W, C) boards -> (B, n_actions) logits and (B,) values."""
        features = self.encoder(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class PPOAgent:
    """The network, its optimizer, and how to act with it."""

    def __init__(self, observation_shape: tuple[int, ...], n_actions: int, config: Config):
        self.config = config
        self.network = ActorCritic(observation_shape, n_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, eps=1e-5)

    @torch.no_grad()
    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        """pi(. | s) of a single board, as a float64 array (for `watch`)."""
        logits, _ = self.network(torch.from_numpy(observation).unsqueeze(0))
        probs = torch.softmax(logits[0].double(), dim=0).numpy()
        return probs / probs.sum()

    def act(self, observation: np.ndarray, rng: np.random.Generator) -> int:
        """A sampled action for a single board (for `watch --sampled`)."""
        return int(rng.choice(len(probs := self.probabilities(observation)), p=probs))

    def greedy_action(self, observation: np.ndarray) -> int:
        """The most likely action: what evaluation plays (as in Weeks 7-8)."""
        return int(self.probabilities(observation).argmax())


# --- collecting a segment ---------------------------------------------------------------------


class Segment(NamedTuple):
    """T steps of N environments, as `collect` stored them. Shapes (T, N) unless noted."""

    observations: torch.Tensor   # (T, N, H, W, C) uint8: s_t, the boards acted on
    actions: torch.Tensor        # int64: a_t
    log_probs: torch.Tensor      # log pi_old(a_t | s_t)
    values: torch.Tensor         # V(s_t)
    rewards: torch.Tensor        # r_t
    terminated: torch.Tensor     # 1.0 where the episode terminated at this step
    truncated: torch.Tensor      # 1.0 where it was cut off by the time limit at this step
    final_values: torch.Tensor   # where an episode ended: V of its last board (before the reset)
    last_values: torch.Tensor    # (N,): V(s_T), after the segment's last step


class Environments:
    """N copies of the training world, each with its own random seed, and the running state
    between segments: the current boards and the returns of the episodes in progress."""

    def __init__(self, config: Config, seed: int):
        self.envs = [make_training_env(config.hint_reward) for _ in range(config.n_envs)]
        seeds = np.random.SeedSequence(seed).generate_state(config.n_envs)
        self.observations = np.stack([env.reset(seed=int(s))[0] for env, s in zip(self.envs, seeds)])
        self.returns = np.zeros(config.n_envs)
        self.true_returns = np.zeros(config.n_envs)


@torch.no_grad()
def collect(envs: Environments, agent: PPOAgent, config: Config, monitor: Monitor, step: int) -> tuple[Segment, int]:
    """Play T steps in each of the N environments with the current policy (sampling actions),
    resetting an environment as soon as its episode ends. Returns the segment and the new step
    count. The Monitor gets every finished episode and evaluates when it is due."""
    T, N = config.n_steps, config.n_envs
    observations = torch.empty((T, N, *envs.observations.shape[1:]), dtype=torch.uint8)
    actions = torch.empty((T, N), dtype=torch.int64)
    log_probs, values, rewards, terminated, truncated, final_values = (torch.zeros(T, N) for _ in range(6))
    for t in range(T):
        observations[t] = torch.from_numpy(envs.observations)
        logits, values[t] = agent.network(observations[t])
        policy = torch.distributions.Categorical(logits=logits)
        actions[t] = policy.sample()
        log_probs[t] = policy.log_prob(actions[t])
        for j, env in enumerate(envs.envs):
            observation, reward, term, trunc, info = env.step(int(actions[t, j]))
            rewards[t, j], terminated[t, j], truncated[t, j] = reward, term, trunc
            envs.returns[j] += reward
            envs.true_returns[j] += info.get("true_reward", reward)
            if term or trunc:
                monitor.record_episode(envs.returns[j])
                if config.hint_reward:
                    monitor.log(true_return=envs.true_returns[j])
                envs.returns[j] = envs.true_returns[j] = 0.0
                last_board = observation
                observation, _ = env.reset()
                if config.bug == "bootstrap_from_reset":     # B1: the next episode's first board
                    last_board = observation
                final_values[t, j] = agent.network(torch.from_numpy(last_board).unsqueeze(0))[1][0]
            envs.observations[j] = observation
            step += 1
            if step <= config.total_steps:
                monitor.evaluate_if_due(step, agent.greedy_action)
    last_values = agent.network(torch.from_numpy(envs.observations))[1]
    return Segment(observations, actions, log_probs, values, rewards, terminated, truncated,
                   final_values, last_values), step


# --- the update ---------------------------------------------------------------------------------


def update(agent: PPOAgent, segment: Segment, config: Config, reference: PPOAgent | None) -> dict:
    """GAE once, then K epochs of minibatch steps on ppo_loss. Returns the diagnostics."""
    terminated, truncated = segment.terminated, segment.truncated
    if config.bug == "truncation_as_termination":      # B1
        terminated, truncated = torch.maximum(terminated, truncated), torch.zeros_like(truncated)
    elif config.bug == "no_episode_cut":                # B1
        terminated, truncated = torch.zeros_like(terminated), torch.zeros_like(truncated)
    advantages, targets = gae(segment.rewards, segment.values, segment.last_values, terminated,
                              truncated, segment.final_values, config.gamma, config.gae_lambda)

    # From here on, the T x N transitions are one batch: flatten the first two dimensions.
    n = config.n_steps * config.n_envs
    observations = segment.observations.flatten(0, 1)
    actions, old_log_probs, old_values = (x.flatten() for x in (segment.actions, segment.log_probs, segment.values))
    advantages, targets = advantages.flatten(), targets.flatten()
    if reference is not None:
        with torch.no_grad():
            reference_log_probs = F.log_softmax(reference.network(observations)[0], dim=1)

    entropies = []
    for _ in range(config.epochs):
        for batch in torch.randperm(n).split(n // config.minibatches):
            logits, values = agent.network(observations[batch])
            all_log_probs = F.log_softmax(logits, dim=1)
            log_probs = all_log_probs.gather(1, actions[batch].unsqueeze(1)).squeeze(1)
            entropy = -(all_log_probs.exp() * all_log_probs).sum(1)
            batch_advantages = advantages[batch]
            if not config.detach:
                # B1: the gradient that A_t = G_t - V(s_t) would carry if V(s_t) came from this
                # forward pass (as in A2C implementations); the numbers stay the same.
                batch_advantages = batch_advantages - (values - values.detach())
            loss = ppo_loss(log_probs, old_log_probs[batch], batch_advantages, values, targets[batch],
                            entropy, config)
            if reference is not None and config.reference_coef:
                # B2: KL(pi || pi_ref) in the visited states, averaged over the minibatch.
                kl_to_reference = (all_log_probs.exp() * (all_log_probs - reference_log_probs[batch])).sum(1).mean()
                loss = loss + config.reference_coef * kl_to_reference
            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.network.parameters(), config.max_grad_norm)
            agent.optimizer.step()
            entropies.append(float(entropy.mean().detach()))

    # How far did this update move the policy? And how good was the critic before it?
    with torch.no_grad():
        new_log_probs = F.log_softmax(agent.network(observations)[0], dim=1)
        diagnostics = policy_change(new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1),
                                    old_log_probs, config.clip_eps)
        diagnostics["entropy"] = float(np.mean(entropies))
        diagnostics["expl_var"] = explained_variance(old_values, targets)
        if reference is not None:
            diagnostics["kl_ref"] = float((new_log_probs.exp() * (new_log_probs - reference_log_probs)).sum(1).mean())
    return diagnostics


# --- the training loop ------------------------------------------------------------------------


def make_agent(config: Config) -> PPOAgent:
    env = make_env()
    return PPOAgent(env.observation_space.shape, int(env.action_space.n), config)


def load_reference() -> PPOAgent:
    return load_agent(REFERENCE_POLICY, make_agent, Config)


def train(config: Config, seed: int) -> RunResult:
    """Train one agent from scratch (or from the reference policy); return what the Monitor recorded."""
    agent_seed, env_seed = np.random.SeedSequence(seed).generate_state(2)
    torch.manual_seed(int(agent_seed))
    agent = make_agent(config)
    reference = None
    if config.from_reference or config.reference_coef:
        reference = load_reference()
        if config.from_reference:
            agent.network.load_state_dict(reference.network.state_dict())
    envs = Environments(config, int(env_seed))
    monitor = Monitor(config.total_steps, seed, eval_every=config.eval_every, gamma=config.gamma)

    step = 0
    while step < config.total_steps:
        segment, step = collect(envs, agent, config, monitor, step)
        monitor.log(**update(agent, segment, config, reference))

    save_agent(agent, config, seed)
    return monitor.finish(agent.greedy_action)


# --- command line -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Train:
    """Train with several seeds in parallel, save the results, and optionally submit them."""

    config: Annotated[Config, tyro.conf.OmitArgPrefixes] = field(default_factory=Config)
    seeds: tuple[int, ...] = (0, 1, 2)
    """One run per seed. Leaderboard runs use the default."""
    workers: int | None = None
    """Processes; default: one per seed (as many as your CPU allows). 1: no extra processes."""
    name: str | None = None
    """Name of the results file; default: 'ppo' or 'a2c', plus the options you changed."""
    submit: bool = False
    """Send the result (and this file) to the leaderboard."""
    nickname: str | None = None
    """Shown on the leaderboard next to your group."""


@dataclass(frozen=True)
class TrainA2C(Train):
    """Train A2C (`train --preset a2c`) with several seeds in parallel."""

    config: Annotated[A2CConfig, tyro.conf.OmitArgPrefixes] = field(default_factory=A2CConfig)


@dataclass(frozen=True)
class Check:
    """Check your implementation, piece by piece."""


@dataclass(frozen=True)
class Watch:
    """Watch a trained policy play (or, with --quiet, only count what happens)."""

    policy: tyro.conf.Positional[Path]
    """A saved agent, e.g. week09_ppo/results/policies/ppo_seed0.pt."""
    episodes: int = 1
    sampled: bool = False
    """Sample the actions from pi (as in training) instead of taking the most likely one."""
    delay: float = 0.15
    """Seconds between frames."""
    quiet: bool = False
    """Show only the summary (e.g. with --episodes 100)."""


@dataclass(frozen=True)
class Score:
    """What do trained policies do (sampled, as in training), and how good are their critics?"""

    policies: tyro.conf.Positional[tuple[Path, ...]]
    """Saved agents, e.g. week09_ppo/results/policies/*_seed0.pt."""
    episodes: int = 100
    greedy: bool = False
    """Play the most likely actions instead of sampling."""
    hint_reward: float = 0.0
    """B2: also report the return with this hint reward."""
    reference: bool = False
    """B2: also report the KL divergence from the reference policy (and score the reference itself)."""


def preset_from_argv(argv: list[str]) -> str:
    """`train --preset a2c` chooses which defaults the options start from, so it has to be
    known before the command line is parsed."""
    for k, arg in enumerate(argv):
        if arg == "--preset" and k + 1 < len(argv):
            return argv[k + 1]
        if arg.startswith("--preset="):
            return arg.split("=", 1)[1]
    return "ppo"


def run_name(config: Config) -> str:
    """'ppo' or 'a2c', plus the options changed from that preset's defaults."""
    changes = {k: v for k, v in config_changes(config).items() if k != "preset"}
    return config.preset if not changes else f"{config.preset}+{default_name(changes)}"


def main() -> None:
    train_command = TrainA2C if preset_from_argv(sys.argv) == "a2c" else Train
    command = tyro.extras.subcommand_cli_from_dict(
        {"train": train_command, "check": Check, "watch": Watch, "score": Score},
        description=__doc__.splitlines()[0],
    )
    if isinstance(command, Check):
        run_checks(Config=Config, A2CConfig=A2CConfig, gae=gae, ppo_loss=ppo_loss, train=train)
    elif isinstance(command, Watch):
        agent = load_agent(command.policy, make_agent, Config)
        watch(agent, episodes=command.episodes, sampled=command.sampled, delay=command.delay,
              quiet=command.quiet)
    elif isinstance(command, Score):
        agents = [(path.stem, load_agent(path, make_agent, Config)) for path in command.policies]
        reference = load_reference() if command.reference else None
        if reference is not None:
            agents.insert(0, ("(reference)", reference))
        score(agents, reference=reference, hint_reward=command.hint_reward, episodes=command.episodes,
              sampled=not command.greedy)
    else:
        name = command.name or run_name(command.config)
        prepare_saving(WEEK_DIR / "results" / "policies", name)
        run_experiment(
            train, command.config, seeds=command.seeds, workers=command.workers, name=name,
            results_dir=WEEK_DIR / "results", submit=command.submit, week_dir=WEEK_DIR,
            nickname=command.nickname, files=[Path(__file__)], leaderboard_kind="ppo",
        )
        policies = WEEK_DIR / "results" / "policies"
        print(f"Agents saved as {policies / name}_seed<k>.pt, e.g. for `ppo.py watch` or `ppo.py score`.")


if __name__ == "__main__":
    main()
