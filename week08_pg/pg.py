"""Policy gradients (REINFORCE) on PragMan: the file you edit this week.

REINFORCE with the two improvements from the lecture (Week 8): the return-to-go and a baseline.
For every step t of every episode in a batch, the action taken is weighted by how good what
followed it was, compared with what we expected:

    return-to-go:   G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ... (to the end of the episode)
    baseline:       b   = the mean of G_t over all steps of the batch (the simplest choice)
    loss:           L(theta) = - mean over all steps of the batch of
                               log pi(a_t | s_t; theta) * (G_t - b)  -  entropy bonus

Gradient descent on L is gradient *ascent* on the expected return. The weights G_t - b are plain
numbers: no gradient flows through them, only through log pi.

Run everything from the repository root:

    uv run python week08_pg/pg.py check              # checks each piece as soon as you write it
    uv run python week08_pg/pg.py train              # 3 seeds side by side (~5 min)
    uv run python week08_pg/pg.py train --help       # every hyperparameter, with its default
    uv run python week08_pg/pg.py train --workers 1 --seeds 0   # one seed, in this process
    uv run python week08_pg/pg.py train --submit     # ... and send the result to the leaderboard
    uv run python week08_pg/pg.py watch week08_pg/results/policies/core_seed0.pt
    uv run python week08_pg/pg.py variance week08_pg/results/policies/core_seed0.pt

Results are saved in week08_pg/results/<name>.json (compare them with
`uv run python -m rlcourse.deeprl.plotting week08_pg/results`), and the trained policies in
week08_pg/results/policies/<name>_seed<k>.pt.

Fill in the places marked todo(...). The shared tools are in src/rlcourse/deeprl/ (the same
world, network body, evaluation, and experiment runner as for DQN last week).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (you will need F.log_softmax in update)
import tyro
from torch import nn

from rlcourse.deeprl.experiment import Monitor, RunResult, config_changes, default_name, run_experiment
from rlcourse.deeprl.networks import GridEncoder
from rlcourse.deeprl.pg_checks import run_checks
from rlcourse.deeprl.pg_tools import gradient_variance, load_agent, prepare_saving, save_agent, watch
from rlcourse.deeprl.world import make_env

WEEK_DIR = Path(__file__).resolve().parent


def todo(what: str):
    raise NotImplementedError(what)


# --- hyperparameters --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """REINFORCE hyperparameters. Every field is also a command-line option of `train`."""

    total_steps: int = 1_000_000
    """Environment steps of training. Leaderboard runs use the default."""
    gamma: float = 0.99
    """Discount factor (the same as for DQN)."""
    learning_rate: float = 2e-3
    """Adam step size for the policy."""
    episodes_per_update: int = 8
    """Whole episodes collected with the current policy before each gradient step."""
    rtg: bool = True
    """Weight each action by the return-to-go G_t; with --no-rtg, by the whole return R(tau)."""
    baseline: Literal["batch_mean", "none"] = "batch_mean"
    """What to subtract from the returns: the mean over the batch, or nothing."""
    entropy_coef: float = 0.01
    """Weight of the entropy bonus, which keeps the policy from becoming deterministic too early."""
    max_grad_norm: float = 10.0
    """Gradients with a larger norm are scaled down to this norm."""
    eval_every: int = 100_000
    """Evaluate the greedy policy every this-many environment steps (for the learning curve)."""

    # --- the world we *train* in (evaluation always uses the standard world) ----------------
    living_reward: float = -0.01
    """Reward for every step that does not end the episode."""
    death_reward: float = -1.0
    """Reward for being caught by the ghost."""
    ghost_policy: Literal["random", "chaser", "proximity", "mix"] = "random"
    """How the ghost moves."""

    # --- Task 4: your group's switches -------------------------------------------------------
    # Add them here, with defaults that keep the core algorithm and a docstring underneath:
    #
    #     baseline_value: float = 0.0
    #     """The value of the 'constant' baseline."""
    #
    # It then appears as `train --baseline-value 1.0` automatically, and in the name of the
    # results file. A new kind of baseline: add its name to the Literal of `baseline`.


# --- episodes ----------------------------------------------------------------------------------


class Episode(NamedTuple):
    """One complete episode s_0, a_0, r_0, s_1, ..., s_{T-1}, a_{T-1}, r_{T-1}, s_T (T steps)."""

    observations: np.ndarray      # (T, H, W, C) uint8: s_0 ... s_{T-1}, the boards we acted on
    actions: np.ndarray           # (T,) int64: a_0 ... a_{T-1}
    rewards: np.ndarray           # (T,) float32: r_t = R(s_t, a_t, s_{t+1})
    terminated: bool              # True: caught or cleared; False: cut off by the time limit
    last_observation: np.ndarray  # (H, W, C) uint8: s_T, the board after the last step


def returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G_t = r_t + gamma * G_{t+1} for t = T-1, ..., 0, with G_T = 0; a float32 array like `rewards`.

    Compute it backwards in one loop: each G_t reuses G_{t+1}."""
    return todo("the discounted sum of the rewards from t to the end, for every t")


# --- the policy network --------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    """pi(. | s; theta) as four logits: one forward pass gives the scores of all four actions;
    softmax turns them into probabilities.

    The same GridEncoder as DQN's Q-network reads the board. The head starts with tiny weights,
    so that the first policy is close to uniform: every action gets tried."""

    def __init__(self, observation_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.encoder = GridEncoder(observation_shape)
        self.head = nn.Linear(self.encoder.n_features, n_actions)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) boards -> (B, n_actions) logits."""
        return self.head(self.encoder(observations))


def stack(episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
    """All steps of a batch of episodes, episode after episode: (observations, actions) tensors
    of shapes (total steps, H, W, C) and (total steps,)."""
    observations = torch.from_numpy(np.concatenate([e.observations for e in episodes]))
    actions = torch.from_numpy(np.concatenate([e.actions for e in episodes]))
    return observations, actions


# --- the agent --------------------------------------------------------------------------------


class PGAgent:
    """The policy network and what to do with it."""

    def __init__(self, observation_shape: tuple[int, ...], n_actions: int, config: Config):
        self.config = config
        self.n_actions = n_actions
        self.policy_network = PolicyNetwork(observation_shape, n_actions)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        """pi(. | s; theta) of a single board: a float64 array of n_actions that sums to 1."""
        logits = self.policy_network(torch.from_numpy(observation).unsqueeze(0))[0]
        probs = torch.softmax(logits.double(), dim=0).numpy()
        return probs / probs.sum()

    def act(self, observation: np.ndarray, rng: np.random.Generator) -> int:
        """Sample an action from pi(. | s; theta), as a Python int. Use `rng` for the randomness
        (`rng.choice` takes the probabilities as `p=`)."""
        return todo("sample from the policy's probabilities")

    def greedy_action(self, observation: np.ndarray) -> int:
        """The most likely action: what evaluation plays (as for DQN)."""
        return int(self.probabilities(observation).argmax())

    def weights(self, episodes: list[Episode]) -> torch.Tensor:
        """The weight of each step's log-probability in the loss, for all steps of the batch,
        episode after episode (the order of `stack`). A float32 tensor of shape (total steps,).

        - config.rtg: the return-to-go G_t of each step; otherwise (--no-rtg) every step of an
          episode gets the episode's whole return R(tau) (which is G_0).
        - config.baseline == "batch_mean": subtract the mean of these numbers over *all steps*
          of the batch; "none": subtract nothing."""
        return todo("returns-to-go (or whole returns), minus the baseline")

    def update(self, episodes: list[Episode]) -> float:
        """One gradient step on the loss (see the top of this file). Returns the mean entropy of
        the policy over the batch's states (before the step), as a Python float (`.item()`).

        Steps: `stack` the episodes; the weights from `self.weights(episodes)`; the log-probabilities
        of all actions (F.log_softmax of the logits), and of the actions taken (Tensor.gather, as
        for DQN); the entropy  -sum_a pi(a|s) log pi(a|s), averaged over the states; the loss
        -mean(log pi(a_t|s_t) * weight_t) - entropy_coef * entropy; then zero the gradients,
        backward, clip the gradient norm to config.max_grad_norm, optimizer step."""
        return todo("loss, backward, clip, step")


# --- the training loop ------------------------------------------------------------------------


def run_episode(env, agent: PGAgent, rng: np.random.Generator) -> Episode:
    """Play one whole episode with the agent's (sampled) policy."""
    observation, _ = env.reset()
    observations, actions, rewards = [], [], []
    while True:
        action = agent.act(observation, rng)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        observation = next_observation
        if terminated or truncated:
            return Episode(np.stack(observations), np.array(actions, dtype=np.int64),
                           np.array(rewards, dtype=np.float32), bool(terminated), observation)


def make_agent(config: Config) -> PGAgent:
    env = make_env()
    return PGAgent(env.observation_space.shape, int(env.action_space.n), config)


def train(config: Config, seed: int) -> RunResult:
    """Train one agent from scratch; return what the Monitor recorded."""
    # Independent random streams for the agent and for the world, both derived from `seed`.
    agent_seed, env_seed = np.random.SeedSequence(seed).generate_state(2)
    rng = np.random.default_rng(agent_seed)
    env = make_env(living_reward=config.living_reward, death_reward=config.death_reward,
                   ghost_policy=config.ghost_policy)
    agent = make_agent(config)
    monitor = Monitor(config.total_steps, seed, eval_every=config.eval_every, gamma=config.gamma)
    value_fn = None   # a function board -> V(board) here adds a "V(s0) vs G0" column to the progress

    env.reset(seed=int(env_seed))
    step = 0
    while step < config.total_steps:
        episodes = [run_episode(env, agent, rng) for _ in range(config.episodes_per_update)]
        # Bookkeeping, step by step as if live (the policy did not change during the batch).
        for episode in episodes:
            for t, reward in enumerate(episode.rewards):
                step += 1
                monitor.record_step(reward, t == len(episode.rewards) - 1)
                monitor.evaluate_if_due(step, agent.greedy_action, value_fn=value_fn)
        entropy = agent.update(episodes)
        monitor.log(entropy=entropy, episode_length=np.mean([len(e.rewards) for e in episodes]))

    save_agent(agent, config, seed)
    return monitor.finish(agent.greedy_action, value_fn=value_fn)


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
    """Name of the results file; default: 'core', or the options you changed."""
    submit: bool = False
    """Send the result (and this file) to the leaderboard."""
    nickname: str | None = None
    """Shown on the leaderboard next to your group."""


@dataclass(frozen=True)
class Check:
    """Check your implementation, piece by piece."""


@dataclass(frozen=True)
class Watch:
    """Watch a trained policy play (or, with --quiet, only count what happens)."""

    policy: tyro.conf.Positional[Path]
    """A saved policy, e.g. week08_pg/results/policies/core_seed0.pt."""
    episodes: int = 1
    sampled: bool = False
    """Sample the actions from pi (as in training) instead of taking the most likely one."""
    ghost: Literal["random", "chaser", "proximity", "mix"] = "random"
    """The ghost to play against."""
    delay: float = 0.15
    """Seconds between frames."""
    quiet: bool = False
    """Show only the summary (e.g. with --episodes 100)."""


@dataclass(frozen=True)
class Variance:
    """How noisy is each policy-gradient estimator, at a fixed policy?"""

    policy: tyro.conf.Positional[Path | None] = None
    """A saved policy; leave out for an untrained one (with the default Config)."""
    batches: int = 40
    episodes_per_batch: int = 8
    seed: int = 0


def main() -> None:
    command = tyro.extras.subcommand_cli_from_dict(
        {"train": Train, "check": Check, "watch": Watch, "variance": Variance},
        description=__doc__.splitlines()[0],
    )
    if isinstance(command, Check):
        run_checks(Config=Config, PGAgent=PGAgent, returns_to_go=returns_to_go, train=train)
    elif isinstance(command, Watch):
        agent = load_agent(command.policy, make_agent, Config)
        watch(agent, episodes=command.episodes, sampled=command.sampled, delay=command.delay,
              quiet=command.quiet, **({} if command.ghost == "random" else {"ghost_policy": command.ghost}))
    elif isinstance(command, Variance):
        agent = make_agent(Config()) if command.policy is None else load_agent(command.policy, make_agent, Config)
        gradient_variance(agent, run_episode, returns_to_go, batches=command.batches,
                          episodes_per_batch=command.episodes_per_batch, seed=command.seed)
    else:
        name = command.name or default_name(config_changes(command.config), "core")
        prepare_saving(WEEK_DIR / "results" / "policies", name)
        run_experiment(
            train, command.config, seeds=command.seeds, workers=command.workers, name=name,
            results_dir=WEEK_DIR / "results", submit=command.submit, week_dir=WEEK_DIR,
            nickname=command.nickname, files=[Path(__file__)], leaderboard_kind="pg",
        )
        policies = WEEK_DIR / "results" / "policies"
        print(f"Policies saved as {policies / name}_seed<k>.pt, e.g. for `pg.py watch` or `pg.py variance`.")


if __name__ == "__main__":
    main()
