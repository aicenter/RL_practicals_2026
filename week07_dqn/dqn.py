"""Deep Q-learning (DQN) on PragMan: the file you edit this week.

DQN is Q-learning with a neural network in place of the table, plus the two fixes from the
lecture (Week 7) that keep it stable: experience replay and a target network.

    Q-learning target:  y = r + gamma * max_a' Q(s', a'; w-)         (w- : the target network)
    loss:               L(w) = mean over a minibatch of  huber(y - Q(s, a; w))

The gradient flows only through Q(s, a; w), never through y (a semi-gradient). If the episode
*terminated* at s' (a ghost caught us, or we ate the last pellet), there is no future: y = r.
If it was only *truncated* by the time limit, s' is an ordinary state and we bootstrap as usual.

Run everything from the repository root:

    uv run python week07_dqn/dqn.py check            # checks each piece as soon as you write it
    uv run python week07_dqn/dqn.py train            # vanilla DQN, 3 seeds side by side (~3 min)
    uv run python week07_dqn/dqn.py train --help     # every hyperparameter, with its default
    uv run python week07_dqn/dqn.py train --workers 1 --seeds 0   # one seed, in this process
    uv run python week07_dqn/dqn.py train --submit   # ... and send the result to the leaderboard

Results are saved in week07_dqn/results/<name>.json; compare them with

    uv run python -m rlcourse.deeprl.plotting week07_dqn/results

Fill in the places marked todo(...). The shared tools (the world, the network that reads the
board, evaluation, and the experiment runner) are in src/rlcourse/deeprl/; you are welcome to
read them, but there is no need to edit them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (you will need F.smooth_l1_loss in update)
import tyro
from torch import nn

from rlcourse.deeprl.dqn_checks import run_checks
from rlcourse.deeprl.experiment import Monitor, RunResult, run_experiment
from rlcourse.deeprl.networks import GridEncoder
from rlcourse.deeprl.world import make_env

WEEK_DIR = Path(__file__).resolve().parent


def todo(what: str):
    raise NotImplementedError(what)


# --- hyperparameters --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """DQN hyperparameters. Every field is also a command-line option (see `train --help`)."""

    total_steps: int = 100_000
    """Environment steps of training. Leaderboard runs use the default."""
    gamma: float = 0.99
    """Discount factor."""
    learning_rate: float = 5e-4
    """Adam step size."""
    batch_size: int = 64
    """Transitions per gradient step."""
    buffer_size: int = 50_000
    """Replay buffer capacity: the most recent this-many transitions are kept."""
    learning_starts: int = 2_000
    """Act (randomly, at first) for this many steps before the first gradient step."""
    train_every: int = 4
    """One gradient step every this-many environment steps."""
    target_update_every: int = 1_000
    """Copy the online network into the target network every this-many environment steps."""
    epsilon_start: float = 1.0
    """Exploration at the start ..."""
    epsilon_end: float = 0.05
    """... and at the end of the linear decay ..."""
    epsilon_decay_steps: int = 30_000
    """... which takes this many environment steps."""
    max_grad_norm: float = 10.0
    """Gradients with a larger norm are scaled down to this norm."""
    eval_every: int = 10_000
    """Evaluate the greedy policy every this-many environment steps (for the learning curve)."""

    # --- Task 4: your group's improvement --------------------------------------------------
    # Add its switch here, with a default that keeps vanilla DQN, and a docstring underneath:
    #
    #     double: bool = False
    #     """Use the Double DQN target."""
    #
    # It then appears as `train --double` automatically, and in the name of the results file.


# --- experience replay ------------------------------------------------------------------------


class Batch(NamedTuple):
    """A minibatch of transitions, as torch tensors; the first dimension is the batch."""

    observations: torch.Tensor       # (B, H, W, C) uint8
    actions: torch.Tensor            # (B,) int64
    rewards: torch.Tensor            # (B,) float32
    next_observations: torch.Tensor  # (B, H, W, C) uint8
    terminated: torch.Tensor         # (B,) float32: 1.0 if the episode terminated at s', else 0.0


class ReplayBuffer:
    """The last `capacity` transitions, in a ring: when it is full, a new transition overwrites
    the oldest one. Boards are kept as uint8 (one byte per cell and plane); `sample` turns the
    chosen rows into tensors."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...]):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.position = 0   # where the next transition goes
        self.size = 0       # how many slots hold a transition

    def __len__(self) -> int:
        return self.size

    def add(self, observation: np.ndarray, action: int, reward: float,
            next_observation: np.ndarray, terminated: bool) -> None:
        """Store one transition at `self.position`, then move the position on (wrapping round at
        the capacity) and update `self.size`."""
        todo("store the transition, advance position (ring), update size")

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """`batch_size` transitions drawn uniformly at random (with replacement) from the filled
        part of the buffer. Use `rng` for the randomness, and torch.from_numpy for the arrays."""
        return todo("uniform random indices among the stored transitions -> Batch of tensors")


# --- the Q-network ----------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Q(s, .; w): one forward pass gives the values of all four actions.

    GridEncoder (two convolutions and a fully-connected layer, see
    src/rlcourse/deeprl/networks.py) reads the board; a linear head turns its features into one
    number per action."""

    def __init__(self, observation_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.encoder = GridEncoder(observation_shape)
        self.head = nn.Linear(self.encoder.n_features, n_actions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) boards -> (B, n_actions) action values."""
        return self.head(self.encoder(observations))


# --- the agent --------------------------------------------------------------------------------


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    """Go linearly from `start` (at step 0) to `end` (at step `duration`), then stay at `end`."""
    return todo("linear interpolation, constant after `duration`")


class DQNAgent:
    """The online network, the target network, and what to do with them."""

    def __init__(self, observation_shape: tuple[int, ...], n_actions: int, config: Config):
        self.config = config
        self.n_actions = n_actions
        self.q_network = QNetwork(observation_shape, n_actions)
        self.target_network = copy.deepcopy(self.q_network)    # w- = w, to begin with
        self.target_network.requires_grad_(False)               # w- is never trained directly
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def q_values(self, observation: np.ndarray) -> torch.Tensor:
        """Q(s, .; w) of a single board, shape (n_actions,)."""
        return self.q_network(torch.from_numpy(observation).unsqueeze(0))[0]

    def greedy_action(self, observation: np.ndarray) -> int:
        """argmax_a Q(s, a; w), as a Python int."""
        return int(self.q_values(observation).argmax())

    def start_value(self, observation: np.ndarray) -> float:
        """max_a Q(s, a; w): what the agent believes the board is worth. Evaluation compares
        it with the discounted return that actually follows."""
        return float(self.q_values(observation).max())

    def act(self, observation: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        """Epsilon-greedy: with probability epsilon a uniformly random action, otherwise the
        greedy one. Use `rng` for both random choices."""
        return todo("epsilon-greedy with respect to the online network")

    def td_targets(self, batch: Batch) -> torch.Tensor:
        """y = r + gamma * max_a' Q(s', a'; w-), or y = r where the episode terminated.
        Shape (B,). No gradient may flow through it (torch.no_grad)."""
        return todo("the Q-learning targets, from the target network")

    def update(self, batch: Batch) -> float:
        """One gradient step on the Huber loss between Q(s, a; w) and the targets. Returns the
        loss as a Python float.

        Steps: the targets; Q(s, a; w) of the actions actually taken (Tensor.gather picks one
        entry per row); F.smooth_l1_loss; zero the gradients, backward, clip the gradient norm
        to config.max_grad_norm (nn.utils.clip_grad_norm_), optimizer step."""
        return todo("loss, backward, clip, step")

    def sync_target(self) -> None:
        """w- <- w: copy the online network's parameters into the target network."""
        todo("one line: load_state_dict")


# --- the training loop ------------------------------------------------------------------------


def train(config: Config, seed: int) -> RunResult:
    """Train one agent from scratch; return what the Monitor recorded."""
    # Independent random streams for the agent and for the world, both derived from `seed`.
    # (Seeding both with the same integer would make the ghost's dice and ours the same numbers.)
    agent_seed, env_seed = np.random.SeedSequence(seed).generate_state(2)
    rng = np.random.default_rng(agent_seed)
    env = make_env()
    shape, n_actions = env.observation_space.shape, int(env.action_space.n)
    agent = DQNAgent(shape, n_actions, config)
    buffer = ReplayBuffer(config.buffer_size, shape)
    monitor = Monitor(config.total_steps, seed, eval_every=config.eval_every, gamma=config.gamma)

    observation, _ = env.reset(seed=int(env_seed))
    for step in range(1, config.total_steps + 1):
        epsilon = linear_schedule(step, config.epsilon_start, config.epsilon_end,
                                  config.epsilon_decay_steps)
        action = agent.act(observation, epsilon, rng)
        next_observation, reward, terminated, truncated, _ = env.step(action)

        todo("store the transition in the buffer (which of the two flags?)")
        monitor.record_step(reward, terminated or truncated)

        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation

        todo("from step `learning_starts` on, every `train_every` steps: sample a batch and "
             "update the agent; pass the loss to monitor.log(loss=...)")
        todo("every `target_update_every` steps: sync the target network")

        monitor.log(epsilon=epsilon)
        monitor.evaluate_if_due(step, agent.greedy_action, value_fn=agent.start_value)

    return monitor.finish(agent.greedy_action, value_fn=agent.start_value)


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
    """Name of the results file; default: 'vanilla', or the options you changed."""
    submit: bool = False
    """Send the result (and this file) to the leaderboard."""
    nickname: str | None = None
    """Shown on the leaderboard next to your group."""


@dataclass(frozen=True)
class Check:
    """Check your implementation, piece by piece."""


def main() -> None:
    command = tyro.extras.subcommand_cli_from_dict(
        {"train": Train, "check": Check}, description=__doc__.splitlines()[0]
    )
    if isinstance(command, Check):
        run_checks(Config=Config, ReplayBuffer=ReplayBuffer, DQNAgent=DQNAgent,
                   linear_schedule=linear_schedule, train=train)
        return
    run_experiment(
        train, command.config, seeds=command.seeds, workers=command.workers, name=command.name,
        results_dir=WEEK_DIR / "results", submit=command.submit, week_dir=WEEK_DIR,
        nickname=command.nickname, files=[Path(__file__)], leaderboard_kind="dqn",
    )


if __name__ == "__main__":
    main()
