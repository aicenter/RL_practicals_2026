"""PPO in one file, as a black-box solver for small environments.

This is a compact version of CleanRL's `ppo.py` (https://github.com/vwxyzjn/cleanrl), with two
changes: it runs a plain list of environments instead of a vectorised one, so that every step
of the rollout is visible, and it treats time-limit truncation correctly (see `collect_rollout`).
How PPO works is Week 9 material; for now, treat it as a solver you point at an environment.

    uv run python week01_foundations/ppo.py --env week01_foundations.corridor:CorridorEnv
    uv run python week01_foundations/ppo.py --env week01_foundations.thermostat:ThermostatEnv
    uv run python week01_foundations/ppo.py --env week01_foundations.roomba:RoombaEnv --env-kwargs '{"reward": "cleanliness"}'
    uv run python week01_foundations/ppo.py --env my_env:MyEnv --max-episode-steps 100

Each run writes a learning curve to `runs/<run-name>/learning_curve.png`, logs to TensorBoard
(`uv run tensorboard --logdir runs`), and finally shows one episode played by the trained agent.
Only discrete action spaces are supported.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from common import make_env, run_episodes


@dataclass
class Args:
    env: str = "week01_foundations.corridor:CorridorEnv"
    """environment spec: a Gymnasium id or `module:ClassName`"""
    env_kwargs: str = "{}"
    """extra constructor arguments as a JSON string"""
    max_episode_steps: int | None = None
    """time limit, for environments that never truncate on their own"""
    seed: int = 1
    run_name: str | None = None
    """name of the output directory under runs/ (default: environment name + time)"""
    watch: bool = True
    """after training, print one episode played by the trained agent"""
    show_plot: bool = False
    """open the learning-curve window at the end (it is always saved to a file)"""

    # Training budget. The defaults suit environments with tens to hundreds of states.
    total_timesteps: int = 100_000
    num_envs: int = 8
    """environments stepped in parallel; each contributes `num_steps` transitions per update"""
    num_steps: int = 64
    """steps per environment between updates"""
    learning_rate: float = 1e-3
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.1
    """entropy bonus, keeps the policy exploring (higher than usual: our environments are tiny and deterministic)"""
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64


# ---- the agent: a small policy network and a small value network ------------------------------


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(obs)

    def greedy_action(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(self.actor(torch.as_tensor(obs, dtype=torch.float32)).argmax())


# ---- collecting experience --------------------------------------------------------------------


class Rollout:
    """Fixed-size buffers for one batch of experience: num_steps x num_envs transitions."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int):
        self.obs = torch.zeros((num_steps, num_envs, obs_dim))
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long)
        self.logprobs = torch.zeros((num_steps, num_envs))
        self.rewards = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))


def collect_rollout(envs, agent, rollout, state, args, finished_episodes):
    """Step every environment `num_steps` times, filling `rollout`.

    `state` carries the current observation, done flag, and running episode return of every
    environment from one rollout to the next. Finished episodes are appended to `finished_episodes`
    as (global_step, return).
    """
    for step in range(args.num_steps):
        state["global_step"] += len(envs)
        rollout.obs[step] = state["next_obs"]
        rollout.dones[step] = state["next_done"]

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(state["next_obs"])
        rollout.actions[step] = action
        rollout.logprobs[step] = logprob
        rollout.values[step] = value

        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, info = env.step(int(action[i]))
            state["episode_return"][i] += reward

            if truncated and not terminated:
                # The episode was cut short by a time limit, but the task did not end: the state
                # we were cut off in still has value. We fold that value into the reward so that
                # the transition can be treated as terminal below without pretending that the
                # future was worth nothing. (Many reference implementations skip this.)
                with torch.no_grad():
                    reward += args.gamma * float(agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32)))

            rollout.rewards[step, i] = reward
            done = terminated or truncated
            if done:
                finished_episodes.append((state["global_step"], state["episode_return"][i]))
                state["episode_return"][i] = 0.0
                next_obs, info = env.reset()
            state["next_obs"][i] = torch.as_tensor(next_obs, dtype=torch.float32)
            state["next_done"][i] = float(done)


def compute_advantages(rollout, agent, state, args):
    """Generalised advantage estimation, working backwards through the rollout."""
    with torch.no_grad():
        next_value = agent.get_value(state["next_obs"])
    advantages = torch.zeros_like(rollout.rewards)
    last_gae = torch.zeros(len(state["next_done"]))
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            next_nonterminal = 1.0 - state["next_done"]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - rollout.dones[t + 1]
            next_values = rollout.values[t + 1]
        delta = rollout.rewards[t] + args.gamma * next_values * next_nonterminal - rollout.values[t]
        last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + rollout.values


# ---- the PPO update ---------------------------------------------------------------------------


def ppo_update(agent, optimizer, rollout, advantages, returns, args) -> dict:
    batch_size = args.num_steps * args.num_envs
    minibatch_size = batch_size // args.num_minibatches
    b_obs = rollout.obs.reshape(batch_size, -1)
    b_actions = rollout.actions.reshape(batch_size)
    b_logprobs = rollout.logprobs.reshape(batch_size)
    b_advantages = advantages.reshape(batch_size)
    b_returns = returns.reshape(batch_size)

    for _ in range(args.update_epochs):
        for start in torch.randperm(batch_size).split(minibatch_size):
            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[start], b_actions[start])
            ratio = (new_logprob - b_logprobs[start]).exp()
            adv = b_advantages[start]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # The clipped surrogate objective: do not move the policy too far in one update.
            policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
            value_loss = 0.5 * ((new_value - b_returns[start]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy_loss.item()}


# ---- reporting --------------------------------------------------------------------------------


def plot_learning_curve(finished_episodes, random_mean, greedy_mean, path: Path, title: str, show: bool):
    import matplotlib.pyplot as plt

    steps = np.array([s for s, _ in finished_episodes])
    rets = np.array([r for _, r in finished_episodes])
    window = max(1, min(50, len(rets) // 10))
    smoothed = np.convolve(rets, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(steps, rets, s=6, color="#9aa0a6", alpha=0.35, linewidths=0, label="episode return")
    ax.plot(steps[window - 1:], smoothed, color="#1f6feb", linewidth=2, label=f"moving average ({window} episodes)")
    ax.axhline(random_mean, color="#6e7781", linestyle="--", linewidth=1, label=f"random policy: {random_mean:.1f}")
    ax.axhline(greedy_mean, color="#1f6feb", linestyle=":", linewidth=1, label=f"trained agent, greedy: {greedy_mean:.1f}")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"learning curve saved to {path}")
    if show:
        plt.show()


def watch_episode(env: gym.Env, agent: Agent, seed: int) -> None:
    """Print one episode played greedily by the trained agent, frame by frame."""
    obs, info = env.reset(seed=seed)
    total, t = 0.0, 0
    print("\n=== one episode with the trained agent ===")
    while True:
        frame = env.unwrapped.render()
        if frame is not None:
            print(frame)
        action = agent.greedy_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        t += 1
        print(f"      action={action}  reward={reward:+.2f}  return so far={total:.2f}")
        if terminated or truncated:
            frame = env.unwrapped.render()
            if frame is not None:
                print(frame)
            print(f"episode over after {t} steps: terminated={terminated} truncated={truncated} return={total:.2f}")
            return


# ---- main -------------------------------------------------------------------------------------


def main(args: Args) -> None:
    env_name = args.env.split(":")[-1].replace("-", "_")
    run_name = args.run_name or f"{env_name}__{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # networks this small run fastest on a single thread

    envs = [make_env(args.env, args.env_kwargs, args.max_episode_steps) for _ in range(args.num_envs)]
    eval_env = make_env(args.env, args.env_kwargs, args.max_episode_steps)
    assert isinstance(envs[0].action_space, gym.spaces.Discrete), "this script only supports discrete actions"
    obs_dim = int(np.prod(envs[0].observation_space.shape))
    n_actions = int(envs[0].action_space.n)
    print(f"{args.env}: {obs_dim}-dimensional observation, {n_actions} actions")

    # A reference point: how well does acting at random do?
    eval_env.action_space.seed(args.seed)
    random_stats = run_episodes(eval_env, lambda obs: eval_env.action_space.sample(), episodes=20, seed=args.seed)
    random_mean = float(np.mean([s["return"] for s in random_stats]))
    print(f"random policy: mean return {random_mean:.2f} over 20 episodes")

    agent = Agent(obs_dim, n_actions, args.hidden_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    rollout = Rollout(args.num_steps, args.num_envs, obs_dim)

    state = {
        "global_step": 0,
        "next_obs": torch.stack([torch.as_tensor(env.reset(seed=args.seed + i)[0], dtype=torch.float32) for i, env in enumerate(envs)]),
        "next_done": torch.zeros(args.num_envs),
        "episode_return": np.zeros(args.num_envs),
    }
    finished_episodes: list[tuple[int, float]] = []
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Linear learning-rate decay, a common PPO stabiliser.
        optimizer.param_groups[0]["lr"] = args.learning_rate * (1 - (update - 1) / num_updates)

        n_before = len(finished_episodes)
        collect_rollout(envs, agent, rollout, state, args, finished_episodes)
        advantages, returns = compute_advantages(rollout, agent, state, args)
        losses = ppo_update(agent, optimizer, rollout, advantages, returns, args)

        for step, ret in finished_episodes[n_before:]:
            writer.add_scalar("charts/episodic_return", ret, step)
        for name, value in losses.items():
            writer.add_scalar(f"losses/{name}", value, state["global_step"])

        if update % max(1, num_updates // 20) == 0 or update == num_updates:
            recent = [r for _, r in finished_episodes[-20:]]
            mean_str = f"{np.mean(recent):8.2f}" if recent else "     n/a"
            print(
                f"update {update:4d}/{num_updates}  step {state['global_step']:7d}  "
                f"mean return (last 20 episodes) {mean_str}  "
                f"entropy {losses['entropy']:.3f}  {int(state['global_step'] / (time.time() - start_time))} steps/s"
            )

    # Final evaluation with the greedy policy, and the learning curve.
    greedy_stats = run_episodes(eval_env, agent.greedy_action, episodes=20, seed=args.seed + 1000)
    greedy_mean = float(np.mean([s["return"] for s in greedy_stats]))
    print(f"\ntrained agent (greedy): mean return {greedy_mean:.2f} over 20 episodes  (random: {random_mean:.2f})")
    torch.save(agent.state_dict(), run_dir / "agent.pt")
    if finished_episodes:
        plot_learning_curve(finished_episodes, random_mean, greedy_mean, run_dir / "learning_curve.png", args.env, args.show_plot)
    if args.watch:
        watch_episode(eval_env, agent, seed=args.seed + 2000)
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.show_plot:
        matplotlib.use("Agg")  # no window needed, just the file
    main(args)
