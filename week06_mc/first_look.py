"""Task 3: a first look. Three plots, one question each.

    uv run python week06_mc/first_look.py

Left: prediction on the maze. Monte Carlo against TD(0), from the same episodes.
Middle: control on the same maze. Monte Carlo against SARSA and Q-learning.
Right: off-policy evaluation of a world we never ran: logs of the tired policy (30 % random),
evaluated for a less tired one (10 % random), with ordinary and weighted importance sampling.
"""
from mc_tools import *
from control import QLearning, SARSA
from importance import OrdinaryIS, WeightedIS, effective_sample_size, tail_ratios
from mc import MC, MCControl
from td import TD0

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

world = load("maze")
print(f"{world}:  {world.description}")
curves = learning_curves({"Monte Carlo": MC, "TD(0), alpha = 0.15": partial(TD0, alpha=0.15)},
                         world, n_episodes=100, runs=20)
plot_curves(curves, ax=axes[0], title="maze, prediction: error of V")
for label, c in curves.items():
    m, se = final_error(c)
    print(f"   prediction, {label:20s} error after 100 episodes: {m:.3f} +- {se:.3f}")

agents = {"Monte Carlo": MCControl, "SARSA": partial(SARSA, alpha=0.3),
          "Q-learning": partial(QLearning, alpha=0.3)}
res = run_control(agents, world, n_episodes=300, runs=10, epsilon=0.1, gap_every=10)
plot_control(res, "gap", ax=axes[1], title="maze, control: how far the greedy policy is from optimal")
for name, r in res.items():
    print(f"   control, {name:12s} gap after 300 episodes: {r['gap'][:, -1].mean():.3f}"
          f"   steps in all: {r['steps'].sum(1).mean():.0f}")

logs, target = load("maze", tired=0.3), load("maze", tired=0.1)
pi, b = target.policy, logs.policy
episodes = logs.episodes(2000, seed=0)
ordinary = OrdinaryIS(logs.n_states, logs.gamma, pi, b)
weighted = WeightedIS(logs.n_states, logs.gamma, pi, b)
path = {"ordinary IS": [], "weighted IS": []}
for ep in episodes:
    ordinary.update(ep)
    weighted.update(ep)
    path["ordinary IS"].append(ordinary.V[logs.start])
    path["weighted IS"].append(weighted.V[logs.start])
for label, y in path.items():
    axes[2].plot(np.arange(1, len(y) + 1), y, label=label)
axes[2].axhline(start_value(target), color="k", ls="--", label="the truth")
axes[2].set_xscale("log")
axes[2].set_xlabel("logged episodes")
axes[2].set_ylabel("estimate of v_pi(start)")
axes[2].set_title("off-policy evaluation of the less tired policy")
axes[2].grid(alpha=0.3)
axes[2].legend()
w = np.array([tail_ratios(ep, pi, b)[0] for ep in episodes])
print(f"   off-policy, truth {start_value(target):.3f}; ordinary {ordinary.V[logs.start]:.3f}, "
      f"weighted {weighted.V[logs.start]:.3f}; {len(w)} episodes, "
      f"effective sample size {effective_sample_size(w):.0f}")
plt.tight_layout()
plt.show()

# Who wins prediction on the maze, and who wins control? Same world, same data budget: write
# down a guess why they differ. Then: what is the effective sample size telling you, and what
# would make it worse? Your group's question is one way of finding out.
