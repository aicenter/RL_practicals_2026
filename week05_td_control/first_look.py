"""Task 3: a first look. Prediction on the left, control on the right.

    uv run python week05_td_control/first_look.py

Left: TD(0) with three step sizes on the 19-state random walk; all estimators learn from
exactly the same episodes (20 runs). Right: SARSA, Expected SARSA and Q-learning acting
epsilon-greedily (eps = 0.1) in the ghost's lair -- the short route past the ghost against
the long safe one -- their online return per episode, and the quality of their greedy policy.
"""
import matplotlib.pyplot as plt

from td_tools import (learning_curves, load, partial, plot_curves, final_error, run_control,
                      plot_control, one_over_n)
from td import TD0
from control import SARSA, ExpectedSARSA, QLearning

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

world = load("random_walk", size=19, left_reward=-1)
print(f"{world}:  {world.description}")
estimators = {"TD(0), alpha = 0.05": partial(TD0, alpha=0.05),
              "TD(0), alpha = 0.3": partial(TD0, alpha=0.3),
              "TD(0), alpha = 1/n^0.7": partial(TD0, alpha=one_over_n(0.7))}
curves = learning_curves(estimators, world, n_episodes=200, runs=20)
plot_curves(curves, ax=axes[0], title="random walk: error of V after each episode")
for label, c in curves.items():
    m, se = final_error(c)
    print(f"   {label:24s} error after 200 episodes: {m:.3f} +- {se:.3f}")
print()

lair = load("ghost_lair")
print(f"{lair}:  {lair.description}")
agents = {"SARSA": SARSA, "Expected SARSA": ExpectedSARSA, "Q-learning": QLearning}
res = run_control(agents, lair, n_episodes=1500, runs=5, epsilon=0.1, alpha=0.3, gap_every=50)
plot_control(res, "return", ax=axes[1], title="ghost lair: online return (eps = 0.1)")
plot_control(res, "gap", ax=axes[2], title="ghost lair: how far the greedy policy is from optimal")
for name, r in res.items():
    print(f"   {name:16s} mean return of the last 300 episodes: {r["return"][:, -300:].mean():7.3f}"
          f"   final gap of the greedy policy: {r['gap'][:, -1].mean():.3f}")
plt.tight_layout()
plt.show()

# Which step size wins early, which late? Which agent earns more while learning, and which
# has the better greedy policy -- and why are these not the same agent? Write down a guess
# before Task 4: your group's question is one way of finding out.
