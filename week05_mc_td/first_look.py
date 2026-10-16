"""Task 3: a first look. Monte Carlo against TD(0) on two worlds.

    uv run python week05_mc_td/first_look.py

Both estimators learn from exactly the same episodes (20 runs, different seeds), and we
plot their error (RMS over states, against the true V_pi) after each episode.
"""
import matplotlib.pyplot as plt

from pred_tools import learning_curves, load, partial, plot_curves, final_error
from mc import MC
from td import TD0

estimators = {
    "MC, alpha = 0.05": partial(MC, alpha=0.05),
    "MC, alpha = 0.15": partial(MC, alpha=0.15),
    "TD(0), alpha = 0.05": partial(TD0, alpha=0.05),
    "TD(0), alpha = 0.15": partial(TD0, alpha=0.15),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, name in zip(axes, ["random_walk", "maze"]):
    world = load(name)
    print(f"{world}:  {world.description}")
    curves = learning_curves(estimators, world, n_episodes=100, runs=20)
    plot_curves(curves, ax=ax, title=f"{name}: error of V after each episode")
    for label, c in curves.items():
        m, se = final_error(c)
        print(f"   {label:22s} error after 100 episodes: {m:.3f} +- {se:.3f}")
    print()
plt.tight_layout()
plt.show()

# Which one wins, on which world? Before you read on, write down a guess *why*.
# Your group's question in Task 4 is one way of finding out.
