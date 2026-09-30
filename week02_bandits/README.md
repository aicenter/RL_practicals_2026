# Week 2: Multi-armed bandits

Goal: get a feel for the exploration–exploitation trade-off, first by playing
bandits yourself and then by implementing, tuning and comparing the algorithms
from the lecture: ε-greedy, explore-then-commit, Boltzmann (softmax)
exploration and UCB.

A bandit is the simplest possible MDP: one state, a few actions ('arms'), and
a random reward for each pull. We therefore use the same Gymnasium interface as
last week (`env.reset()`, `env.step(action)`); the observation is always 0.
One episode is one whole learning run of T pulls.

## What's in this directory

| File | What it is |
| --- | --- |
| `agents.py` | **The file you edit.** A working (but poor) greedy agent and one empty class for each of the four algorithms. |
| `bandit_tools.py` | Running, evaluating, tuning and plotting agents. No need to edit it. |
| `play.py` | Play a bandit by hand and record your results. |

The bandits themselves live in `src/rlcourse/envs/bandits.py` (read it only
*after* Task 1).

## Groups and problem classes

We work in five groups of 3–4. Each group gets two **problem classes** (out of
five) and two **algorithms**. A problem class is not a single bandit but a
*distribution* of similar bandits: every run draws a fresh instance (new means,
new noise levels, shuffled arms). You are tuning for the class, not for one
lucky instance. Groups with the same algorithms form a super-group and compare
notes at the end.

| Group | Problem classes | Algorithms | Play by hand |
| --- | --- | --- | --- |
| A1 | 1, 2 | ε-greedy, explore-then-commit | 2 |
| A2 | 1, 3 | ε-greedy, explore-then-commit | 3 |
| A3 | 4, 5 | ε-greedy, explore-then-commit | 4, 5 |
| B1 | 2, 4 | Boltzmann, UCB | 2, 4 |
| B2 | 3, 5 | Boltzmann, UCB | 3, 5 |

## Tasks

0. **Setup** *(5 min)*. From the repository root: `git pull`, then `uv sync`,
   then `uv run python check_setup.py`. (If you have not set up the course
   environment yet, see the [main README](../README.md).)

1. **Be the bandit** *(15 min)*. Each of you plays at least two rounds of
   T = 50 pulls on your 'play by hand' class(es), trying to keep your regret
   as low as possible:

   ```bash
   uv run python week02_bandits/play.py 2 --name yourname
   ```

   Every call plays your next round. Round *k* is the same bandit, with the
   same luck, for everybody, so you can compare. Then see how you did against
   some simple algorithms *on exactly the rounds you played*:

   ```bash
   uv run python week02_bandits/play.py --summary
   ```

   Discuss in your group: how did you decide when to stop exploring? Did
   anything about the rewards surprise you? What would you do differently with
   T = 1000 pulls?

2. **Read the class descriptions** *(2 min)*: the docstrings of
   `sample_class1` … `sample_class5` in `src/rlcourse/envs/bandits.py`, for
   your two classes. Does what you saw in Task 1 make sense now?

3. **Implement your two algorithms** *(25 min)* in `agents.py`, e.g. one pair
   of you per algorithm. Look at `Greedy` first: it shows the whole interface.
   Check your implementation with

   ```bash
   uv run python week02_bandits/bandit_tools.py
   ```

   Hints:
   - All four algorithms need to deal with actions that have not been tried
     yet. For UCB, N(a) = 0 would even mean dividing by zero. A common fix is
     to pull every action once first.
   - Explore-then-commit knows the horizon: it is in `self.horizon`. What
     should happen if m·K is larger than T?
   - Boltzmann: `np.exp(Q / tau)` overflows for small τ. Subtract `Q.max()`
     first; it does not change the probabilities. Sampling with
     `self.rng.choice(K, p=p)` is correct but slow; with `cdf = np.cumsum(p)`,
     `np.searchsorted(cdf, self.rng.random() * cdf[-1])` does the same about
     3× faster.
   - Use `self.rng` for randomness and `argmax_random`, not `np.argmax`,
     for maximizing (see the top of `agents.py`).

4. **Tune** *(30 min)* your algorithms' hyperparameters on your two classes
   (T = 1000). Write your experiments in a script of your own in this
   directory, e.g.

   ```python
   # week02_bandits/tune_A1.py  --  run with: uv run python week02_bandits/tune_A1.py
   import matplotlib.pyplot as plt
   from bandit_tools import *
   from agents import *

   tuned = tune(lambda eps: EpsilonGreedy(eps), [0.003, 0.01, 0.03, 0.1, 0.3], problems=[1, 2])
   plot_tuning(tuned, xlabel="epsilon")
   plt.show()
   ```

   One setting takes 1–6 s with the default 200 runs; for a first coarse
   sweep, pass `seeds=range(50)`. Things to look at:
   - Does the best value differ between your two classes? By how much, and
     why? (Think about the units of each hyperparameter.)
   - How flat is the tuning curve around the optimum? Which is worse: too
     little exploration or too much?
   - Is the mean regret the whole story? Compare the distributions with
     `plot_histogram([...])`, and look at the median and the 90th percentile
     (printed by `evaluate`).
   - To compare two agents, use `paired_difference(evaluate(a, p), evaluate(b, p))`:
     both face the same bandits and the same luck, so the comparison is far
     more precise than two separate means.
   - When you have made your choice, check it **once** on the test seeds:
     `evaluate(agent, p, seeds=TEST_SEEDS)`. Why is the number usually a bit
     worse than on the tuning seeds?

5. **Final choice and sharing** *(15 min)*.
   - In `final_agents()` at the bottom of `agents.py`, fill in **one** setting
     per algorithm. We will run it on **all five** classes, including the three
     you have never seen, on secret seeds, and score it relative to a UCB tuned
     separately for each class. Before you submit, write down your predicted
     rank. Submit your `agents.py` as instructed in class.
   - Super-group discussion (5 min), then each group has 2 minutes: the best
     hyperparameters for each of your classes, which algorithm won where, and
     one thing that surprised you.

## If you have time left (or at home)

- **Explore-then-commit** explores all arms equally. Can you stop exploring
  arms that are clearly bad early (*successive elimination*)?
- **ε-greedy**: let ε decay over time (e.g. ε_t = min(1, c/t) or c/√t).
  Relate what you see to the lecture.
- **Boltzmann**: use a temperature schedule (e.g. τ_t = C / log(1 + t)), or
  scale the rewards so that one τ works across classes.
- **UCB** assumes all arms are equally noisy. Estimate each arm's standard
  deviation and use it in the bonus. Which class should this help?
- **Robust means**: in class 4, one enormous reward can ruin the sample mean
  for a long time. Replace it by the *median of means*: split an arm's rewards
  into k blocks, average each block, and take the median of the averages.
- **Optimism**: give `Greedy` optimistic initial values Q_0(a) (e.g. counting
  them as one fake observation). How good does plain greedy get, and why?

## Food for thought

Every group tunes its algorithms on two classes, and then we test them on all
five. How well your choices carry over to classes you never saw is a small
version of a big question in machine learning and AI safety: a system that is
optimized hard against one benchmark tends to exploit the benchmark's quirks,
and can be surprisingly fragile elsewhere.
