# Week 5: Monte Carlo and TD prediction

Goal: learn the value function V_π of a fixed policy π from *episodes alone* -- no model, no
P, no R -- in the two ways of the Week 5 lecture: Monte Carlo, which waits for the actual
return, and temporal-difference (TD) learning, which bootstraps from its own estimates. Then
take them apart: which wins, when, and why? How should the step size shrink? What does each
of them assume about the world? How far does a reward travel? And can you trust an estimate
when the thing you worry about rarely happens?

This week is about **prediction** only: π is fixed and we just want to know how good it is.
Learning to act better (control) comes in Week 6.

## The setting

Our worlds are small enough that we know the true V_π exactly (we have the model, as in
Week 4), so we can measure how wrong an estimate is. The estimators never look at the model:
all they get is a list of episodes.

| World | States | γ | Policy π (default) | What it is |
| --- | --- | --- | --- | --- |
| `random_walk` | 5 | 1 | random | Sutton & Barto's random walk (Example 6.2): states A–E, start in C, step left or right at random, +1 for leaving on the right. `load("random_walk", size=19, left_reward=-1)` is their 19-state version |
| `gridworld` | 16 | 1 | random | the 4×4 gridworld of Week 4, reward −1 per step |
| `maze` | 52 | 0.99 | tired | the 8×8 maze of Week 4 (10 % slip) |
| `corridor` | 20 | 0.99 | tired | a corridor with the goal at the end, no reward until then (`length=...`) |
| `ghost_lair` | 841 | 0.99 | tired | the ghost's lair of Week 4: a short route past the lair, and a long one around it |

Policies: `"random"`, `"optimal"`, `"tired"` (the optimal action 70 % of the time, a random one
otherwise; `tired=0.1` changes the 30 %), `"risky"` (the tired version of the policy that is
optimal for γ = 0.95, which takes the short route past the ghost), or an array of your own.

```python
from pred_tools import *
world = load("maze")                    # load("maze", policy="random"), load("ghost_lair", tired=0.05), ...
episodes = world.episodes(100, seed=0)  # 100 episodes of the policy; same seed, same episodes
ep = episodes[0]
ep.states, ep.rewards, ep.terminated    # S_0..S_T;  rewards[t] = reward of the step S_t -> S_{t+1};  ended or cut off?
world.V_true                            # the true V_pi (never shown to the estimators)
world.error(V)                          # RMS error of V, weighted by how often pi visits each state
world.start, world.outcomes()           # the start state; P(goal), P(caught)
```

`rewards[t]` is R_{t+1} in Sutton & Barto's notation (and R_t in the Week 5 notes). An episode
ends either because S_T is terminal (`ep.terminated` is True; the value of a terminal state
is 0) or because it hit the time limit `world.max_steps` (1000 by default; then S_T is an
ordinary state).

**Estimators.** Both files define classes with the same interface: `est = MC(n_states, gamma,
alpha=..., V0=...)`, then `est.update(episode)` once per episode, and `est.V` is the current
estimate. The step size `alpha` is a number (constant) or a *schedule*: `one_over_n()` gives
α_n = 1/n, where n counts the updates of that state so far, and `one_over_n(beta, c)` gives
α_n = min(1, c/n^β).

**Learning curves.** `learning_curves({name: factory}, world, n_episodes, runs)` gives the
error after every episode in several runs; within a run, all estimators learn from *the same
episodes*. A factory is anything that makes a fresh estimator from `(n_states, gamma)`: the
class itself, or e.g. `partial(TD0, alpha=0.1)`.

```python
from mc import MC
from td import TD0, NStepTD
curves = learning_curves({"MC": MC, "TD(0), 0.1": partial(TD0, alpha=0.1)}, world, 100, runs=20)
plot_curves(curves); plt.show()
print(final_error(curves["MC"]))        # (mean, standard error) after the last episode
```

## What's in this directory

| File | What it is |
| --- | --- |
| `mc.py` | **Edited by half of your group:** `discounted_returns` and Monte Carlo prediction (`MC`). |
| `td.py` | **Edited by the other half:** `TD0` and `NStepTD`. |
| `pred_tools.py` | Worlds, episodes, true values, errors, learning curves, plots, and the checker. No need to edit it. |
| `first_look.py` | Task 3. |

Other useful things in `pred_tools`: `visits(episodes, n_states)` (how often each state was
visited), `returns_from_start(episodes, gamma)` (G_0 of each episode), `plot_world(world, V)`
(values -- or errors, or visit counts -- on the map), and `pretend_terminated(episodes)`.

## Groups

The same five groups as before. In each group, two of you take `mc.py` and two take `td.py`
(in a group of three, one person takes `mc.py`). Task 4 differs per group:

| Group | Task 4: the question you own |
| --- | --- |
| A1 | The n dial: where between TD(0) and Monte Carlo should we be? |
| A2 | Step sizes: what do the Robbins–Monro conditions buy us in practice? |
| A3 | Batch learning and the Markov property |
| B1 | How far does a reward travel? |
| B2 | Rare disasters: can you trust the estimate? |

## Tasks

0. **Setup** *(3 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python check_setup.py`.

1. **Implement** *(25 min)* the places marked `todo(...)`: the `mc.py` half implements
   `discounted_returns` and `MC.update`, the `td.py` half `TD0.update` and `NStepTD.update`.
   Check with

   ```bash
   uv run python week05_mc_td/pred_tools.py
   ```

   It checks each piece as soon as you have written it, first on a hand-made episode you can
   follow on paper, then on the random walk, whose true values are 1/6, 2/6, …, 5/6.

   - **Monte Carlo** moves V(S_t) towards the return G_t that actually followed. *Every-visit*
     MC does this for every t; *first-visit* MC only for the first time each state appears in
     the episode. With α_n = 1/n, V(s) is exactly the average of the returns seen from s --
     the incremental mean of the Week 5 lecture.
   - **TD(0)** moves V(S_t) towards R_{t+1} + γ V(S_{t+1}), one step at a time, in time
     order. Careful at the end: if the episode terminated, the value of S_T is 0, whatever
     `V` says; if it was cut off by the time limit, S_T is an ordinary state and we bootstrap
     from V(S_T).
   - **n-step TD** uses n rewards before it bootstraps:
     G_{t:t+n} = R_{t+1} + γR_{t+2} + … + γ^{n−1}R_{t+n} + γ^n V(S_{t+n}). Near the end of the
     episode there are fewer than n rewards left: take what there is, and bootstrap from
     V(S_T) only if the episode was cut off. n = 1 must give exactly TD(0), and n = ∞
     exactly every-visit MC with the same constant α -- the checker tests both, so the two
     halves of your group check each other.

   The `mc.py` half will be done first: then run the check of `NStepTD` against your `MC`,
   and read `td.py` with its authors. You will need to understand both files in Task 4 (and
   at the presentation, anyone may be asked about either).

2. **Teach-back** *(5 min)*. Each half explains its code to the other half: what happens at
   the end of an episode, and why the order of the updates matters.

3. **First look** *(7 min)*. Everyone:

   ```bash
   uv run python week05_mc_td/first_look.py
   ```

   It compares MC and TD(0) with two step sizes on the random walk and on the maze. Which
   wins where? Write down a guess *why* before you start Task 4 -- your question is one way
   of finding out.

4. **Your question** *(the rest of the practical, then at home)*. Write your experiments in a
   script of your own in this directory (e.g. `week05_mc_td/a1.py`). Use several runs (seeds)
   and show the spread; if two curves differ by less than their noise, say so. Prepare
   **one plot and one sentence** for the class.

   - **A1 -- The n dial.** n-step TD runs from TD(0) (n = 1) to Monte Carlo (n = ∞, e.g.
     `n=10**6`). For n in 1, 2, 4, 8, 16, 32, ∞ and α in 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1,
     measure the error averaged over the first 10 episodes (50–100 runs) on
     `load("random_walk", size=19, left_reward=-1)`, and plot it against α, one line per n
     (you are reproducing Sutton & Barto's Figure 7.2). Then do the same on `maze` (first 25
     episodes) and `ghost_lair` (first 50). Which n is best on which world? Form a hypothesis
     about what decides it, and test it. One test: start the estimators not at 0 but at the
     true values plus some noise (in a factory: `e = NStepTD(S, g, n=n, alpha=a); e.V[:] =
     world.V_true + 0.2 * rng.normal(size=S); return e`). Relate what you find to the bias
     and variance of the targets in the lecture.
   - **A2 -- Step sizes.** Compare, for both MC and TD(0): constant α (0.01, 0.03, 0.1, 0.3),
     `one_over_n()` (α_n = 1/n), and `one_over_n(beta, c)` for β in 0.5, 0.6, 0.8 and c in
     1, 2, 5. Use `load("random_walk", size=19, left_reward=-1)` and `maze`, 2000 episodes,
     log-log plots. Which schedules satisfy the Robbins–Monro conditions? Which is best for MC,
     and which for TD -- and why is 1/n so different for the two? (Think about what V(s) is
     an average of, under 1/n.) Constant α never converges: measure the error it settles at,
     and for MC, predict it (the stationary variance of V(s) under the update
     V ← V + α(G − V) is α/(2 − α) · Var(G | s) -- derive it; on the 5-state random walk the
     returns are 0 or 1). Finally, let the world change: 500 episodes of the tired policy,
     then 500 of `load("maze", policy="optimal")`, measuring the error against the *current*
     policy's values. Who keeps up? What does this mean for control (Week 6), where the policy
     changes all the time?
   - **A3 -- Batch learning and the Markov property.** You have N episodes and may reuse them
     as often as you like. *Batch MC*: the average of the returns from each state (MC with 1/n,
     one pass). *Batch TD*: go through all N episodes, add up the TD(0) increments
     R + γV(S') − V(S) for each state without changing V, then apply α times the sums, and
     repeat until V stops changing (Sutton & Barto, Section 6.3). *Certainty equivalence*:
     count transitions to estimate the Markov chain of the policy -- P̂(s′|s), the mean reward
     R̂(s), and how often s leads to the end -- and solve V = R̂ + γP̂V exactly, as in Week 4.
     Check that batch TD and certainty equivalence agree. Then compare batch MC and batch TD on
     `random_walk`, `maze`, and `ghost_lair`, for N = 5, 20, 100, 500, 2000. Now hide the ghost:
     `load("ghost_lair", observe="position")` gives the estimators only the agent's cell, one
     value per cell. No table over cells can be perfect any more: `world.error(world.V_true)`
     is the best any of them can do. Which method gets close to it, and which does not, even
     with lots of data? Which assumption does the loser rely on? (In Week 7, V is a neural
     network that cannot tell all states apart either.)
   - **B1 -- How far does a reward travel?** On `load("corridor", length=L)` (the only reward is
     at the goal), for L = 5, 10, 20, 40: how many episodes do MC and TD(0) (α = 0.5 and 1)
     need until the error is below 0.05? Explain the numbers. Then write `BackwardTD0`, a
     subclass of `TD0` that processes the steps of each episode from the last to the first
     (you met the importance of update order in Week 4). What changes, on the corridor and on
     `maze`? What do you give up -- could an agent do this while the episode is still
     running? Finally, time limits: with `load("corridor", max_steps=25)` and
     `load("maze", max_steps=20)`, many episodes are cut off. `world.V_true` is still the value
     *without* a time limit. Compare MC, TD(0), and TD(0) on `pretend_terminated(episodes)` (a
     common bug: treating "out of time" as "the end"). Who is right, and why? Which value do
     we actually want here? What happens with `load("corridor", max_steps=15)`?
   - **B2 -- Rare disasters.** `load("ghost_lair", tired=0.05)`: the ghost catches the agent in
     2 % of episodes (`world.outcomes()`). Suppose you evaluate this policy with N = 10, 30,
     100, or 300 episodes (300 different seeds each). How often does an evaluation see no
     catch at all? What is the distribution of the MC estimate of V(start): its mean, its
     median, how often it is too optimistic? If you saw no catch in N episodes, what can you
     honestly say about the probability of being caught (look up the "rule of three")? Then:
     is the mean all we care about? Estimate quantiles of the return from the start (τ = 0.01,
     0.05, 0.25, 0.5) with the quantile version of the lecture's Robbins–Monro median
     example, θ ← θ + α(τ − 1[G < θ]), one θ per state, updated with G_t as in MC (θ starting
     at 0, α_n = 1/n^0.6 per state, 5000 episodes; check against `np.quantile` of
     `returns_from_start` of 50 000 episodes).
     Then try a "TD" version, with the target R_{t+1} + γθ(S_{t+1}) instead of G_t. Are its
     answers even possible? Why does bootstrapping work for the mean but not for a quantile?
     What would you need to estimate instead? Finally: what evaluation would you want to see
     before deploying a policy whose failures are rare but bad?

## Presentations and BRUTE

- **Presentations** at the start of the Week 6 practical, on **4 November** (28 October is a
  public holiday): 3 minutes per group, one plot and one sentence, then questions -- to any
  member of the group, about any part of the work, including the half of the code they did
  not write.
- **BRUTE**, by **9:15 on 4 November**, each of you individually: a few sentences on your
  group's question -- what you did, what you found (with at least one number), and why you
  think it happens.

## If you have time left (or at home)

- **First-visit or every-visit?** On the random walk and the maze, does it matter?
- **The whole spectrum at once.** n-step TD makes us choose n. TD(λ) averages over all n with
  weights (1 − λ)λ^{n−1} (Sutton & Barto, Chapter 12). Implement the offline λ-return
  algorithm and compare it with the best n of A1.
- **Is the error measure fair?** `world.error(V, weights="uniform")` weighs every state the
  same, instead of by how often π visits it. Does it change any of your conclusions?
