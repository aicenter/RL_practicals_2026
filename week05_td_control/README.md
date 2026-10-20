# Week 5: Temporal-difference learning, prediction and control

Goal: learn from *one transition at a time* -- no model, no P, no R -- the way the Week 5
lecture does it: replace the expectation in a Bellman backup by the transition you just saw,
shrink the step size, and trust the Robbins–Monro theorem. First for prediction (TD(0): how
good is a fixed policy?), then for control (SARSA, Expected SARSA, Q-learning: how do I act
well?). Then take them apart: what do the step-size conditions buy, what does TD(0) actually
compute from a finite batch, what is the difference between the policy that *acts* and the
policy that is *learned*, why does exploration have to vanish and why is that so slow, and can
you learn to act well from data somebody else collected?

Monte Carlo methods, which wait for the whole return, are next week. Everything here
bootstraps.

## The setting

Our worlds are small enough that we know the truth exactly (we have the model, as in Week 4):
the true V_π of a policy, and the optimal q_*. So we can measure how wrong an estimate is and
how far a learned policy is from optimal. The learners never look at the model: they get a
list of episodes (prediction), or one sampled transition at a time (control).

| World | States | γ | Policy π (default) | What it is |
| --- | --- | --- | --- | --- |
| `random_walk` | 5 | 1 | random | Sutton & Barto's random walk (Example 6.2): states A–E, start in C, step left or right at random, +1 for leaving on the right. `load("random_walk", size=19, left_reward=-1)` is their 19-state version |
| `gridworld` | 16 | 1 | random | the 4×4 gridworld of Week 4, reward −1 per step |
| `maze` | 52 | 0.99 | tired | the 8×8 maze of Week 4 (10 % slip) |
| `corridor` | 20 | 0.99 | tired | a corridor with the goal at the end, no reward until then (`length=...`) |
| `ghost_lair` | 841 | 0.99 | tired | the ghost's lair of Week 4: a short route past the lair, and a long one around it. **Our cliff walking**: the short route is optimal only if you never slip |
| `maxbias` | 3 | 1 | random | Sutton & Barto's Example 6.7: from A, RIGHT ends with reward 0; LEFT leads to B, where each of 10 actions ends with a reward N(−0.1, 1) |

Policies: `"random"`, `"optimal"`, `"tired"` (the optimal action 70 % of the time, a random one
otherwise; `tired=0.1` changes the 30 %), `"risky"` (the tired version of the policy that is
optimal for γ = 0.95, which takes the short route past the ghost), or an array of your own.

```python
from td_tools import *
world = load("maze")                    # load("maze", policy="random"), load("ghost_lair", tired=0.05), ...
episodes = world.episodes(100, seed=0)  # 100 episodes of the policy; same seed, same episodes
ep = episodes[0]
ep.states, ep.actions, ep.rewards, ep.terminated   # S_0..S_T; A_0..A_{T-1}; rewards[t] = R_{t+1}; ended or cut off?
world.V_true                            # the true V_pi (never shown to the estimators)
world.error(V)                          # RMS error of V, weighted by how often pi visits each state
optimal_values(world)                   # (v_*, q_*) from value iteration on the model
gap(world, greedy_policy(Q))            # v_*(start) - v_greedy(start): how far a policy is from optimal
s2, r, done = world.step(s, a, rng)     # one sampled transition, for control
```

`rewards[t]` is R_{t+1} in the notation of the lecture. An episode ends either because S_T is
terminal (`ep.terminated` is True; the value of a terminal state is 0) or because it hit the
time limit `world.max_steps` (1000 by default; then S_T is an ordinary state).

**Prediction.** `est = TD0(n_states, gamma, alpha=..., V0=...)`, then `est.update(episode)`
once per episode; `est.V` is the current estimate. The step size `alpha` is a number
(constant) or a *schedule*: `one_over_n()` gives α_n = 1/n, where n counts the updates of
that state so far, and `one_over_n(beta, c)` gives α_n = min(1, c/n^β).
`BatchTD0(n_states, gamma).fit(episodes)` runs batch TD(0) on a fixed set of episodes to
convergence.

**Control.** `agent = SARSA(n_states, n_actions, gamma, alpha=...)` (or `ExpectedSARSA`,
`QLearning`), with `agent.Q` the table and `agent.update(s, a, r, s2, a2, done, eps)` one
learning step. You do not call `update` yourself in the experiments: `run_control` lets the
agents act ε-greedily in a world and learn online,

```python
res = run_control({"SARSA": SARSA, "Q-learning": QLearning}, load("ghost_lair"),
                  n_episodes=1000, runs=10, epsilon=0.1, alpha=0.3)
res["SARSA"]["return"]        # (runs, episodes): the online return of every episode
res["SARSA"]["gap"]           # (runs, episodes): v_*(start) - v_greedy(start) after each episode
plot_control(res, "return"); plot_control(res, "gap"); plt.show()
plot_policy(world, res["SARSA"]["Q"])                      # greedy arrows over max_a Q on the map
```

`epsilon` is a number (fixed) or a schedule over episodes, e.g. `glie(eps0=0.1, tau=100)` for
ε_k = ε₀·τ/(τ+k). `start="random"` starts every episode in a random state (exploring starts).
To learn from *logged* episodes instead, call `agent.learn_from(episode, eps)`.

**Learning curves** for prediction work as in Week 4's tools: `learning_curves({name: factory},
world, n_episodes, runs)`, where a factory makes a fresh estimator from `(n_states, gamma)`,
e.g. `partial(TD0, alpha=0.1)`; then `plot_curves`, `final_error`.

## What's in this directory

| File | What it is |
| --- | --- |
| `td.py` | **Edited by half of your group:** `TD0` and `BatchTD0`. |
| `control.py` | **Edited by the other half:** `SARSA`, `ExpectedSARSA`, `QLearning` (and, optional, `DoubleQLearning`). |
| `td_tools.py` | Worlds, episodes, true values, errors, learning curves, control runs, plots, and the checker. No need to edit it. |
| `first_look.py` | Task 3. |

Other useful things in `td_tools`: `visits(episodes, n_states)`, `returns_from_start(episodes,
gamma)`, `plot_world(world, V)` (values, errors or visit counts on the map),
`pretend_terminated(episodes)`, `empirical_model(episodes, n_states)` (the counts R̂, P̂, N),
`certainty_equivalence(episodes, n_states, gamma)` (the Week 4 direct solve on them),
`two_state_batch()` (the eight episodes of the lecture).

## Groups

The same five groups as before. In each group, two of you take `td.py` and two take
`control.py` (in a group of three, one person takes `td.py`). Task 4 differs per group:

| Group | Task 4: the question you own |
| --- | --- |
| A1 | Step sizes: what do the Robbins–Monro conditions buy, and what does a constant α buy instead? |
| A2 | What does TD(0) compute from a finite batch? (Proposition 5.2, and when it is wrong) |
| A3 | On-policy or off-policy: who learns the better policy, who earns more while learning? |
| B1 | Exploration: GLIE, and why "greedy in the limit" takes so long |
| B2 | Learning from somebody else's data: behaviour policy, target policy, coverage |

## Tasks

0. **Setup** *(3 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python check_setup.py`.

1. **Implement** *(25 min)* the places marked `todo(...)`: the `td.py` half implements
   `TD0.update` and `BatchTD0.sweep`, the `control.py` half the three `target` methods.
   Check with

   ```bash
   uv run python week05_td_control/td_tools.py
   ```

   It checks each piece as soon as you have written it, first on hand-made transitions you
   can follow on paper, then on the random walk (true values 1/6, …, 5/6) and the gridworld.

   - **TD(0)** moves V(S_t) towards R_{t+1} + γ V(S_{t+1}), one step at a time, in time
     order. Careful at the end: if the episode terminated, the value of S_T is 0, whatever
     `V` says; if it was cut off by the time limit, S_T is an ordinary state and we bootstrap
     from V(S_T).
   - **Batch TD(0)** does the same over a fixed batch, but *accumulates* the increments of a
     whole sweep and applies them at once, then repeats until nothing moves. The checker
     verifies the lecture's Proposition 5.2 on the two-state example (V(A) = V(B) = 3/4) and
     on the random walk: the result is the value function of the empirical model, whatever
     the step size (as long as α ≤ 1/max_s N(s), the default).
   - **The three targets** are one line each. SARSA uses `a2`, the action the behaviour has
     already chosen for `s2`. Expected SARSA averages Q(s2, ·) under the ε-greedy policy at
     s2 (probability ε/|A| for every action, plus 1 − ε for the greedy one; split ties). With
     ε = 0 it must equal Q-learning -- the checker tests that, so the two authors check each
     other. All three: if `done`, the target is r alone.

   The `control.py` half will be done first: read `td.py` with its authors, then try
   `uv run python week05_td_control/first_look.py` on their behalf.

2. **Teach-back** *(5 min)*. Each half explains its code to the other: what happens at the end
   of an episode; which policy chooses `a2`, and which policy's value each target plugs in.

3. **First look** *(7 min)*. Everyone:

   ```bash
   uv run python week05_td_control/first_look.py
   ```

   Left: TD(0) on the random walk with three step sizes. Right: the three control algorithms
   in the ghost's lair with ε = 0.1, their online return and the quality of their greedy
   policy. Which step size wins early, which late? Which agent earns more *while learning*,
   and which ends with the better greedy policy -- and why are these not the same agent?
   Write down a guess before Task 4; your question is one way of finding out.

4. **Your question** *(the rest of the practical, then at home)*. Write your experiments in a
   script of your own in this directory (e.g. `week05_td_control/a1.py`). Use several runs
   (seeds) and show the spread; if two curves differ by less than their noise, say so.
   Prepare **one plot and one sentence** for the class.

   - **A1 -- Step sizes.** For TD(0) compare constant α (0.01, 0.03, 0.1, 0.3),
     `one_over_n()` (α_n = 1/n), and `one_over_n(beta, c)` for β in 0.5, 0.6, 0.8 and c in
     1, 2, 5, on `load("random_walk", size=19, left_reward=-1)` and `maze`, 2000 episodes,
     log-log plots. Which schedules satisfy the Robbins–Monro conditions (lecture, Theorem
     3.1)? Which is best early, which late? Why is 1/n, the sample mean, a poor choice for
     TD(0) although it was the natural one for bandits -- what is V(s) an average *of*, and
     does that quantity stay put? Constant α never converges: measure the error it settles at
     as a function of α (predict the slope on a log-log plot before you measure it). Then
     let the world change: 500 episodes of the tired policy, then 500 of
     `load("maze", policy="optimal")`, measuring the error against the *current* policy's
     values. Who keeps up? What does this mean for control, where the policy changes after
     every update, and for deep RL, where everybody uses constant steps?
   - **A2 -- What does TD(0) compute from a finite batch?** You have N episodes and may reuse
     them as often as you like. Start from the lecture's Proposition 5.2: batch TD(0)
     converges to the value function of the empirical model. Verify it beyond the checker,
     on `maze` and `ghost_lair` for N = 5, 20, 100, 500: `BatchTD0(...).fit(eps)` against
     `certainty_equivalence(eps, ...)`. How many sweeps does it take, and how does the step
     size (≤ 1/max N(s), the proposition's condition) affect that? What happens with a larger
     α? Now the online algorithm: how far is plain `TD0` after *one* pass over the same
     episodes from the batch fixed point, and from the truth? After k passes with a decaying
     α? Then break the proposition's assumption: `load("ghost_lair", observe="position")`
     gives the learner only the agent's cell, not the ghost's. No table over cells can be
     exact any more: `world.error(world.V_true)` is the best any estimator can do. How close
     does batch TD(0) get, even with N = 2000, and why not closer? (The empirical model it
     solves is a Markov chain over cells. Is the process over cells Markov?) In Week 7, V is
     a neural network that cannot tell all states apart either.
   - **A3 -- On-policy or off-policy?** The ghost's lair is our cliff: `load("ghost_lair")`.
     Run SARSA, Expected SARSA and Q-learning with fixed ε in 0.02, 0.1, 0.3 (α = 0.3, then
     0.1; a few thousand episodes, 10 runs). For each: the online return, the gap of the
     greedy policy, and *which route* the greedy policy takes (`plot_policy`; or build a
     `World` with the greedy policy as in `td_tools.policy_values` and ask `outcomes()` how
     often it is caught). Who has the shortest route, who the best online return, and why are
     these different agents? How does the answer depend on ε? Relate it to what each target
     evaluates: the greedy policy, or the one that explores. Expected SARSA has the same
     target *in expectation* as SARSA: compare the variance of their learning curves, and
     their speed, against α. Which of the three would you run on a robot that breaks when
     caught, and which in a simulator?
   - **B1 -- Exploration and GLIE.** Start on `gridworld` and `maze`, then move to the lair.
     Compare fixed ε = 0.1 with `glie(0.1, tau)` for τ in 30, 100, 300, 1000, for SARSA and
     Q-learning: which converges to q_* (compare the whole table with `optimal_values`), and
     which to something else -- the values of which policy? Which needs the decay at all?
     Then watch the greedy policy of SARSA in the lair under the GLIE schedule for many
     episodes: does it move to the short route as ε → 0, as the theorem promises? If not
     (expect not), explain why: how often are the states on the short route visited once a
     safe route is learned, and what would have to happen to their values for the policy to
     move? "Nothing happens" is a legitimate finding here, if you can say why. Two remedies
     to test: `start="random"` (exploring starts), and optimistic initialisation (`Q0=...`
     at the largest value any pair can have; is Q0 = 0 already optimistic in this world?).
     Which remedy works for which algorithm, and what does each assume that a robot does not
     have?
   - **B2 -- Learning from somebody else's data.** You get logged episodes of a *behaviour*
     policy b and want a good policy without ever acting: `agent.learn_from(episode, eps)`.
     On `maze`, log 2000 episodes of `policy="random"` and of `"tired"`, and let SARSA (with
     `eps` = the behaviour's randomness) and Q-learning learn from them, several passes. What
     does each table converge to -- compare with `optimal_values(world)` and with
     `world.V_true` of the behaviour -- and how good is each greedy policy (`gap`)? Why can
     the greedy policy of SARSA be nearly optimal although its table is not q_*? (Week 4 has
     a theorem about one step of improvement.) Then shrink the coverage: in `ghost_lair`, log
     `policy="risky"` with `tired` = 0.3, 0.1, 0.03, 0 (the behaviour takes the short route
     more and more deterministically). Can Q-learning still find the safe route, and when
     does it stop being able to? Count the (s, a) pairs it never saw (`agent.N == 0`). State
     the coverage condition of the lecture in terms of what you observed. Finally the
     *optional* part, the lecture's Section 6.5: implement `DoubleQLearning` in `control.py`
     (the checker has a test) and run Q-learning against it on `load("maxbias")` with ε =
     0.1, α = 0.1, 300 episodes, 200 runs: plot how often each takes LEFT from A over the
     episodes (`res[...]["first_action"] == 1`; the optimal rate is ε/10). Why does Q-learning
     go wrong first and recover later, and what does Double Q-learning change?

## Presentations and BRUTE

- **Presentations** at the start of the Week 6 practical: 3 minutes per group, one plot and
  one sentence, then questions -- to any member of the group, about any part of the work,
  including the half of the code they did not write.
- **BRUTE**, by 9:15 before that practical, each of you individually: a few sentences on your
  group's question -- what you did, what you found (with at least one number), and why you
  think it happens.

## If you have time left (or at home)

- **How far does a reward travel?** On `load("corridor", length=L)` (the only reward is at
  the goal), for L = 5, 10, 20, 40: how many episodes does TD(0) (α = 0.5 and 1) need until
  the error is below 0.05? Explain the numbers. Then write `BackwardTD0`, a subclass of `TD0`
  that processes the steps of each episode from the last to the first (you met the
  importance of update order in Week 4). What changes? What do you give up -- could an agent
  do this while the episode is still running?
- **Time limits.** With `load("corridor", max_steps=25)` and `load("maze", max_steps=20)`,
  many episodes are cut off. `world.V_true` is the value *without* a time limit. Compare
  TD(0) on the episodes and on `pretend_terminated(episodes)` (a common bug: treating "out
  of time" as "the end"). Who is right, and why? Which value do we actually want here?
- **One update, three targets, one class.** Expected SARSA with a target policy you choose
  (not the behaviour): pass the target's ε separately. With target ε = 0 you have Q-learning,
  with target ε = behaviour ε you have Expected SARSA. What happens in between?
