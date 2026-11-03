# Week 6: Monte Carlo, and learning from somebody else's data

Goal: last week every algorithm bootstrapped. This week nothing does. Monte Carlo waits for the
whole return, which is unbiased and needs no Markov assumption. You will implement it, for
prediction and for control, and put it against your SARSA and Q-learning from Week 5. Then you
will use the return for something TD cannot do as directly: evaluating a policy we never ran,
from episodes somebody else logged, by importance sampling.

Along the way: why can an algorithm have the more accurate table and the worse policy? Where
between TD and Monte Carlo should control sit? What happens to Monte Carlo when an episode
refuses to end? And how much should you trust a log when it says that a dangerous policy is fine?

Importance sampling comes back in Weeks 8 and 9: policy gradients meet the same ratio as soon as
they reuse old data.

## The setting

The worlds, policies, and tools are those of Week 5: `maze`, `ghost_lair`, `corridor`,
`random_walk`, and `gridworld`, with the policies `"random"`, `"optimal"`, `"tired"`, and
`"risky"`. We still know the truth, the learners still don't.

```python
from mc_tools import *                  # always first: it makes the Week 5 tools available
from mc import MC, MCControl
from importance import OrdinaryIS, WeightedIS, tail_ratios, effective_sample_size
from control import SARSA, QLearning    # your Week 5 code
from td import TD0

world = load("maze")                    # as in Week 5
episodes = world.episodes(100, seed=0)
ep = episodes[0]
ep.states, ep.actions, ep.rewards, ep.terminated   # S_0..S_T; A_0..A_{T-1}; rewards[t] = R_{t+1}

start_value(world)                      # the true v_pi at the start
world.policy                            # (states, actions): the probabilities pi(a | s)
returns_from_start(episodes, world.gamma)   # G_0 of each episode
```

**Prediction**, as in Week 5: `est = MC(n_states, gamma, alpha=..., first_visit=...)`, then
`est.update(episode)` once per episode; `est.V` is the estimate. The default step size is
`one_over_n()`, the running average of the returns.

**Control**, as in Week 5, with `MCControl` as one more agent. Use `partial` to give each agent
its own settings:

```python
agents = {"MC": MCControl, "SARSA": partial(SARSA, alpha=0.3)}
res = run_control(agents, load("maze"), n_episodes=500, runs=5, epsilon=0.1)
plot_control(res, "gap"); plot_control(res, "return"); plot_steps(res); plt.show()
res["MC"]["steps"]                      # (runs, episodes): the length of every episode
res["MC"]["agent"].N                    # the agent of the last run, and its visit counts
```

`gap` is v_\*(start) − v_greedy(start), as before, computed every `gap_every` episodes (50 by
default). `start="exploring"` gives exploring starts: a random state and a random first action.

To learn from logged episodes, call `agent.learn_from(episode)`, on Week 5 agents and on
`MCControl` alike. Three new measures for a table Q: `q_error(world, Q)` compares it with q_π of
the world's policy (weighted by how often π uses each pair), `q_true(world)` is that q_π, and
`count_loops(world, Q)` counts the states that the greedy policy sends round a loop of two.

**Off-policy evaluation.** Target and behaviour are arrays like `world.policy`:

```python
logs, target = load("maze", tired=0.3), load("maze", tired=0.1)
pi, b = target.policy, logs.policy
est = WeightedIS(logs.n_states, logs.gamma, pi, b)    # or OrdinaryIS
for ep in logs.episodes(1000, seed=0):
    est.update(ep)
est.V[logs.start], start_value(target)                # the estimate, and the truth
tail_ratios(ep, pi, b)[0]                             # the weight of a whole episode
```

## What's in this directory

| File | What it is |
| --- | --- |
| `mc.py` | **Edited by half of your group:** `discounted_returns`, `MC`, and `MCControl`. |
| `importance.py` | **Edited by the other half:** `tail_ratios`, `effective_sample_size`, `OrdinaryIS`, and `WeightedIS`. |
| `nstep.py` | For group A2: `NStepSARSA`. |
| `mc_tools.py` | The Week 5 tools, what is new this week, and the checker. No need to edit it. |
| `first_look.py` | Task 3. |

## Groups

The same five groups as before. In each group, two of you take `mc.py` and two take
`importance.py`. Task 4 differs per group:

| Group | Task 4: the question you own |
| --- | --- |
| A1 | The reversal: a better table, a worse policy? |
| A2 | The n dial: where between TD and Monte Carlo should control sit? |
| A3 | The episode that never ends |
| B1 | How far can logs reach? |
| B2 | A dangerous policy, judged from safe logs |

## Tasks

0. **Setup** *(3 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python check_setup.py`. Then bring your Week 5 code along, one person's working
   version per group:

   ```bash
   cp week05_td_control/td.py week06_mc/
   cp week05_td_control/control.py week06_mc/
   ```

   If your `td.py` or `control.py` from Week 5 is not finished, tell us now and we will help you
   out: this week needs them working.

1. **Implement** *(25 min)* the places marked `todo(...)`, except in `nstep.py`. Check with

   ```bash
   uv run python week06_mc/mc_tools.py
   ```

   It checks each piece as soon as you have written it, first on hand-made episodes you can
   follow on paper.

   - **`discounted_returns`**: one backward loop. **`MC.update`** moves V(S_t) towards G_t,
     forwards in time; first-visit skips repeated visits within one episode.
   - **`MCControl.learn`** is the same for Q(S_t, A_t). It is first-visit by default, and a
     "visit" is to a *pair* (s, a) here.
   - **`tail_ratios`**: ρ_{t:T−1}, one backward loop, like the returns.
   - **`effective_sample_size`**: (Σw)² / Σw². Watch out for all-zero weights.
   - **`OrdinaryIS`** keeps V(s) the running average of ρG. **`WeightedIS`** adds ρ to C(s) and
     moves V(s) by (ρ / C(s)) (G − V(s)). If C(s) is still 0, there is nothing to learn yet.

   The `importance.py` half does not need `mc.py`, so you can work in parallel.

2. **Teach-back** *(5 min)*. Each half explains its code to the other: why does `MCControl` learn
   only at the end of an episode? What is a weight of 0, and what is a weight of 50?

3. **First look** *(7 min)*. Everyone:

   ```bash
   uv run python week06_mc/first_look.py
   ```

   Left: prediction on the maze, Monte Carlo against TD(0). Middle: control on the same maze.
   Right: importance sampling. Who wins on the left? Does that carry over to the middle, and at
   what cost in steps? What does the effective sample size say about the 2000 logged episodes?
   Write down a guess before Task 4; your question is one way of finding out.

4. **Your question** *(30 min, finish at home if needed)*. Write your experiments in a script of
   your own in this directory (e.g. `week06_mc/a1.py`). The bullets say what to measure, the
   questions say what to explain. Use at least 5 runs (seeds) and show the spread: some of these
   results swing a lot, and one lucky run proves nothing. Prepare **one plot and one sentence**
   for the class.

   - **A1 -- The reversal.** Monte Carlo won prediction on the maze. First, control: MC against
     SARSA (α = 0.3) in `load("ghost_lair")`, ε = 0.1, 6000 episodes, 5 runs, `plot_control(res,
     "gap")`. Maybe control is hard for MC because the policy keeps changing under it? Test that
     by removing the change: **freeze the policy**. Log 1000 episodes of `load("maze")` (10
     seeds), and let `MCControl` and SARSA (`alpha=one_over_n(0.6)`) learn from the same log with
     `learn_from`. After 30, 100, 300, and 1000 episodes record `q_error(world, agent.Q)`,
     `gap(world, greedy_policy(agent.Q))`, and `count_loops(world, agent.Q)`. Who has the better
     table? Who has the better policy? Was it the moving target? Look at one loop (`plot_policy`
     shows the arrows). Why can MC estimates point two neighbouring states at each other, and what
     stops TD estimates from doing so? Relate this to "they solve different equations".
   - **A2 -- The n dial.** Implement `NStepSARSA` in `nstep.py` (the checker tests n = 1 against
     SARSA and n = ∞ against MC). Then n = 1, 2, 4, 8, 16, ∞ (`n=10**6`), α = 0.1 and 0.3, ε =
     0.1, on `maze` (500 episodes) and `ghost_lair` (1500 episodes), 5 runs: the final gap (mean
     of the last three checks) and the mean gap over all of learning, against n. Which n ends
     best, and which is best on the way there? Does the best n depend on α? For prediction from V =
     0, large n wins on these same worlds, because with n = 1 a reward travels back only one state
     per episode. What is different about control?
   - **A3 -- The episode that never ends.** On `load("maze")`, ε = 0, 1000 episodes, 8 runs: MC
     and SARSA (α = 0.3). Compare gaps and `plot_steps`. Why does MC fail, and how does it get
     stuck? What would have to happen for it to get out, within one episode? Now try exploring
     starts for MC (`start="exploring"`). Then look harder at SARSA: why does it work at all with
     no exploration? Run it with `Q0=-1`. What did Q0 = 0 do for you, given that a step costs
     about −0.01? Finally, time limits: MC, SARSA, and Q-learning with ε = 0.1 and
     `load("maze", max_steps=m)` for m = 20, 30, 60. Who survives, and what does each of them do
     with an episode that gets cut off?
   - **B1 -- How far can logs reach?** Log 20 000 episodes of `load("maze", tired=0.3)`, target
     `tired=0.1`. Compute the weight and the return of every episode:

     ```python
     w = np.array([tail_ratios(ep, pi, b)[0] for ep in episodes])
     G = returns_from_start(episodes, logs.gamma)
     ```

     Split them into chunks of N = 10, 30, 100, 300, 1000. Per chunk, the ordinary estimate
     (mean of wG), the weighted one (sum of wG / sum of w), and the ESS. Plot the RMSE of both
     against N. What does the mean of the ordinary estimates do, and what do the medians do?
     Then the collapse: corridors with `length=L`, L = 5, 10, 20, 40, 80, tired 0.3 to 0.1, 1000
     episodes per chunk: ESS against L. What law is that, and why? Finally the cut: target
     `load("maze", policy="optimal")`, behaviour `tired=t` for t = 0.02, 0.05, 0.1, 0.3. How many
     episodes keep a non-zero weight, and how many of their steps can be used
     (`tail_ratios(...) > 0`)? Why does nobody run off-policy Monte Carlo control?
   - **B2 -- A dangerous policy, judged from safe logs.** The company's logged policy is
     `load("ghost_lair", tired=0.1)`. Somebody proposes the short route,
     `load("ghost_lair", policy="risky", tired=0.1)`, and wants to know if it is safe without
     trying it. We know the truth (`.outcomes()`, `start_value`); pretend you don't. Log 60 000
     episodes, split them into chunks of N = 30, 100, 300, 1000, 3000, and compute the weighted
     estimate per chunk (as in B1). How often does the estimate say that the new policy is at
     least as good as the logged one? Plot that against N. Compare the ordinary estimate over the
     whole log with the truth, and look at the largest weights: which episode is it, and what
     share of the total weight does it hold? Now make the logs safer: logging `tired=0.03`, `0.01`,
     and `0.0`, N = 1000. Does safer logging help or hurt, and why? Print the ESS and the mean of
     the weights (what should that be?): which of them warns you, and when? What would you demand
     to see before signing off on this policy?

   *Optional, for whoever finishes early* (any group): see the end of this page.

## Presentations and BRUTE

- **Presentations** at the start of next week's practical, on **11 November**: 3 minutes per
  group, one plot and one sentence, then questions -- to any member of the group, about any part
  of the work, including the half of the code they did not write.
- **BRUTE**, by **9:15 on 11 November**, each of you individually: a few sentences on your group's
  question -- what you did, what you found (with at least one number), and why you think it
  happens.

## If you have time left (or at home)

- **First-visit or every-visit?** Try both in control: `partial(MCControl, first_visit=False)` on
  the maze, ε = 0.1, 500 episodes, 10 runs. Does it matter? Which episodes dominate the averages?
- **The hidden ghost.** `load("ghost_lair", observe="position")`, 2000 episodes: compare `MC` with
  `certainty_equivalence` (what batch TD(0) computes), and with the best any table can do,
  `world.error(world.V_true)`. Which assumption does the loser rely on? Then try control in the
  same world: does the winner of prediction win control? We don't think the answer is simple.
- **Per-decision importance sampling.** A reward at step k cannot depend on actions after k, so
  G can be reweighted term by term: Σ γ^k ρ_{0:k} R_{k+1}. Does it beat ordinary IS?
- **Clipping instead of cutting.** Weight each step by min(1, π/b) instead of π/b, as Retrace does
  (see the lecture). What happens to the variance on B1's corridors, and what do you pay for it?
