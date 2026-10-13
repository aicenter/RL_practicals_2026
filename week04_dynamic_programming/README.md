# Week 4: Dynamic programming

Goal: implement the dynamic programming algorithms of the Week 4 lecture -- policy
evaluation, value iteration, and policy iteration -- and then take them apart: when can
you stop, how much should you evaluate before you improve, in which order should you
update, what does γ do, and how far does it all scale? At the end, every group builds the
cheapest solver it can, and we see whose survives worlds it has never seen.

## The worlds

We use **Prag-man** (`src/pragman/`), the Pac-Man-like gridworld that will stay with us for
the rest of the course. Here we need it as a *known* MDP: `dp_tools.load` writes a world out as
the arrays of the lecture, P(s' | s, a) and R(s, a), over all states reachable from the start.

| World | States | γ | What it is |
| --- | --- | --- | --- |
| `gridworld` | 16 | 1 | the 4×4 gridworld of the lecture: reward −1 per step, two terminal corners (with γ = 1, only for policy evaluation) |
| `maze` | 52 | 0.99 | an 8×8 maze; 10 % of the time the agent slips sideways |
| `ghost_lair` | 841 | 0.99 | two routes to the goal: short, past the ghost's lair, or long, around it. The ghost wanders until you come within 4 steps; then it chases you |
| `pellets` | 752 | 0.999 | the maze with 4 pellets to eat; the episode ends when the last one is eaten |

Rewards (Prag-man defaults): −0.01 per step, +1 for the goal, +0.1 per pellet and +1 for
the last one, −1 for being caught. The state includes the ghost's position and the
remaining pellets -- that is why `ghost_lair` has 841 states and not 29.

```python
from dp_tools import *
model, gamma = load("ghost_lair")       # load("maze", slip_prob=0.3), load("maze", gamma=0.9), ...
model.R                                 # (S, A) expected rewards
model.expect(V)                         # (S, A): sum_s' P(s'|s,a) V(s')
model.backups                           # how many backups the model has computed so far
print(model.describe(0))                # the board of state 0
```

**Backups.** One *backup* is one expectation Σ_s' P(s'|s,a) V(s') for one pair (s, a): one
look at the model. A sweep of value iteration costs S × A backups; a sweep of policy
evaluation for a deterministic policy costs S. `model.expect` counts them. Backups are our
measure of the cost of an algorithm: unlike seconds, they don't depend on your laptop, nor
on how cleverly you use numpy. (`model.P` gives the whole array, uncounted; we need it only for
exact policy evaluation, and it is switched off for `solve` in Task 3.)

## What's in this directory

| File | What it is |
| --- | --- |
| `dp.py` | **The file you edit.** Task 1 (`bellman_q`, `policy_evaluation`, `policy_evaluation_exact`, `value_iteration`, `policy_iteration`) and Task 3 (`solve`). |
| `dp_tools.py` | Loading worlds, checking your code, measuring, plotting, and the final evaluation. No need to edit it. |

Useful functions in `dp_tools` (all of them leave `model.backups` alone):

- `gap(model, gamma, policy)`: max_s (V\*(s) − V_π(s)), how far a policy is from optimal in its
  worst state. A policy is *ε-optimal* if the gap is at most ε = 10⁻³.
- `measure(solver, "maze")`: runs `solver(model, gamma, eps)` on a fresh model and returns
  its backups, gap, `ok` (ε-optimal or not), and seconds.
- `optimal(model, gamma)`: V\* and an optimal policy, for comparisons.
- `outcomes(model, policy)`: how episodes end under a policy (goal, caught, never ending).
- `plot_world(model, V, policy)`: values and policy on the map.

## Groups

The same five groups as last week. Task 2 differs per group:

| Group | Task 2: the question you own |
| --- | --- |
| A1 | When can we stop? |
| A2 | How much evaluation before improving? (modified policy iteration) |
| A3 | Update in place -- and in which order? (Gauss–Seidel, asynchronous DP) |
| B1 | What does γ do? |
| B2 | How far does it scale? |

## Tasks

0. **Setup** *(3 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python check_setup.py`.

1. **Implement** *(22 min)* the functions marked `todo(...)` in `dp.py`, e.g. one pair of you
   on evaluation and one on value and policy iteration. Check with

   ```bash
   uv run python week04_dynamic_programming/dp_tools.py
   ```

   It checks each function as soon as you have written it. Your iterative policy evaluation
   should reproduce the lecture's values for the random policy on the 4×4 gridworld
   (−14, −20, −22, …).

   Hints:
   - Everything is built on one line: Q(s, a) = R(s, a) + γ Σ_s' P(s'|s,a) V(s'), which is
     `bellman_q`. Then (T\*V)(s) = max_a Q(s, a) and (T^π V)(s) = Σ_a π(a|s) Q(s, a).
   - No loops over states are needed in Task 1: `model.expect` works on whole arrays.
   - `value_iteration` stops when max_s |V_{k+1}(s) − V_k(s)| < ε(1 − γ)/(2γ). Why this
     threshold? If δ = ‖V_{k+1} − V_k‖∞ and π is greedy with respect to V_k, then both
     ‖V\* − V_{k+1}‖∞ and ‖V_π − V_{k+1}‖∞ are at most γδ/(1 − γ) (use the contraction
     property of the lecture), so V_π is within 2γδ/(1 − γ) < ε of V\*.

   Then look at what you computed:

   ```python
   # week04_dynamic_programming/look.py  --  uv run python week04_dynamic_programming/look.py
   from dp_tools import *
   import dp
   model, gamma = load("ghost_lair")
   V, policy, deltas = dp.value_iteration(model, gamma)
   print(len(deltas), "sweeps,", model.backups, "backups;", outcomes(model, policy))
   plot_world(model, V, policy)
   plt.show()
   ```

2. **Your question** *(22 min)*. Write your experiments in a script of your own in this
   directory (e.g. `week04_dynamic_programming/a1.py`). Use `measure` to compare algorithms by
   backups, on all three worlds `maze`, `ghost_lair` and `pellets` -- and on at least one world
   you make yourself, e.g. `load("maze", slip_prob=0.0)` or `load("maze", slip_prob=0.4,
   gamma=0.999)`. Prepare **one plot and one sentence** for the class.

   - **A1 -- When can we stop?** Run value iteration for many sweeps and record, for each sweep
     k, δ_k and the gap of the greedy policy (`gap(...)`). Plot both on a log scale, with the
     threshold of the stopping rule. When does the greedy policy become ε-optimal, and when
     does the rule notice? Then try cheaper rules: a looser threshold, "δ < ε", "the greedy
     policy did not change in the last sweep", … Which of them are *safe* on all the worlds you
     tried, and how much do they save? Is a rule that works on our three worlds safe?
   - **A2 -- How much evaluation?** *Modified policy iteration:* after each improvement (one
     sweep of T\*, which also gives you the greedy policy π), do k − 1 sweeps of T^π (cheap:
     S backups each) before improving again. k = 1 is value iteration; k → ∞ approaches policy
     iteration. Stop as in value iteration, on the T\* sweep. Plot backups against k (1, 2, 3,
     5, 10, 20, 50). Also measure policy iteration with *iterative* evaluation run to
     convergence (you can't use the exact solve here: it needs `model.P`). Explain the
     `pellets` result. Hint: what does the first policy, "UP everywhere", do?
   - **A3 -- In place, and in which order?** *Gauss–Seidel* value iteration: loop over the
     states and overwrite V[s] immediately (`model.expect(V, [s])[0]` gives the A expectations
     for one state), so later states in the same sweep already use the new values. To stop
     safely, finish with one ordinary full sweep and apply the usual rule to it (why does that
     work?). Compare orders: index order (the states are numbered breadth-first from the
     start), reversed, random, and sorted by a value function (e.g. V\* from `optimal`, or the V
     of a cheap first run). Also compare the seconds. Which order is best, and why -- think
     about where the information comes from. Try it on `load("maze", slip_prob=0.0)`.
   - **B1 -- What does γ do?** The lecture's bound says ‖V_k − V\*‖∞ ≤ γ^k ‖V_0 − V\*‖∞. For γ
     in 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, count the sweeps value iteration needs, and
     the sweeps iterative policy evaluation needs for the *uniformly random* policy; compare
     both with what the bound predicts. Plot ‖V_k − V\*‖∞ against k for one world, with the
     bound. Why is value iteration so much faster than the bound, and than evaluating the
     random policy? Then solve `ghost_lair` for γ from 0.5 to 0.999: which route does the
     optimal policy take, and how likely is it to be caught (`outcomes`)? Why does γ change
     the *behaviour*, not just the speed?
   - **B2 -- How far does it scale?** `pellet_maze(n)` is the maze with n pellets
     (`load(pellet_maze(n), gamma=0.99, max_states=3000)`, n = 0 … 5). For each n, measure the
     number of states, the time to build the model, the time of one sweep of value iteration,
     of value iteration, of one exact policy evaluation, and of policy iteration. How do they
     grow with S? Is policy iteration (few iterations, each expensive) or value iteration
     (many cheap sweeps) faster in seconds, and does that change with S? Then extrapolate:
     how many states does `classic_11x11` have (62 floor cells, 51 pellets, 2 ghosts), and how
     long would one sweep take? What would you need to change about DP to get anywhere near it?

3. **Sharing** *(8 min)*. Each group: one plot, one sentence, 90 seconds. Listen well: you
   need the other groups' results for the final.

4. **The final** *(12 min)*. Write `solve(model, gamma, eps)` in `dp.py`: return a policy that
   is ε-optimal in every state, with as few backups as you can. Combine what the five groups
   found. Rules: use only `model.R` and `model.expect` (plus `model.n_states`,
   `model.n_actions`, `model.terminal`, `model.rho0`); no `model.P`, and nothing hard-coded
   about particular worlds. Check it with `dp_tools.py` (part 6). Before running the final,
   write down the score you expect. Then run it, **once**:

   ```bash
   uv run python week04_dynamic_programming/dp_tools.py --final
   ```

   It runs `solve` on the three worlds you know and several you don't (γ between 0.9 and
   0.999), and submits the results and your `dp.py` to the class leaderboard. The score is
   your backups relative to plain value iteration, geometric mean over the worlds (lower is
   better); a world where your policy is not ε-optimal counts as 10× value iteration.

## If you have time left (or at home)

- **Play it.** `pragman.play(pragman.PragmanEnv(model.cfg))` with `model` from
  `load("ghost_lair")` opens a window; arrow keys to move. Which route do you take, and
  with which γ would the optimal policy agree with you?
- **Prioritized sweeping.** Update first the states whose Bellman error |(T\*V)(s) − V(s)| is
  largest. Can you do that without paying for a full sweep to find them? What would you need
  to know about the model?
- **The certificate.** Any algorithm, however it got its V, can prove its greedy policy is
  ε-optimal with one full sweep of T\*. Use this to make the risky stopping rules of A1 safe:
  how much do they still save?
- **Reward design.** In `ghost_lair` with γ = 0.95, the optimal policy accepts a 10 % chance
  of being caught to save ten steps. Change `death_reward` (`load("ghost_lair", gamma=0.95,
  death_reward=...)`) until it takes the safe route. What does that tell you about what the
  reward said before?
