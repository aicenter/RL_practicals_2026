# Week 7: Deep Q-Networks

Goal: Q-learning without the table. You will implement vanilla DQN (the network, experience
replay, and the target network from the lecture), train it on a PragMan maze far too big for a
table, and put it on the class leaderboard. Then each group adds one improvement to DQN from the
research literature and finds out how much it helps here, and why.
After the deadline, the leaderboard also shows our own runs of every improvement, and of
several of them together.

Along the way: why does the agent believe the start of the maze is worth three times what it
actually gets? How much of a difference between two runs is just the seed? And does an idea that
worked on Atari work on your laptop?

## The setting

The world is PragMan's `pellets_8x8`: 29 pellets and one ghost that moves at random, at most 100
steps per episode. The 2^29 pellet patterns alone rule out a table. The agent sees the board as
four 8×8 binary planes (walls, agent, ghost, pellets) and has four actions.

```
########
#Pooooo#        a pellet is worth 0.1, the last one another 1.0,
#o##o#o#        being caught -1.0, and every other step -0.01:
#oooooo#
#o#oo#o#        random policy               about -0.4
#ooGooo#        clearing the board          up to about 3.6
#oooooo#
########
```

The same world comes back in Weeks 8 and 9 (policy gradients and actor–critic), so you will be
able to compare DQN with them.

## What's in this directory, and where the rest lives

| File | What it is |
| --- | --- |
| `dqn.py` | **The file you edit**: the replay buffer, the agent, and the training loop; also the command line. |
| `improvements/` | One page per improvement, in more detail than below: the idea, why it should help, how to fit it into `dqn.py`, and what to look at. Optional reading, but start there for Task 4. |
| `results/` | Created by your runs: one `.json` per experiment (not in git). |

The parts you don't need to edit are in the course package, `src/rlcourse/deeprl/`, which the
next two weeks reuse. You are welcome to read them:

| Module | What it is |
| --- | --- |
| `world.py` | `make_env()`: the world above. |
| `networks.py` | `GridEncoder`: two convolutions and a fully-connected layer that read the board. |
| `evaluation.py` | `evaluate(policy)`: the greedy policy on fixed episodes (the same ghost dice for everyone). |
| `experiment.py` | `Monitor` (bookkeeping inside the training loop) and `run_experiment` (several seeds in parallel, results to disk, leaderboard). |
| `plotting.py` | Learning curves of saved results. |
| `dqn_checks.py` | The checks behind `dqn.py check`. |

`dqn.py` imports what it needs by name (`from rlcourse.deeprl.world import make_env`); your IDE
should find all of it, and you can Ctrl+click through to the source. Selecting the interpreter
in `.venv` (see the main README) is all the setup it needs.

## The commands

From the repository root:

```bash
uv run python week07_dqn/dqn.py check                  # checks each piece as you write it
uv run python week07_dqn/dqn.py train                  # vanilla DQN, seeds 0 1 2 in parallel
uv run python week07_dqn/dqn.py train --help           # every option, with its default
uv run python week07_dqn/dqn.py train --seeds 0 --workers 1   # one seed, in this process (for a debugger)
uv run python week07_dqn/dqn.py train --submit --nickname ... # ... and send the result to the leaderboard
uv run python -m rlcourse.deeprl.plotting week07_dqn/results  # learning curves of all your runs
uv run python -m rlcourse.deeprl.plotting week07_dqn/results --metric overestimation
```

A training run prints a line every 10 000 steps for each seed: the return of the greedy policy on
50 evaluation episodes, the pellets it ate, how often it died, and **V(s0) vs G0**: what the
agent thinks the start is worth (max_a Q(s_0, a)), against the discounted return that actually
followed. Three seeds of 100 000 steps take about 3 minutes on a laptop with four cores.

At the end, a run reports two numbers, each as a mean ± standard error over the seeds: the
**area under the curve** (AUC: the mean evaluation return over the ten checkpoints of training)
and the **final return** (the policy at the end, on 100 episodes). The AUC is the leaderboard's
main number. It rewards learning fast, and it is much less noisy: a DQN policy can change a lot
between two checkpoints (watch the progress lines), so the final policy alone is partly luck.

**Leaderboard runs** use the default budget and seeds (100 000 steps; seeds 0, 1, and 2), so
that every group's numbers are comparable. The leaderboard compares your improved runs with
*your own* vanilla run. Other options (a different `--total-steps`, other seeds) are fine for
experimenting, but those submissions are not shown.

## Groups

The same five groups as before. Every group implements vanilla DQN (Tasks 1–3); then each group
owns one improvement (Task 4):

| Group | Improvement | Lecture link |
| --- | --- | --- |
| A1 | Double DQN, and measuring overestimation | maximization bias and Double DQN (Week 7) |
| A2 | Maxmin Q-learning | maximization bias, independent estimators (Week 7) |
| A3 | Munchausen DQN | the max in the target (Week 7); softmax policies (Week 8) |
| B1 | Averaged-DQN | the target network, maximization bias (Week 7) |
| B2 | Evaluating averaged weights | stochastic approximation (Week 5); the target network (Week 7) |

Three of the five (A1, A2, and B1) attack the same weakness of DQN, each in a different way, so
compare notes with each other.

## Tasks

0. **Setup** *(2 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python week07_dqn/dqn.py check`. You should see seven TODOs.

1. **Implement** *(25 min)* the places marked `todo(...)` in `dqn.py`. Split the work: two of you
   take `linear_schedule`, `ReplayBuffer`, and `DQNAgent.act`; the other two take
   `DQNAgent.td_targets`, `DQNAgent.update`, and `DQNAgent.sync_target`. Then fill in the three
   todos in `train` together. Check with `uv run python week07_dqn/dqn.py check`, which tests each
   piece on a small hand-made example as soon as you have written it.

   - **`ReplayBuffer`**: a ring. `position` is where the next transition goes; it wraps round to
     0 at the capacity. `sample` draws from the filled part only.
   - **`td_targets`**: under `torch.no_grad()`. `batch.terminated` is 1.0 where the episode
     terminated: use it to switch off the bootstrap term.
   - **`update`**: `Tensor.gather(1, actions.unsqueeze(1))` picks Q(s, a) of the action taken in
     each row. The order is `zero_grad`, `backward`, clip, `step`.
   - **`train`**: which flag goes into the buffer, `terminated` or `terminated or truncated`? Why
     does it matter? (Hint: what is the last board of an episode that ran out of time worth?)

2. **Teach-back** *(5 min)*. Each half explains its code to the other. Why does `td_targets` use
   the target network, and what would go wrong without it? Why sample from a buffer instead of
   learning from each transition as it comes?

3. **Vanilla DQN on the leaderboard** *(5 min, plus 3 min of training)*. Everyone in the group
   with a working `check` (one submission per group is enough):

   ```bash
   uv run python week07_dqn/dqn.py train --submit --nickname <your nickname>
   ```

   The first time, you are asked for your group's password for this week (you get it in class).
   While it trains, watch the **V(s0) vs G0** column. Write down a guess: why might the agent
   consistently believe that it will get more than it gets? (Groups A1, A2, and B1 will find
out.) When it is
   done, compare your vanilla result with the other groups'. Same algorithm, same
   hyperparameters, same seeds: how different are the numbers, and what does that tell you about
   reading a difference of 0.2 between two runs?

4. **Your improvement** *(30 min, finish at home)*. Implement your group's improvement in
   `dqn.py`, behind a switch in `Config` whose default keeps vanilla DQN (see the comment at the
   end of `Config`). For example, a field `double: bool = False` makes `train --double` work, and
   the results are saved and submitted as `double`. Then:

   - train with the defaults and your switch on, and `--submit` it;
   - compare the learning curves with your vanilla run
     (`uv run python -m rlcourse.deeprl.plotting week07_dqn/results`);
   - answer your group's questions below, with numbers.

   Each improvement has its own page in [`improvements/`](improvements/), with the idea in more
   detail, implementation hints, and the original paper.

   You may tune your improvement's own hyperparameters (and submit the variants: each gets its
   own row), but report what you tried. Leave the shared hyperparameters at their defaults for
   the main comparison.

   - **A1 -- Double DQN, and overestimation.** In the target, let the *online* network choose
     the next action and the *target* network evaluate it:
     y = r + γ Q(s′, argmax_a′ Q(s′, a′; w); w⁻). It is a few lines in `td_targets`. The
     `V(s0) vs G0` column is your measuring instrument: plot it for both runs with
     `--metric overestimation`. How large is the overestimation of vanilla DQN, and does it
     shrink over training? How much of it does Double DQN remove? Does the smaller bias buy a
     better policy? Why does taking a max over noisy estimates lead to optimism, and why is a
     *random* ghost the right world to see it? Where else, outside RL, does choosing the best
     of many noisy estimates disappoint in the same way?
   - **A2 -- Maxmin Q-learning.** Train two Q-networks side by side, each with its own target
     network, and bootstrap from their *minimum*:
     y = r + γ max_a′ min_i Q_i(s′, a′; w⁻_i). Both networks regress towards this same target
     (add the two Huber losses; one optimizer over the parameters of both), `sync_target` syncs
     both, and the agent acts on min_i Q_i(s, a; w_i) (change `q_values`, and acting and
     evaluation follow). Take the minimum over the networks *first*, then the max over actions.
     Why is the minimum of two noisy estimates pessimistic, and how can a pessimistic min and an
     optimistic max cancel out? Does Maxmin over- or underestimate in the **V(s0) vs G0**
     column, and how does that compare with vanilla and with group A1's Double DQN? Double DQN
     also uses two networks: what is the difference? How much slower is a run, and is it worth
     it? If you generalize to N networks, what do N = 3 and 4 do?
   - **A3 -- Munchausen DQN.** The idea in one sentence: add to the reward a penalty for how much
     worse the action taken was than the best one, *as the agent currently sees it*. With the
     softmax policy π(a | s) ∝ exp(Q(s, a; w⁻) / τ) at a small temperature τ,
     τ ln π(a | s) ≈ Q(s, a) − max_b Q(s, b), so the change is:
     - add the bonus α · clip(τ ln π(a | s), −1, 0) to the reward, for the action a that was
       taken: zero for the greedy action, negative for the others;
     - in the bootstrap, replace the hard max over next actions with a soft one:
       Σ_a′ π(a′ | s′) [Q(s′, a′; w⁻) − τ ln π(a′ | s′)], which equals τ ln Σ_a′ exp(Q(s′, a′; w⁻) / τ).

     Both changes are in `td_targets`; `F.log_softmax` computes ln π stably. τ = 0.03, α = 0.9,
     and the clip at −1 are the paper's values. Mind what is clipped: τ ln π, not ln π. (Why
     the method works is deeper, and the [page](improvements/munchausen.md) explains it; you
     don't need it to implement it.) What does the bonus do to the gap between the best
     action's value and the others'? How does Munchausen compare with vanilla DQN on the
     **V(s0) vs G0** column, and with group A1's Double DQN? Is the final policy better, the
     learning faster, or both? What happens with α = 0 (the soft bootstrap alone), and with a
     larger τ?
   - **B1 -- Averaged-DQN.** Bootstrap from the *average* of the last K target networks:
     y = r + γ max_a′ (1/K) Σ_k Q(s′, a′; w⁻_k), where w⁻_1 is the current target network,
     w⁻_2 the one before the last sync, and so on. Keep the K − 1 older ones in a
     `collections.deque(maxlen=K - 1)`; in `sync_target`, append a `copy.deepcopy` of the target
     network before overwriting it. Average the Q-values *before* taking the max. Start with
     K = 5. Why does a single snapshot of the network give optimistic targets, and why does
     averaging several help? Which is more optimistic: the max of the average, or the average
     of the maxima? How do the **V(s0) vs G0** column and the AUC change with K (1, 2, 5, 10),
     and how do they compare with group A1's Double DQN? Is the evaluation return more stable
     from one checkpoint to the next? Would syncing the target network less often
     (`--target-update-every`) do the same job?
   - **B2 -- Evaluating averaged weights.** Keep a running average of the online network's
     weights, w̄ ← (1 − β) w̄ + β w after every gradient step (β = 0.01; `Tensor.lerp_` does it
     in place, parameter by parameter), and *evaluate* the averaged network instead of the
     online one: give `evaluate_if_due` and `finish` functions that use it. Training, acting
     included, stays exactly as it is. Check that: your run's loss curve
     (`--metric loss`) should be identical to your vanilla run's. So where does a difference
     on the leaderboard come from? How often does the online network's greedy action change
     from one checkpoint to the next (try a few hundred boards), and does the averaged
     network's change less? The loss is far from convex, so why does averaging the weights of a
     network make sense at all, and why would averaging two *independently* trained networks
     be nonsense? Try β = 0.001 and 0.1. Finally: if your leaderboard number goes up although
     the agent has learned nothing new, is that a real improvement? What would you deploy?

   *Optional, for whoever finishes early* (any group): see the end of this page.

## What to hand in

Your group's **leaderboard submissions**: the vanilla run (Task 3) and at least one run with
your improvement (Task 4), before next week's practical. That is all; the questions above are
for understanding what you see, so do discuss them in your group.

## If you have time left (or at home)

- **Take DQN apart.** What do the two fixes from the lecture buy here? Turn the target network
  off (`--target-update-every 1`) and replay off (`--buffer-size 64 --batch-size 64`), one at a
  time. Which one matters more, and what goes wrong without it?
- **The time-limit bug.** Store `terminated or truncated` instead of `terminated`. Does it
  matter in this world, and when would it matter more?
- **Combine.** Take another group's improvement into your file (they will happily tell you how)
  and run both together. Do the gains add up?
- **Harder ghost.** `make_env(ghost_policy="chaser")` in `train` (not for the leaderboard).
  Which of the improvements helps most against a ghost that hunts you?
- **The Rainbow classics.** Dueling networks, n-step returns, prioritized replay, and noisy
  networks (the lecture's backup slides) each have a page in [`improvements/`](improvements/).
  They were the original group topics, but in our runs none of them clearly helped in this
  world with the default hyperparameters, and some made things worse. Can you find out why, or
  make one of them help? (Dueling is the quickest: a dozen lines in `QNetwork`.)
