# Week 7: Deep Q-Networks

Goal: Q-learning without the table. You will implement vanilla DQN (the network, experience
replay, and the target network from the lecture), train it on a PragMan maze far too big for a
table, and put it on the class leaderboard. Then each group adds one improvement from the
"Rainbow" family (and beyond) and finds out whether it actually helps here, how much, and why.
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
| A1 | Double DQN, and measuring overestimation | maximization bias (Week 6), Double DQN (Week 7) |
| A2 | n-step returns | the n dial (Week 6) |
| A3 | Munchausen DQN | the max in the target (Week 7); softmax policies (Week 8, preview) |
| B1 | Prioritized experience replay | importance sampling (Week 6), Week 7, improvements |
| B2 | Noisy networks | exploration (Weeks 2–3), Week 7, backup slides |

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
   consistently believe that it will get more than it gets? (Group A1 will find out.) When it is
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
   - **A2 -- n-step returns.** Store n-step transitions:
     (s_t, a_t, r_t + γ r_{t+1} + … + γ^{n−1} r_{t+n−1}, s_{t+n}), and bootstrap with γ^n.
     You need a small queue of the last n steps in `train` (`collections.deque`), the number of
     steps (or γ^k) per transition in the buffer, and care at the end of an episode: every window
     still in the queue must be stored, with fewer than n rewards, and must bootstrap only if the
     episode was truncated, not terminated. Try n = 3 (and 2, 5, 10, if you have time). Last
     week n = 1 won control in the tabular worlds; what is different with a network? The n-step
     return here is collected by the ε-greedy behaviour policy but used to learn about the
     greedy one: what bias does that introduce, and why does Rainbow get away with ignoring it?
   - **A3 -- Munchausen DQN.** Two changes to the target, both in `td_targets`. First, replace
     the hard max over next actions with a soft one: with the softmax policy
     π(a | s) ∝ exp(Q(s, a; w⁻) / τ), bootstrap with Σ_a′ π(a′ | s′) [Q(s′, a′; w⁻) − τ ln π(a′ | s′)].
     Second, add a bonus α · clip(τ ln π(a | s), −1, 0) to the reward, for the action a that
     was taken (`F.log_softmax` computes ln π stably). τ = 0.03, α = 0.9, and the clip at −1
     are the paper's values.
     The bonus is never positive: for which actions is it zero, and for which is it most
     negative? What does it do to the gap between the best action's value and the others'? How
     does Munchausen compare with vanilla DQN on the **V(s0) vs G0** column (and with group A1's
     Double DQN)? Is the final policy better, the learning faster, or both? What happens with
     α = 0 (soft targets alone), and with a larger τ?
   - **B1 -- Prioritized experience replay.** Sample transition i with probability
     P(i) ∝ p_i^α, where p_i = |δ_i| + ε is its last TD error (new transitions get the largest
     priority so far), and correct the bias with importance-sampling weights
     w_i = (N · P(i))^(−β) / max_j w_j in the loss. A sum tree makes sampling and updating
     O(log N); write one (an array of size 2N whose inner nodes hold the sums of their
     children), or start with a slow version on `rng.choice(p=...)` and see whether it is fast
     enough. `update` needs to return the TD errors, and the training loop to give them back to
     the buffer. α = 0.6 and β annealed from 0.4 to 1 are the paper's values. What do the
     weights correct, and why is that the same problem as off-policy evaluation in Week 6? Which
     transitions end up with the highest priorities? Does prioritization help here, and if not,
     what does the learning rate have to do with it?
   - **B2 -- Noisy networks.** Replace the ε-greedy exploration by noise in the weights: a
     `NoisyLinear` layer with learned means μ and learned noise scales σ,
     w = μ_w + σ_w ⊙ ε_w (factorized Gaussian noise, see the paper), in place of the head's
     `nn.Linear`, with ε = 0 throughout. Resample the noise before every action and every
     update; for evaluation, use the means (switch the noise off). How does exploration with
     noisy weights differ from ε-greedy: is it per step or more consistent? Do the learned σ
     shrink during training, and in which layer? Does the agent explore more or less than with
     ε-greedy, and does it pay off here? Compare several seeds: is noisy exploration more or
     less reliable?

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
- **Dueling networks** ([`improvements/dueling.md`](improvements/dueling.md)): replace the head
  of `QNetwork` with a value stream V(s) and an advantage stream A(s, a), combined as
  Q(s, a) = V(s) + A(s, a) − mean_a′ A(s, a′). A dozen lines; does it change anything here?
