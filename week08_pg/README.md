# Week 8: Policy gradients

Goal: learn the policy directly, without a Q-function. You will implement REINFORCE with the two
improvements from the lecture (the return-to-go and a baseline), train it on the same PragMan
maze as DQN last week, and put it on the leaderboard next to your DQN, with returns *and*
run times. Then each group takes one question about what makes policy gradients work, or fail,
and answers it with experiments.

Along the way: why does REINFORCE without a baseline sometimes not learn at all? Why can a
smaller batch beat a bigger one? Why does a policy sometimes freeze into doing something dumb
forever? And what does an agent learn when the reward says something slightly different from
what we meant?

## The setting

The same world as in Week 7: PragMan's `pellets_8x8`, 29 pellets, one ghost that moves at
random, at most 100 steps per episode. A pellet is worth 0.1, the last one another 1.0, being
caught −1.0, and every other step −0.01. A random policy gets about −0.4; clearing the board
pays up to about 3.6. The network that reads the board (`GridEncoder`) and the evaluation (the
same fixed episodes for everyone) are the same as for DQN, so the numbers compare directly.

## What's in this directory, and where the rest lives

| File | What it is |
| --- | --- |
| `pg.py` | **The file you edit**: the policy, the agent, and the training loop; also the command line. |
| `results/` | Created by your runs: one `.json` per experiment, and the trained policies in `results/policies/` (not in git). |

The shared tools are in the course package, `src/rlcourse/deeprl/`, as last week: `world.py`,
`networks.py`, `evaluation.py`, `experiment.py`, and `plotting.py`, plus two new ones:

| Module | What it is |
| --- | --- |
| `pg_checks.py` | The checks behind `pg.py check`. |
| `pg_tools.py` | Saving and loading trained agents, `watch` (see a policy play), and `variance` (how noisy is a gradient estimate?). |

## A map of `pg.py`

REINFORCE is simpler than DQN: no replay buffer, no target network, no ε. It is **on-policy**:
collect a few whole episodes with the current policy, take one gradient step, throw the
episodes away, repeat. Exploration comes from the policy itself, which is random: it *samples*
its actions.

```
train(config, seed)
 └─ repeat until 1M steps:
     ├─ run_episode(env, agent, rng)     x 8   play whole episodes; agent.act samples each action
     └─ agent.update(episodes)                 one gradient step
         ├─ stack(episodes)                    all steps of the batch, as tensors
         └─ agent.weights(episodes)            G_t - b for every step
             └─ returns_to_go(rewards, gamma)
```

Two choices differ slightly from the lecture's algorithm boxes, as in most implementations:

- **The loss averages over all steps of the batch**, instead of summing over the steps and dividing
  by the number of episodes. The direction of the gradient is the same; only its scale differs,
  and Adam does not care about the scale. It matters for the **entropy bonus**, which is a mean
  over the steps too: in a sum over ~70 steps per episode, it would be drowned out.
- **No γ^t factor** in front of each step (see the footnote in the lecture notes).

The **entropy bonus**: the loss subtracts `entropy_coef` times the mean entropy of π(· | s) over
the batch's states, which rewards the policy for staying random a while longer. The entropy of
a policy over four actions is between 0 (deterministic) and ln 4 = 1.39 (uniform); the progress
lines print it.

## The commands

From the repository root:

```bash
uv run python week08_pg/pg.py check                   # checks each piece as you write it
uv run python week08_pg/pg.py train                   # the core, seeds 0 1 2 in parallel
uv run python week08_pg/pg.py train --help            # every option, with its default
uv run python week08_pg/pg.py train --seeds 0 --workers 1    # one seed, in this process (for a debugger)
uv run python week08_pg/pg.py train --submit --nickname ...  # ... and send the result to the leaderboard
uv run python week08_pg/pg.py watch week08_pg/results/policies/core_seed0.pt     # see it play
uv run python week08_pg/pg.py watch week08_pg/results/policies/core_seed0.pt --episodes 100 --quiet
uv run python week08_pg/pg.py variance week08_pg/results/policies/core_seed0.pt  # Task 5
uv run python -m rlcourse.deeprl.plotting week08_pg/results               # learning curves
uv run python -m rlcourse.deeprl.plotting week08_pg/results --metric entropy
uv run python -m rlcourse.deeprl.plotting week08_pg/results/core.json week08_pg/results/rtg=False.json --save core.png
```

Each run saves `week08_pg/results/<name>.json` and its trained policies as
`week08_pg/results/policies/<name>_seed<k>.pt`; the name is `core` for the defaults, otherwise
the changed options in the order of `Config` (e.g. `rtg=False`, `baseline=none`,
`gamma=0.9`). `train` prints the names at the end.

A run trains for **1 000 000 steps**, ten times DQN's budget: REINFORCE uses every step once,
DQN replays each one many times. But a step of REINFORCE is much cheaper, so three seeds take
about **4–6 minutes** on a recent laptop with four cores, about one and a half times
DQN's 100 000 steps (`check` prints how long its short run took: about 8 s on such a laptop; if yours takes
much longer, plan fewer runs). A run uses three cores, so one run per laptop at a time.
A run prints a line every 100 000 steps for each seed: the return of the greedy policy (the most
likely action in each state) on 50 evaluation episodes, the pellets it ate, how often it died,
the **entropy** of the policy during training, and the mean length of the training episodes. At the end, as last week: the **area under the
curve** (AUC, the mean evaluation return over the ten checkpoints) and the **final return**, each
± standard error over the seeds.

**Leaderboard runs** use the default budget and seeds (1 000 000 steps; seeds 0, 1, and 2). The
leaderboard shows your group's PG runs next to your DQN runs from last week, with their run
times. Run times depend on the laptop, so the board also shows each PG run's time relative to
your own DQN run; that comparison is fair only if both ran on the same machine (say which in
your presentation).

## Groups

The same five groups. Every group implements the core (Tasks 1–3); then each group owns one
question (Task 4):

| Group | Question | Lecture link |
| --- | --- | --- |
| A1 | The entropy bonus, and constant baselines: unbiased is not the same as usable | baselines (Week 8) |
| A2 | Big batches or small ones? And how noisy is a gradient estimate? | variance of the estimator (Week 8) |
| A3 | The discount factor as a variance knob | return-to-go and discounting (Week 8) |
| B1 | A learned baseline V(s), and episodes cut off by the time limit | learned baseline (Week 8); terminated vs truncated (Week 7) |
| B2 | Reward design: what does the agent learn when we change the reward? | the reward as the definition of the task |

The A groups mostly change settings and need more runs and more analysis; B1 writes the most
code. Each group has four laptops: split the runs between them.

## Tasks

0. **Setup** *(2 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python week08_pg/pg.py check`. You should see five lines marked TODO: the four
   places you implement in Task 1, and `train`, which works as soon as they do.

1. **Implement** *(20 min)* the four places marked `todo(...)` in `pg.py`. Split the work: two of
   you take `returns_to_go` and `PGAgent.weights`, the other two `PGAgent.act` and
   `PGAgent.update`. Check with `uv run python week08_pg/pg.py check`, which tests each piece on a
   small hand-made example as soon as you have written it.

   - **`returns_to_go`**: backwards, `G_t = r_t + γ G_{t+1}`. With rewards `[1, 0, 2]` and γ = 0.5,
     the answer is `[1.5, 1.0, 2.0]`.
   - **`weights`**: one number per step, in the order of `stack`: episode after episode. With
     `--no-rtg`, every step of an episode gets the whole return R(τ) = G_0. The batch mean is
     taken over *all steps* of the batch.
   - **`act`**: `self.probabilities(observation)` gives π(· | s); `rng.choice(n, p=...)` samples.
   - **`update`**: like DQN's, but the loss is the one at the top of `pg.py`.
     `F.log_softmax(logits, dim=1)` gives log π of all actions; `.gather(1, actions.unsqueeze(1))`
     picks those taken. Use `self.weights(episodes)` (the check substitutes its own weights).
     The order is `zero_grad`, `backward`, clip, `step`.

2. **Teach-back** *(5 min)*. Each pair explains its code to the other. Where in `update` is the
   log-derivative trick from the lecture? Why may no gradient flow through the weights? Why is
   it fine to sample the actions in training but play the most likely one in evaluation?

3. **The core and two ablations on the leaderboard** *(5 min, plus about 5 min of training)*.
   Three of you, each on your own laptop (one submission of each per group is enough):

   ```bash
   uv run python week08_pg/pg.py train --submit --nickname <your nickname>                   # the core
   uv run python week08_pg/pg.py train --no-rtg --submit --nickname <your nickname>          # whole return R(tau)
   uv run python week08_pg/pg.py train --baseline none --submit --nickname <your nickname>   # no baseline
   ```

   The first time, you are asked for your group's password for this week (you get it in class).
   While they train, the fourth of you starts on Task 4, and everyone watches the **entropy**
   column. Write down a guess: which of the three will learn best, and which not at all? When
   they are done, look at the board. How does the core compare with your DQN from last week, per
   step and per minute? If a run did not learn, look at its entropy: did the policy collapse onto
   one action, or stay random? Then `watch` one of its policies.

4. **Your question** *(the rest of the practical, finish at home)*. Answer your group's questions
   below with numbers, and `--submit` your runs (each variant gets its own row on the board).
   Learning curves: `uv run python -m rlcourse.deeprl.plotting week08_pg/results`, also with
   `--metric entropy` or `--metric steps` (the length of the evaluation episodes).

   - **A1 -- The entropy bonus, and constant baselines.** Five runs: `--entropy-coef 0`,
     `--entropy-coef 0.05`, and `--learning-rate 0.008` (with the default bonus) for the first
     half; b = −1 and b = 1 for the second (more if you have time: b = 0.5, 2, or entropy
     coefficients between 0 and 0.01, to see where things break). What happens to the entropy
     column, and what does the policy do (`watch` it)? Does the bonus change the result a little,
     or decide between learning and not learning? Why can a deterministic policy no longer learn
     (what new information does it collect)? Then the second half: the lecture shows that subtracting *any* baseline b that
     does not depend on the action keeps the gradient estimate unbiased. Add a `constant`
     baseline (subtract a fixed number `baseline_value` from G_t; see the comment in `Config`)
     (`--baseline none`, from Task 3, is b = 0; the batch mean is about 0.3). Which of them learn,
     and why do the others fail, although every one of them is unbiased? Hint: what does a
     gradient step do to the probabilities when every action taken gets a *negative* weight? A
     positive one? And how does this connect with the first half?
   - **A2 -- Big batches or small ones?** Train with `--episodes-per-update` 2, 4, 16, and 32 (the
     budget of 1 000 000 steps stays the same). Which learns fastest? How many gradient steps does
     each run take? (Steps per update = episodes per update × the episode length in the progress
     lines.) Then measure the noise itself: `pg.py variance` (see Task 5) on your core
     policy, with `--episodes-per-batch` 2, 8, and 32. How does the cosine with the true gradient
     grow with the batch? (The cosine is estimated from how much independent batches agree with
     each other, so it compares across batch sizes; repeat with `--seed 1` to see how much the
     numbers move.) Does the variance shrink as 1/N? So the large batch gives the better
     direction: why does it lose anyway? Where else in machine learning is this trade-off made,
     and what is the usual answer?
   - **A3 -- The discount factor.** Train with `--gamma` 0.9, 0.95, and 1.0 (the core has 0.99). The
     evaluation always reports the *undiscounted* return, so the comparison is fair. Which γ
     gives the best real return? Roughly how many steps ahead does G_t look at each γ, and what
     does that do to its variance? What is the price: what does a γ = 0.9 agent not care about?
     (Look at the episode lengths and the cleared rate, and `watch` the policies.) Is a smaller γ
     a better algorithm, or a different problem? Describe a world where γ = 0.9 would fail badly.
     If you have time: does DQN gain as much from a smaller γ (`week07_dqn/dqn.py train
     --gamma 0.9`)?
   - **B1 -- A learned baseline, and the time limit.** Implement the lecture's learned baseline
     V(s; φ): a `ValueNetwork` (a `GridEncoder` and a linear layer with one output), its own Adam
     optimizer (step size 1e-3), and after each policy step one gradient step on the mean squared
     error between V(s_t) and G_t. Add `"value"` to the choices of `baseline` (the weights become
     G_t − V(s_t), computed without gradient), and set `value_fn` in `train` to a function that
     returns V of a board: the progress lines then show **V(s0) vs G0**, as last week. Does V beat
     the batch mean? Does V over- or underestimate the value of the start, and how does that
     compare with DQN's estimate last week? Careful: G0 is the return of the *greedy* policy that
     evaluation plays; V is learned from the episodes of which policy? Then the time limit: an episode cut off after
     100 steps did not end, but `returns_to_go` treats its last board as worth nothing. Add a
     switch `bootstrap_truncated` that starts the backward sum of a truncated episode
     (`episode.terminated` is False) from V(s_T) of its `last_observation` instead of 0. Does it
     help, and why? (Hint: without it, what is G_t of the last steps before the time limit,
     whatever pellets are left?) You have now built half of the next lecture's actor–critic:
     what is still missing?

     Hints for the code: create the network and its optimizer in `PGAgent.__init__` as attributes
     (e.g. `self.value_network`; saving and loading picks up every network the agent holds). The
     value step needs G_t, the policy step G_t − V: compute the returns once, in a helper used by
     both. Give `returns_to_go` an extra argument with a default, e.g. `last_value: float = 0.0`,
     so that the existing calls keep working. `bootstrap_truncated` only makes sense with
     `--baseline value`.
   - **B2 -- Reward design.** The `Config` options `--living-reward` and `--death-reward` change the
     rewards of the world the agent *trains* in; the evaluation (and the board) always uses the
     standard rewards. Train with `--living-reward -0.05` (time is expensive),
     `--living-reward 0.02` (being alive pays), and `--death-reward -0.25` (being caught is
     cheaper). For each, `watch` a policy, and count with `watch ... --episodes 100 --quiet`: how
     many pellets, how often caught, how long the episodes. What has each agent learned to do,
     and is it doing anything *wrong*, given the reward it was trained on? Which change makes the
     agent better at the real task, and why? Before you run them, write down your predictions;
     afterwards, which ones surprised you? Where outside PragMan does an agent get rewarded for
     something slightly different from what we want?

   *Optional, for whoever finishes early* (any group): see the end of this page.

5. **Closing: how noisy is a gradient?** *(the last 10 minutes, together)*. Each group runs

   ```bash
   uv run python week08_pg/pg.py variance week08_pg/results/policies/core_seed0.pt
   uv run python week08_pg/pg.py variance            # an untrained, uniform policy
   ```

   (about 30 s each). Both freeze a policy, collect 40 batches of 8 episodes, and compute the
   gradient estimate of several estimators on the *same* episodes. They print how much the
   estimate varies from batch to batch and how well a single batch's estimate points in the
   direction of the true gradient. Which estimator is least noisy? At the untrained policy, why
   could the return-to-go be *noisier* than the whole return? (Hint: the step cost.) Compare the
   size of the baseline's effect here with its effect in Task 3: if it only reduces the variance
   a little, why did it decide between learning and not learning? We will discuss it together.

## What to hand in

- **Leaderboard submissions** before the Week 9 practical: the core and the two ablations
  (Task 3), and the runs behind your group's answers (Task 4).
- **Presentations** at the start of the Week 9 practical: 3 minutes per group, one plot and one
  sentence, then questions -- to any member of the group, about any part of the work.
- **BRUTE**, by 9:15 before that practical, each of you individually: a few sentences on your
  group's question -- what you did, what you found (with at least one number), and why you
  think it happens.

## If you have time left (or at home)

- **A different ghost.** `train --ghost-policy chaser` trains against a ghost that hunts you;
  `watch --ghost chaser` plays any policy against it. How does a policy trained against the random
  ghost fare against the chaser, and the other way round? Which one would you deploy if you did
  not know the ghost?
- **Greedy or sampled?** `watch ... --episodes 100 --quiet` with and without `--sampled`. Which plays
  better, and why do we train with one and evaluate the other?
- **Leave-one-out baseline.** For each step t of episode j, subtract the mean of G_t over the
  *other* episodes of the batch (at the same t). It needs no network and is unbiased. Does it
  beat the batch mean? The same idea, with whole answers in place of episodes, is GRPO, the
  algorithm behind several recent reasoning language models (more in a later lecture).
- **A shared encoder.** Group B1's value network as a second head on the policy's `GridEncoder`
  (add the value loss, times 0.5, to the policy loss). Fewer parameters; does it learn faster
  or slower?
- **Normalizing the weights.** Divide the weights by their standard deviation over the batch, as
  many implementations do. Does it matter here? (Hint: which optimizer do we use?)
