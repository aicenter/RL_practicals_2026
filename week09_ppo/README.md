# Week 9: Actor–critic (A2C and PPO)

Goal: put a critic next to the policy and reuse each batch of experience several times. You will
implement the two pieces of PPO that the lecture added to Week 8's REINFORCE, generalized
advantage estimation (GAE) and the clipped loss, train PPO and A2C on the same PragMan maze, and
put them on the leaderboard next to your DQN and REINFORCE. Then each group takes one question
about what makes PPO work, or fail, and answers it with experiments.

Along the way: how can one algorithm be both A2C and PPO? Why does reusing a batch help, and
when does it stop helping? What does a bug in three lines of bookkeeping do to an agent? And
what does a policy learn when we give it a hint?

## The setting

The world of Weeks 7 and 8: PragMan's `pellets_8x8`, 29 pellets, one ghost that moves at random,
at most 100 steps per episode. A pellet is worth 0.1, the last one another 1.0, being caught
−1.0, and every other step −0.01. A random policy gets about −0.4; clearing the board pays up to
about 3.6. The network body (`GridEncoder`) and the evaluation (the same fixed episodes, the
greedy policy) are those of the last two weeks, so the numbers compare directly.

## What's in this directory, and where the rest lives

| File | What it is |
| --- | --- |
| `ppo.py` | **The file you edit**: the two pieces of PPO, and (given) the network, the data collection, the update, and the command line. |
| `reference_policy.pt` | A policy for group B2 (PPO trained for 100 000 steps); everyone else can ignore it. |
| `results/` | Created by your runs: one `.json` per experiment, and the trained agents in `results/policies/` (not in git). |

The shared tools are in the course package, `src/rlcourse/deeprl/`, as before, plus two new ones:

| Module | What it is |
| --- | --- |
| `ppo_checks.py` | The checks behind `ppo.py check`. |
| `ppo_tools.py` | The diagnostics of an update (KL, clip fraction, explained variance), group B2's training world, and `score` (what a trained policy does, and how good its critic is). Saving, loading, and `watch` are Week 8's (`pg_tools.py`). |

## A map of `ppo.py`, as a diff against Week 8's `pg.py`

```
train(config, seed)
 └─ repeat until 300 000 steps:
     ├─ collect(...)      T steps in each of N environments (not whole episodes);
     │                    stores s_t, a_t, log pi_old(a_t|s_t), V(s_t), r_t, the episode ends
     └─ update(...)
         ├─ gae(...)                    YOU: advantages and critic targets, once per segment
         └─ K epochs x M minibatches:
             └─ ppo_loss(...)           YOU: clipped surrogate + value loss − entropy bonus
                + Adam step (gradient clipped)
```

What is new compared with `pg.py`:

- **One network, two heads** (`ActorCritic`): a shared `GridEncoder`, a policy head (four logits,
  as last week), and a value head V(s).
- **N environments side by side, segments of T steps** instead of whole episodes. Episodes run
  across segment boundaries; an environment that finishes an episode in the middle of a segment
  is reset at once and carries on. So a segment can contain the end of one episode and the start
  of the next, and an episode can end in two ways: **terminated** (caught, or the board cleared)
  or **truncated** (the time limit). `collect` records which, and also stores the value of the
  last board *before* the reset (the observation an environment returns after the reset already
  belongs to the new episode).
- **A batch is used K times** (epochs), in M minibatches each. The old log-probabilities stored
  during collection give the ratio ρ_t = π(a_t|s_t) / π_old(a_t|s_t).
- **A2C is the same file** with `--preset a2c`: one epoch, one minibatch, λ = 1, no clipping, and
  A2C's own short segments (T = 5) and no advantage normalization. Why a preset and not just
  these flags? Because each algorithm deserves the settings that suit it: with PPO's long
  segments and normalization, A2C looks much worse than it is.

The loss is the lecture's (Week 9, PPO): the **mean over the minibatch** of
−min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t) + c_V (V(s_t) − Ĝ_t)² − β H(π(·|s_t)), with the
advantages standardized per minibatch (one of PPO's implementation details).

## The commands

From the repository root:

```bash
uv run python week09_ppo/ppo.py check                    # checks each piece as you write it
uv run python week09_ppo/ppo.py train                    # PPO, seeds 0 1 2 in parallel
uv run python week09_ppo/ppo.py train --preset a2c       # A2C
uv run python week09_ppo/ppo.py train --help             # every option, with its default
uv run python week09_ppo/ppo.py train --preset a2c --help  # ... and A2C's defaults
uv run python week09_ppo/ppo.py train --seeds 0 --workers 1     # one seed, in this process (for a debugger)
uv run python week09_ppo/ppo.py train --submit --nickname ...   # ... and send the result to the leaderboard
uv run python week09_ppo/ppo.py watch week09_ppo/results/policies/ppo_seed0.pt       # see it play
uv run python week09_ppo/ppo.py score week09_ppo/results/policies/*_seed0.pt         # compare policies
uv run python -m rlcourse.deeprl.plotting week09_ppo/results                         # learning curves
uv run python -m rlcourse.deeprl.plotting week09_ppo/results --metric kl
```

Each run saves `week09_ppo/results/<name>.json` and its trained agents as
`week09_ppo/results/policies/<name>_seed<k>.pt`. The name is the preset (`ppo` or `a2c`),
plus the options you changed from that preset's defaults, e.g. `ppo+gae_lambda=1.0` or
`a2c+n_envs=1+n_steps=80`.

A run trains for **300 000 steps** (all environments together). Three seeds take about
**3–4 minutes** on a recent laptop with four cores, about as long as Week 7's DQN (`check` prints
how long its two short runs took: about 15 s in all on such a laptop); A2C takes about two
thirds of that, and runs with many epochs (group A2) longer. A run uses three cores, so one run
per laptop at a time.

A run prints a line every 30 000 steps for each seed: the return of the greedy policy on 50
evaluation episodes, the pellets it ate, how often it died, and the lecture's diagnostics,
averaged over the updates since the last line:

| column | what it is | what to look for |
| --- | --- | --- |
| `kl` | approximate KL divergence between the policy before and after an update, the mean of (ρ − 1) − log ρ over the batch | how far one update moves the policy; a sudden jump means the updates are too large |
| `clip_frac` | fraction of the batch's ratios ρ outside [1 − ε, 1 + ε] after the update | how much the clip had to do |
| `ratio_max` | the largest ratio ρ in the batch after the update | is the clip a hard limit? |
| `entropy` | of π(·\|s), averaged over the batch (between 0 and ln 4 = 1.39) | premature collapse (Week 8) |
| `expl_var` | explained variance of the critic for its targets Ĝ_t: 1 − Var(Ĝ − V) / Var(Ĝ), before the update | 1 = perfect, 0 = no better than a constant |

At the end, as in the last two weeks: the **area under the curve** (AUC, the mean evaluation
return over the ten checkpoints) and the **final return**, each ± standard error over the seeds.

**Leaderboard runs** use the default budget and seeds (300 000 steps; seeds 0, 1, and 2). The
leaderboard shows your PPO and A2C next to your group's DQN (Week 7) and REINFORCE (Week 8), with
their budgets and run times. Run times depend on the laptop, so compare them within your group,
and say which machine made which run.

## Groups

The same five groups. Every group implements the core (Tasks 1–3); then each group owns one
question (Task 4):

| Group | Question | Lecture link |
| --- | --- | --- |
| A1 | λ or γ? Two ways to shorten the horizon, and what the critic's numbers really say | GAE (Week 9); discounting (Week 8, A3) |
| A2 | How often can we reuse a batch? Epochs, clipping, and the trust region | TRPO and PPO's clip (Week 9) |
| A3 | Many environments or long segments? Per sample and per minute | A2C's parallel environments (Week 9) |
| B1 | Three lines of bookkeeping: what do the classic bugs do to an agent? | detaching; termination vs truncation (Weeks 7–9) |
| B2 | A hint in the reward, and a leash to a reference policy | the KL penalty to a reference (Week 9, PPO) |

Each group has four laptops: split the runs between them.

## Tasks

0. **Setup** *(2 min)*. From the repository root: `git pull`, `uv sync`, and
   `uv run python week09_ppo/ppo.py check`. You should see three lines marked TODO: the two
   functions you write in Task 1, and `train`, which works as soon as they do.

1. **Implement** *(20 min)* the two places marked `todo(...)` in `ppo.py`. Split the work: two of
   you take `gae`, the other two `ppo_loss`. Check with `uv run python week09_ppo/ppo.py check`,
   which tests each piece on a small hand-made example as soon as you have written it.

   - **`gae`**: backwards over the segment, t = T−1, …, 0, for all N environments at once (each
     row of the tensors is one time step). The recursion is the lecture's:
     δ_t = r_t + γ (1 − d_t) V(s_{t+1}) − V(s_t), Â_t = δ_t + γλ (1 − d_t) Â_{t+1}, where
     d_t = 1 if the episode ended at step t, V(s_T) is `last_values`, and Â_T = 0. Then the targets
     Ĝ_t = Â_t + V(s_t). The one subtle part: where an episode was **truncated** at step t, its
     future is not worth nothing. Add γ times the value of its last board (`final_values`) to
     the reward, then cut the recursion as for a termination. Where it **terminated**,
     `final_values` must not be used. Check your loop on paper: with λ = 0 you should get the
     one-step TD errors δ_t, with λ = 1 the returns to the end of the segment minus V(s_t).
   - **`ppo_loss`**: the docstring lists the steps. `torch.clamp` clips, `torch.min` takes the
     elementwise minimum. Which inputs are constants, and why? (The check finds out whether a
     gradient flows where it should not.)

2. **Teach-back** *(5 min)*. Each pair explains its code to the other. Then together: write down
   A2C's loss for the policy, −Â_t log π(a_t|s_t), and explain why PPO with one epoch, one
   minibatch, and no advantage normalization takes *exactly* the same gradient step. (Hint: what
   is ρ_t at the only step, and what are the gradients of ρ_t and of log π there?) Where are
   the detaches in our code, and what would go wrong without each?

3. **PPO and A2C on the leaderboard** *(8 min, of which 4 of training)*. First, each of you writes
   down a prediction: which of the three runs below learns best per sample, and which per minute?
   Then three of you, each on your own laptop (one submission of each per group is enough):

   ```bash
   uv run python week09_ppo/ppo.py train --submit --nickname <your nickname>                     # PPO
   uv run python week09_ppo/ppo.py train --preset a2c --submit --nickname <your nickname>        # A2C
   uv run python week09_ppo/ppo.py train --gae-lambda 1 --submit --nickname <your nickname>      # PPO without GAE
   ```

   The first time, you are asked for your group's password for this week (you get it in class).
   While they train, the fourth of you starts on Task 4, and everyone watches the diagnostic
   columns: how do `kl` and `clip_frac` of PPO compare with A2C's? When they are done, look at
   the board. How do PPO and A2C compare with your DQN and REINFORCE, per step and per minute?
   PPO with λ = 1 and A2C both use the returns to the end of the segment as targets: why
   does one do so much better than the other? (Hint: look at `--help` for each.)

4. **Your question** *(the rest of the practical, finish at home)*. Answer your group's questions
   below with numbers, and `--submit` your runs (each variant gets its own row on the board).
   Learning curves: `uv run python -m rlcourse.deeprl.plotting week09_ppo/results`, also with
   `--metric kl`, `--metric clip_frac`, `--metric entropy`, or `--metric expl_var`.

   - **A1 -- λ or γ?** GAE's λ and the discount γ both shorten how far ahead the advantage
     looks. Train with `--gae-lambda` 0, 0.5, and 1 (the default is 0.95), and with `--gamma 0.9`
     at λ = 0, 0.95, and 1. Which λ works best at γ = 0.99, and how much does λ matter at
     γ = 0.9? Why? (How many steps ahead does each look: 1/(1 − γλ)?) Which knob would you
     rather turn, and what is the price of each? (Look at the deaths and the episode lengths:
     what does a γ = 0.9 agent not care about? The lecture said that λ costs bias only where
     the critic is wrong; does the data agree?) Then the critic: the `expl_var` column says how
     well V explains its targets Ĝ_t. Compare it across λ. Is the critic of λ = 0 really the best?
     What *is* Ĝ_t at λ = 0, and why is it easy to predict? Measure the critics independently:
     `ppo.py score week09_ppo/results/policies/*_seed0.pt` plays 100 episodes with each policy
     and compares V(s_t) with the returns that actually followed (`critic EV`). Which critic is
     best now? What does this tell you about diagnostics that measure a model against its own
     targets?
   - **A2 -- How often can we reuse a batch?** Train with `--epochs` 1, 10, and 20 (the default
     is 4), then `--epochs 10 --clip-eps 0.1`, and `--no-clip` with the default 4 epochs and with
     `--epochs 1`. (Runs with many epochs take longer: start them first.) How does the return
     depend on K? Watch `kl`, `clip_frac`, and `ratio_max`: how far does one update move the
     policy, and does the clip keep the ratios inside [1 − ε, 1 + ε]? What happens without the
     clip, and why does it make no difference at K = 1? What does that say about what the clip
     is *for*? Where is the sweet spot here, and how does it depend on ε? Connect it with the
     lecture's figure of the surrogate along a line: why does the surrogate become a bad guide
     far from π_old? If you have time: implement **early stopping**, i.e. stop the epochs of an
     update as soon as the approximate KL of the batch exceeds a threshold (e.g. 0.02), add it
     behind a `Config` switch, and test it at K = 20.
   - **A3 -- Many environments or long segments?** Keep the batch at N × T = 1024 and change the
     shape: `--n-envs 1 --n-steps 1024`, `--n-envs 4 --n-steps 256`, `--n-envs 64 --n-steps 16`,
     and `--n-envs 256 --n-steps 4` (the default is 16 × 64). Compare the learning curves per step
     *and* the minutes per run. Then A2C, at its batch of N × T = 80: `--preset a2c --n-envs 1
     --n-steps 80` and `--preset a2c --n-envs 80 --n-steps 1` (the default is 16 × 5). The
     lecture says that parallel environments decorrelate the batch. Do your numbers show that,
     or something else? Careful: with λ = 1 (A2C's default), T also sets how far ahead the
     targets look. Separate the two effects with `--gae-lambda 0.8` at 1 × 80 and at 16 × 5.
     What do parallel environments buy *here*, and where would they matter more? Finally, put
     PPO and A2C next to your DQN and REINFORCE: which algorithm would you choose if samples
     were cheap and time was short, and which if every sample were expensive?
   - **B1 -- Three lines of bookkeeping.** `ppo.py` has switches for four classic bugs (see
     `--help`): `--no-detach` (the advantages are not treated as constants: the gradient flows
     into the critic through Â_t), `--bug truncation_as_termination`,
     `--bug bootstrap_from_reset` (a truncated episode is bootstrapped from the first board of
     the next episode, which is what a vectorized environment that resets automatically hands
     you), and `--bug no_episode_cut` (GAE runs on across the ends of episodes). Train each,
     plus `--no-normalize-advantages` alone and together with `--no-detach`, and
     `--preset a2c --no-detach`. Before each run, write down what you expect. Which bugs hurt,
     which do not, and why? For those that hurt, `watch` a policy and count with
     `watch ... --episodes 100 --quiet`: what has the agent learned to do, and how does the bug
     make that the *right* thing to do, given the numbers it was trained on? For those that do
     not hurt: why not, here? (Look at how often episodes are truncated.) Would you have found
     each of them from the return alone? Read the code of `--no-detach` in `update`: what does
     the missing detach do to the critic, and why does the normalization of the advantages
     matter for it? (Hint: what is the sum of normalized advantages over a minibatch?)
   - **B2 -- A hint in the reward, and a leash.** `--hint-reward 0.05` adds a reward of 0.05 to
     every training step that ends next to a pellet (evaluation never pays it): a hint that good
     positions are near food. `--from-reference` starts training from `reference_policy.pt`
     (PPO after 100 000 steps) instead of from scratch, as one fine-tunes a pretrained model, and
     `--reference-coef β` adds β · KL(π ‖ π_ref) (averaged over the visited states) to the loss: a
     leash that keeps the policy close to the reference. Train with `--from-reference
     --hint-reward 0.05` and β = 0, 0.03, 0.1, 0.3, and 1; and, for comparison,
     `--from-reference` alone (no hint, no leash). Before you run them, write down what you expect.
     Then score each run's policies against the reference:
     `ppo.py score --reference --hint-reward 0.05 week09_ppo/results/policies/*hint*seed0.pt`
     (and the same for seeds 1 and 2). It prints the true return, the return with the hint, and
     the KL divergence from the reference. Plot both returns against the KL (one point per run
     and seed). What does the agent do at β = 0 (`watch` it)? Is it doing anything wrong, given
     the reward it was trained on? How do the two returns change as the leash gets longer? Is
     there a best β, and could you have found it if you could only see the hinted return? The
     lecture mentioned a KL penalty to a fixed reference model in the training of language
     models from human feedback: what might play the role of the hint there, and of the leash?

   *Optional, for whoever finishes early* (any group): see the end of this page.

5. **Closing: four algorithms on one board** *(the last 8 minutes, together)*. We put the board on
   the projector, with our own runs of all four algorithms on one laptop. Per sample and per
   minute, which algorithm wins, and why? What did PPO buy over A2C, and at what cost? We will
   discuss it together; each group says in one sentence what its question added.

## What to hand in

- **Leaderboard submissions** before the Week 10 practical: PPO, A2C, and PPO with λ = 1 (Task
  3), and the runs behind your group's answers (Task 4). The Week 10 practical starts with a
  test, so there are no presentations of this week's questions; but keep your notes: the test
  may ask about them.

## If you have time left (or at home)

- **Separate networks.** Give the critic its own `GridEncoder` (two networks, one optimizer). The
  lecture says that separate networks often work better: do they here, and what do they cost
  per minute? (Hint: what does `value_coef` do when nothing is shared, and with the gradient
  clipping?)
- **The learning rate.** PPO and A2C here share the step size 2e-3. Is that the best for both?
  Try 1e-3 and 4e-3 for each, and compare with Week 8's REINFORCE (same step size).
- **Learning-rate annealing**, linearly to 0 over training, as in many PPO implementations.
  What does it do to `kl` and `clip_frac` late in training?
- **The value of the start.** `ppo.py score` prints V(s0), the critic's value of the start, next
  to the discounted return that followed it, G0. Does PPO's critic over- or underestimate, and
  how does it compare with DQN's (Week 7) and the learned baseline of Week 8 (B1)?
