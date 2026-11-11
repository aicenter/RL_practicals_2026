# The improvements, in more detail

One page per improvement from Task 4. Each page covers four things:

- what the improvement changes, in one equation;
- which weakness of vanilla DQN it addresses, and why it should help;
- how to fit it into `dqn.py`, and where people trip up;
- what to look at in your runs.

It ends with where to read more. The pages are optional: the README is enough to start. Come
here when you want to understand *why*, or when your implementation doesn't behave.

| Page | Group | What it changes | Where in `dqn.py` |
| --- | --- | --- | --- |
| [Double DQN](double_dqn.md) | A1 | which network chooses the next action in the target | `td_targets` |
| [n-step returns](n_step.md) | A2 | what a transition in the buffer contains | `train`, `ReplayBuffer`, `td_targets` |
| [Munchausen DQN](munchausen.md) | A3 | a soft max in the target, plus a bonus on the reward | `td_targets` |
| [Prioritized replay](prioritized_replay.md) | B1 | which transitions are replayed, and with what weight | `ReplayBuffer`, `update`, `train` |
| [Noisy networks](noisy_nets.md) | B2 | how the agent explores | `QNetwork`, `act`, `train` |
| [Dueling networks](dueling.md) | optional | the head of the network | `QNetwork` |
| [Averaged-DQN](averaged_dqn.md) | optional | the target averages several old target networks | `DQNAgent`, `sync_target`, `td_targets` |
| [Maxmin Q-learning](maxmin.md) | optional | two networks; the target takes their minimum | `DQNAgent`, `td_targets`, `update` |

## Notation

As in the Week 7 lecture:

| Symbol | Meaning |
| --- | --- |
| $Q(s, a; \mathbf{w})$ | the online network |
| $\mathbf{w}^-$ | the parameters of the target network |
| $\mathcal{D}$ | the replay buffer |
| $(s, a, r, s', d)$ | a transition, where $d = 1$ if the episode *terminated* at $s'$ (it is `batch.terminated`) |

The vanilla DQN target, which every page modifies, is

$$y = r + \gamma (1 - d) \max_{a'} Q(s', a'; \mathbf{w}^-),$$

and the loss is the Huber loss between $Q(s, a; \mathbf{w})$ and $y$, averaged over the batch.

## Advice for all of them

- **Keep vanilla one switch away.** Put your improvement behind a `Config` field whose default
  keeps vanilla DQN. With the switch off, your file should reproduce your vanilla run to the
  last digit (same seeds, same numbers). If it doesn't, something else has changed. This is the
  cheapest bug check there is.
- **Keep `check` passing.** `check` uses the default `Config`, and it builds its own small
  batches with only the five standard fields of `Batch`. If you add a field to `Batch` (n-step,
  prioritized replay), read it with `getattr(batch, "weights", None)` and fall back to the
  vanilla behaviour when it is missing.
- **Compare against your own vanilla run**, with the same seeds. With three seeds, a
  difference smaller than about twice its standard error is not evidence of anything.
- **Change one thing at a time.** If you tune your improvement's own hyperparameters (n, α, σ₀,
  τ, …), give each variant its own run and write down what you tried. If you also change a
  shared hyperparameter, such as the learning rate, run vanilla with the same change. Otherwise
  you cannot tell which change did the work.
