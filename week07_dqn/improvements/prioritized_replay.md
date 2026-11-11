# Prioritized experience replay (group B1)

**In equations.** Replay transition $i$ with probability

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad p_i = |\delta_i| + \varepsilon_p ,$$

where $\delta_i$ is the TD error of transition $i$ the last time it was replayed, and
$\varepsilon_p$ is a small constant (say $10^{-6}$). Multiply its loss by the importance-sampling
weight

$$w_i = \frac{\big(N \cdot P(i)\big)^{-\beta}}{\max_j w_j},$$

where $N$ is the number of transitions in the buffer, and $\beta$ is annealed from
$\beta_0 = 0.4$ to 1 over training. The paper's values for the proportional variant are
$\alpha = 0.6$ and $\beta_0 = 0.4$.

## The problem: uniform replay spends most updates on what is already known

Once the buffer is full, each transition is replayed about 16 times during its stay with our
settings: a batch of 64 every 4 steps, and 50 000 steps in the buffer. Most transitions are dull.
A step down an empty corridor for −0.01 is soon predicted almost perfectly, and replaying it
teaches nothing. A few are surprising: the last pellet (+1.0), or being caught (−1.0). Uniform
replay gives the dull and the surprising the same 16 replays.

Prioritized replay replays transitions *in proportion to how surprising they were*: where the TD
error is large, there is something left to learn. The idea comes from *prioritized sweeping* in
planning (Sutton and Barto, Section 8.4), which updates first the states whose values are
about to change most.

Details that matter:

- **New transitions** get the largest priority seen so far, so that each is replayed at least
  once before its priority is known.
- **α** interpolates between uniform sampling ($\alpha = 0$) and pure prioritization
  ($\alpha = 1$).
- **$\varepsilon_p > 0$** keeps every transition sampleable, including those with zero error.
- **Priorities go stale.** A transition's priority is updated only when it is replayed, so a
  transition that *was* surprising long ago keeps its high priority until it comes up again.

## The bias, and why the weights correct it

The loss we want to minimize is an average over the buffer, $\frac{1}{N} \sum_i L_i$. Sampling
with probabilities $P(i)$ instead gives $\sum_i P(i)\, L_i$, which is a different objective. It
over-weights the surprising transitions. That matters because the TD target is an average over
what happens next.

Take a state next to the ghost, and an action after which the ghost catches you one time in four.
The "caught" transitions have large errors and get replayed much more often than the "escaped"
ones. Uniform replay would teach the agent that the action gets caught 25 % of the time.
Prioritized replay teaches it that the action is more dangerous than it is.

This is Week 6's problem again: we sample from one distribution, $P$, but want expectations under
another, uniform. The importance-sampling ratio is $(1/N)/P(i) = (N P(i))^{-1}$, which is the
weight with $\beta = 1$. Smaller $\beta$ corrects only partly. The paper argues that full
correction matters most near the end of training, when the targets have settled, hence the
annealing. Dividing by the largest weight keeps all weights at most 1, so that they only ever
scale updates *down*. (In the paper's words, this is weighted importance sampling, not
ordinary.)

## The learning rate

A prioritized batch is full of large errors, and so produces large gradients. The same few
transitions also come back again and again. Schaul et al. (2016) therefore reduced the step
size by a factor of 4 relative to their Double DQN setup: their only change to the baseline's
hyperparameters. Rainbow, too, uses a much smaller learning rate than DQN. If you try a smaller
learning rate with prioritization, run vanilla with the same learning rate as well (see the
[index page](README.md)).

## Implementing it

**A sum tree.** To sample in proportion to $p_i^\alpha$ quickly, keep an array `tree` of size
$2C$ for a capacity $C$. Leaf $i$ lives at `tree[C + i]` and holds $p_i^\alpha$. Every inner node
`k` holds `tree[2k] + tree[2k + 1]`, so `tree[1]` is the total.

- *Updating* a leaf: set it, then walk up (`k //= 2`) and recompute each parent. This is
  $O(\log C)$.
- *Sampling*: draw $u$ uniformly in $[0, \text{total})$ and walk down from the root. If $u$ is
  less than the left child's sum, go left. Otherwise subtract the left sum and go right. This is
  also $O(\log C)$.
- *Stratified sampling*, as in the paper: split $[0, \text{total})$ into $B$ equal segments and
  draw one $u$ in each. A batch then covers the whole range of priorities.

Start with a power of two for $C$; it keeps the indexing simple. Or skip the tree at first: a
flat array of priorities and `rng.choice(size, size=B, p=priorities / priorities.sum())` is
$O(N)$ per batch. Time it before deciding you need the tree.

**The buffer.** Write `PrioritizedReplayBuffer` as a subclass of `ReplayBuffer`:

- `add` also sets the new transition's priority to the maximum so far.
- `sample(batch_size, rng, beta)` returns a `Batch` that also carries the `weights` and the
  buffer `indices` of the sampled transitions.
- A new method, `update_priorities(indices, td_errors)`, writes the new priorities back.

**The update.** Use `F.smooth_l1_loss(..., reduction="none")` to get one loss per transition,
multiply by `batch.weights`, then take the mean. The TD errors $y - Q(s, a; \mathbf{w})$, detached,
must get back to the buffer. `update` must still return a float (for `check`), so either keep
the last errors in an attribute or add a second method that returns both.

**The loop.** $\beta$ comes from `linear_schedule(step, beta_start, 1.0, total_steps)`; you
already have it. After each update, call `buffer.update_priorities(batch.indices, errors)`.

## What to look at

- **Which transitions get the highest priorities?** Look at the top 20 leaves: their rewards,
  their `terminated` flags.
- **Learning curves against your vanilla run**, at the default learning rate, and possibly at a
  lower one (both runs).
- **What $\alpha$ does** (0.3, 0.6, 1.0), and what happens without the weights ($\beta = 0$).
- **The wall-clock time per run.** Is prioritization worth its cost here?

## Reading

- T. Schaul, J. Quan, I. Antonoglou, D. Silver (2016). *Prioritized experience replay.* ICLR.
  [arXiv:1511.05952](https://arxiv.org/abs/1511.05952). Section 3.4 has the weights; Section 4
  has the hyperparameters and the step-size change.
- R. S. Sutton, A. G. Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Section
  8.4, "Prioritized sweeping".
- M. Hessel et al. (2018). *Rainbow: combining improvements in deep reinforcement learning.*
  AAAI. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)
