# Averaged-DQN

**In one equation.** Bootstrap from the *average* of the last $K$ target networks:

$$y = r + \gamma (1 - d) \max_{a'} \frac{1}{K} \sum_{k=1}^{K} Q(s', a'; \mathbf{w}^-_k),$$

where $\mathbf{w}^-_1$ is the current target network, $\mathbf{w}^-_2$ the one before it (one
target update ago), and so on. With $K = 1$ this is DQN. The paper uses $K$ between 2 and 20;
start with $K = 5$.

## The problem: the target network is a noisy snapshot

Every 1 000 steps, DQN copies the online network into the target network, and for the next 1 000
steps every target is computed from that one snapshot. But the online network at a particular
moment is the result of a particular sequence of minibatches. A slightly different sequence
would have given a somewhat different network, with different errors. The snapshot's errors go
straight into every target computed from it.

Two things follow.

- **The errors are optimistic on average.** The target takes a max over actions, so a snapshot
  that happens to overrate some action passes that optimism on (see the
  [Double DQN page](double_dqn.md)). The noisier the snapshot, the larger the bias.
- **The targets jump at every sync.** Every 1 000 steps the whole set of targets changes with the
  new snapshot's errors. Q-values chase those jumps, and the greedy policy flips with them. You
  can see this in the progress lines: the evaluation return changes a lot between checkpoints.

## The idea: average out the snapshot noise

Successive target networks have partly independent errors, because each one saw different
minibatches since the last. Averaging $K$ of them reduces the variance of the error, and with it
both the optimism of the max and the jumps at each sync. In a simplified model, Anschel et
al. (2017) show that averaging $K$ snapshots reduces the variance of the target-approximation
error by at least a factor of $K$. That is more than training $K$ independent networks and
averaging those would achieve, at a fraction of the cost.

The price is $K$ forward passes per target instead of one, and $K - 1$ extra copies of the
network in memory. The number of gradient steps is unchanged, so this costs much less than
training more networks.

## Implementing it

- Keep the older target networks in a `collections.deque(maxlen=K - 1)`.
- In `sync_target`, *before* overwriting the target network, append a copy of it to the deque
  (`copy.deepcopy(self.target_network)`). The deque drops the oldest by itself.
- In `td_targets`, compute $Q(s', \cdot)$ with every network in
  `[self.target_network, *older]`. Average them (`torch.stack(...).mean(dim=0)`), *then* take
  the max over actions.

Pitfalls:

- **Averaging after the max.** The average of $K$ maxima is *more* optimistic than the max of
  the average, which defeats the purpose. Average the Q-values first, then take the max.
- **Appending the wrong network.** Appending `self.target_network` itself, without the copy,
  stores the same object $K - 1$ times. `load_state_dict` changes it in place, so all copies
  would follow.
- **Breaking vanilla.** With $K = 1$, the deque has `maxlen=0` and must stay empty; the target
  is then exactly DQN's.

The paper also outputs the averaged network as the final policy. In the simple version above,
the agent acts and is evaluated with the online network. Acting with the average is a natural
variant to try.

## What to look at

- **Learning curves and AUC** against your vanilla run, and their spread over seeds.
- **The V(s0) vs G0 column.** How does the overestimation change with $K$ (1, 2, 5, 10)?
  Compare with group A1's Double DQN, which attacks the same bias differently.
- **The jumps at each sync.** Plot `--metric loss`, or compare consecutive checkpoints: is the
  evaluation return more stable from one checkpoint to the next?
- **Cost.** How much slower is a run with $K = 5$ or $K = 10$?

## Reading

- O. Anschel, N. Baram, N. Shimkin (2017). *Averaged-DQN: variance reduction and stabilization
  for deep reinforcement learning.* ICML. [arXiv:1611.01929](https://arxiv.org/abs/1611.01929).
  Algorithm 2 is the method; Section 5 analyses the variance.
- T. Schaul, A. Barreto, J. Quan, G. Ostrovski (2022). *The phenomenon of policy churn.*
  NeurIPS. [arXiv:2206.00730](https://arxiv.org/abs/2206.00730). How much the greedy policy of a
  DQN agent changes from one update to the next, and why.
