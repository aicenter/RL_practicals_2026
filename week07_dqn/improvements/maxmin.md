# Maxmin Q-learning

**In one equation.** Train $N$ Q-networks side by side, each with its own target network, and
bootstrap from their *minimum*:

$$y = r + \gamma (1 - d) \max_{a'} \min_{i = 1, \dots, N} Q_i(s', a'; \mathbf{w}^-_i).$$

All $N$ networks regress towards this same target. The agent acts ε-greedily with respect to
$\min_i Q_i(s, a; \mathbf{w}_i)$. With $N = 1$ this is DQN. Start with $N = 2$.

## The problem, and a blunt fix

The max in the DQN target is optimistic: it prefers whichever estimate happens to be too high
(see the [Double DQN page](double_dqn.md)). Maxmin Q-learning counters it head-on. Before taking
the max over actions, it takes the minimum over several independent estimates of each action's
value. The minimum of $N$ noisy estimates is *pessimistic*, and more so the larger $N$. One
minimum pulls down, the following maximum pulls up, and $N$ sets the balance between them.

- $N = 1$ overestimates (DQN).
- A large $N$ underestimates.
- Lan et al. (2020) show that, under their simple model of the errors, some $N$ in between
  removes the bias.

Underestimation is not harmless either. A pessimistic agent avoids what it knows little about,
which can slow down exploration. That is why $N$ is a dial and not "the bigger the better".

Maxmin with $N = 2$ is *not* Double DQN, although it also uses two networks. Double DQN
separates *choosing* the action from *evaluating* it. Maxmin takes the minimum *before*
choosing. (A related idea, *clipped double Q-learning*, uses the minimum of two critics in the
actor–critic method TD3, so you will meet it again.)

## Implementing it

- **Networks.** In `DQNAgent.__init__`, create a second `QNetwork` with its own target network.
  Give the optimizer the parameters of both, e.g.
  `torch.optim.Adam([*net1.parameters(), *net2.parameters()], lr=...)`, and clip the gradient
  norm over both.
- **`td_targets`.** Compute `torch.minimum(target1(s'), target2(s'))`, which has shape (B, 4),
  *then* take the max over actions.
- **`update`.** Compute $Q_i(s, a)$ for both networks, and add the two Huber losses against the
  same targets. One `backward` then trains both.
- **`sync_target`.** Sync both target networks.
- **Acting and evaluation.** Use the minimum of the two online networks. The simplest place to do
  this is `q_values`, which `greedy_action` and `start_value` both use.

Pitfalls:

- **Min and max in the wrong order.** Take the minimum over the networks first, then the max
  over actions. $\min_i \max_{a'} Q_i$ is a different, weaker correction.
- **One optimizer per network, but only one stepped.** If you use two optimizers, step both.
- **The same initialization.** The two networks are independent only if they start differently.
  Two `QNetwork(...)` calls do that; a `deepcopy` of the first network does not.

**A simplification.** The paper picks one network at random at each update and trains it on its
own minibatch, so that the networks see different data and stay nearly independent. Training
both on the same minibatch, as above, is simpler. Their differences then come only from the
initialization (and from where that leads them). Comparing the two versions is a good
experiment.

The cost: two forward and backward passes per update instead of one, so a run takes noticeably
longer than vanilla. Measure how much.

## What to look at

- **The V(s0) vs G0 column.** Does Maxmin over- or underestimate, and by how much? Compare with
  group A1's Double DQN, which attacks the same bias differently.
- **Learning curves and AUC** against your vanilla run. Is the final policy better? Does it die
  more or less often?
- **Different $N$** (1, 2, 3, 4), if you generalize your code to $N$ networks: where is the bias
  closest to zero, and is that also where the agent does best?
- **Random-network updates** as in the paper, against both on the same batch.

## Reading

- Q. Lan, Y. Pan, A. Fyshe, M. White (2020). *Maxmin Q-learning: controlling the estimation bias
  of Q-learning.* ICLR. [arXiv:2002.06487](https://arxiv.org/abs/2002.06487). Algorithm 1 is the
  method; Section 4 explains how $N$ controls the bias.
- S. Fujimoto, H. van Hoof, D. Meger (2018). *Addressing function approximation error in
  actor-critic methods.* ICML. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477). TD3's
  clipped double Q-learning.
