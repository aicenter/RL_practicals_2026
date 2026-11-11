# Double DQN (group A1)

**In one equation.** Let the *online* network choose the next action, and the *target* network
evaluate it:

$$y = r + \gamma (1 - d)\, Q\big(s', \arg\max_{a'} Q(s', a'; \mathbf{w});\ \mathbf{w}^-\big)$$

instead of $y = r + \gamma (1 - d) \max_{a'} Q(s', a'; \mathbf{w}^-)$.

## The problem: the maximum of noisy estimates is optimistic

Suppose that in some state $s'$ all four actions are truly worth 1.0. Our network has seen
$s'$ only a few times, so its estimates are the truth plus some noise, say independent with a
standard deviation of 0.1. Each estimate on its own is unbiased. Their maximum is not: it picks
whichever estimate happens to be too high. The expected maximum of four independent standard
normal variables is about 1.03, so on average the target uses about 1.10 where the truth is 1.0.
In general, because the maximum is a convex function,

$$\mathbb{E}\Big[\max_{a'} \hat Q(s', a')\Big] \ \ge\ \max_{a'} \mathbb{E}\big[\hat Q(s', a')\big],$$

with equality only when there is no noise (or one action always wins anyway).

Bootstrapping makes it worse. The optimistic target trains $Q(s, a)$. That inflated value then
appears inside the maximum of the target for the state before $s$, which adds its own bit of
optimism, and so on backwards. With $\gamma = 0.99$, the chain is long.

The optimism matters for the *policy* because it is not uniform. If every value were inflated
by the same amount, the argmax, and with it the policy, would not change. In fact the bias is
largest where the estimates are noisiest. Those are states the agent has rarely visited, and
states whose outcome is random. In our world, that means anything near a ghost that moves at
random. So the agent ends up preferring actions whose values it knows *least* well.

You met the tabular version in Week 6 as *maximization bias* (Sutton and Barto, Section 6.7).

## The fix: separate choosing from evaluating

**Double Q-learning** (van Hasselt, 2010) keeps two independent estimates, $Q_A$ and $Q_B$. It
chooses the action with one and evaluates it with the other. If the noise in $Q_B$ is
independent of the choice made by $Q_A$, then $Q_B$ is an unbiased estimate of the chosen
action's value, which is at most the best action's value. The upward bias is gone; if anything,
the result is slightly pessimistic.

**Double DQN** (van Hasselt, Guez, and Silver, 2016) does not train a second network. It reuses
the two that DQN already has: the online network chooses and the target network evaluates. The
two are not independent, since the target network is an older copy of the online one. So the
bias is reduced, not removed. The cost is one extra forward pass of the online network on the
batch of next states.

## Implementing it

Everything happens in `td_targets`, which already runs under `torch.no_grad()`:

1. Run the online network on the next states, `self.q_network(batch.next_observations)`, of
   shape (B, 4). Take the argmax over the actions with `keepdim=True`, which gives shape (B, 1).
2. Run the target network on the same states. Use `gather(1, best)` to pick the value of the
   chosen action, then `squeeze(1)`, which gives shape (B,).
3. The rest is as before: rewards, plus $\gamma$, times one minus `terminated`, times that value.

Pitfalls:

- **Swapping the roles.** Letting the target network choose and the online network evaluate is
  also "double" in a sense, but it is not Double DQN.
- **Leaving gradients on.** If you compute the argmax outside `td_targets`, compute it under
  `torch.no_grad()` too.
- **Breaking vanilla.** With `--double` off, the code must give exactly the vanilla target.

## Measuring overestimation: the V(s0) vs G0 column

At the start of every evaluation episode, the Monitor records the agent's own estimate of the
start, $\max_a Q(s_0, a; \mathbf{w})$. After the episode, it records what the greedy policy
actually got from there, the discounted return $\sum_t \gamma^t r_t$. Both are averaged over the
evaluation episodes, so their difference estimates how much the agent overestimates the value of
its own policy at the start. `--metric overestimation` plots this difference over training.

Things to keep in mind:

- $G_0$ is random (the ghost), so the difference has a standard error. Look at it before reading
  much into a small gap.
- The start is only one state. The bias can be larger elsewhere, for example next to the ghost.
  To look there, play an episode with `make_env()` and print `agent.q_values(observation)` along
  the way, next to what then happens.

## What to look at

- How large is the gap for vanilla DQN, in absolute terms and relative to $G_0$? Does it grow,
  shrink, or stay put during training?
- How much of the gap does Double DQN remove? Does it do so in every seed?
- Does a smaller bias buy a better policy (AUC, final return, death rate)? These are different
  questions: the argmax only cares about the *ordering* of the actions in each state.

## The bigger picture: the optimizer's curse

Choosing the best of many noisy estimates, and then being disappointed, is not specific to RL.
In decision analysis it is called the **optimizer's curse** (Smith and Winkler, 2006). The option
that looks best is disproportionately often the one whose estimate erred upwards. The more
options there are, and the noisier the estimates, the larger the expected disappointment. You
have seen it already: the model that won a large hyperparameter search does worse on the test
set than on the validation set it was selected on.

It comes back later in this course. When a policy is optimized against a *learned* reward model,
as in RLHF, the policy searches for the outputs that the model scores highest. Those are
disproportionately the outputs where the model is wrong in the policy's favour. As
optimization pressure grows, the learned reward keeps rising while the true quality first
rises and then falls (Gao, Schulman, and Hilton, 2023). The remedy has the same shape as Double
DQN: don't let the same noisy estimate both choose and grade.

## Reading

- H. van Hasselt, A. Guez, D. Silver (2016). *Deep reinforcement learning with double
  Q-learning.* AAAI. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- H. van Hasselt (2010). *Double Q-learning.* NeurIPS. The tabular original.
- R. S. Sutton, A. G. Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Section
  6.7, "Maximization bias and double learning".
- J. E. Smith, R. L. Winkler (2006). *The optimizer's curse: skepticism and postdecision surprise
  in decision analysis.* Management Science 52(3).
- L. Gao, J. Schulman, J. Hilton (2023). *Scaling laws for reward model overoptimization.* ICML.
  [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
