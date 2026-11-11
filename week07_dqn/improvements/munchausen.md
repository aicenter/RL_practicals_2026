# Munchausen DQN (group A3)

**In one equation.** Let $\bar\pi$ be the softmax policy of the target network at temperature $\tau$:

$$\bar\pi(a \mid s) = \frac{\exp\big(Q(s, a; \mathbf{w}^-)/\tau\big)}{\sum_b \exp\big(Q(s, b; \mathbf{w}^-)/\tau\big)} .$$

Then replace the target by

$$y = r + \alpha \big[\tau \ln \bar\pi(a \mid s)\big]_{-1}^{0} + \gamma (1 - d) \sum_{a'} \bar\pi(a' \mid s') \Big( Q(s', a'; \mathbf{w}^-) - \tau \ln \bar\pi(a' \mid s') \Big),$$

where $[x]_{-1}^{0}$ clips $x$ to the interval $[-1, 0]$. The paper's values are $\tau = 0.03$
and $\alpha = 0.9$.

The name refers to Baron Munchausen, who pulled himself out of a swamp by his own hair. The
agent uses *its own policy* as an extra learning signal.

## Ingredient 1: a soft maximum in the bootstrap (that is, "soft DQN")

The last term replaces $\max_{a'} Q(s', a')$. It looks complicated but is simple. Since
$\tau \ln \bar\pi(a' \mid s') = Q(s', a') - \tau \ln \sum_b \exp(Q(s', b)/\tau)$, the bracket is
the same number for every $a'$, and so is its average:

$$\sum_{a'} \bar\pi(a' \mid s') \big( Q(s', a') - \tau \ln \bar\pi(a' \mid s') \big) = \tau \ln \sum_{a'} \exp\big(Q(s', a')/\tau\big).$$

This is the *log-sum-exp*, a smooth version of the maximum. With four actions it lies between
$\max_{a'} Q(s', a')$ and that maximum plus $\tau \ln 4 \approx 0.04$. It also equals the
average of $Q(s', \cdot)$ under $\bar\pi$ plus $\tau$ times the entropy of $\bar\pi$. That is the
value in a problem that pays a small bonus for keeping options open (maximum-entropy RL). As $\tau \to 0$ you get DQN back.

With $\alpha = 0$, this ingredient alone is called Soft DQN. Its target is the discrete-action
version of the one in Soft Actor-Critic (SAC). This is one of the places where value-based and
policy-based methods meet; Weeks 8 and 9 come at it from the other side.

## Ingredient 2: the Munchausen bonus

The second term is added to the reward. It is the (scaled) log-probability, under the agent's
current policy, of the action $a$ that was actually taken in $s$. For a small $\tau$,

$$\tau \ln \bar\pi(a \mid s) \approx Q(s, a) - \max_b Q(s, b) ,$$

so the bonus is about $\alpha$ times *minus the gap* between the action taken and the best
action, as the agent currently sees them. It is zero for the greedy action and negative for the
others. The clip at −1 stops it from becoming enormous for actions the agent considers hopeless.

A worked example: the agent thinks going left in $s$ is worth 0.2 less than the best move. If it
does go left (an ε-greedy step), the target for $Q(s, \text{left})$ is lowered by
$0.9 \times 0.2 = 0.18$. That is almost two pellets. So every update pushes the values of
non-greedy actions *further down* relative to the greedy one, and the gap between the best action
and the rest (the *action gap*) grows. A larger action gap makes the argmax robust: small
estimation errors no longer flip the greedy action. The idea of increasing the action gap is
older (*advantage learning*; Bellemare et al., 2016); Munchausen gets it through the policy.

The deeper reason it works, shown in the paper: the bonus makes the algorithm equivalent to
regularizing each new policy towards the previous one, with a KL penalty of weight $\alpha\tau$,
plus an entropy bonus of weight $(1 - \alpha)\tau$. KL regularization between successive
policies makes the errors of successive Q-estimates *average out* rather than accumulate. The
same idea, keeping the new policy close to the old one, is at the heart of TRPO and PPO in
Week 9.

## Implementing it

Everything happens in `td_targets`, which already runs under `torch.no_grad()`:

1. `next_q = self.target_network(batch.next_observations)`, then
   `log_pi_next = F.log_softmax(next_q / tau, dim=1)`. The bootstrap is
   `(log_pi_next.exp() * (next_q - tau * log_pi_next)).sum(dim=1)`, or equivalently
   `tau * torch.logsumexp(next_q / tau, dim=1)`.
2. `log_pi = F.log_softmax(self.target_network(batch.observations) / tau, dim=1)`. Pick the
   action taken with `gather(1, batch.actions.unsqueeze(1)).squeeze(1)`. The bonus is
   `alpha * (tau * log_pi_taken).clamp(-1.0, 0.0)`.
3. `y = rewards + bonus + gamma * (1 - terminated) * bootstrap`.

Pitfalls:

- **Clipping the wrong thing.** Clip $\tau \ln \bar\pi$, not $\ln \bar\pi$. Clipping $\ln \bar\pi$
  to $[-1, 0]$ and then multiplying by $\alpha\tau$ caps the bonus at −0.027, which all but
  switches it off.
- **Computing the softmax yourself.** Don't compute `softmax` and then `log`. With
  $\tau = 0.03$, a Q-difference of 1 gives $e^{-33}$, and the log of a rounded-off zero is
  `-inf`. `F.log_softmax` and `torch.logsumexp` subtract the maximum first.
- **Using the online network for the bonus.** Both policies, $\bar\pi(\cdot \mid s)$ and
  $\bar\pi(\cdot \mid s')$, come from the *target* network, as in the paper. The bonus costs one
  extra forward pass, on the current states.

Acting and evaluation stay as they are: ε-greedy for acting, greedy for evaluation. The paper
does the same, although the softmax policy would be a natural alternative.

## What to look at

- The **V(s0) vs G0** column, against vanilla (and group A1's Double DQN). Careful: $Q$ now
  estimates a *regularized* return. It includes the entropy and bonus terms, not only the
  rewards. The difference is small where the agent is confident (action gaps much larger than
  $\tau$), but not zero.
- The **action gap** at the start: $Q(s_0, \cdot)$ sorted, best minus second best. Compute it
  for vanilla and for Munchausen.
- Faster learning (AUC), a better final policy, or both? The death rate?
- The ablation: $\alpha = 0$ is Soft DQN. Which ingredient does the work? What does a larger
  $\tau$ (0.1, 0.3) do?

## Reading

- N. Vieillard, O. Pietquin, M. Geist (2020). *Munchausen reinforcement learning.* NeurIPS.
  [arXiv:2007.14430](https://arxiv.org/abs/2007.14430). Equation (2) is the target above;
  Section 4 gives the clipping and the hyperparameters.
- M. G. Bellemare, G. Ostrovski, A. Guez, P. S. Thomas, R. Munos (2016). *Increasing the action
  gap: new operators for reinforcement learning.* AAAI.
  [arXiv:1512.04860](https://arxiv.org/abs/1512.04860)
