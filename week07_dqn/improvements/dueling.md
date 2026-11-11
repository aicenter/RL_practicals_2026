# Dueling networks (optional)

**In one equation.** Split the head of the Q-network into two streams, a state value $V(s)$ and
advantages $A(s, a)$, and combine them as

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') .$$

Nothing else changes: the target, acting, and the loss all use $Q$ as before.

## The idea: most of the time, the action barely matters

In many states, all actions are worth about the same. Far from the ghost, with pellets in every
direction, it hardly matters which way you go first. What matters is the *state*: how many
pellets are left, and where the ghost is. In other states the action matters a lot, for example
when the ghost is one step away.

A plain head has one output per action. When the agent updates $Q(s, a)$, only that action's
output is moved directly; the shared features change as a side effect. With a dueling head,
every update moves $V(s)$, *whichever* action was taken. So the value of a state is learned from
all the data about it, and the advantage stream only has to learn the (often small) differences
between the actions.

## Why subtract the mean

Without the last term, $V$ and $A$ are not *identifiable*. Adding a constant $c$ to $V(s)$ and
subtracting it from every $A(s, a)$ gives the same $Q$. The network can then shift value back and
forth between the two streams freely, and $V$ need not mean anything.

Subtracting $\max_{a'} A(s, a')$ instead would pin it down most cleanly: the best action gets
advantage 0, and $V(s) = \max_a Q(s, a)$. The paper uses the mean because it is more stable.
With the mean, the advantages only need to change as fast as their average, not as fast as the
best action's advantage. With the mean, $V$ is the average of the $Q$-values, not their maximum.

## Implementing it

In `QNetwork.__init__`, keep the encoder and replace `self.head` by two layers:

- `self.value = nn.Linear(n_features, 1)`
- `self.advantage = nn.Linear(n_features, n_actions)`

In `forward`, combine them:

```python
features = self.encoder(observations)
value, advantage = self.value(features), self.advantage(features)
return value + advantage - advantage.mean(dim=1, keepdim=True)
```

Mind the shapes: (B, 1) + (B, 4) − (B, 1) broadcasts to (B, 4). The paper gives each stream a
hidden layer of its own (on Atari, 512 units). You can try a small hidden layer per stream, but
then compare against a plain head of the same size. Otherwise you are measuring the extra
capacity, not the dueling architecture.

## What to look at

- Learning curves (AUC) and final return against your vanilla run.
- $V$ and $A$ in different states. Add a method that returns both streams, take a board from
  `make_env().reset(...)`, and step to a position next to the ghost. Where does the advantage
  stream spread the actions apart, and where does it hardly bother?
- Does the dueling head combine well with Double DQN?

## Reading

- Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, N. de Freitas (2016). *Dueling
  network architectures for deep reinforcement learning.* ICML.
  [arXiv:1511.06581](https://arxiv.org/abs/1511.06581). Section 3 covers the architecture and
  the choice between mean and max.
