# Evaluating averaged weights (group B2)

**In one equation.** Keep an exponential moving average of the online network's weights,

$$\bar{\mathbf{w}} \leftarrow (1 - \beta)\, \bar{\mathbf{w}} + \beta\, \mathbf{w} \qquad \text{after every gradient step,}$$

starting from $\bar{\mathbf{w}} = \mathbf{w}$, and *evaluate* the greedy policy of
$Q(s, a; \bar{\mathbf{w}})$ instead of that of $Q(s, a; \mathbf{w})$. Start with $\beta = 0.01$.
With $\beta = 1$ this is DQN.

Unlike every other page here, this one changes nothing about how the agent learns. Acting,
the replay buffer, the targets, and the gradient steps all stay exactly as in vanilla DQN. Only
the policy we *report* (and would deploy) changes.

## The problem: the online network never sits still

SGD does not settle at a minimum. With a constant step size, the weights keep jittering around
it, driven by the noise of the minibatches. In DQN there is more. Every 1 000 steps the target
network jumps, the targets jump with it, and the online network starts chasing new targets. And
as the policy changes, so does the data in the buffer.

For a Q-network, small changes in the weights can have large effects on the *policy*. Where two
actions have nearly the same value, a tiny change in the Q-values swaps the argmax. Schaul et
al. (2022) call this *policy churn*: in DQN agents, a surprisingly large fraction of states
changes its greedy action after a single gradient step. So the policy at any particular moment,
for example at an evaluation checkpoint, is partly an accident of the last few minibatches. You
can see it in your vanilla run: the evaluation return goes up and down from one checkpoint to
the next, even late in training.

## The idea: report the average, not the last iterate

The jitter is noise around something better. Averaging the weights over the last few hundred
gradient steps cancels much of it, and the averaged network's greedy policy is the policy the
agent has been converging towards, without the accident of the last few steps.

This is an old idea in stochastic approximation (Week 5). Polyak and Juditsky (1992) showed that
averaging the iterates of a noisy stochastic approximation method gives the best possible
asymptotic rate. You get the fast progress of a large step size *and* the low noise of a small
one. In deep learning the same trick, under names like EMA weights or weight averaging (Izmailov
et al., 2018), is widely used for the final model.

You have seen this formula before: it is the *soft* (Polyak) target update, which some DQN
variants use instead of copying the online network every 1 000 steps. The difference is where
the averaged network goes. There it is used for the targets, and changes learning. Here it is
used only for evaluation.

Two things to think about:

- **Why may we average the weights of a neural network at all?** The loss is far from convex,
  and the average of two good networks can be a bad one. Averaging two *independently* trained
  networks is nonsense: their hidden units can be in any order, so the average mixes unrelated
  features. Averaging works here because the iterates of one run, a few hundred steps apart,
  are close to each other and in the same "valley" of the loss.
- **How long a memory?** The average effectively covers the last $1/\beta$ gradient steps. With
  $\beta = 0.01$ that is about 100 updates, or 400 environment steps (one update every 4 steps).
  A much smaller $\beta$ keeps old, worse weights in the average. A larger one hardly averages
  at all.

## Implementing it

- **The averaged network.** In `DQNAgent.__init__`:
  `self.averaged_network = copy.deepcopy(self.q_network).requires_grad_(False)`.
- **The update.** Add a method, decorated with `@torch.no_grad()`, that loops over
  `zip(self.averaged_network.parameters(), self.q_network.parameters())` and calls
  `averaged.lerp_(online, beta)` on each pair. That is $\bar w \leftarrow \bar w + \beta (w -
  \bar w)$, in place. Call it in `train` after every gradient step.
- **Evaluation.** The Monitor evaluates whatever functions you pass to `evaluate_if_due` and
  `finish`, at present `agent.greedy_action` and `agent.start_value`. Write two more methods
  that do the same with the averaged network, and pass those instead. Keep `act` as it is: the
  agent still explores with the online network.

Pitfalls:

- **Starting the average from zero**, or from a fresh `QNetwork`. Start from a copy of the
  online network.
- **Averaging without `torch.no_grad()`.** The online parameters require gradients, so autograd
  would record the averaging as part of a graph.
- **Acting with the averaged network** by accident (for example by changing `greedy_action`,
  which `act` uses). That is a different method, and it changes learning. It is a fine
  experiment, but not this one.

**Check it.** With the same seeds, your training run should be *identical* to your vanilla
run, step for step: compare the loss curves (`--metric loss`). If they differ, something other
than evaluation has changed.

## What to look at

- **Learning curves and AUC** against your vanilla run. Since the training is identical, every
  difference comes from which weights are evaluated.
- **The V(s0) vs G0 column.** Does the averaged network overestimate less? Why might it?
- **Policy churn.** Collect a few hundred boards (for example from evaluation episodes). At each
  checkpoint, record the greedy actions of the online network and of the averaged network on
  them. In what fraction of the boards does the greedy action change between two checkpoints,
  for each network?
- **Different $\beta$** (0.001, 0.01, 0.1): where is the sweet spot, and why does a very small
  $\beta$ hurt early in training?
- **Is it a real improvement?** If the leaderboard number goes up although the agent has
  learned nothing new, has the algorithm improved? Which network would you deploy? And what
  does this say about reading any leaderboard: what exactly does its number measure?

## Reading

- B. T. Polyak, A. B. Juditsky (1992). *Acceleration of stochastic approximation by averaging.*
  SIAM Journal on Control and Optimization 30(4), 838–855.
  [doi:10.1137/0330046](https://doi.org/10.1137/0330046). The theory of iterate averaging.
- T. Schaul, A. Barreto, J. Quan, G. Ostrovski (2022). *The phenomenon of policy churn.*
  NeurIPS. [arXiv:2206.00730](https://arxiv.org/abs/2206.00730). How much the greedy policy of a
  DQN agent changes from one update to the next, and why.
- P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, A. G. Wilson (2018). *Averaging weights
  leads to wider optima and better generalization.* UAI.
  [arXiv:1803.05407](https://arxiv.org/abs/1803.05407). Weight averaging in supervised deep
  learning.
