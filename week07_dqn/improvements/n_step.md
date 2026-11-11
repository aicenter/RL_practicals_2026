# n-step returns (optional)

**In one equation.** Learn from $n$ real rewards before bootstrapping:

$$y_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} \;+\; \gamma^n (1 - d)\, \max_{a'} Q(s_{t+n}, a'; \mathbf{w}^-),$$

where $d = 1$ if the episode terminated within those $n$ steps. If the episode ended after
$m < n$ steps, the sum has only $m$ rewards and the bootstrap uses $\gamma^m$ and $s_{t+m}$.
With $n = 1$ this is the vanilla target.

## The problem: one-step targets pass news backwards slowly

The largest reward in this world, the 1.0 for the last pellet, comes at the end of an episode.
With one-step targets, a state $k$ steps earlier learns about it only after the value has been
passed back through the $k - 1$ states in between, one link per update. The target network
slows this further. Between two syncs it is frozen, so along a chain of states the value can move
back by at most one step per sync, which is every 1 000 steps in our runs. A 100 000-step run has
100 syncs, which is just about the length of one episode. The network can generalize across
similar boards and so do better than this, but the chain is real. An $n$-step target moves the
news $n$ steps per sync.

There is a second advantage. Early in training, $Q$ is mostly wrong. The one-step target trusts
it with weight $\gamma$. The $n$-step target trusts it with weight $\gamma^n$ and puts $n$ real
rewards in front of it.

## The price: variance, and learning about the wrong policy

**Variance.** A sum of $n$ random rewards is noisier than one. In our world the randomness is
mainly the ghost: whether it happens to come your way within the next $n$ steps.

**Off-policy bias.** The one-step Q-learning target is off-policy for free. It uses the action
$a_t$, whatever chose it, and then the max: what the greedy policy would do. The $n$-step return
also contains the actions $a_{t+1}, \dots, a_{t+n-1}$. These were chosen by the *behaviour*
policy: ε-greedy, and an old version of it, since the transition may have sat in the buffer for
up to 50 000 steps. So the target estimates "follow the old ε-greedy policy for $n - 1$ steps,
then act greedily", not $Q^*$. In a world where one random step can walk you into the ghost, that
estimate is *pessimistic* about states near the ghost.

Week 6 showed the textbook correction: importance sampling. With a greedy target policy,
though, the ratio $\pi(a \mid s) / b(a \mid s)$ is zero whenever the behaviour took a non-greedy
action, and $1/(1 - \varepsilon + \varepsilon/4)$ otherwise. Practical corrections, such as tree
backup or Retrace (Munos et al., 2016), cut the return short where the behaviour deviated.
Rainbow (Hessel et al., 2018) ignores the problem, with $n = 3$. Whether that is safe depends on
$n$, on $\varepsilon$, and on how much a single random action can cost: a good question to test.

## Implementing it

Three places change.

**The buffer** stores, per transition, how many rewards it contains (or the discount
$\gamma^m$ directly). Add an array next to `rewards`, an argument to `add`, and a field to `Batch`.
Give the field a default and read it with `getattr`, so that `check` keeps working (see the
[index page](README.md)).

**`td_targets`** uses `gamma ** batch.n_steps` instead of `gamma`. Nothing else changes: the
stored reward is already the discounted sum.

**`train`** collects the windows. Keep a `collections.deque(maxlen=n)` of the last steps
(observation, action, reward). After each environment step:

```
append (observation, action, reward) to the window
if the episode ended (terminated or truncated):
    for each start still in the window, oldest first:
        store (its observation, its action, the discounted sum of the rewards from it to the end,
               next_observation, terminated, number of those rewards)
    empty the window
elif the window holds n steps:
    store the oldest start with all n rewards, next_observation, terminated = False, n
    (the deque drops it at the next append)
```

The common bugs are all at the end of an episode:

- **Losing the last windows.** When the episode ends, every window still open must be stored,
  each with fewer than $n$ rewards.
- **Storing a full window twice.** If the window is full on the step that ends the episode,
  store it only once, as part of the flush.
- **Bootstrapping wrongly.** A window that reaches a *terminated* end must not bootstrap
  ($d = 1$). One that reaches a *truncated* end must bootstrap, with $\gamma^m$, from the final
  board.
- **Mixing episodes.** Empty the window after the flush, so that the next episode starts afresh.

The Monitor's `record_step` keeps receiving the one-step reward. It only does bookkeeping.

Check yourself: with `--n-step 1`, your run must reproduce your vanilla run exactly.

## What to look at

- Learning speed (AUC) against the final return: does $n$-step help early and hurt late, or the
  reverse?
- The death rate of the greedy policy. Is it more careful, or less, than vanilla's? Why?
- The V(s0) vs G0 column. The maximum is applied once per $n$ steps, and the behaviour policy's
  random steps pull the other way. Which effect wins?
- Different $n$ (2, 3, 5, 10). If you suspect the off-policy bias, lower $\varepsilon$ at the end
  of training (`--epsilon-end 0.01`) and run vanilla with the same change.

## Reading

- R. S. Sutton, A. G. Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Chapter
  7, "n-step bootstrapping": Sections 7.3 (off-policy $n$-step learning) and 7.5 (tree backup).
- M. Hessel et al. (2018). *Rainbow: combining improvements in deep reinforcement learning.*
  AAAI. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298). Uses $n = 3$ without correction;
  its ablation finds $n$-step returns among the most important components on Atari.
- R. Munos, T. Stepleton, A. Harutyunyan, M. G. Bellemare (2016). *Safe and efficient
  off-policy reinforcement learning.* NeurIPS. [arXiv:1606.02647](https://arxiv.org/abs/1606.02647).
  Retrace.
