# Noisy networks (optional)

**In one equation.** Replace a linear layer $y = W x + b$ by a *noisy* one,

$$y = (\mu_W + \sigma_W \odot \varepsilon_W)\, x + \mu_b + \sigma_b \odot \varepsilon_b ,$$

where $\mu$ and $\sigma$ are learned parameters, $\varepsilon$ is random noise drawn afresh from
time to time, and $\odot$ is element-wise multiplication. Then switch ε-greedy off: the agent
explores because its network is randomly perturbed.

## The problem: ε-greedy explores by dithering

ε-greedy replaces the greedy action, independently at each step, by a uniformly random one.
This has three weaknesses.

- **It doesn't go far.** Reaching a region of the maze that the greedy policy avoids takes
  several random actions in a row. At $\varepsilon = 0.05$, three in a row happen with probability
  $0.05^3 \approx 10^{-4}$, and even then a random walk mostly turns back.
- **It is the same everywhere.** It explores as much in a state the agent understands well as in
  one it has never seen.
- **It follows a hand-made schedule** (1 → 0.05 over 30 000 steps), whatever the agent has
  actually learned.

Perturbing the *weights* instead of the actions changes the greedy policy as a whole. As long as
the noise is held fixed, the perturbed policy is *consistent*: it is a different but coherent
plan, followed for several steps, not a coin flipped at each one. And the change depends on the
state: where the Q-values of the actions are far apart, a small perturbation does not flip the
choice. Where they are close, it does.

## How much noise: learned

The noise scales $\sigma$ are parameters like any other, trained by gradient descent on the usual
TD loss. If noise in some weight hurts the predictions, the gradient shrinks that $\sigma$. So
the network decides, weight by weight, how much randomness to keep, and the hand-made ε schedule
disappears. Note what this does *not* say. The loss is a prediction loss, not an exploration
objective, so nothing guarantees that $\sigma$ shrinks at the right speed, or at all. You can
measure what it does.

## Factorized Gaussian noise

A layer with $p$ inputs and $q$ outputs has $pq$ weights. Drawing $pq$ independent noise values
every time is expensive, so the paper factorizes:

1. Draw $\varepsilon^{\text{in}} \in \mathbb{R}^p$ and $\varepsilon^{\text{out}} \in \mathbb{R}^q$
   from standard normals.
2. Apply $f(x) = \operatorname{sgn}(x)\sqrt{|x|}$.
3. Set $\varepsilon_W = f(\varepsilon^{\text{out}})\, f(\varepsilon^{\text{in}})^\top$ (an outer
   product, `torch.outer`) and $\varepsilon_b = f(\varepsilon^{\text{out}})$.

That is $p + q$ random numbers instead of $pq + q$.

Initialization, as in the paper: each element of $\mu_W$ and $\mu_b$ is drawn from
$U[-1/\sqrt{p}, 1/\sqrt{p}]$, and every element of $\sigma_W$ and $\sigma_b$ starts at
$\sigma_0/\sqrt{p}$, with $\sigma_0 = 0.5$.

## Implementing it

**`NoisyLinear(nn.Module)`.**

- Make `weight_mu`, `weight_sigma`, `bias_mu`, and `bias_sigma` `nn.Parameter`s.
- Keep the noise vectors in buffers: `self.register_buffer("eps_in", torch.zeros(p))`, and the
  same for `eps_out`. Buffers are copied along with the parameters by `copy.deepcopy` and
  `load_state_dict`, but are not trained.
- Add a method `resample_noise()` that draws fresh noise.
- Add a switch that makes `forward` use the means only, for evaluation.

**`QNetwork`.** Use `NoisyLinear` for the head instead of `nn.Linear`. The paper makes *all*
fully connected layers noisy. Ours hidden inside `GridEncoder` is not, which is fine to start
with. If you want more noise, give the head a noisy hidden layer of its own.

**No ε.** With the switch on, act greedily with respect to the noisy network: set
`epsilon = 0.0` in `train`, instead of following the schedule.

**When to resample.** The paper's rule: draw new noise after every optimization step, and act
with fixed parameters in between. Its DQN took one optimization step per action, so the noise
changed before every action. The target network gets *independent* noise of its own in the
target, and acting uses a third, independent sample.

We update only every 4 steps, so there are two reasonable readings:

- **Resample only at updates.** This follows the general rule, and each perturbed policy is
  kept for 4 steps.
- **Resample before every action as well.** This is closer to the paper's DQN in effect.

Pick one and say which.

**Evaluation without noise.** The Monitor evaluates the function you pass to
`evaluate_if_due` and `finish`, which is `agent.greedy_action` and `agent.start_value`. Give the
agent methods that switch the noise off, call those, and switch the noise back on afterwards.
Otherwise every evaluation measures a random perturbation of your policy.

**Logging.** `monitor.log(sigma=...)` with, for example, the mean of `|weight_sigma|` puts it in
the results. Plot it with `--metric sigma`.

## What to look at

- **Does $\sigma$ shrink during training?** Compare it with the size of $\mu$.
- **How often does the noisy greedy action differ** from the noise-free one? That is the
  agent's effective exploration rate; compare it with the ε schedule. Log it like $\sigma$.
- **Is exploration more consistent?** Look at a few steps of an episode with fixed noise.
- **Several seeds.** Is noisy exploration more or less reliable than ε-greedy? What changes
  with $\sigma_0$ (0.1, 0.5, 1.0)?

## Reading

- M. Fortunato et al. (2018). *Noisy networks for exploration.* ICLR.
  [arXiv:1706.10295](https://arxiv.org/abs/1706.10295). Section 3 covers the layer and the noise;
  Section 3.1 covers NoisyNet-DQN; Section 3.2 covers initialization.
- M. Plappert et al. (2018). *Parameter space noise for exploration.* ICLR.
  [arXiv:1706.01905](https://arxiv.org/abs/1706.01905). The same idea, developed independently,
  with a fixed rather than learned noise scale that is adapted to a target amount of
  exploration.
