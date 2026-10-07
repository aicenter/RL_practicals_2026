# Week 3: Contextual bandits

Goal: learn to recognize handwritten digits **from rewards alone**, and compare
the two families of contextual bandit algorithms from the lecture -- optimism
and posterior sampling on a model of the reward (LinUCB, linear Thompson
sampling), and learning the policy directly (REINFORCE) -- with a supervised
learner that is simply told the right answer.

## The problem

At every step the agent sees an image of a handwritten digit from MNIST (the
context x_t), names a digit (the action A_t), and gets the reward
R_t = 1 if it was right and 0 otherwise. It is **never told the right answer**.

This is classification seen as a bandit. A supervised learner gets the label,
which tells it the reward of *every* action at once ("full information"); a
bandit gets the reward of the one action it took. So every classification
problem can be posed as a contextual bandit -- the bandit framework is the more
general one -- but posing it that way throws information away. How much does
that cost? That is one of the questions for today.

We use two problems (`rlcourse/Digits1-v0` and `rlcourse/Digits2-v0`):

| Problem | Actions | Reward |
| --- | --- | --- |
| 1: digits | 10: the digits 0-9 | 1 if right, 0 if wrong |
| 2: digits or pass | 11: the digits, and 10 = "pass" | as in problem 1; passing always gives 0.5 |

One run is **T = 60 000 steps: one pass over the MNIST training images**, in a
random order (the seed). We measure two things:

- **Regret** L_T = Σ_t (1 − R_t): the reward lost while learning, against an
  oracle that knows every label. On problem 1 it is just the number of wrong
  guesses; a pass costs 0.5. This is what a bandit algorithm is judged on.
- **Test reward**: after the run, the agent's greedy policy (no exploration) is
  applied to the 10 000 MNIST test images it has never seen. On problem 1 it is
  the test accuracy -- what a supervised learner is judged on.

**Contexts.** Agents choose the form in which they see the image (their
`features` attribute): `"pixels"`, the 784 pixel intensities in [0, 1], or
`"pca"`, 51 numbers: the first 50 principal components of the images and a
constant 1. The ridge-regression agents keep a d × d matrix per action, so with
d = 784 instead of 51 they would be about 200× slower. (The principal components
were computed from the training images alone, without their labels, so they
reveal nothing about the labels.)

## What's in this directory

| File | What it is |
| --- | --- |
| `agents.py` | **The file you edit.** Two working examples (`GreedyRidge`, `SupervisedNet`) and three empty algorithms. |
| `cb_tools.py` | Running, evaluating, tuning and plotting agents. No need to edit it. |
| `play.py` | Be the bandit: learn to read digits from rewards yourself. |

The environment is in `src/rlcourse/envs/digits.py`, the data loader in
`src/rlcourse/mnist.py`.

## Groups

We work in five groups of 3–4. Every group implements one algorithm of each
family, on both problems; e.g. one pair of you per algorithm.

| Group | Algorithms |
| --- | --- |
| A1, A2, A3 | LinUCB, REINFORCE |
| B1, B2 | linear Thompson sampling, REINFORCE |

## Tasks

0. **Setup** *(5 min, ideally before class)*. From the repository root:
   `git pull`, `uv sync`, then

   ```bash
   uv run python -m rlcourse.mnist      # downloads MNIST (11 MB) into data/, once
   uv run python check_setup.py
   ```

1. **Be the bandit** *(10 min)*. Each of you plays at least two games of 50
   images:

   ```bash
   uv run python week03_contextual_bandits/play.py --name yournickname
   ```

   You press one of ten buttons, A–J. Each button stands for one digit, but
   which is a secret -- a new one every game. Your games go to the class
   leaderboard (the first time, you are asked for your group's password for this
   week). Then compare yourself with the algorithms *on exactly the images you
   saw*:

   ```bash
   uv run python week03_contextual_bandits/play.py --summary
   ```

   Discuss: why do you need so few mistakes, and the algorithms so many? What do
   you know before the first image that they don't? (The summary also shows an
   "ideal player"; see `ideal_mistakes` in `play.py`.)

2. **Implement your two algorithms** *(25 min)* in `agents.py`. Read the two
   examples first: `LinUCB` and `LinearTS` inherit everything from
   `GreedyRidge` except `act`; `Reinforce` inherits the network, the optimizer
   and `predict` from `SupervisedNet`. Check your implementation with

   ```bash
   uv run python week03_contextual_bandits/cb_tools.py
   ```

   Hints:
   - LinUCB needs ||x||_{V_a^{-1}} = sqrt(x^T V_a^{-1} x) for every action:
     `(self.Vinv @ x) @ x` computes x^T V_a^{-1} x for all a at once.
   - Linear Thompson sampling: drawing each θ̃_a with
     `self.rng.multivariate_normal` works, but factorizes a matrix for every
     action at every step. The agent only ever uses x^T θ̃_a. What is the
     distribution of this number? Sampling it is as cheap as LinUCB.
   - REINFORCE, `act`: sample from the softmax of `self.logits(x)` (there is a
     `softmax` function at the top of the file), using `self.rng`, e.g.
     `self.rng.choice(self.n_actions, p=p)`.
   - REINFORCE, `update`: collect (x, A, R − b) in `self.batch` and every
     `self.batch_size` steps call `self.learn(...)`. Compare this with
     `SupervisedNet.update`: what plays the role of the label? Don't call
     `SupervisedNet.update` (there is no label: it is `None`); but do increase
     `self.t`, e.g. with `CBAgent.update(self, x, action, reward)`.
   - Use `self.rng` for all randomness, and `argmax_random` rather than
     `np.argmax` where ties are possible.

3. **Experiment and tune** *(30 min)* on both problems. Write your experiments
   in a script of your own in this directory, e.g.

   ```python
   # week03_contextual_bandits/tune_A1.py  --  run with: uv run python week03_contextual_bandits/tune_A1.py
   import matplotlib.pyplot as plt
   from cb_tools import *
   from agents import *

   tuned = tune(lambda beta: LinUCB(beta), [0.1, 0.3, 1, 3])
   plot_tuning(tuned, xlabel="beta")
   res = compare({"greedy": GreedyRidge(), "LinUCB": LinUCB(1.0), "supervised": SupervisedNet()},
                 problems=[1])
   plot_learning_curve(list(res[1].values()))
   plt.show()
   ```

   One run takes 3–5 s for the ridge agents and 10–15 s for the networks; `tune`
   and `compare` do two runs (seeds) per setting and problem. For a first coarse
   sweep, pass `horizon=20_000`, then check the winner at the full horizon.
   Things to look at:
   - **The price of bandit feedback.** Compare like with like. `Reinforce` and
     `SupervisedNet` use the same network and optimizer; the only difference is
     what they learn from. How many more mistakes does REINFORCE make, and when
     -- early, or all the way through (`plot_learning_curve`)? Relate this to
     the lecture: the expected policy gradient is the supervised gradient
     multiplied by π(y | x). For LinUCB and Thompson sampling, the supervised
     counterpart with the same features is a linear classifier on "pca":
     `SupervisedNet(hidden=0, features="pca", lr=0.03)`. Is the price the same
     for both families?
   - **Does exploration matter?** Compare `GreedyRidge` (no exploration at all)
     with your LinUCB or Thompson sampling, on problem 1 and on problem 2.
   - **Problem 2.** Watch how often the agents pass while learning (`passes` in
     the printout). Is the best setting the same for both problems? You will
     have to choose one setting for both.
   - **Regret vs test reward.** Which agent has the lowest regret, and which the
     best test reward? When would you care about which?
   - **REINFORCE.** If its policy collapses (e.g. it ends up passing on almost
     every image), look at the lecture's remedy: `Reinforce(entropy_bonus=...)`
     adds it to the loss. Also try the linear softmax policy of the lecture,
     `Reinforce(hidden=0, features="pca", lr=...)` (it needs a larger learning
     rate).
   - To compare two agents, use `paired_difference(evaluate(a, p), evaluate(b, p))`:
     with the same seeds, both see the same images in the same order.

4. **Final choice and sharing** *(15 min)*.
   - In `final_agents()` at the bottom of `agents.py`, fill in **one** setting
     per algorithm, used on **both** problems. Before running anything, write
     down how well you expect each of your agents to do.
   - Then run the final evaluation, **once**:

     ```bash
     uv run python week03_contextual_bandits/cb_tools.py --final
     ```

     It runs your agents and two references on new seeds (a few minutes) and
     submits the results and your `agents.py` to the class leaderboard. The
     score is your regret relative to `SupervisedNet`, which is told the label
     after every step (geometric mean over the two problems). Part of that gap
     is bandit feedback, part is the model: which part, for your agents?
   - Each group has 2 minutes: your settings, the price of bandit feedback you
     measured (like with like), and one thing that surprised you.

## If you have time left (or at home)

- **Is full information always better?** Write a `GreedyRidge` that is told the
  label (`full_information = True`) and so updates the models of *all* actions
  after every step, with the reward each would have got. Compare its regret and
  test reward with plain `GreedyRidge`. Can you explain the result?
- **A learned baseline.** Replace REINFORCE's average reward by a baseline b(x)
  that depends on the context (a small network trained on (R − b(x))², a
  *critic*). Does it help here? Why might it, or might it not?
- **Neural-Linear** (lecture): use the hidden layer of a network as the
  features of linear Thompson sampling. Where do the features come from, if you
  are not allowed to use the labels?
- **When should you pass?** With the true probabilities of each digit, the best
  policy would pass exactly when it is less than 50 % sure of its best guess.
  How often do your trained agents pass on the test images, and is it on the
  right images (`predict` and the test labels, from `test_data`)?
- **Rewards from a judge.** In practice, rewards often come from a model, not
  from the truth (e.g. a reward model trained on human ratings -- Week 11).
  Train a weak classifier on a few hundred labelled images, and give the agent
  reward 1 when its answer agrees with the classifier. How does the agent's
  *true* test accuracy compare with the reward it collects?
