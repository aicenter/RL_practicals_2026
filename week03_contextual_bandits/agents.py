"""Contextual bandit agents -- the file you will edit this week.

Every agent follows the same interface, used by cb_tools.py:

    agent = LinUCB(beta=0.3)                    # hyperparameters go to __init__
    agent.reset(n_actions, dim, horizon, rng)   # called at the start of every run
    a = agent.act(x)                            # choose A_t for the context x_t
    agent.update(x, a, reward)                  # learn from R_t
    actions = agent.predict(X)                  # greedy actions for a whole array of
                                                # contexts, without exploring; used to
                                                # measure the test reward after a run

Contexts: every agent has an attribute `features`, and the tools give it contexts
of that kind (see the Week 3 README):
    "pca"     51 numbers: 50 principal components of the image and a constant 1;
    "pixels"  784 numbers: the pixel intensities in [0, 1].

Notation follows the Week 3 lecture: x is the context, A_t the action, R_t the
reward; theta_a, V_a, b_a and lambda are the quantities of ridge regression
(LinUCB section), pi(a | x) the policy (policy-gradient section).

Rules of the game:
- Use `self.rng` for all randomness (torch agents: their network is initialized
  from a seed drawn from `self.rng`), so that runs are reproducible.
- A bandit agent sees only the contexts, its own actions and their rewards. The
  true label is passed to `update` only for agents with `full_information = True`
  -- that is, supervised learners such as SupervisedNet, which are references,
  not bandit algorithms.
- `np.argmax` breaks ties by always picking the *lowest* index -- use
  `argmax_random` below when ties are possible (e.g. at the start).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest entry of `values`, ties broken uniformly at random."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def softmax(z: np.ndarray) -> np.ndarray:
    """pi(a) = exp(z_a) / sum_b exp(z_b), computed without overflow (in float64)."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


class CBAgent:
    """Base class: what all agents share."""

    features = "pca"                 # which contexts the agent gets: "pca" or "pixels"
    full_information = False         # True only for supervised references (they see labels)

    def reset(self, n_actions: int, dim: int, horizon: int, rng: np.random.Generator) -> None:
        """Forget everything and get ready for a new run of `horizon` steps."""
        self.n_actions = n_actions
        self.dim = dim               # length of a context vector x
        self.horizon = horizon
        self.rng = rng
        self.t = 0                   # number of steps taken so far

    def act(self, x: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, x: np.ndarray, action: int, reward: float, label: int | None = None) -> None:
        self.t += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        skip = {"n_actions", "dim", "horizon", "rng", "t"}
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                           if k not in skip and not k.startswith("_")
                           and isinstance(v, (int, float, str, bool, type(None))))
        return f"{type(self).__name__}({params})"


class RandomAgent(CBAgent):
    """Guesses uniformly at random. Any algorithm should beat this."""

    def act(self, x):
        return int(self.rng.integers(self.n_actions))

    def predict(self, X):
        return self.rng.integers(self.n_actions, size=len(X))


# -----------------------------------------------------------------------------
# Example 1: greedy ridge regression. A complete bandit algorithm -- but it
# never explores on purpose. When does that hurt?
# -----------------------------------------------------------------------------

class GreedyRidge(CBAgent):
    """One linear model of the reward per action, fitted by ridge regression; act greedily.

    For every action a we keep (Week 3 lecture, LinUCB):
        V_a     = lambda I + sum_{s: A_s = a} x_s x_s^T    -- we store its inverse, self.Vinv[a]
        b_a     = sum_{s: A_s = a} R_s x_s                -- self.b[a]
        theta_a = V_a^{-1} b_a                            -- the ridge estimate, self.theta[a]
    so that x^T theta_a estimates the expected reward r(x, a). After each step only
    the model of the action taken changes; the Sherman-Morrison formula updates
    V_a^{-1} in O(d^2) operations instead of inverting V_a again.
    """

    features = "pca"

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def reset(self, n_actions, dim, horizon, rng):
        super().reset(n_actions, dim, horizon, rng)
        self.Vinv = np.tile(np.eye(dim) / self.lam, (n_actions, 1, 1))   # V_a^{-1}, shape (K, d, d)
        self.b = np.zeros((n_actions, dim))                                # b_a, shape (K, d)
        self.theta = np.zeros((n_actions, dim))                            # theta_a, shape (K, d)

    def act(self, x):
        return argmax_random(self.theta @ x, self.rng)      # x^T theta_a for all actions a

    def update(self, x, action, reward, label=None):
        super().update(x, action, reward)
        Vx = self.Vinv[action] @ x                          # Sherman-Morrison:
        self.Vinv[action] -= np.outer(Vx, Vx) / (1.0 + x @ Vx)   # (V + x x^T)^{-1}
        self.b[action] += reward * x
        self.theta[action] = self.Vinv[action] @ self.b[action]

    def predict(self, X):
        return np.argmax(X @ self.theta.T, axis=1)


# -----------------------------------------------------------------------------
# Example 2: a supervised classifier. NOT a bandit algorithm: it is told the
# right answer after every step. It is here as a reference ("how well could we
# do with full information?") and as the starting point for REINFORCE.
# -----------------------------------------------------------------------------

class SupervisedNet(CBAgent):
    """A neural-network classifier trained online with the true labels.

    Network: x -> Linear -> ReLU -> Linear -> one logit H(x, a) per action
    (with hidden=0: x -> Linear, i.e. multinomial logistic regression). The softmax
    of the logits is a policy pi(a | x), as in the lecture's policy-gradient section.
    Learning: every `batch_size` steps, one Adam step on the mean cross-entropy
    loss -log pi(label | x) of the last `batch_size` examples.
    """

    full_information = True

    def __init__(self, lr: float = 1e-3, hidden: int = 128, batch_size: int = 32,
                 features: str = "pixels"):
        self.lr, self.hidden, self.batch_size, self.features = lr, hidden, batch_size, features

    def reset(self, n_actions, dim, horizon, rng):
        super().reset(n_actions, dim, horizon, rng)
        torch.manual_seed(int(rng.integers(2**31)))
        if self.hidden:
            self.net = nn.Sequential(nn.Linear(dim, self.hidden), nn.ReLU(),
                                     nn.Linear(self.hidden, n_actions))
        else:
            self.net = nn.Linear(dim, n_actions)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.batch = []              # examples since the last gradient step

    def logits(self, x: np.ndarray) -> np.ndarray:
        """The logits H(x, a) of one context x, as a numpy array (no gradient)."""
        with torch.no_grad():
            return self.net(torch.as_tensor(x)).numpy()

    def act(self, x):
        return int(np.argmax(self.logits(x)))     # it is told the labels: no need to explore

    def update(self, x, action, reward, label=None):
        super().update(x, action, reward)
        self.batch.append((x, label))
        if len(self.batch) == self.batch_size:
            X, labels = zip(*self.batch)
            self.batch = []
            self.learn(np.array(X), np.array(labels), np.ones(len(labels)))

    def learn(self, X: np.ndarray, targets: np.ndarray, weights: np.ndarray,
              entropy_bonus: float = 0.0) -> None:
        """One optimizer step on the weighted cross-entropy loss

            mean_i  weights_i * ( -log pi(targets_i | X_i) )  -  entropy_bonus * mean_i H(pi(. | X_i)),

        where H(pi) = -sum_a pi(a) log pi(a) is the entropy of the policy (subtracting it
        from the loss rewards policies that stay random; see the lecture).
        """
        logits = self.net(torch.as_tensor(X))
        ce = F.cross_entropy(logits, torch.as_tensor(targets, dtype=torch.long), reduction="none")
        loss = (torch.as_tensor(weights, dtype=torch.float32) * ce).mean()
        if entropy_bonus:
            logp = F.log_softmax(logits, dim=1)
            loss = loss + entropy_bonus * (logp.exp() * logp).sum(dim=1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.as_tensor(X)).argmax(dim=1).numpy()


# -----------------------------------------------------------------------------
# Your algorithms. Complete the two your group was assigned (see README.md).
# -----------------------------------------------------------------------------

class LinUCB(GreedyRidge):
    """Optimism (LinUCB): choose

        A_t = argmax_a  x^T theta_a + beta * ||x||_{V_a^{-1}},    ||x||_{V^{-1}} = sqrt(x^T V^{-1} x).

    The models are fitted exactly as in GreedyRidge (inherited); only `act` changes.
    """

    def __init__(self, beta: float = 1.0, lam: float = 1.0):
        super().__init__(lam)
        self.beta = beta

    def act(self, x):
        raise NotImplementedError


class LinearTS(GreedyRidge):
    """Thompson sampling for linear bandits: for every action draw

        theta~_a ~ N(theta_a, sigma^2 V_a^{-1})      (the posterior of theta_a)

    and choose A_t = argmax_a x^T theta~_a. The posterior mean theta_a and V_a^{-1}
    are fitted exactly as in GreedyRidge (inherited); only `act` changes.
    """

    def __init__(self, sigma: float = 0.3, lam: float = 1.0):
        super().__init__(lam)
        self.sigma = sigma

    def act(self, x):
        raise NotImplementedError


class Reinforce(SupervisedNet):
    """REINFORCE with a softmax policy (Week 3 lecture, policy gradient for contextual bandits).

    Act: sample A_t ~ pi(. | x_t), the softmax of the network's logits.
    Learn: every `batch_size` steps, one optimizer step on the loss

        mean_i  ( -(R_i - b) log pi(A_i | x_i) ),

    whose gradient is minus the policy gradient. This is SupervisedNet's loss with the
    *sampled action* A_i in place of the label, weighted by the advantage R_i - b:
    you can reuse `self.learn`. For the baseline b, start with the average reward so far.
    Pass `entropy_bonus=self.entropy_bonus` to `self.learn` (it does nothing while it is 0).
    The network, optimizer and `predict` are inherited from SupervisedNet.
    """

    full_information = False

    def __init__(self, lr: float = 1e-3, hidden: int = 128, batch_size: int = 32,
                 features: str = "pixels", entropy_bonus: float = 0.0):
        super().__init__(lr, hidden, batch_size, features)
        self.entropy_bonus = entropy_bonus

    def act(self, x):
        raise NotImplementedError

    def update(self, x, action, reward, label=None):
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Final choice (Task 5): one setting per algorithm, used on BOTH problems.
# Fill in the algorithms you implemented; leave the others out.
# -----------------------------------------------------------------------------

def final_agents() -> dict[str, CBAgent]:
    return {
        # "LinUCB": LinUCB(beta=...),
        # "Reinforce": Reinforce(lr=..., hidden=..., features=...),
    }
