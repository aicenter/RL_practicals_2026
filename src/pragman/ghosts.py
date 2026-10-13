"""Ghost personalities, and the two rules the environment applies to all of them.

A ghost policy returns a **distribution** over the four actions; it never
samples. That single decision is what makes Prag-man a genuinely *known* MDP:
:func:`pragman.dynamics.successors` is then just a product over the slip
distribution and the ghosts' distributions, and a learned ghost only has to
return a softmax over four numbers.

Two rules are applied by the environment rather than by each policy, so that
nobody has to reimplement them:

* mass a policy puts on a move into a wall is legal, and resolves to staying
  put (:func:`resolve_illegal`);
* ``cfg.ghost_epsilon`` is mixed in afterwards as
  ``(1 - eps) * policy + eps * uniform(legal moves)`` (:func:`mix_epsilon`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from .errors import ConfigError
from .state import ACTIONS, Action, Cell, PragmanState

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .config import PragmanConfig
    from .layout import Layout

__all__ = [
    "GhostPolicy",
    "RandomGhost",
    "ChaserGhost",
    "ProximityChaser",
    "uniform",
    "mix_epsilon",
    "resolve_illegal",
    "check_distribution",
    "ghost_policy_by_name",
    "policy_to_json",
    "policy_from_json",
    "register_ghost_policy",
    "BUILTIN_GHOST_POLICIES",
    "MIX_ORDER",
]

#: Distributions are allowed to be this far from summing to 1 before we complain.
PROBABILITY_TOLERANCE = 1e-9


@runtime_checkable
class GhostPolicy(Protocol):
    """What every ghost personality must provide.

    Implementations should be **frozen dataclasses with a ``name``**: the name
    is what a config compares, hashes, and serializes by, and freezing is what
    keeps :class:`~pragman.config.PragmanConfig` hashable.

    A policy must be a function of the state alone -- there is no step counter
    to consult, because the episode clock is not part of the state.
    """

    name: str

    def distribution(
        self, cfg: PragmanConfig, state: PragmanState, ghost_index: int
    ) -> dict[Action, float]:
        """The ghost's action distribution, before epsilon-mixing.

        Args:
            cfg: the configuration (mostly for ``cfg.layout``).
            state: the world *after* the agent has moved, *before* any ghost
                moves. All ghosts see the same state, because they move
                simultaneously.
            ghost_index: which ghost this is; ``state.ghosts[ghost_index]`` is
                its cell and ``cfg.ghost_policies[ghost_index]`` is this policy.

        Returns:
            A dictionary over :class:`~pragman.state.Action` summing to 1. Mass
            on a move into a wall is allowed and resolves to staying put.
        """
        ...


@dataclass(frozen=True)
class RandomGhost:
    """Uniform over the moves that actually go somewhere.

    What "random ghost" means to everyone, and the default: it wanders, so a
    student can watch a policy learn to avoid it without also having to solve a
    pursuit problem.
    """

    name: ClassVar[str] = "random"

    def distribution(
        self, cfg: PragmanConfig, state: PragmanState, ghost_index: int
    ) -> dict[Action, float]:
        """Uniform over :meth:`~pragman.layout.Layout.legal_actions`."""
        return uniform(cfg.layout.legal_actions(state.ghosts[ghost_index]))


@dataclass(frozen=True)
class ChaserGhost:
    """Takes a step that minimizes the BFS distance to the agent.

    The distances come from the layout's cached breadth-first tables, so this
    is a dictionary lookup rather than a search, and it stays a pure function
    of the state. With ``ghost_epsilon = 0`` a chaser is inescapable on an open
    grid once it is adjacent, which is why the default epsilon is 0.1.

    Attributes:
        tie_break: what to do when two moves are equally close. ``"uniform"``
            (the default) splits the probability between them, which is the
            natural reading of "it chases you". ``"first"`` takes the lowest
            action index instead, which makes the ghost a *deterministic*
            function of the state -- the whole environment then has no
            randomness left once ``slip_prob`` and ``ghost_epsilon`` are zero,
            which is what keeps Monte Carlo tree search off chance nodes in the
            planning lab. Ties are common (roughly two moves in five on an open
            maze), so this is not a detail: with ``"uniform"`` a "deterministic"
            planning configuration still branches.
    """

    tie_break: str = "uniform"
    name: ClassVar[str] = "chaser"

    def __post_init__(self) -> None:
        if self.tie_break not in ("uniform", "first"):
            raise ConfigError(
                f"ChaserGhost tie_break must be 'uniform' or 'first', got {self.tie_break!r}"
            )

    def distribution(
        self, cfg: PragmanConfig, state: PragmanState, ghost_index: int
    ) -> dict[Action, float]:
        """Over the legal moves that come closest to the agent."""
        layout = cfg.layout
        cell = state.ghosts[ghost_index]
        legal = layout.legal_actions(cell)
        if not legal:
            return uniform(ACTIONS)
        distances = layout.distances(state.agent)
        scores = {a: distances.get(layout.move(cell, a), math.inf) for a in legal}
        best = min(scores.values())
        if best == math.inf:
            # The agent is walled off from this ghost; nothing to chase.
            return uniform(legal)
        closest = [a for a in legal if scores[a] == best]
        if self.tie_break == "first":
            return {min(closest): 1.0}
        return uniform(closest)


@dataclass(frozen=True)
class ProximityChaser:
    """A :class:`ChaserGhost` when close to the agent, a :class:`RandomGhost` otherwise.

    Behaviourally distinct from both, still Markov in the state, and it gives a
    nice partial-observability story for Week 13: the ghost that only wakes up
    when you come near.

    Attributes:
        wake_distance: it chases when the BFS distance to the agent is at most
            this many steps.
    """

    wake_distance: int = 4
    name: ClassVar[str] = "proximity"

    def distribution(
        self, cfg: PragmanConfig, state: PragmanState, ghost_index: int
    ) -> dict[Action, float]:
        """Chase if ``distance <= wake_distance``, otherwise wander."""
        distances = cfg.layout.distances(state.agent)
        distance = distances.get(state.ghosts[ghost_index], math.inf)
        if distance <= self.wake_distance:
            return _CHASER.distribution(cfg, state, ghost_index)
        return _RANDOM.distribution(cfg, state, ghost_index)


# ──────────────────────────────────────────────────────────────────────────────
# The rules the environment applies to every policy
# ──────────────────────────────────────────────────────────────────────────────


def uniform(actions: Iterable[Action]) -> dict[Action, float]:
    """A uniform distribution over *actions*, in :class:`Action` order.

    A ghost sealed into a single cell has no legal move at all; rather than
    return an empty distribution we spread the mass over all four actions,
    every one of which resolves to staying put.
    """
    actions = sorted(set(actions))
    if not actions:
        actions = list(ACTIONS)
    probability = 1.0 / len(actions)
    return {action: probability for action in actions}


def check_distribution(dist: Mapping[Action, float], who: object) -> None:
    """Complain, naming *who*, unless *dist* is a probability distribution.

    Checked on every call: it is four floats, and a learned ghost that quietly
    stops summing to 1 is an unpleasant bug to find by other means.
    """
    name = getattr(who, "name", None) or type(who).__name__
    total = 0.0
    for action, probability in dist.items():
        try:
            action = Action(action)
        except ValueError:
            raise ConfigError(
                f"ghost policy {name!r} put mass on {action!r}, "
                "which is not one of the four actions"
            ) from None
        if probability < 0.0:
            raise ConfigError(
                f"ghost policy {name!r} returned a negative probability "
                f"{probability!r} for action {action.name}"
            )
        total += probability
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ConfigError(
            f"ghost policy {name!r} returned a distribution summing to {total!r}, not 1"
        )


def mix_epsilon(
    dist: Mapping[Action, float], legal: Iterable[Action], epsilon: float
) -> dict[Action, float]:
    """Mix a policy's distribution with uniform noise: ``(1-eps)*dist + eps*uniform(legal)``.

    The environment does this so that every policy, built in or written by a
    student, is shakeable off. Actions left with exactly zero probability are
    dropped, which keeps the enumeration in
    :func:`pragman.dynamics.successors` small.
    """
    if epsilon == 0.0:
        return {a: p for a, p in sorted(dist.items()) if p > 0.0}
    noise = uniform(legal)
    mixed = {
        action: (1.0 - epsilon) * dist.get(action, 0.0) + epsilon * noise.get(action, 0.0)
        for action in ACTIONS
    }
    return {a: p for a, p in mixed.items() if p > 0.0}


def resolve_illegal(
    dist: Mapping[Action, float], layout: Layout, cell: Cell
) -> dict[Cell, float]:
    """Turn a distribution over *actions* into one over *destination cells*.

    This is where "mass on a move into a wall resolves to staying put" happens:
    two actions that both bump into a wall land on the same cell and their
    probabilities are merged. Working in cell space keeps
    :func:`pragman.dynamics.successors` from enumerating outcomes it would only
    have to merge again afterwards.
    """
    destinations: dict[Cell, float] = {}
    for action, probability in dist.items():
        target = layout.move(cell, action)
        destinations[target] = destinations.get(target, 0.0) + probability
    return destinations


# ──────────────────────────────────────────────────────────────────────────────
# The registry: policies by name
# ──────────────────────────────────────────────────────────────────────────────

_RANDOM = RandomGhost()
_CHASER = ChaserGhost()

#: The personalities you can name in ``cfg.ghost_policy``.
BUILTIN_GHOST_POLICIES: dict[str, GhostPolicy] = {
    "random": _RANDOM,
    "chaser": _CHASER,
    "proximity": ProximityChaser(),
}

#: The order ``ghost_policy="mix"`` cycles through, in row-major spawn order.
MIX_ORDER: tuple[str, ...] = ("random", "chaser", "proximity")

_REGISTRY: dict[str, GhostPolicy] = dict(BUILTIN_GHOST_POLICIES)


def register_ghost_policy(policy: GhostPolicy, name: str | None = None) -> None:
    """Make a custom policy findable by name.

    Only needed for serialization: an :class:`~pragman.state.Episode` stores its
    ghosts by name, so a dataset recorded with your own ghost can only be
    loaded again in a process where that ghost has been registered.
    """
    _REGISTRY[name or policy.name] = policy


def ghost_policy_by_name(name: str) -> GhostPolicy:
    """Look up a policy by name.

    Raises:
        ConfigError: naming the unknown policy and listing the known ones,
            rather than quietly substituting a random ghost.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ConfigError(
            f"unknown ghost policy {name!r}; known policies are {known} "
            "(register a custom one with pragman.ghosts.register_ghost_policy)"
        ) from None


def policy_to_json(policy: GhostPolicy) -> dict[str, Any]:
    """Serialize a policy as its name *and* its settings.

    Storing only the name would quietly turn a ``ProximityChaser(wake_distance=6)``
    back into the default on load, changing the dynamics an offline dataset came
    from without saying so. Policies are frozen dataclasses, so their settings
    are exactly their fields.
    """
    params = {f.name: getattr(policy, f.name) for f in dc_fields(policy)}
    return {"name": policy.name, "params": params}


def policy_from_json(data: dict[str, Any] | str) -> GhostPolicy:
    """Inverse of :func:`policy_to_json`; a bare name is accepted too.

    Raises:
        ConfigError: naming the policy, if it is not registered in this process
            or if its stored settings no longer fit its fields.
    """
    if isinstance(data, str):
        return ghost_policy_by_name(data)
    prototype = ghost_policy_by_name(data["name"])
    params = data.get("params") or {}
    try:
        return type(prototype)(**params)
    except TypeError as exc:
        raise ConfigError(
            f"stored settings {params!r} do not fit ghost policy {data['name']!r}: {exc}"
        ) from None
