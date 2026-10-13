"""The white-box model: the MDP written out as matrices.

Prag-man is a *known* MDP. The environment samples from its dynamics, but
:func:`pragman.dynamics.successors` also hands you the exact one-step
distribution, so the whole Markov decision process can be written out as the
three objects a textbook uses::

    P[s, a, s']   the transition kernel,   p(s' | s, a)
    R[s, a]       the expected reward,     R(s, a)
    rho0[s]       the initial distribution, rho_0(s)

That is what :class:`TabularModel` holds, and it is the object Week 4 builds
value iteration on top of: you write your own sweeps against ``.P``, ``.R``
and ``.rho0``.

**The episode clock is not part of the MDP,** so ``cfg.max_steps`` is simply
ignored here. A time limit is a property
of the *experiment*, not of the decision process: if the clock were in the
state, the same board at t = 0 and t = 1 would be two different states,
breadth-first enumeration would never merge them, and the tables below would be
``max_steps`` times larger for no gain. The finite-horizon MDP really does have
a time-augmented state space -- that is worth saying in the lecture, and it is
not worth paying for in the API.

**Dynamic programming does not scale, and this module demonstrates it** rather
than merely asserting it: enumeration stops with
:class:`~pragman.errors.StateSpaceTooLarge` above ``max_states``. Adding
pellets multiplies the state count by 2 per pellet, so ``pellets_8x8`` is out of
reach while ``haunted_5x5`` is comfortable. That is the whole argument for
function approximation, in one traceback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import PragmanConfig
from .dynamics import successors
from .errors import StateSpaceTooLarge
from .state import ACTIONS, Cell, PragmanState

__all__ = [
    "enumerate_states",
    "TabularModel",
    "build_tabular_model",
]

#: The default cap on the enumerated state space. At this size a dense ``P`` is
#: ``1500**2 * 4 * 8`` bytes, about 72 MB, which a laptop tolerates -- which is
#: why no sparse representation (and hence no scipy) is introduced anywhere.
DEFAULT_MAX_STATES = 1500


# ──────────────────────────────────────────────────────────────────────────────
# Enumerating the reachable states
# ──────────────────────────────────────────────────────────────────────────────


def enumerate_states(
    cfg: PragmanConfig, max_states: int = DEFAULT_MAX_STATES
) -> tuple[list[PragmanState], dict[PragmanState, int]]:
    """Every state reachable from the reset distribution, breadth-first.

    The search starts from the states ``reset()`` can produce -- one per cell in
    ``cfg.start_cells()``, with the ghosts on their spawn cells and a pellet on
    the agent's own start cell removed, exactly as the environment clears it --
    and follows :func:`pragman.dynamics.successors` for all four actions until
    nothing new turns up. Only outcomes with non-zero probability are followed,
    so an unreachable pocket of the maze does not appear in the table.

    The order is deterministic: start cells in the layout's row-major order,
    then successors in the order ``successors`` returns them. Re-running the
    enumeration therefore gives the same ``index`` mapping, which is what makes
    the ``"index"`` observation encoding reproducible across sessions and lets a
    student's printed value table be compared with a neighbour's.

    ``cfg.max_steps`` is ignored by construction: the episode clock is not part
    of the state (see the module docstring), so there is no ``ignore_time_limit``
    flag to remember.

    Args:
        cfg: the world to enumerate.
        max_states: give up above this many states. Enumeration stops as soon
            as the cap is exceeded, rather than grinding through the 2**40
            states of a pellet-filled maze.

    Returns:
        ``(states, index)`` -- the list of states, and the inverse mapping
        ``index[state] == states.index(state)``.

    Raises:
        StateSpaceTooLarge: if more than ``max_states`` states are reachable.
            Its message points at :func:`pragman.dynamics.successors`, which
            gives the exact one-step model for a *single* state -- all that
            asynchronous dynamic programming, Dyna-Q, and MCTS ever need.
    """
    states: list[PragmanState] = []
    index: dict[PragmanState, int] = {}
    frontier: deque[PragmanState] = deque()

    def discover(state: PragmanState) -> None:
        if state in index:
            return
        index[state] = len(states)
        states.append(state)
        frontier.append(state)
        if len(states) > max_states:
            raise StateSpaceTooLarge(count=len(states), max_states=max_states)

    for cell in cfg.start_cells():
        discover(initial_state(cfg, cell))

    while frontier:
        state = frontier.popleft()
        for action in ACTIONS:
            for probability, next_state, _reward, _terminated in successors(cfg, state, action):
                if probability > 0.0:
                    discover(next_state)

    return states, index


def initial_state(cfg: PragmanConfig, agent_cell: Cell) -> PragmanState:
    """The state ``reset()`` produces when the agent is drawn onto ``agent_cell``.

    The one subtlety is the pellet on the agent's own cell, which is cleared
    here as it is at reset: pellets are otherwise eaten by *moving* onto a cell,
    so an agent spawning on one would leave it uneaten underneath itself.
    """
    return PragmanState(
        agent=agent_cell,
        ghosts=cfg.initial_ghosts(),
        pellets=cfg.initial_pellets() - {agent_cell},
        terminated=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# The model itself
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class TabularModel:
    """The MDP as dense arrays: ``P``, ``R``, ``rho0``.

    Attributes:
        states: every reachable state, in enumeration order.
        index: the inverse mapping, ``index[states[i]] == i``.
        P: shape ``(S, A, S)``. ``P[s, a, s']`` is ``p(s' | s, a)``; each row
            ``P[s, a]`` sums to 1.
        R: shape ``(S, A)``. ``R[s, a]`` is the *expected* immediate reward.
        rho0: shape ``(S,)``. The initial distribution: uniform over the states
            ``reset()`` can produce, zero everywhere else. Value iteration never
            looks at it -- ``V*`` and the optimal policy do not depend on where
            you start -- so it appears only when a lab reports the expected
            return from the start, and in Week 8's visitation measure.
        terminal: shape ``(S,)``, ``True`` for terminal states. These are
            **absorbing with zero reward** (``P[s, a, s] = 1``, ``R[s, a] = 0``),
            so value iteration is a plain contraction with no special cases and
            ``V[s] = 0`` falls out of the recursion rather than being patched in.

    **What the expectation in ``R`` hides.** ``R[s, a]`` averages over the two
    sources of randomness in a step, the agent's slip and the ghosts' moves.
    This matches the ``R(s_t, a_t)`` of the lecture notes, but it is worth
    knowing that Prag-man's rewards are natively ``R(s, a, s')``: on
    ``haunted_5x5`` the death reward depends on where the *ghost* moved, not on
    what the agent did, and the expectation is where that dependence goes. If
    the notes ever switch convention the change here is additive -- an
    ``R_sas`` array of shape ``(S, A, S)``, with ``R`` kept as its expectation
    under ``P`` -- so nothing written against ``R`` would break.

    **Memory.** ``P`` is dense, which at the 1500-state cap means
    ``1500**2 * 4 * 8`` bytes, about 72 MB. That is deliberate: a dense array is
    what a student can print, slice, and check by hand, and it is the reason
    Prag-man needs no sparse matrices and no scipy.
    """

    states: list[PragmanState]
    index: dict[PragmanState, int]
    P: np.ndarray
    R: np.ndarray
    rho0: np.ndarray
    terminal: np.ndarray

    @property
    def n_states(self) -> int:
        """``S``, the number of enumerated states."""
        return len(self.states)

    @property
    def n_actions(self) -> int:
        """``A``, which is always 4 -- ``UP``, ``RIGHT``, ``DOWN``, ``LEFT``."""
        return int(self.P.shape[1])

    def __repr__(self) -> str:  # the arrays themselves are far too big for a repr
        return f"TabularModel(n_states={self.n_states}, n_actions={self.n_actions})"


def build_tabular_model(
    cfg: PragmanConfig, max_states: int = DEFAULT_MAX_STATES
) -> TabularModel:
    """Enumerate the reachable states and fill in ``P``, ``R`` and ``rho0``.

    Every entry comes from :func:`pragman.dynamics.successors`, so the model is
    exact rather than estimated -- it is the ground truth that Week 10's Dyna-Q
    compares its *learned* model against.

    Args:
        cfg: the world. ``cfg.max_steps`` is ignored; see the module docstring.
        max_states: passed to :func:`enumerate_states`.

    Returns:
        The :class:`TabularModel`.

    Raises:
        StateSpaceTooLarge: if the state space does not fit under the cap.
    """
    states, index = enumerate_states(cfg, max_states=max_states)
    n_states, n_actions = len(states), len(ACTIONS)

    P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R = np.zeros((n_states, n_actions), dtype=np.float64)
    rho0 = np.zeros(n_states, dtype=np.float64)
    terminal = np.zeros(n_states, dtype=bool)

    for s, state in enumerate(states):
        terminal[s] = state.terminated
        for a, action in enumerate(ACTIONS):
            for probability, next_state, reward, _terminated in successors(cfg, state, action):
                P[s, a, index[next_state]] += probability
                R[s, a] += probability * reward  # the expectation over slips and ghost moves

    # rho_0 is uniform over the cells reset() can draw; distinct start cells give
    # distinct states, but accumulate rather than assign so that stays an
    # observation about this environment and not an assumption of this code.
    start_cells = cfg.start_cells()
    for cell in start_cells:
        rho0[index[initial_state(cfg, cell)]] += 1.0 / len(start_cells)

    return TabularModel(states=states, index=index, P=P, R=R, rho0=rho0, terminal=terminal)
