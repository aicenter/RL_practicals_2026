"""The Gymnasium environment, and the two lines of boilerplate every lab starts with.

::

    import pragman
    env = pragman.make("classic_11x11", render_mode="human")
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(action)

:class:`PragmanEnv` is a thin, stateful shell around the pure functions of
:mod:`pragman.dynamics`. It owns exactly three things the maths does not: the
random number generator, the episode clock, and the window. Everything else --
the transition, the reward, the successor distribution -- is a pure function you
can call yourself, which is what makes Prag-man a genuinely *known* MDP rather
than a black box with a peephole.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

from . import configs as _configs
from .config import PragmanConfig
from .dynamics import apply, sample_moves, successors
from .errors import StateInjectionDisabled
from .observations import channel_names, observation_space, observe, pixel_frame
from .render import ansi, ghost_names_from_config
from .state import ACTIONS, Action, Cell, PragmanState, as_action

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .model import TabularModel

__all__ = ["PragmanEnv", "make", "make_vec"]

#: The message :meth:`PragmanEnv.step` raises with once an episode is over.
EPISODE_OVER = "episode has ended; call reset()"


class PragmanEnv(gym.Env):
    """One Prag-man world, wrapped in the Gymnasium API.

    Args:
        cfg: the world. Build one with a factory
            (``pragman.configs.maze_8x8()``) or by hand
            (``PragmanConfig(layout="...")``).
        render_mode: ``None``, ``"rgb_array"``, ``"ansi"``, or ``"human"``.
            ``"human"`` opens a pygame window and needs ``pragman[play]``.

    Attributes:
        cfg: the config this environment was built from (read-only in spirit:
            it is a frozen dataclass, so ``dataclasses.replace`` is how you
            change one field, and that gives you a *new* world).
        elapsed_steps: how many steps the current episode has taken. The clock
            lives here rather than in the state, exactly as Gymnasium's own
            ``TimeLimit`` wrapper has it: a time limit is a property of the
            experiment, not of the Markov decision process. Were it in the
            state, the same board at t=0 and t=1 would be different dictionary
            keys and tabular learning would never generalize across time.

    The environment never truncates *and* terminates on the same step, and it
    refuses to step an episode that has already ended -- returning nonsense
    would only produce a bug report two weeks later.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 8,
    }

    def __init__(self, cfg: PragmanConfig, render_mode: str | None = None) -> None:
        if not isinstance(cfg, PragmanConfig):
            raise TypeError(
                "PragmanEnv takes a PragmanConfig; for a named world use "
                f"pragman.make({cfg!r}) instead of PragmanEnv({cfg!r})"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            legal = ", ".join(repr(mode) for mode in self.metadata["render_modes"])
            raise ValueError(f"render_mode must be None or one of {legal}, got {render_mode!r}")

        self.cfg = cfg
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = observation_space(cfg)

        self.elapsed_steps = 0
        self._state: PragmanState | None = None
        self._over = False  # terminated or truncated: step() refuses either way
        self._model: TabularModel | None = None
        self._window: Any = None
        self._ghost_names = ghost_names_from_config(cfg)

    # ── convenience, because students hold an env and not a cfg ──────────────

    @property
    def layout(self):
        """The map this world is played on (``cfg.layout``)."""
        return self.cfg.layout

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Names of the ``"channels"`` planes, in order; see :mod:`pragman.observations`."""
        return channel_names(self.cfg)

    @property
    def state(self) -> PragmanState:
        """The current true state. Also in ``info["state"]`` after every call."""
        return self._require_state()

    @property
    def model(self):
        """The exact tabular model of this world, built once and cached.

        ``model.P``, ``model.R`` and ``model.rho0`` are the ``(S, A, S)``,
        ``(S, A)`` and ``(S,)`` arrays Week 4 does dynamic programming on. The
        clock is not part of the MDP, so the model ignores ``cfg.max_steps``.

        Raises:
            StateSpaceTooLarge: on a world with too many reachable states --
                which is the point of Week 6, and why the error message points
                at :func:`pragman.dynamics.successors` instead.
        """
        if self._model is None:
            from .model import build_tabular_model  # lazy: only the DP labs need it

            self._model = build_tabular_model(self.cfg)
        return self._model

    def successors(self, state: PragmanState, action: int | Action):
        """Every outcome of one step from *state*, with its exact probability.

        The white-box model for a single state: ``((probability, next_state,
        reward, terminated), ...)``. This is all that asynchronous dynamic
        programming, Dyna-Q and MCTS ever need, and unlike :attr:`model` it
        costs nothing to call on a world with a billion states.
        """
        return successors(self.cfg, state, action)

    # ── the Gymnasium API ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: seeds ``self.np_random``. Resetting twice with the same seed
                and replaying the same actions reproduces a trajectory exactly.
            options: ``{"state": s}`` places the environment in a state of your
                choosing instead of drawing one from rho_0. ``s`` may be a
                :class:`~pragman.state.PragmanState` or an ASCII position (the
                position form, see the README), which makes a hand-written
                fixture readable. Gated by ``cfg.allow_state_injection``.

        Returns:
            ``(observation, info)``.
        """
        super().reset(seed=seed)  # this is what seeds self.np_random
        injected = self._injected_state(options)
        self._state = self._initial_state() if injected is None else injected
        self.elapsed_steps = 0
        self._over = self._state.terminated
        self._render_human()
        return observe(self._state, self.cfg), self._info(self._state)

    def step(self, action: int | Action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take one step: slip, move, eat, die, and let the ghosts move.

        The numbered order of events is in the :mod:`pragman.dynamics` module
        docstring, and the sampling happens there too -- this method only turns
        the result into the five-tuple Gymnasium expects and looks after the
        clock.

        Raises:
            RuntimeError: if the episode has already ended.
        """
        state = self._require_state()
        if self._over:
            raise RuntimeError(EPISODE_OVER)

        chosen = as_action(action)
        # Sampling in two stages (rather than calling dynamics.transition) so
        # that info["slipped"] can say whether the world took you somewhere you
        # did not ask to go. Students chasing a "my policy is broken" bug find
        # this flag before they find the slip probability in the config.
        realized, ghost_actions = sample_moves(self.cfg, state, chosen, self.np_random)
        next_state, reward, terminated = apply(self.cfg, state, realized, ghost_actions)

        self.elapsed_steps += 1
        truncated = bool(
            not terminated
            and self.cfg.max_steps is not None
            and self.elapsed_steps >= self.cfg.max_steps
        )
        self._state = next_state
        self._over = bool(terminated or truncated)

        self._render_human()
        info = self._info(next_state, slipped=realized != chosen)
        return observe(next_state, self.cfg), float(reward), bool(terminated), truncated, info

    def render(self) -> np.ndarray | str | None:
        """Draw the current state in whichever mode the environment was built with.

        ``"rgb_array"`` returns exactly the picture the ``"pixels"`` observation
        is made of (uncropped: ``view_radius`` narrows what the *agent* sees,
        not what the lecture hall does), ``"ansi"`` returns a coloured terminal
        string, and ``"human"`` puts a pygame window on the screen and returns
        ``None``.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "render() was called on an environment with render_mode=None; build it with "
                'e.g. pragman.make("maze_8x8", render_mode="rgb_array") to get a picture'
            )
            return None
        state = self._require_state()
        if self.render_mode == "rgb_array":
            return pixel_frame(state, self.cfg)
        if self.render_mode == "ansi":
            return ansi(
                state,
                self.cfg.layout,
                palette=self.cfg.palette,
                ghost_names=self._ghost_names,
            )
        self._render_human()
        return None

    def close(self) -> None:
        """Shut the pygame window, if one was ever opened. Safe to call twice."""
        if self._window is not None:
            self._window.close()
            self._window = None

    # ── state access ─────────────────────────────────────────────────────────

    def clone_state(self) -> PragmanState:
        """The current state, for saving and restoring.

        No copying happens: :class:`~pragman.state.PragmanState` is frozen, so
        the object you get back can never change under you. That is the payoff
        for immutability.
        """
        return self._require_state()

    def restore_state(self, state: PragmanState, elapsed_steps: int = 0) -> PragmanState:
        """Put the environment into *state*, as a simulator would.

        **The random number generator is not part of the state**, deliberately:
        restoring does not rewind the random stream, so a restored environment
        continues the same stream rather than repeating the same draws. An MCTS
        example that wants repeatable rollouts therefore either calls
        :func:`pragman.dynamics.transition` with its own generator or uses a
        second :class:`PragmanEnv` as a simulator.

        Args:
            state: where to put the world.
            elapsed_steps: what to set the episode clock to. It defaults to 0
                because the usual caller is a planner asking "suppose the world
                were like this" and wants a fresh horizon; pass the clock back
                explicitly if you are resuming a specific episode.

        Raises:
            StateInjectionDisabled: if ``cfg.allow_state_injection`` is False.
        """
        self._check_injection_allowed("restore_state()")
        self._state = self._coerce_state(state)
        self.elapsed_steps = int(elapsed_steps)
        self._over = self._state.terminated
        return self._state

    # ── internals ────────────────────────────────────────────────────────────

    def _initial_state(self) -> PragmanState:
        """Draw a state from rho_0: a start cell, the ghost spawns, the pellets."""
        cells = self.cfg.start_cells()
        agent: Cell = cells[int(self.np_random.integers(len(cells)))]
        # The agent's own cell is cleared of its pellet. Pellets are otherwise
        # only collected by *moving* onto a cell, so one spawning underneath the
        # agent would sit there uneaten -- harmless but confusing, and it
        # matters as soon as start="random" drops the agent into a full maze.
        pellets = self.cfg.initial_pellets() - {agent}
        return PragmanState(agent, self.cfg.initial_ghosts(), pellets, False)

    def _injected_state(self, options: dict[str, Any] | None) -> PragmanState | None:
        """Read ``options={"state": ...}``, if the config allows it."""
        if not options:
            return None
        unknown = sorted(set(options) - {"state"})
        if unknown:
            raise ValueError(
                f"unknown reset option(s) {unknown}; the only option Prag-man understands "
                "is 'state'"
            )
        if "state" not in options:
            return None
        self._check_injection_allowed('reset(options={"state": ...})')
        return self._coerce_state(options["state"])

    def _coerce_state(self, state: PragmanState | str | Iterable[str]) -> PragmanState:
        """Accept a state object or a hand-drawn ASCII position."""
        if isinstance(state, PragmanState):
            return state
        if isinstance(state, str) or (
            isinstance(state, (list, tuple)) and all(isinstance(row, str) for row in state)
        ):
            return PragmanState.from_ascii(state, self.cfg.layout, n_ghosts=self.cfg.n_ghosts)
        raise TypeError(
            "a state must be a PragmanState or an ASCII position (a string of rows), "
            f"not {type(state).__name__}"
        )

    def _check_injection_allowed(self, what: str) -> None:
        if not self.cfg.allow_state_injection:
            raise StateInjectionDisabled(
                f"{what} is disabled for this config (allow_state_injection=False). "
                "This world is meant to be solved model-free, so setting the state by hand "
                "would be cheating. If you mean to, build it with "
                "dataclasses.replace(cfg, allow_state_injection=True). Note that the flag is "
                "a signpost rather than a sandbox: info['state'] and "
                "pragman.dynamics.successors() are always available."
            )

    def _require_state(self) -> PragmanState:
        if self._state is None:
            raise RuntimeError("call reset() before using the environment")
        return self._state

    def _info(self, state: PragmanState, slipped: bool = False) -> dict[str, Any]:
        """The ``info`` dict, identical in shape after ``reset`` and after ``step``.

        Exposing the true state costs nothing and makes debugging, recording
        demonstrations, reward wrappers, and offline datasets trivial:
        ``Q[info["state"]]`` in a lab script does what you expect, because a
        state is hashable.
        """
        return {
            "state": state,
            "pellets_left": state.pellets_left,
            # A death is exactly "the agent and a ghost are in the same cell":
            # whether the agent walked into the ghost or the ghost walked into
            # the agent, nothing else moves once the episode ends.
            "death": bool(state.terminated and state.agent in state.ghosts),
            "slipped": bool(slipped),
        }

    def _render_human(self) -> None:
        """Show the current frame in the pygame window, opening it on first use."""
        if self.render_mode != "human" or self._state is None:
            return
        if self._window is None:
            from .render import PygameWindow  # lazy: pygame is an optional extra

            self._window = PygameWindow(fps=self.metadata["render_fps"], title="Prag-man")
        self._window.show(pixel_frame(self._state, self.cfg))


# ──────────────────────────────────────────────────────────────────────────────
# Making environments
# ──────────────────────────────────────────────────────────────────────────────


def make(
    name_or_cfg: str | PragmanConfig,
    render_mode: str | None = None,
    **overrides: Any,
) -> PragmanEnv:
    """Build an environment from a factory name or a config. The one-liner.

    ::

        env = pragman.make("maze_8x8", slip_prob=0.2, render_mode="human")

    Any keyword that is not ``render_mode`` is a :class:`PragmanConfig` field,
    so every world ships with every knob: ``pragman.make("classic_11x11",
    ghost_policy="mix")`` is the most vivid thing to put on a projector.

    Unlike ``gym.make(...)`` this returns the :class:`PragmanEnv` itself, with
    no wrappers around it -- truncation is built in, so there is no
    ``TimeLimit`` to add -- which is why ``env.model``, ``env.successors`` and
    ``env.clone_state`` are reachable without ``env.unwrapped``.

    Args:
        name_or_cfg: a factory name from :mod:`pragman.configs`, or a config.
        render_mode: ``None``, ``"rgb_array"``, ``"ansi"`` or ``"human"``.
        **overrides: config fields to change.

    Raises:
        ConfigError: naming the available worlds, if the name is unknown, or
            naming the field, if an override is not a config field.
    """
    if isinstance(name_or_cfg, PragmanConfig):
        cfg = replace(name_or_cfg, **overrides) if overrides else name_or_cfg
    else:
        cfg = _configs.by_name(name_or_cfg)(**overrides)
    return PragmanEnv(cfg, render_mode=render_mode)


def make_vec(
    name: str,
    n: int = 1,
    *,
    autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
    vectorization_mode: str = "sync",
    **overrides: Any,
) -> gym.vector.VectorEnv:
    """Run *n* copies of a world in parallel, for Week 9's PPO.

    ::

        envs = pragman.make_vec("classic_11x11", 8)
        observations, infos = envs.reset(seed=0)
        observations, rewards, terminations, truncations, infos = envs.step(actions)

    **The autoreset mode is pinned on purpose.** Gymnasium 1.0 changed the
    default from same-step to *next-step* autoreset, and rollout buffers
    written against the old behaviour silently learn from garbage. This
    function therefore states the mode rather than inheriting it, and the Week
    9 PPO skeleton assumes the pinned default,
    :attr:`~gymnasium.vector.AutoresetMode.NEXT_STEP`:

    * when a sub-environment terminates or truncates on step *t*, that step
      returns the **final** observation and reward of the episode;
    * the *following* step ignores the action you pass for that
      sub-environment, resets it, and returns the **first** observation of the
      new episode with reward 0 and both flags ``False``.

    So the value bootstrap belongs on the step that reported ``terminated`` or
    ``truncated``, and the transition spanning the reset must be dropped from
    the buffer. Pass ``autoreset_mode=AutoresetMode.SAME_STEP`` for the old
    behaviour, in which the reset observation arrives in the same step and the
    final one is tucked into ``infos["final_obs"]``.

    ``info["state"]`` survives vectorization as a **numpy object array** of
    :class:`~pragman.state.PragmanState`, one entry per sub-environment, so
    ``infos["state"][i]`` is the true state of environment *i*. (Gymnasium
    builds arrays out of the sub-environment infos; anything that is not a
    number ends up in an object array.)

    Args:
        name: a factory name from :mod:`pragman.configs`.
        n: how many copies.
        autoreset_mode: see above.
        vectorization_mode: ``"sync"`` (one process, the default and by far the
            fastest for a world this small) or ``"async"``.
        **overrides: config fields, applied to every copy.
    """
    from .registration import env_id  # lazy: registration imports this module

    return gym.make_vec(
        env_id(name),
        num_envs=n,
        vectorization_mode=vectorization_mode,
        vector_kwargs={"autoreset_mode": AutoresetMode(autoreset_mode)},
        **overrides,
    )
