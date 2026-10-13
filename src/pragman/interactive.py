"""Playing Prag-man by hand, and recording what a policy does.

Two functions:

:func:`play`
    Opens a window, gives you the arrow keys, prints the reward stream as you
    go, and hands back the :class:`~pragman.state.Episode` you just played.
    Playing a level for two minutes teaches more about a reward function than a
    page of notes -- try ``living_reward`` at 0 and at -0.05 and watch what you
    do differently.
:func:`record_episode`
    Runs a policy for one episode, returns the same ``Episode`` container, and
    optionally writes a GIF for a slide or for Week 11's preference labelling.

Both return the *same* object, so a lab that consumes demonstrations does not
care whether they came from a human or from a policy.
Keyboard play needs pygame (``pip install 'pragman[play]'``); writing a GIF needs
imageio (``pip install 'pragman[viz]'``).  Both are imported lazily.

Example
-------
>>> import pragman                                        # doctest: +SKIP
>>> env = pragman.make("classic_11x11")                   # doctest: +SKIP
>>> episode = pragman.play(env)                           # doctest: +SKIP
>>> episode.save("my_demo.json")                          # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from pragman.render import PygameWindow, frame, ghost_names_from_config, save_gif

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from pragman.config import PragmanConfig
    from pragman.env import PragmanEnv
    from pragman.state import Episode, PragmanState

__all__ = ["play", "record_episode", "ACTION_NAMES"]

#: Action names in ``Action`` order, for the printed reward stream.
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers shared by the two entry points
# ─────────────────────────────────────────────────────────────────────────────


def _config_of(env: PragmanEnv) -> PragmanConfig:
    """The ``PragmanConfig`` behind a (possibly wrapped) environment."""
    for candidate in (env, getattr(env, "unwrapped", None)):
        cfg = getattr(candidate, "cfg", None)
        if cfg is not None:
            return cfg
    raise TypeError(
        "expected a Prag-man environment with a .cfg attribute; "
        f"got {type(env).__name__}"
    )


def _state_of(env: PragmanEnv, info: dict[str, Any]) -> PragmanState:
    """The true state of the environment, which Prag-man always puts in ``info``."""
    state = info.get("state")
    if state is None:
        raise KeyError(
            "info['state'] is missing: play() and record_episode() need the true "
            "state, which a plain Prag-man environment always provides"
        )
    return state


def _make_episode(
    cfg: PragmanConfig,
    states: list[PragmanState],
    actions: list[int],
    rewards: list[float],
    terminated: bool,
    truncated: bool,
) -> Episode:
    """Build the shared container, importing it late to avoid an import cycle."""
    from pragman.state import Episode

    return Episode(
        config=cfg,
        states=states,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
    )


def _frame_arguments(cfg: PragmanConfig, cell_pixels: int, palette: str) -> dict[str, Any]:
    """The keyword arguments that make every picture of this world look alike."""
    try:
        ghost_names = ghost_names_from_config(cfg)
    except AttributeError:  # a config without ghost policies: colour them alike
        ghost_names = None
    return {"cell_pixels": cell_pixels, "palette": palette, "ghost_names": ghost_names}


def _print_step(
    step: int, action: int, reward: float, total: float, info: dict[str, Any]
) -> None:
    """One line of the reward stream, aligned so that the numbers form a column."""
    note = ""
    if info.get("death"):
        note = "  <- eaten"
    elif info.get("slipped"):
        note = "  (slipped)"
    pellets = info.get("pellets_left")
    left = "" if pellets is None else f"  pellets left {pellets:3d}"
    print(
        f"step {step:4d}  {ACTION_NAMES[action]:>5}  "
        f"reward {reward:+7.3f}  return {total:+8.3f}{left}{note}"
    )


def _print_summary(rewards: list[float], terminated: bool, truncated: bool) -> None:
    ending = "terminated" if terminated else "truncated" if truncated else "stopped early"
    steps = len(rewards)
    print(
        f"--- {ending} after {steps} step{'' if steps == 1 else 's'}, "
        f"undiscounted return {float(np.sum(rewards)):+.3f} ---"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard play
# ─────────────────────────────────────────────────────────────────────────────


def play(
    env: PragmanEnv,
    fps: int = 30,
    cell_pixels: int = 16,
    palette: str = "high_contrast",
    scale: int | None = None,
    seed: int | None = None,
) -> Episode:
    """Play one episode with the arrow keys and return it as an ``Episode``.

    Controls: **arrow keys** (or **WASD**) to move, **Escape** or closing the
    window to stop early, **Q** likewise.  The environment steps once per key
    press -- Prag-man is turn-based, so there is no need to hold a key down.

    Parameters
    ----------
    env:
        A Prag-man environment.  Construct it with ``render_mode=None``: this
        function opens and drives its own window, so a second one would only
        get in the way.
    fps:
        How often the window is polled for key presses.  This is *not* a speed
        limit on the game, which advances one step per key press.
    cell_pixels, palette, scale:
        Passed to :func:`~pragman.render.frame` and
        :class:`~pragman.render.PygameWindow`; the defaults give a large, high
        contrast board suitable for a lecture hall.
    seed:
        Forwarded to ``env.reset``, so a demonstration can be reproduced.

    Returns
    -------
    Episode
        The recorded demonstration, ready for ``episode.save(path)`` or for
        Week 12's behavioural cloning.  Quitting early gives an episode with
        both ``terminated`` and ``truncated`` False -- a trajectory that was
        simply cut short.
    """
    cfg = _config_of(env)
    layout = cfg.layout
    drawing = _frame_arguments(cfg, cell_pixels, palette)

    window = PygameWindow(fps=fps, scale=scale, title="Prag-man  --  arrow keys, Esc to stop")
    pygame = window.pygame
    keys = {
        pygame.K_UP: 0,
        pygame.K_w: 0,
        pygame.K_RIGHT: 1,
        pygame.K_d: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
    }

    observation, info = env.reset(seed=seed)
    states: list[PragmanState] = [_state_of(env, info)]
    actions: list[int] = []
    rewards: list[float] = []
    terminated = truncated = False
    total = 0.0

    print("Arrow keys or WASD to move, Escape or Q to stop.")
    try:
        window.show(frame(states[-1], layout, **drawing), tick=False)
        while not window.closed:
            action = _wait_for_key(window, pygame, keys)
            if action is None:  # the window was closed
                break

            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            rewards.append(float(reward))
            states.append(_state_of(env, info))
            total += float(reward)
            _print_step(len(actions), int(action), float(reward), total, info)

            window.show(frame(states[-1], layout, **drawing), tick=False)
            if terminated or truncated:
                break
    finally:
        window.close()

    _print_summary(rewards, terminated, truncated)
    return _make_episode(cfg, states, actions, rewards, terminated, truncated)


def _wait_for_key(window: PygameWindow, pygame: Any, keys: dict[int, int]) -> int | None:
    """Block until a movement key is pressed; return ``None`` if the window closed."""
    while not window.closed:
        for event in window.events():
            if event.type == pygame.KEYDOWN:
                if event.key in keys:
                    return keys[event.key]
                if event.key == pygame.K_q:
                    window.closed = True
        pygame.time.wait(15)  # be kind to the CPU while nothing is happening
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Recording a policy
# ─────────────────────────────────────────────────────────────────────────────


def record_episode(
    env: PragmanEnv,
    policy: Callable[[Any], int],
    path: str | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    cell_pixels: int = 16,
    palette: str = "classic",
    fps: int = 8,
    verbose: bool = False,
) -> Episode:
    """Run ``policy`` for one episode; optionally write a GIF of it.

    Parameters
    ----------
    env:
        A Prag-man environment (``render_mode`` is irrelevant: the frames are
        drawn straight from the states).
    policy:
        A callable mapping an observation to an action, ``0 <= a < 4``.  A
        random policy is ``lambda obs: env.action_space.sample()``; a trained
        network is ``lambda obs: int(net(obs).argmax())``.
    path:
        Where to write an animated GIF, or ``None`` for no file.  Needs imageio.
    seed:
        Forwarded to ``env.reset``, so a clip can be reproduced exactly.
    max_steps:
        Stop after this many steps even if the environment has not finished --
        a safety net for a policy that never terminates.  ``None`` relies on
        the environment's own ``max_steps``.
    cell_pixels, palette, fps:
        Appearance of the GIF.  16 pixels per cell is a good compromise between
        a readable slide and a small file.

    Returns
    -------
    Episode
        The trajectory, with its config, exactly as :func:`play` returns.  This
        is what Week 11 labels for preference learning and Week 12 trains on.
    """
    cfg = _config_of(env)
    layout = cfg.layout
    drawing = _frame_arguments(cfg, cell_pixels, palette)

    observation, info = env.reset(seed=seed)
    states: list[PragmanState] = [_state_of(env, info)]
    actions: list[int] = []
    rewards: list[float] = []
    terminated = truncated = False
    total = 0.0

    while not (terminated or truncated):
        if max_steps is not None and len(actions) >= max_steps:
            break
        action = int(policy(observation))
        if not 0 <= action < 4:
            raise ValueError(f"the policy returned action {action}; actions are 0, 1, 2, 3")

        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        states.append(_state_of(env, info))
        total += float(reward)
        if verbose:
            _print_step(len(actions), action, float(reward), total, info)

    if verbose:
        _print_summary(rewards, terminated, truncated)
    if path is not None:
        save_gif([frame(state, layout, **drawing) for state in states], path, fps=fps)
    return _make_episode(cfg, states, actions, rewards, terminated, truncated)
