"""Every exception Prag-man raises deliberately.

They all derive from :class:`PragmanError`, so a lab script can write
``except pragman.PragmanError`` and catch anything the environment complains
about without also swallowing genuine bugs such as ``TypeError``.
"""

from __future__ import annotations


class PragmanError(Exception):
    """Base class for all Prag-man errors."""


class ConfigError(PragmanError):
    """A :class:`~pragman.config.PragmanConfig` field is missing or illegal.

    The message always names the offending field and the values it will accept,
    because the config is the object students edit most often.
    """


class LayoutError(PragmanError):
    """An ASCII map (or an ASCII position) could not be parsed.

    The message names the offending row and column wherever the problem is
    local to one cell, so that a typo in a hand-drawn maze is easy to find.
    """


#: Printed at the end of every :class:`StateSpaceTooLarge` message.
DEFAULT_ADVICE = (
    "Dynamic programming needs the whole state space in memory, and pellets make "
    "that space grow as 2**n_pellets. Either shrink the problem (fewer pellets, "
    "fewer ghosts, a smaller layout, pellets='none'), raise max_states if you really "
    "have the memory, or stop enumerating: dynamics.successors(cfg, state, action) "
    "gives the exact one-step model for a single state, which is all that asynchronous "
    "dynamic programming, Dyna-Q, and MCTS ever need."
)


class StateSpaceTooLarge(PragmanError):
    """Raised when enumerating the reachable states would exceed ``max_states``.

    This exception is a teaching device as much as a safety net: the point of
    Week 6 is that tabular dynamic programming stops working long before the
    problems get interesting, and hitting this error is a more convincing
    demonstration of that than any slide.

    Attributes:
        count: how many states had been found when enumeration gave up.
        max_states: the cap that was exceeded, if the caller supplied one.
        advice: the human-readable suggestion appended to the message.
    """

    def __init__(
        self,
        count: int,
        advice: str | None = None,
        max_states: int | None = None,
    ) -> None:
        self.count = int(count)
        self.max_states = max_states
        self.advice = DEFAULT_ADVICE if advice is None else advice
        if max_states is None:
            preamble = f"state space too large: enumeration reached {self.count} states"
        else:
            preamble = (
                f"state space too large: enumeration reached {self.count} states, "
                f"above the limit of {max_states}"
            )
        super().__init__(f"{preamble}. {self.advice}")


class StateInjectionDisabled(ConfigError):
    """Raised by ``reset(options={'state': ...})`` and ``restore_state()``.

    It subclasses :class:`ConfigError` rather than ``PermissionError`` because
    nothing about the operating system is involved: the environment was simply
    configured with ``allow_state_injection=False``. The flag is a signpost that
    says "you are about to cheat", not a sandbox -- ``info['state']`` and the
    pure functions in ``dynamics`` remain available to anyone who wants them.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "state injection is disabled for this config; construct the environment "
            "with allow_state_injection=True (or dataclasses.replace(cfg, "
            "allow_state_injection=True)) if you really want to set the state by hand"
        )
