"""Calibration.  Marked ``slow``.

Excluded from the default run (``pyproject.toml`` sets ``-m 'not slow'``); run
it with ``pytest -m slow``.  This is not a correctness test: it is a
measurement of whether the environment is fast enough for a 90-minute practical
on a laptop.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import pragman

# ── throughput ────────────────────────────────────────────────────────────────

#: Ten thousand steps a second, single core.
REQUIRED_STEPS_PER_SECOND = 10_000

#: Long enough that the measurement is not dominated by the first-call costs of
#: numpy and the observation encoder.
MEASURED_STEPS = 50_000


@pytest.mark.slow
def test_at_least_ten_thousand_steps_per_second_on_classic_11x11():
    """>= 10 000 steps/second single-core, with channel observations.

    ``classic_11x11`` with ``observation="channels"`` is the workhorse of Weeks
    7-9, so it is the configuration whose speed decides how much a student can
    train inside a practical.  Actions come from a pre-drawn array so that the
    policy costs nothing and what is timed is the environment alone.
    """
    env = pragman.make("classic_11x11")
    assert env.cfg.observation == "channels"
    actions = np.random.default_rng(0).integers(0, 4, size=MEASURED_STEPS)

    env.reset(seed=0)
    started = time.perf_counter()
    for action in actions:
        _obs, _reward, terminated, truncated, _info = env.step(int(action))
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - started

    steps_per_second = MEASURED_STEPS / elapsed
    print(f"\nclassic_11x11: {steps_per_second:,.0f} steps/s")
    assert steps_per_second >= REQUIRED_STEPS_PER_SECOND, (
        f"only {steps_per_second:,.0f} steps/s; the target is "
        f"{REQUIRED_STEPS_PER_SECOND:,}"
    )
