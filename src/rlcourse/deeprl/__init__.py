"""Shared code for the deep-RL practicals (Weeks 7-9): import what you need from its modules.

- `rlcourse.deeprl.world`: the PragMan world all of them train on (`make_env`).
- `rlcourse.deeprl.networks`: `GridEncoder`, the network body that reads a board.
- `rlcourse.deeprl.evaluation`: `evaluate`, a policy on fixed evaluation episodes.
- `rlcourse.deeprl.experiment`: `Monitor` for a training loop, `run_experiment` for several
  seeds in parallel, results on disk, and leaderboard submission.
- `rlcourse.deeprl.plotting`: learning curves of saved results
  (`uv run python -m rlcourse.deeprl.plotting week07_dqn/results`).
- `rlcourse.deeprl.dqn_checks`: the checks behind `uv run python week07_dqn/dqn.py check`.
"""
