"""Be the bandit agent: play a problem class by hand and record your regret.

From the repository root:

    uv run python week02_bandits/play.py 2 --name alice        # play problem class 2
    uv run python week02_bandits/play.py 2 --name alice        # ...again: next round
    uv run python week02_bandits/play.py --summary             # you vs the algorithms

Each round is one run of `--horizon` pulls (default 50) on a fresh instance of
the problem class. Rounds use fixed seeds (round 1, 2, ... are the same
instances, with the same luck, for everybody), so the summary can compare you
with the algorithms *on exactly the bandits you played*.

Results are appended to week02_bandits/results/handplay.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path

import numpy as np

from bandit_tools import WEEK_DIR, make_env, run_episode
from rlcourse import leaderboard

RESULTS = Path(__file__).parent / "results" / "handplay.csv"
PLAY_SEED_BASE = 5_000
FIELDS = ["name", "problem", "round", "seed", "horizon", "regret", "total_reward",
          "actions", "time"]


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    with open(RESULTS, newline="") as f:
        return list(csv.DictReader(f))


def save_result(row: dict) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def show(N, S, last):
    K = len(N)
    print("\n arm   " + "".join(f"{a:>8d}" for a in range(K)))
    print(" pulls " + "".join(f"{int(n):>8d}" for n in N))
    print(" mean  " + "".join(f"{s / n:>8.2f}" if n else "       -" for s, n in zip(S, N)))
    if last is not None:
        print(f" last reward: {last:.2f}")


def play(problem: int, name: str, horizon: int, round_: int) -> None:
    seed = PLAY_SEED_BASE + round_
    env = make_env(problem, horizon)
    env.reset(seed=seed)
    K = env.action_space.n
    N, S = np.zeros(K), np.zeros(K)
    actions, total, last, t = [], 0.0, None, 0
    print(f"Problem class {problem}, round {round_}: {K} arms, {horizon} pulls. "
          f"Type an arm number and Enter; 'q' quits without saving.")
    truncated = False
    while not truncated:
        show(N, S, last)
        s = input(f" pull {t + 1}/{horizon} -> arm: ").strip()
        if s.lower() == "q":
            print("Quit, nothing saved.")
            return
        if not s.isdigit() or int(s) >= K:
            print(f" Please type a number between 0 and {K - 1}.")
            continue
        a = int(s)
        _, last, _, truncated, _ = env.step(a)
        N[a] += 1
        S[a] += last
        total += last
        actions.append(a)
        t += 1
    show(N, S, last)

    gaps = env.unwrapped.gaps()
    regret = float(np.dot(N, gaps))
    print(f"\nDone. Total reward {total:.1f}. Your regret L_T = sum_a N(a) * Delta_a = {regret:.1f}")
    print(" true means q(a): " + " ".join(f"{m:.2f}" for m in env.unwrapped.means))
    print(f" best arm: {int(np.argmax(env.unwrapped.means))}")
    save_result(dict(name=name, problem=problem, round=round_, seed=seed, horizon=horizon,
                     regret=f"{regret:.3f}", total_reward=f"{total:.3f}",
                     actions=" ".join(map(str, actions)),
                     time=datetime.datetime.now().isoformat(timespec="seconds")))
    print(f"Saved to {RESULTS}.")
    leaderboard.submit(WEEK_DIR, "handplay",
                       {"problem": problem, "round": round_, "seed": seed, "horizon": horizon,
                        "regret": regret, "total_reward": total, "actions": actions},
                       nickname=name)


def algorithm_regrets(problem: int, horizon: int, seeds: list[int]) -> dict[str, float]:
    """Mean regret of the baselines and your implemented agents on the given seeds
    (a seed listed twice counts twice)."""
    import agents as A
    candidates = {"Random": A.RandomAgent(), "Greedy": A.Greedy(),
                  "EpsilonGreedy(0.1)": A.EpsilonGreedy(0.1),
                  "ExploreThenCommit(2)": A.ExploreThenCommit(2),
                  "Boltzmann(0.1)": A.Boltzmann(0.1), "UCB(0.5)": A.UCB(0.5)}
    out = {}
    env = make_env(problem, horizon)
    for label, agent in candidates.items():
        try:
            per_seed = {s: run_episode(env, agent, s)["regret"][-1] for s in set(seeds)}
            out[label] = float(np.mean([per_seed[s] for s in seeds]))
        except NotImplementedError:
            pass
    return out


def summary() -> None:
    rows = load_results()
    if not rows:
        print("No results yet.")
        return
    groups = {}
    for r in rows:
        groups.setdefault((int(r["problem"]), int(r["horizon"])), []).append(r)
    for (problem, horizon), rs in sorted(groups.items()):
        seeds = [int(r["seed"]) for r in rs]
        human = np.array([float(r["regret"]) for r in rs])
        print(f"\nProblem {problem}, T = {horizon}: {len(rs)} game(s) by "
              f"{len({r['name'] for r in rs})} player(s), on {len(set(seeds))} different round(s)")
        print(f"   {'humans':22s} {human.mean():7.1f}  (per game: "
              + ", ".join(f"{r['name']} r{r['round']}: {float(r['regret']):.1f}" for r in rs) + ")")
        for label, reg in algorithm_regrets(problem, horizon, seeds).items():
            print(f"   {label:22s} {reg:7.1f}")
    print("\nThe algorithms play exactly the same rounds (instances and luck) as the humans.\n"
          "Single games are very noisy: compare averages over many games, not one.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", type=int, nargs="?", choices=[1, 2, 3, 4, 5])
    ap.add_argument("--name", default=None,
                    help="your nickname (shown on the class leaderboard, so a nickname is fine)")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--round", type=int, default=None,
                    help="which round to play (default: your next unplayed one)")
    ap.add_argument("--summary", action="store_true", help="compare recorded games with algorithms")
    args = ap.parse_args()
    if args.summary:
        summary()
    elif args.problem is None:
        ap.error("give a problem class (1-5) or --summary")
    else:
        while not args.name:
            args.name = input("Your nickname for the leaderboard: ").strip()
        played = {int(r["round"]) for r in load_results()
                  if r["name"] == args.name and int(r["problem"]) == args.problem
                  and int(r["horizon"]) == args.horizon}
        rnd = args.round or next(i for i in range(1, 10_000) if i not in played)
        play(args.problem, args.name, args.horizon, rnd)
