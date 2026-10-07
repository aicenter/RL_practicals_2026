"""Be the bandit agent: learn to read digits from rewards alone.

From the repository root:

    uv run python week03_contextual_bandits/play.py --name alice     # play a game
    uv run python week03_contextual_bandits/play.py --name alice     # ...again: next game
    uv run python week03_contextual_bandits/play.py --summary        # you vs the algorithms

You see an image of a handwritten digit and press one of ten buttons, A-J. Each
button stands for one digit, but which one is a secret (a new secret every game):
all you learn is whether your answer was right. A game is 50 images. Games use
fixed seeds (game 1, 2, ... show the same images, with the same secret, to
everybody), so the summary can compare you with the algorithms *on exactly the
images you saw*.

If the digits look garbled in your terminal, add --ascii.
Results are appended to week03_contextual_bandits/results/handplay.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path

import numpy as np

from cb_tools import WEEK_DIR, make_env, run_episode
from rlcourse import leaderboard

RESULTS = Path(__file__).parent / "results" / "handplay.csv"
PLAY_SEED_BASE = 5_000
PLAY_HORIZON = 50
BUTTONS = "ABCDEFGHIJ"
FIELDS = ["name", "round", "seed", "horizon", "mistakes", "presses", "time"]


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


def draw(pixels: np.ndarray, ascii_only: bool = False) -> str:
    """A 28x28 image as text: two pixel rows per line with half blocks, or '##' in ASCII."""
    on = pixels.reshape(28, 28) > 0.5
    rows = [r for r in range(28) if on[r].any()]
    cols = [c for c in range(28) if on[:, c].any()]
    if not rows:
        return ""
    on = on[max(rows[0] - 1, 0):rows[-1] + 2, max(cols[0] - 2, 0):cols[-1] + 3]
    if ascii_only:
        return "\n".join("    " + "".join("##" if p else "  " for p in r) for r in on)
    if len(on) % 2:
        on = np.vstack([on, np.zeros((1, on.shape[1]), bool)])
    chars = {(0, 0): " ", (1, 0): "▀", (0, 1): "▄", (1, 1): "█"}
    return "\n".join("    " + "".join(chars[int(a), int(b)] for a, b in zip(top, bot))
                     for top, bot in zip(on[0::2], on[1::2]))


def secret(seed: int) -> np.ndarray:
    """button_of[digit]: the secret button (0-9 = A-J) of each digit in this game."""
    return np.random.default_rng([seed, 31337]).permutation(10)


def play(name: str, round_: int, ascii_only: bool) -> None:
    seed = PLAY_SEED_BASE + round_
    env = make_env(1, PLAY_HORIZON)
    obs, _ = env.reset(seed=seed)
    button_of = secret(seed)
    digit_of = np.argsort(button_of)                  # digit_of[button]
    print(f"Game {round_}: {PLAY_HORIZON} images. Each of the buttons {', '.join(BUTTONS)} stands "
          "for one digit -- a secret one.\nType a letter and Enter; 'q' quits without saving.")
    presses, mistakes, t, last = [], 0, 0, ""
    while t < PLAY_HORIZON:
        print("\n" + draw(obs["pixels"], ascii_only))
        s = input(f" {last}image {t + 1}/{PLAY_HORIZON}, mistakes so far {mistakes} -> button: ").strip().upper()
        if s == "Q":
            print("Quit, nothing saved.")
            return
        if len(s) != 1 or s not in BUTTONS:
            print(f" Please type one of the letters {BUTTONS}.")
            continue
        action = int(digit_of[BUTTONS.index(s)])
        obs, r, _, _, _ = env.step(action)
        mistakes += int(r == 0)
        last = f"{s}: {'RIGHT' if r else 'wrong'}.  "
        presses.append(s)
        t += 1
    print(f"\n {last}\nDone: {mistakes} mistakes in {PLAY_HORIZON} images (= your regret L_T).")
    print(" The secret was:  " + "  ".join(f"{d}={BUTTONS[button_of[d]]}" for d in range(10)))
    save_result(dict(name=name, round=round_, seed=seed, horizon=PLAY_HORIZON, mistakes=mistakes,
                     presses="".join(presses), time=datetime.datetime.now().isoformat(timespec="seconds")))
    print(f"Saved to {RESULTS}.")
    leaderboard.submit(WEEK_DIR, "handplay",
                       {"round": round_, "seed": seed, "horizon": PLAY_HORIZON,
                        "mistakes": mistakes, "presses": "".join(presses)},
                       nickname=name)


def ideal_mistakes(seed: int, horizon: int = PLAY_HORIZON, sims: int = 200) -> float:
    """Expected mistakes of a player who reads every digit correctly, remembers everything
    and knows that the buttons stand for different digits: for an unknown digit, they press
    a random button that is neither known to be wrong for it nor known to belong to another
    digit."""
    env = make_env(1, horizon)
    env.reset(seed=seed)
    labels = env.unwrapped.labels[env.unwrapped.order]
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(sims):
        known, wrong = {}, {d: set() for d in range(10)}       # digit -> button; excluded buttons
        for d in labels:
            if d in known:
                continue
            taken = set(known.values())
            options = [b for b in range(10) if b not in taken and b not in wrong[d]]
            b = options[rng.integers(len(options))]
            if b == d:                     # w.l.o.g. digit d's button is d
                known[d] = b
            else:
                wrong[d].add(b)
                total += 1
    return total / sims


def algorithm_mistakes(seeds: list[int], agent_seeds: int = 20) -> dict[str, float]:
    """Mean mistakes of a few algorithms on the given games (a game listed twice counts twice),
    averaged over `agent_seeds` runs of each algorithm on each game."""
    import agents as A
    candidates = {"Random": A.RandomAgent(), "GreedyRidge": A.GreedyRidge(),
                  "LinUCB(1)": A.LinUCB(1.0), "LinearTS(0.3)": A.LinearTS(0.3),
                  "Reinforce": A.Reinforce(), "SupervisedNet (sees labels)": A.SupervisedNet()}
    env = make_env(1, PLAY_HORIZON)
    out = {}
    for label, agent in candidates.items():
        try:
            per_game = {s: np.mean([run_episode(env, agent, s, agent_seed=k, with_test=False)["regret"][-1]
                                    for k in range(agent_seeds)]) for s in set(seeds)}
        except NotImplementedError:
            continue
        out[label] = float(np.mean([per_game[s] for s in seeds]))
    return out


def summary() -> None:
    rows = load_results()
    if not rows:
        print("No games yet.")
        return
    seeds = [int(r["seed"]) for r in rows]
    human = np.array([float(r["mistakes"]) for r in rows])
    print(f"{len(rows)} game(s) by {len({r['name'] for r in rows})} player(s); mean mistakes "
          f"in {PLAY_HORIZON} images:")
    print(f"   {'humans':30s} {human.mean():6.1f}  (per game: "
          + ", ".join(f"{r['name']} g{r['round']}: {r['mistakes']}" for r in rows) + ")")
    ideal = np.mean([ideal_mistakes(s) for s in seeds])
    print(f"   {'ideal human (see play.py)':30s} {ideal:6.1f}")
    for label, m in algorithm_mistakes(seeds).items():
        print(f"   {label:30s} {m:6.1f}")
    print("\nThe algorithms play exactly the same images as the humans (each averaged over 20 runs).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None,
                    help="your nickname (shown on the class leaderboard, so a nickname is fine)")
    ap.add_argument("--round", type=int, default=None,
                    help="which game to play (default: your next unplayed one)")
    ap.add_argument("--ascii", action="store_true", help="draw digits with plain ASCII characters")
    ap.add_argument("--summary", action="store_true", help="compare recorded games with algorithms")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        while not args.name:
            args.name = input("Your nickname for the leaderboard: ").strip()
        played = {int(r["round"]) for r in load_results() if r["name"] == args.name}
        rnd = args.round or next(i for i in range(1, 10_000) if i not in played)
        play(args.name, rnd, args.ascii)
