"""Submitting results to the course leaderboard.

Some practicals send results to a leaderboard shown in class. The first time
you submit in a given week, you are asked for your group's password for that
week (you get it in class). It is remembered in that week's `results/` folder,
so you only type it once per computer and week.

Submissions never block your work: if the server cannot be reached, the
submission is saved locally and sent with the next one. You can also retry by
hand, check what is stored, or change the password:

    uv run python -m rlcourse.leaderboard status week02_bandits
    uv run python -m rlcourse.leaderboard flush  week02_bandits   # retry waiting submissions
    uv run python -m rlcourse.leaderboard login  week02_bandits   # (re-)enter the password

What is sent: your group (derived from the password), the nickname you choose,
the results of the script that submits, and, for final submissions, the files
you were asked to edit that week. No other personal data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import ssl
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Iterable

SERVER_URL = os.environ.get("RLCOURSE_LEADERBOARD_URL", "https://bajgar.org/lb/api.php")
TIMEOUT = 8                   # seconds
MAX_FILE_BYTES = 100_000      # per submitted file


class ServerError(Exception):
    """The server answered, but refused the request (e.g. wrong password)."""


class Unreachable(Exception):
    """No answer from the server (offline, firewall, server down, ...)."""


# --- low-level HTTP -----------------------------------------------------------------------

def _ssl_context():
    ctx = ssl.create_default_context()
    if not ctx.get_ca_certs():                       # some Python builds find no CA certificates
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return ctx


def _request(action: str, week: str | None = None, password: str | None = None,
             body: dict | None = None) -> dict:
    url = f"{SERVER_URL}?action={action}" + (f"&week={week}" if week else "")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    req.add_header("User-Agent", "rlcourse-leaderboard/1")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if password:
        req.add_header("X-Group-Password", password)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:              # the server answered with an error
        try:
            msg = json.loads(e.read().decode()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ServerError(msg) from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise Unreachable(str(getattr(e, "reason", e))) from None


def ping() -> dict:
    """Health check (used by check_setup.py)."""
    return _request("")


# --- local state (per week: <week_dir>/results/) -------------------------------------------

def week_id(week_dir: Path) -> str:
    """'week02_bandits' -> 'w02'."""
    m = re.match(r"week(\d{2})", Path(week_dir).resolve().name)
    if not m:
        raise ValueError(f"{week_dir} does not look like a weekNN_... directory")
    return f"w{m.group(1)}"


def _paths(week_dir: Path):
    results = Path(week_dir) / "results"
    return results / "leaderboard_login.json", results / "leaderboard_outbox.jsonl"


def _load_login(week_dir: Path) -> dict | None:
    login_file, _ = _paths(week_dir)
    try:
        login = json.loads(login_file.read_text())
        return login if login.get("week") == week_id(week_dir) else None
    except (OSError, ValueError):
        return None


def login(week_dir: Path) -> dict | None:
    """Ask for the group password (up to 3 tries), check it with the server, remember it."""
    week = week_id(week_dir)
    login_file, _ = _paths(week_dir)
    for _ in range(3):
        try:
            pw = input(f"Leaderboard: your group's password for {week} (from your instructors; "
                       "Enter to skip): ").strip()
        except EOFError:
            pw = ""
        if not pw:
            print("Leaderboard: skipped. Your results are kept locally and sent next time.")
            return None
        try:
            group = _request("whoami", week, pw)["group"]
            print(f"Leaderboard: OK, you are group {group}.")
        except ServerError as e:
            print(f"Leaderboard: {e}. Please try again.")
            continue
        except Unreachable as e:
            print(f"Leaderboard: cannot reach the server ({e}); "
                  "keeping the password and trying again later.")
            group = None
        login_file.parent.mkdir(parents=True, exist_ok=True)
        info = {"week": week, "password": pw, "group": group}
        login_file.write_text(json.dumps(info))
        return info
    return None


def _outbox(week_dir: Path) -> list[dict]:
    _, outbox = _paths(week_dir)
    if not outbox.exists():
        return []
    return [json.loads(line) for line in outbox.read_text().splitlines() if line.strip()]


def _write_outbox(week_dir: Path, records: list[dict]) -> None:
    _, outbox = _paths(week_dir)
    outbox.parent.mkdir(parents=True, exist_ok=True)
    outbox.write_text("".join(json.dumps(r) + "\n" for r in records))


def flush(week_dir: Path, ask: bool = True) -> tuple[int, int]:
    """Send waiting submissions. Returns (sent, still waiting)."""
    records = _outbox(week_dir)
    if not records:
        return 0, 0
    info = _load_login(week_dir) or (login(week_dir) if ask else None)
    if info is None:
        return 0, len(records)
    sent = 0
    for i, rec in enumerate(records):
        try:
            reply = _request("submit", week_id(week_dir), info["password"], rec)
        except ServerError as e:
            print(f"Leaderboard: the server refused the submission: {e}.\n"
                  f"  If the password is wrong: uv run python -m rlcourse.leaderboard login "
                  f"{Path(week_dir).name}")
            break
        except Unreachable as e:
            print(f"Leaderboard: cannot reach the server ({e}).")
            break
        sent += 1
        if info.get("group") is None and reply.get("group"):     # first contact after offline login
            info["group"] = reply["group"]
            _paths(week_dir)[0].write_text(json.dumps(info))
    else:
        i = len(records)
    _write_outbox(week_dir, records[i:])
    return sent, len(records) - i


def submit(week_dir: Path, kind: str, payload: dict, nickname: str | None = None,
           files: Iterable[Path] = ()) -> bool:
    """Queue one submission and try to send everything waiting. Never raises.

    `files` are uploaded as text (e.g. the week's agents.py), as a record of your work."""
    try:
        week_dir = Path(week_dir)
        rec = {"id": uuid.uuid4().hex, "kind": kind, "nickname": nickname, "payload": payload,
               "files": {}, "client_time": datetime.datetime.now().astimezone().isoformat(),
               "client": {"python": platform.python_version(), "os": platform.system()}}
        for f in map(Path, files):
            text = f.read_text(errors="replace")
            rec["files"][f.name] = text[:MAX_FILE_BYTES]
        _write_outbox(week_dir, _outbox(week_dir) + [rec])
        sent, waiting = flush(week_dir)
        info = _load_login(week_dir)
        if waiting == 0:
            print(f"Leaderboard: submitted ({sent} record(s)) as group {info and info.get('group')}.")
            return True
        print(f"Leaderboard: {waiting} submission(s) saved locally; they will be sent with your next "
              f"one, or run: uv run python -m rlcourse.leaderboard flush {week_dir.name}")
        return False
    except Exception as e:                           # never break the student's script
        print(f"Leaderboard: could not submit ({type(e).__name__}: {e}). Your local results are safe.")
        return False


# --- command line -----------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(description="Course leaderboard: status, login, retry.")
    ap.add_argument("command", choices=["status", "login", "flush"])
    ap.add_argument("week_dir", type=Path, help="e.g. week02_bandits")
    args = ap.parse_args()
    if args.command == "login":
        login(args.week_dir)
    elif args.command == "flush":
        sent, waiting = flush(args.week_dir)
        print(f"Sent {sent}, still waiting: {waiting}.")
    else:
        info = _load_login(args.week_dir)
        print(f"Week {week_id(args.week_dir)}, server {SERVER_URL}")
        print(f"Group: {info.get('group') if info else '(no password entered yet)'}")
        print(f"Submissions waiting to be sent: {len(_outbox(args.week_dir))}")
        try:
            ping()
            print("Server: reachable")
        except (ServerError, Unreachable) as e:
            print(f"Server: NOT reachable ({e})")


if __name__ == "__main__":
    _main()
