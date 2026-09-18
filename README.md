# Reinforcement Learning: Practicals 2026

Code for the practicals of the Reinforcement Learning course at CTU. We use
Python 3.13, [PyTorch](https://pytorch.org/),
[Gymnasium](https://gymnasium.farama.org/) and a few supporting libraries.
[uv](https://docs.astral.sh/uv/) manages the environment.

The setup works on Linux, Windows and macOS on Apple Silicon (M1 or newer).
Intel Macs are not supported (see [Troubleshooting](#troubleshooting)).

## Repository layout

```
├── README.md            # this file
├── pyproject.toml       # dependencies
├── uv.lock              # exact pinned versions, identical for everyone
├── check_setup.py       # verifies your installation
├── src/rlcourse/        # shared course code (e.g. the Pacman environment)
└── week01_foundations/  # one directory per week of practicals
```

## Setup

Please complete this **before the first practical**, ideally at home, so we
can sort out any problems in class.

### 1. Install git and uv

**Linux / macOS** (in a terminal):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Git is usually preinstalled. If not, install it with your package manager
(e.g. `sudo apt install git`), or on macOS run `xcode-select --install`.

**Windows** (in PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install git from [git-scm.com](https://git-scm.com/download/win) (the
default options are fine).

**After installing, close and reopen your terminal** so that the `uv`
command is found. Check with `uv --version`.

You do **not** need to install Python yourself. uv downloads the right
version automatically.

### 2. Get the code and install the dependencies

```bash
git clone https://github.com/aicenter/RL_practicals_2026.git
cd RL_practicals_2026
uv sync
```

`uv sync` creates a virtual environment in `.venv/` with Python 3.13 and all
dependencies. The first run downloads about 1 GB.

### 3. Check the installation

```bash
uv run python check_setup.py --render
```

A CartPole window should appear briefly, and the script should finish with
`All checks passed`. If a check fails, see
[Troubleshooting](#troubleshooting).

## Everyday use

Always work from the repository root.

| Task | Command |
| --- | --- |
| Run a script | `uv run python week01_foundations/some_script.py` |
| Interactive console | `uv run ipython` |
| Get the latest practicals | `git pull` then `uv sync` |

`uv run` automatically uses the course environment, so there is nothing to
activate. If you prefer, you can activate the environment manually with
`source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate`
(Windows) and then use `python` directly.

**Editors:** In VS Code or PyCharm, select the interpreter at
`.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python.exe` (Windows).

**Your own work:** To avoid conflicts when pulling updates, put your
solutions in new files (or your own branch) rather than editing the
provided files in place.

## Troubleshooting

**`uv: command not found`**: Open a new terminal after installing uv. If that
does not help, the installer printed which directory to add to your `PATH`.

**Intel Mac:** PyTorch no longer supports Intel Macs. Please use the lab
computers, or a Linux machine you have access to.

**No window appears / display errors:** Rendering needs a graphical session.
It won't work over plain SSH or inside WSL without GUI support. On Windows,
run everything natively in PowerShell rather than in WSL, unless you know
WSLg is set up. The checks without `--render` should still pass.

**Disk quota exceeded (lab computers):** uv keeps a download cache in your
home directory. Point it to local disk before running `uv sync`:

```bash
export UV_CACHE_DIR=/tmp/$USER/uv-cache
```

**Warning `Failed to hardlink files`:** This is harmless. It happens when the
cache and the project are on different filesystems. Silence it with
`export UV_LINK_MODE=copy`.

**GPU:** The course environment installs CPU-only PyTorch on Linux and
Windows, which is enough for all practicals. On Apple Silicon, PyTorch can
use the GPU through MPS.

**Something else:** Delete the environment with `rm -rf .venv` (Linux/macOS)
or `Remove-Item -Recurse -Force .venv` (Windows), then run `uv sync` again.
If the problem persists, post the full output of `check_setup.py` on the
course forum.

## For instructors

- Add a dependency with `uv add <package>` and commit both `pyproject.toml`
  and `uv.lock`. `uv lock --upgrade` refreshes all pinned versions. Do this
  between weeks, not the evening before a practical.
- Course environments live in `src/rlcourse/envs/` and are registered with
  Gymnasium in `src/rlcourse/envs/__init__.py`. The package is installed in
  editable mode, so changes appear immediately after `git pull`.
- For students who can't use uv, you can generate a pip-compatible file with
  `uv export --no-hashes --no-emit-project -o requirements.txt`.
