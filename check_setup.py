"""Check that the course environment is installed correctly.

Run from the repository root:

    uv run python check_setup.py            # quick check, no window
    uv run python check_setup.py --render   # also opens a rendering window
"""

import argparse
import platform
import sys
import traceback

OK = "[ OK ]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def check_python() -> None:
    print(f"{INFO} Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(
            f"Expected Python 3.13, got {platform.python_version()}. "
            "Run the script via `uv run` from the repository root."
        )


def check_imports() -> None:
    import gymnasium
    import matplotlib
    import numpy
    import pygame  # noqa: F401  (pygame prints its own version banner)
    import torch

    import rlcourse.envs  # noqa: F401

    for mod in (numpy, torch, gymnasium, matplotlib):
        print(f"{INFO} {mod.__name__} {mod.__version__}")


def check_torch() -> None:
    import torch

    x = torch.randn(64, 32)
    layer = torch.nn.Linear(32, 4)
    loss = layer(x).pow(2).mean()
    loss.backward()
    assert layer.weight.grad is not None

    if torch.cuda.is_available():
        accel = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        accel = "Apple MPS"
    else:
        accel = "none, using CPU (this is fine for the course)"
    print(f"{INFO} torch accelerator: {accel}")


def check_gymnasium() -> None:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def check_rendering() -> None:
    import gymnasium as gym

    print(f"{INFO} A CartPole window should appear for a few seconds...")
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset(seed=0)
    for _ in range(150):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", action="store_true", help="also test opening a rendering window")
    args = parser.parse_args()

    checks = [
        ("Python version", check_python),
        ("Package imports", check_imports),
        ("PyTorch forward/backward pass", check_torch),
        ("Gymnasium environment step", check_gymnasium),
    ]
    if args.render:
        checks.append(("Rendering window", check_rendering))

    failed = []
    for name, check in checks:
        try:
            check()
            print(f"{OK} {name}")
        except Exception:
            print(f"{FAIL} {name}")
            traceback.print_exc()
            failed.append(name)

    print()
    if failed:
        print(f"Some checks failed: {', '.join(failed)}")
        print("See the Troubleshooting section of README.md, or ask on the course forum.")
        return 1
    print("All checks passed. You're ready for the practicals!")
    if not args.render:
        print("Tip: run with --render to check that environment windows work too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
