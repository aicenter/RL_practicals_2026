"""The MNIST data set of handwritten digits (Week 3 onwards).

On first use the data (one ~11 MB file, the same `mnist.npz` that Keras uses) is
downloaded into the `data/` folder at the root of the repository and reused from
there. To fetch it in advance (e.g. before a practical), run from the repository root

    uv run python -m rlcourse.mnist

Typical use:

    from rlcourse import mnist
    d = mnist.load()                       # x_train (60000, 28, 28) uint8, y_train, x_test, y_test
    F_train, F_test = mnist.features("pca")  # (60000, 51) and (10000, 51) float32

MNIST: Y. LeCun, C. Cortes and C. J. C. Burges, http://yann.lecun.com/exdb/mnist/,
available under the Creative Commons Attribution-Share Alike 3.0 licence.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RLCOURSE_DATA_DIR", REPO_ROOT / "data"))
FILE = DATA_DIR / "mnist.npz"
URLS = [u for u in (os.environ.get("RLCOURSE_MNIST_URL"),
                    "https://bajgar.org/data/mnist.npz",
                    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz") if u]
KNOWN_SHA256 = {   # checksums of the copies we know
    "06af5dfe73b70e3bc1c36439982b18a8d76803db9b544f6fdf838a37fbb01b64",   # course copy (bajgar.org)
    "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",   # Keras' mnist.npz
}
PCA_DIMS = 100           # number of principal components stored in the cache


def _ssl_context():
    from rlcourse.leaderboard import _ssl_context as ctx     # same CA-certificate fallback
    return ctx()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(path: Path) -> None:
    """Raise if `path` is not a usable MNIST file."""
    with np.load(path) as z:
        for split in ("train", "test"):
            x, y = z[f"x_{split}"], z[f"y_{split}"]
            if x.ndim != 3 or x.shape[1:] != (28, 28) or x.dtype != np.uint8 or len(x) != len(y):
                raise ValueError(f"{path}: unexpected x_{split}/y_{split} {x.shape} {x.dtype}")


def download(quiet: bool = False) -> Path:
    """Download mnist.npz into DATA_DIR unless it is there already. Returns its path."""
    if FILE.exists():
        return FILE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".part")
    errors = []
    for url in URLS:
        try:
            if not quiet:
                print(f"Downloading MNIST from {url} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "rlcourse/1"})
            with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r, \
                    open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            _check(tmp)
            digest = _sha256(tmp)
            if digest not in KNOWN_SHA256 and not quiet:
                print(f"  note: unknown checksum {digest[:12]}...; the file looks fine, using it anyway.")
            tmp.replace(FILE)
            if not quiet:
                print(f"  saved to {FILE}")
            return FILE
        except Exception as e:                       # try the next mirror
            errors.append(f"{url}: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Could not download MNIST:\n  " + "\n  ".join(errors)
                       + f"\nDownload mnist.npz by hand from one of these URLs and put it in {DATA_DIR}.")


@lru_cache(maxsize=1)
def load() -> dict[str, np.ndarray]:
    """{'x_train', 'y_train', 'x_test', 'y_test'}: 28x28 uint8 images (0-255) and labels 0-9."""
    with np.load(download()) as z:
        return {k: z[k] for k in ("x_train", "y_train", "x_test", "y_test")}


@lru_cache(maxsize=1)
def _pca() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, principal components (rows) and their variances, fitted on the training images.

    Uses the images only, never the labels. Cached in DATA_DIR/mnist_pca.npz."""
    cache = DATA_DIR / "mnist_pca.npz"
    source = f"{FILE.stat().st_size}:{FILE.stat().st_mtime_ns}"
    if cache.exists():
        with np.load(cache) as z:
            if str(z["source"]) == source:
                return z["mean"], z["components"], z["variances"]
    x = load()["x_train"].reshape(-1, 784).astype(np.float64) / 255.0
    mean = x.mean(0)
    xc = x - mean
    var, vec = np.linalg.eigh(xc.T @ xc / len(x))      # ascending eigenvalues
    var, comps = var[::-1][:PCA_DIMS], vec[:, ::-1][:, :PCA_DIMS].T
    comps *= np.sign(comps[np.arange(PCA_DIMS), np.abs(comps).argmax(1)])[:, None]  # fix signs
    np.savez(cache, mean=mean, components=comps, variances=var, source=source)
    return mean, comps, var


def features(kind: str = "pca", dim: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Context vectors for the training and test images, as float32 arrays.

    kind="pixels": the 784 pixel intensities, scaled to [0, 1].
    kind="pca":    the first `dim` principal components of the images, scaled so that
                   the mean squared length of a training vector is 1, followed by a
                   constant 1 (a bias / intercept term): shape (n, dim + 1).
    """
    d = load()
    x_tr = d["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    x_te = d["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    if kind == "pixels":
        return x_tr, x_te
    if kind != "pca":
        raise ValueError(f"unknown feature kind {kind!r} (use 'pca' or 'pixels')")
    if not 1 <= dim <= PCA_DIMS:
        raise ValueError(f"dim must be between 1 and {PCA_DIMS}")
    mean, comps, var = _pca()
    proj = comps[:dim].T / np.sqrt(var[:dim].sum())

    def f(x):
        z = (x.astype(np.float64) - mean) @ proj
        return np.hstack([z, np.ones((len(z), 1))]).astype(np.float32)
    return f(x_tr), f(x_te)


if __name__ == "__main__":
    try:
        download()
    except RuntimeError as e:
        sys.exit(str(e))
    d = load()
    print(f"MNIST OK: {len(d['x_train'])} training and {len(d['x_test'])} test images in {FILE}")
    print(f"sha256 {_sha256(FILE)}")
    features("pca")
    print("PCA features OK.")
