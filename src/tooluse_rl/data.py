"""Locate (and fetch) the tau2-bench data directory.

The ``tau2`` wheel ships code only. Its data (domain DBs, policies, task sets and
the English language pack the user simulator prompt is built from) lives in the
repository's ``data/`` tree, which tau2 finds through ``TAU2_DATA_DIR``.

We do not vendor that data (about 10 MB of JSON for retail + airline). Instead
``fetch_tau2_data`` does a sparse, pinned checkout into ``data/tau2-bench`` and
``ensure_tau2_data_dir`` exports ``TAU2_DATA_DIR`` before tau2 is imported.
The commit is the same one ``uv.lock`` pins the ``tau2`` package to, so code and
data cannot drift apart (``tests/test_data.py`` checks this).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TAU2_BENCH_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_BENCH_COMMIT = "7233d6a7a7c9b0e0ba551da2105d0b60661aefe5"
# Only what we need: the two domains and the English pack (user-sim guidelines).
TAU2_SPARSE_PATHS = (
    "/data/tau2/domains/retail/",
    "/data/tau2/domains/airline/",
    "/data/tau2/multilingual/en/",
)
_CHECKOUT_DIRNAME = "tau2-bench"
_SENTINEL = Path("tau2") / "domains" / "retail" / "db.json"


def repo_root() -> Path:
    """The repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Where fetched datasets live: ``$TOOLUSE_RL_DATA_DIR`` or ``<repo>/data``."""
    override = os.environ.get("TOOLUSE_RL_DATA_DIR")
    return Path(override).expanduser() if override else repo_root() / "data"


def tau2_data_dir() -> Path:
    """The directory ``TAU2_DATA_DIR`` should point at (contains ``tau2/domains/...``)."""
    return data_root() / _CHECKOUT_DIRNAME / "data"


def has_tau2_data(path: Path | None = None) -> bool:
    env = os.environ.get("TAU2_DATA_DIR")
    candidates = [path] if path else ([Path(env)] if env else []) + [tau2_data_dir()]
    return any((c / _SENTINEL).exists() for c in candidates)


def ensure_tau2_data_dir() -> Path | None:
    """Export ``TAU2_DATA_DIR`` for tau2 if the pinned checkout exists.

    Returns the directory in use, or ``None`` when no data is available (tau2
    will then warn at import time and fail on first use).
    """
    if os.environ.get("TAU2_DATA_DIR"):
        return Path(os.environ["TAU2_DATA_DIR"])
    local = tau2_data_dir()
    if (local / _SENTINEL).exists():
        os.environ["TAU2_DATA_DIR"] = str(local)
        return local
    return None


def require_tau2_data() -> Path:
    found = ensure_tau2_data_dir()
    if found is None or not (found / _SENTINEL).exists():
        raise FileNotFoundError(
            "tau2-bench data not found. Run `uv run tooluse-rl-fetch-data` "
            f"(fetches commit {TAU2_BENCH_COMMIT[:12]} into {tau2_data_dir()}) "
            "or set TAU2_DATA_DIR to a tau2-bench `data/` directory."
        )
    return found


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def fetch_tau2_data(
    dest: Path | None = None, commit: str = TAU2_BENCH_COMMIT, *, force: bool = False
) -> Path:
    """Sparse-checkout the needed tau2-bench data paths at ``commit``.

    Uses a blobless partial clone so only the selected files are downloaded.
    Returns the resulting ``data`` directory.
    """
    root = (dest or data_root()) / _CHECKOUT_DIRNAME
    data_dir = root / "data"
    stamp = root / "COMMIT"
    if (data_dir / _SENTINEL).exists() and not force:
        if stamp.exists() and stamp.read_text().strip() == commit:
            return data_dir
        raise RuntimeError(
            f"{root} exists but is not at commit {commit[:12]}; re-run with --force to replace it."
        )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _git(["init", "-q"], root)
    _git(["remote", "add", "origin", TAU2_BENCH_REPO], root)
    _git(["sparse-checkout", "set", "--no-cone", *TAU2_SPARSE_PATHS], root)
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], root)
    _git(["checkout", "-q", "FETCH_HEAD"], root)
    stamp.write_text(commit + "\n")
    if not (data_dir / _SENTINEL).exists():
        raise RuntimeError(f"fetch succeeded but {data_dir / _SENTINEL} is missing")
    return data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the pinned tau2-bench data directory.")
    parser.add_argument(
        "--dest", type=Path, default=None, help=f"data root (default: {data_root()})"
    )
    parser.add_argument("--commit", default=TAU2_BENCH_COMMIT)
    parser.add_argument("--force", action="store_true", help="replace an existing checkout")
    args = parser.parse_args(argv)
    path = fetch_tau2_data(args.dest, args.commit, force=args.force)
    print(f"tau2 data ready at {path} (commit {args.commit[:12]})")
    print(f"export TAU2_DATA_DIR={path}  # optional; tooluse_rl sets it automatically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
