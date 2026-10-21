"""The data pin, the code pin and the fetch helper must agree."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tooluse_rl.data import TAU2_BENCH_COMMIT, TAU2_BENCH_REPO, data_root, tau2_data_dir

ROOT = Path(__file__).resolve().parents[1]


def test_lockfile_pins_same_tau2_commit_as_data_module():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    tau2 = next(p for p in lock["package"] if p["name"] == "tau2")
    source = tau2["source"]["git"]
    assert source.startswith(TAU2_BENCH_REPO.removesuffix(".git"))
    assert source.endswith("#" + TAU2_BENCH_COMMIT), source


def test_pyproject_pins_same_commit():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["uv"]["sources"]["tau2"]["rev"] == TAU2_BENCH_COMMIT


def test_commit_looks_like_a_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", TAU2_BENCH_COMMIT)


def test_data_dir_layout():
    assert tau2_data_dir() == data_root() / "tau2-bench" / "data"
