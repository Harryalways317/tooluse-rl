from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse_rl.data import has_tau2_data

FIXTURES_DIR = Path(__file__).parent / "fixtures"

requires_data = pytest.mark.skipif(
    not has_tau2_data(),
    reason="tau2-bench data not fetched: run `uv run tooluse-rl-fetch-data`",
)


def fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


def load_fixture(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def fixtures() -> dict[str, dict]:
    return {p.stem: load_fixture(p) for p in fixture_paths()}
