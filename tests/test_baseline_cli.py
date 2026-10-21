from __future__ import annotations

from tooluse_rl.eval.baseline import main
from tooluse_rl.eval.mock_scripts import SCRIPTS
from tooluse_rl.trajectory import read_jsonl

from .conftest import requires_data

pytestmark = requires_data


def test_mock_run_prints_pass_k_table(capsys, tmp_path):
    out = tmp_path / "mock.jsonl"
    assert main(["--mock", "--trials", "2", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "| script |" in text and "pass^1" in text and "pass^4" in text
    for name in SCRIPTS:
        assert name in text
    assert "ALL (mock)" in text
    records = list(read_jsonl(out))
    assert len(records) == 2 * len(SCRIPTS)
    assert {r["task_id"] for r in records} == {"88", "24"}


def test_null_policy_mock_user_smoke(capsys, tmp_path):
    out = tmp_path / "null.jsonl"
    assert (
        main(
            [
                "--policy",
                "null",
                "--user",
                "mock",
                "--split",
                "test",
                "--n-tasks",
                "2",
                "--trials",
                "1",
                "--out",
                str(out),
                "--per-task",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "| policy | user | domain | split |" in text
    assert "null-policy" in text and "mock-user" in text
    records = list(read_jsonl(out))
    assert len(records) == 2 and all(r["termination_reason"] == "user_stop" for r in records)
