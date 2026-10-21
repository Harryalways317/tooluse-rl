from __future__ import annotations

import pytest

from tooluse_rl.env.metrics import EpisodeResult, markdown_table, pass_hat_k, summarize


def _result(task: str, trial: int, success: bool, steps: int = 10) -> EpisodeResult:
    return EpisodeResult(task, trial, success, 1.0 if success else 0.0, steps, 0, "user_stop")


@pytest.mark.parametrize(
    ("trials", "successes", "k", "expected"),
    [
        (4, 4, 4, 1.0),
        (4, 4, 1, 1.0),
        (4, 2, 1, 0.5),
        (4, 2, 4, 0.0),
        (4, 3, 2, 0.5),
        (1, 1, 1, 1.0),
    ],
)
def test_pass_hat_k(trials, successes, k, expected):
    assert pass_hat_k(trials, successes, k) == pytest.approx(expected)


def test_pass_hat_k_needs_enough_trials():
    with pytest.raises(ValueError):
        pass_hat_k(2, 2, 4)


def test_summarize_pass_k_over_tasks():
    results = [
        *(_result("a", t, True) for t in range(4)),  # always passes
        *(_result("b", t, t < 2) for t in range(4)),  # passes half the time
    ]
    summary = summarize(results)
    assert summary.num_tasks == 2 and summary.num_trials == 4
    assert summary.pass_hat[1] == pytest.approx((1.0 + 0.5) / 2)
    assert summary.pass_hat[4] == pytest.approx((1.0 + 0.0) / 2)
    assert summary.avg_task_success == pytest.approx(0.75)


def test_summarize_reports_none_when_too_few_trials():
    summary = summarize([_result("a", 0, True), _result("a", 1, False)])
    assert summary.pass_hat[1] == pytest.approx(0.5)
    assert summary.pass_hat[4] is None


def test_summarize_empty():
    summary = summarize([])
    assert summary.num_tasks == 0 and summary.pass_hat == {1: None, 4: None}


def test_markdown_table():
    table = markdown_table(
        [{"model": "x", "pass^1": 0.5, "pass^4": None}, {"model": "y", "pass^1": 1.0}]
    )
    lines = table.splitlines()
    assert lines[0] == "| model | pass^1 | pass^4 |"
    assert lines[1].startswith("|")
    assert "| x | 0.500 | — |" in lines[2]
    assert "| y | 1.000 | — |" in lines[3]
