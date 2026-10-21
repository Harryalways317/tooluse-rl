"""The reward on the recorded fixtures and on a live re-run must match the hand-written expectations."""

from __future__ import annotations

import pytest

from tooluse_rl.env.domain import task_by_id
from tooluse_rl.env.episode import run_episode
from tooluse_rl.env.reward import compute_reward, evaluate_task, format_signals
from tooluse_rl.eval.mock_scripts import SCRIPTS
from tooluse_rl.trajectory import record_to_episode

from .conftest import fixture_paths, load_fixture, requires_data

pytestmark = requires_data

FIXTURES = [pytest.param(load_fixture(p), id=p.stem) for p in fixture_paths()]


def test_fixtures_exist_for_every_script():
    assert {p.stem for p in fixture_paths()} == set(SCRIPTS), "run scripts/record_fixtures.py"


def _check(reward, episode, expected):
    assert reward.task_success == pytest.approx(expected["task_success"])
    assert reward.format_score == pytest.approx(expected["format_score"])
    assert reward.total == pytest.approx(expected["total"])
    assert reward.termination_reason == expected["termination_reason"]
    assert episode.tool_errors == expected["tool_errors"]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_recorded_episode_regrades_to_expected(fixture):
    episode = record_to_episode(fixture["record"])
    task = task_by_id(fixture["domain"], fixture["task_id"])
    _check(compute_reward(episode, task), episode, fixture["expected"])


@pytest.mark.parametrize("fixture", FIXTURES)
def test_live_rerun_matches_recording(fixture):
    script = SCRIPTS[fixture["name"]]
    task = task_by_id(script.domain, script.task_id)
    episode = run_episode(
        task, script.make_policy(), script.make_user(), max_steps=script.max_steps
    )
    reward = compute_reward(episode, task)
    _check(reward, episode, fixture["expected"])
    assert episode.final_db_hash == fixture["record"]["final_db_hash"]
    assert len(episode.messages) == len(fixture["record"]["messages"])


def test_task_success_is_binary_and_shaping_is_bounded():
    for fixture in (load_fixture(p) for p in fixture_paths()):
        reward = fixture["record"]["reward"]
        assert reward["task_success"] in (0.0, 1.0)
        assert 0.0 <= reward["format_score"] <= 1.0
        assert reward["total"] == pytest.approx(
            reward["task_success"] + 0.1 * reward["format_score"]
        )


def test_retail_uses_db_only_and_skips_the_llm_judge():
    fixture = load_fixture(
        next(p for p in fixture_paths() if p.stem == "cancel_single_order_success")
    )
    episode = record_to_episode(fixture["record"])
    task = task_by_id("retail", fixture["task_id"])
    info = evaluate_task(episode, task)
    assert info.db_check is not None and info.db_check.db_match is True
    assert info.nl_assertions is None  # the judge did not run
    assert compute_reward(episode, task).evaluation_type == "env"


def test_format_signals_components():
    fixture = load_fixture(
        next(p for p in fixture_paths() if p.stem == "hallucinated_tool_then_recovers")
    )
    signals = format_signals(record_to_episode(fixture["record"]))
    assert signals.clean_termination and signals.no_mixed_turns and signals.within_step_budget
    assert not signals.no_tool_errors
    assert signals.score == pytest.approx(0.75)

    fixture = load_fixture(next(p for p in fixture_paths() if p.stem == "runs_out_of_steps"))
    signals = format_signals(record_to_episode(fixture["record"]))
    assert not signals.clean_termination and signals.no_tool_errors


def test_premature_termination_scores_zero_without_replay():
    fixture = load_fixture(
        next(p for p in fixture_paths() if p.stem == "cancel_single_order_success")
    )
    record = dict(fixture["record"], termination_reason="too_many_errors")
    episode = record_to_episode(record)
    task = task_by_id("retail", fixture["task_id"])
    assert compute_reward(episode, task).task_success == 0.0
