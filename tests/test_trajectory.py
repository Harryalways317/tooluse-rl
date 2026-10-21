from __future__ import annotations

from tooluse_rl.env.domain import task_by_id
from tooluse_rl.env.episode import run_episode
from tooluse_rl.env.reward import compute_reward
from tooluse_rl.eval.mock_scripts import SCRIPTS
from tooluse_rl.trajectory import (
    SCHEMA,
    episode_to_record,
    messages_from_record,
    messages_to_dicts,
    read_jsonl,
    record_to_episode,
    write_jsonl,
)

from .conftest import requires_data

pytestmark = requires_data


def test_record_round_trip(tmp_path):
    script = SCRIPTS["hallucinated_tool_then_recovers"]
    task = task_by_id(script.domain, script.task_id)
    episode = run_episode(task, script.make_policy(), script.make_user())
    reward = compute_reward(episode, task)
    record = episode_to_record(episode, reward, run_id="t", policy_model="p", user_model="u")

    assert record["schema"] == SCHEMA and record["task_id"] == "88" and record["domain"] == "retail"
    assert record["policy_model"] == "p" and record["user_model"] == "u"
    assert record["reward"]["task_success"] == 1.0 and record["tool_errors"] == 1
    assert record["messages"][0] == {
        "role": "assistant",
        "content": "Hi! How can I help you today?",
        "turn_idx": 0,
    }
    error_entry = next(m for m in record["messages"] if m["role"] == "tool" and m["error"])
    assert error_entry["content"].startswith("Error:")

    path = tmp_path / "out.jsonl"
    assert write_jsonl(path, [record, record]) == 2
    loaded = list(read_jsonl(path))
    assert len(loaded) == 2 and loaded[0] == record

    rebuilt = record_to_episode(loaded[0])
    assert messages_to_dicts(rebuilt.messages) == record["messages"]
    assert messages_to_dicts(messages_from_record(record)) == messages_to_dicts(episode.messages)
    assert rebuilt.termination_reason == episode.termination_reason
    assert compute_reward(rebuilt, task).total == reward.total
