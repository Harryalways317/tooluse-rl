"""JSONL trajectory records: one JSON object per episode.

Schema ``tooluse-rl/trajectory/v1`` (see ``docs/notes.md`` for the field list).
Messages use OpenAI chat shapes (``role``/``content``/``tool_calls``/
``tool_call_id``) plus tau2's ``error`` flag on tool results, so a record can be
re-graded by tau2's evaluator (``messages_from_record``) and read by any
OpenAI-style tooling. Keep this in sync with the Terminal-Bench harness repo if
the two ever share an eval-table generator.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import TerminationReason

from tooluse_rl.env.episode import Episode
from tooluse_rl.env.reward import RewardBreakdown

SCHEMA = "tooluse-rl/trajectory/v1"

__all__ = [
    "SCHEMA",
    "episode_to_record",
    "messages_from_record",
    "messages_to_dicts",
    "read_jsonl",
    "record_to_episode",
    "write_jsonl",
]


def messages_to_dicts(messages: Iterable[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content, "turn_idx": m.turn_idx})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content, "turn_idx": m.turn_idx})
        elif isinstance(m, AssistantMessage):
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": m.content,
                "turn_idx": m.turn_idx,
            }
            if m.is_tool_call():
                entry["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
                ]
            out.append(entry)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.id,
                    "content": m.content,
                    "error": bool(m.error),
                    "turn_idx": m.turn_idx,
                }
            )
        else:
            raise TypeError(f"unsupported message type {type(m).__name__}")
    return out


def messages_from_record(record: dict[str, Any]) -> list[Message]:
    messages: list[Message] = []
    for entry in record["messages"]:
        role = entry["role"]
        turn_idx = entry.get("turn_idx")
        if role == "system":
            messages.append(
                SystemMessage(role="system", content=entry.get("content"), turn_idx=turn_idx)
            )
        elif role == "user":
            messages.append(
                UserMessage(role="user", content=entry.get("content"), turn_idx=turn_idx)
            )
        elif role == "assistant":
            calls = entry.get("tool_calls")
            messages.append(
                AssistantMessage(
                    role="assistant",
                    content=entry.get("content"),
                    tool_calls=(
                        [
                            ToolCall(id=c["id"], name=c["name"], arguments=dict(c["arguments"]))
                            for c in calls
                        ]
                        if calls
                        else None
                    ),
                    turn_idx=turn_idx,
                )
            )
        elif role == "tool":
            messages.append(
                ToolMessage(
                    id=entry["tool_call_id"],
                    role="tool",
                    content=entry.get("content"),
                    requestor="assistant",
                    error=bool(entry.get("error", False)),
                    turn_idx=turn_idx,
                )
            )
        else:
            raise ValueError(f"unknown role {role!r}")
    return messages


def episode_to_record(
    episode: Episode,
    reward: RewardBreakdown | None = None,
    *,
    run_id: str | None = None,
    policy_model: str | None = None,
    user_model: str | None = None,
    cost_usd: float | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "id": uuid.uuid4().hex,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark": "tau2-bench",
        "domain": episode.domain,
        "task_id": str(episode.task_id),
        "trial": episode.trial,
        "seed": episode.seed,
        "policy_model": policy_model or episode.policy_name,
        "user_model": user_model or episode.user_name,
        "system_prompt": episode.system_prompt,
        "messages": messages_to_dicts(episode.messages),
        "termination_reason": episode.termination_reason.value,
        "steps": episode.steps,
        "tool_errors": episode.tool_errors,
        "final_db_hash": episode.final_db_hash,
        "reward": reward.to_dict() if reward is not None else None,
        "usage": {"policy": episode.policy_usage, "user": episode.user_usage},
        "cost_usd": cost_usd,
        "duration_s": round(episode.duration_s, 3),
    }


def record_to_episode(record: dict[str, Any]) -> Episode:
    """Rebuild an ``Episode`` (for re-grading) from a record."""
    return Episode(
        domain=record["domain"],
        task_id=str(record["task_id"]),
        messages=messages_from_record(record),
        termination_reason=TerminationReason(record["termination_reason"]),
        steps=int(record["steps"]),
        tool_errors=int(record.get("tool_errors", 0)),
        trial=int(record.get("trial", 0)),
        seed=record.get("seed"),
        policy_name=record.get("policy_model") or "",
        user_name=record.get("user_model") or "",
        system_prompt=record.get("system_prompt") or "",
        final_db_hash=record.get("final_db_hash"),
        policy_usage=dict((record.get("usage") or {}).get("policy") or {}),
        user_usage=dict((record.get("usage") or {}).get("user") or {}),
    )


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
