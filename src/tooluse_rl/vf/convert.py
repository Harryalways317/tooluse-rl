"""verifiers trace messages -> tau2 messages, so tau2's evaluator can grade a rollout."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import verifiers.v1 as vf
from tau2.data_model.message import AssistantMessage, Message, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason
from verifiers.v1.types import content_text

from tooluse_rl.env.users import STOP, is_stop_message

__all__ = ["TraceEpisode", "episode_from_trace", "vf_messages_to_tau2"]


def _arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return parsed if isinstance(parsed, dict) else {"__invalid_json__": raw}


def vf_messages_to_tau2(messages: Sequence[vf.Message]) -> list[Message]:
    """Drop system messages, keep user/assistant/tool. Tool errors are recognised by tau2's ``Error:`` prefix."""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, vf.SystemMessage):
            continue
        if isinstance(m, vf.UserMessage):
            out.append(UserMessage(role="user", content=content_text(m.content)))
        elif isinstance(m, vf.AssistantMessage):
            calls = (
                [
                    ToolCall(id=tc.id, name=tc.name, arguments=_arguments(tc.arguments))
                    for tc in m.tool_calls
                ]
                if m.tool_calls
                else None
            )
            out.append(AssistantMessage(role="assistant", content=m.content, tool_calls=calls))
        elif isinstance(m, vf.ToolMessage):
            text = content_text(m.content)
            out.append(
                ToolMessage(
                    id=m.tool_call_id,
                    role="tool",
                    content=text,
                    requestor="assistant",
                    error=text.startswith("Error:"),
                )
            )
    return out


@dataclass
class TraceEpisode:
    """The ``EpisodeLike`` view of a verifiers trace."""

    domain: str
    task_id: str
    messages: list[Message]
    termination_reason: TerminationReason
    steps: int
    tool_errors: int
    trial: int = 0
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def episode_from_trace(
    trace: vf.Trace, *, domain: str, task_id: str, done_marker: str = STOP
) -> TraceEpisode:
    messages = vf_messages_to_tau2(trace.messages)
    stopped = any(
        isinstance(m, UserMessage) and (done_marker in (m.content or "") or is_stop_message(m))
        for m in messages
    )
    return TraceEpisode(
        domain=domain,
        task_id=task_id,
        messages=messages,
        termination_reason=TerminationReason.USER_STOP if stopped else TerminationReason.MAX_STEPS,
        steps=len(messages),
        tool_errors=sum(1 for m in messages if isinstance(m, ToolMessage) and m.error),
    )
