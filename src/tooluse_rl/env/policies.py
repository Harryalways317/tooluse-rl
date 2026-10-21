"""Policies (the agent under evaluation or training).

``OpenAIPolicy`` talks to any OpenAI-compatible endpoint with native tool
calling (vLLM, OpenRouter, OpenAI). ``MockPolicy`` replays a script and
``NullPolicy`` always sends the same text; both exist so the whole pipeline runs
offline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task

__all__ = [
    "MockPolicy",
    "NullPolicy",
    "OpenAIPolicy",
    "Policy",
    "Step",
    "policy_from_env",
    "to_openai_messages",
]


class Policy(Protocol):
    name: str

    def reset(self, task: Task, system_prompt: str, tools: list[dict]) -> None: ...

    def act(self, conversation: list[Message]) -> AssistantMessage: ...


@dataclass
class Step:
    """One scripted agent turn: a text reply or one or more tool calls (never both)."""

    say: str | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        if "say" in data:
            return cls(say=data["say"])
        if "call" in data:
            return cls(calls=[(data["call"], dict(data.get("args") or {}))])
        if "calls" in data:
            return cls(calls=[(c["name"], dict(c.get("args") or {})) for c in data["calls"]])
        raise ValueError(f"step needs 'say', 'call' or 'calls': {data!r}")

    def to_dict(self) -> dict[str, Any]:
        if self.say is not None:
            return {"say": self.say}
        if len(self.calls) == 1:
            name, args = self.calls[0]
            return {"call": name, "args": args}
        return {"calls": [{"name": n, "args": a} for n, a in self.calls]}


class MockPolicy:
    """Replays ``steps``; once exhausted keeps replying ``exhausted`` (text)."""

    name = "mock-policy"

    def __init__(
        self,
        steps: Sequence[Step | dict[str, Any]] = (),
        *,
        exhausted: str = "Is there anything else I can help you with?",
    ) -> None:
        self.steps = [s if isinstance(s, Step) else Step.from_dict(s) for s in steps]
        self.exhausted = exhausted
        self._cursor = 0
        self._call_counter = 0

    def reset(self, task: Task, system_prompt: str, tools: list[dict]) -> None:
        self._cursor = 0
        self._call_counter = 0

    def act(self, conversation: list[Message]) -> AssistantMessage:
        step = (
            self.steps[self._cursor] if self._cursor < len(self.steps) else Step(say=self.exhausted)
        )
        self._cursor += 1
        if step.calls:
            tool_calls = []
            for name, args in step.calls:
                self._call_counter += 1
                tool_calls.append(
                    ToolCall(id=f"call_{self._call_counter}", name=name, arguments=args)
                )
            return AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)
        return AssistantMessage(role="assistant", content=step.say)


class NullPolicy:
    """Always replies the same text and never calls a tool (the floor of the benchmark)."""

    name = "null-policy"

    def __init__(self, reply: str = "I'm sorry, I'm not able to help with that.") -> None:
        self.reply = reply

    def reset(self, task: Task, system_prompt: str, tools: list[dict]) -> None:
        return None

    def act(self, conversation: list[Message]) -> AssistantMessage:
        return AssistantMessage(role="assistant", content=self.reply)


def to_openai_messages(conversation: list[Message]) -> list[dict[str, Any]]:
    """tau2 messages -> OpenAI chat-completions message dicts (no extra keys)."""
    out: list[dict[str, Any]] = []
    for message in conversation:
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.content or ""})
        elif isinstance(message, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.is_tool_call():
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in message.tool_calls
                ]
            out.append(entry)
        elif isinstance(message, ToolMessage):
            out.append(
                {"role": "tool", "tool_call_id": message.id, "content": message.content or ""}
            )
        else:
            raise TypeError(f"unsupported message type {type(message).__name__}")
    return out


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as a dict; malformed JSON is kept so the env reports an error."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return parsed if isinstance(parsed, dict) else {"__invalid_json__": raw}


class OpenAIPolicy:
    """An agent served by any OpenAI-compatible endpoint with native tool calling."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self._system_prompt = ""
        self._tools: list[dict] = []

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def reset(self, task: Task, system_prompt: str, tools: list[dict]) -> None:
        self._system_prompt = system_prompt
        self._tools = tools

    def act(self, conversation: list[Message]) -> AssistantMessage:
        messages = [
            {"role": "system", "content": self._system_prompt},
            *to_openai_messages(conversation),
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        tool_calls = None
        if getattr(choice, "tool_calls", None):
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_arguments(tc.function.arguments),
                )
                for tc in choice.tool_calls
            ]
        usage = getattr(response, "usage", None)
        return AssistantMessage(
            role="assistant",
            content=choice.content,
            tool_calls=tool_calls,
            usage=(
                {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
                if usage is not None
                else None
            ),
        )


def policy_from_env(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> OpenAIPolicy:
    """Build an ``OpenAIPolicy`` from ``POLICY_MODEL`` / ``POLICY_BASE_URL`` / ``POLICY_API_KEY``.

    ``POLICY_BASE_URL`` falls back to ``OPENAI_BASE_URL`` and ``POLICY_API_KEY``
    to ``OPENAI_API_KEY`` so a single local vLLM can serve both roles.
    """
    from openai import OpenAI

    model = model or os.environ.get("POLICY_MODEL")
    if not model:
        raise OSError("set POLICY_MODEL (or pass model=) to use a policy model")
    client = OpenAI(
        base_url=base_url
        or os.environ.get("POLICY_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None,
        api_key=api_key
        or os.environ.get("POLICY_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY",
    )
    return OpenAIPolicy(client, model, **kwargs)
