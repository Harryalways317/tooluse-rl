"""Simulated users.

``OpenAIUser`` is tau2's text user simulator (same system prompt, same role
flipping, same ``###STOP###`` convention) driven through any OpenAI-compatible
chat endpoint, so a cheap API model or a local vLLM can play the customer.
``MockUser`` is a scripted stand-in for tests and offline smoke runs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.tasks import Task
from tau2.user.guidelines import simulation_guidelines
from tau2.user.user_simulator import build_user_system_prompt
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER

__all__ = [
    "STOP",
    "MockUser",
    "OpenAIUser",
    "User",
    "flip_roles",
    "is_stop_message",
    "user_from_env",
    "user_system_prompt",
]

STOP_MARKERS: tuple[str, ...] = (STOP, TRANSFER, OUT_OF_SCOPE)


def is_stop_message(message: UserMessage) -> bool:
    """tau2's ``UserSimulator.is_stop``: any of the three end-of-call markers in a text reply."""
    if message.is_tool_call() or message.content is None:
        return False
    return any(marker in message.content for marker in STOP_MARKERS)


class User(Protocol):
    """A simulated customer. ``respond`` sees the whole conversation so far."""

    name: str

    def reset(self, task: Task) -> None: ...

    def respond(self, conversation: list[Message]) -> UserMessage: ...


class MockUser:
    """Deterministic scripted user.

    Replies with ``lines`` in order; once they are exhausted it replies
    ``exhausted_reply`` (default ``###STOP###``, which ends the episode).
    """

    name = "mock-user"

    def __init__(self, lines: Sequence[str] = (), *, exhausted_reply: str = STOP) -> None:
        self.lines = list(lines)
        self.exhausted_reply = exhausted_reply
        self._cursor = 0

    def reset(self, task: Task) -> None:
        self._cursor = 0

    def respond(self, conversation: list[Message]) -> UserMessage:
        if self._cursor < len(self.lines):
            content = self.lines[self._cursor]
        else:
            content = self.exhausted_reply
        self._cursor += 1
        return UserMessage(role="user", content=content)


def user_system_prompt(task: Task) -> str:
    """The exact tau2 text-mode user-simulator prompt for ``task``.

    Global simulation guidelines (from the English language pack) + the task's
    persona and structured instructions, assembled by tau2's own builder.
    """
    persona = PersonaConfig()
    guidelines = simulation_guidelines(persona, mode="text", use_tools=False)
    return build_user_system_prompt(
        str(task.user_scenario), persona_config=persona, mode="text", guidelines=guidelines
    )


def flip_roles(conversation: list[Message]) -> list[dict[str, Any]]:
    """The conversation as the user model sees it (tau2 ``UserState.flip_roles``).

    Assistant text becomes the ``user`` role, the simulated user's own lines
    become ``assistant``; tool calls and tool results are not shown to the user.
    """
    flipped: list[dict[str, Any]] = []
    for message in conversation:
        if isinstance(message, AssistantMessage):
            if not message.is_tool_call() and message.has_content():
                flipped.append({"role": "user", "content": message.content})
        elif isinstance(message, UserMessage) and message.content is not None:
            flipped.append({"role": "assistant", "content": message.content})
    return flipped


class OpenAIUser:
    """tau2's user simulator over an OpenAI-compatible ``chat.completions`` endpoint."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt: str | None = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def reset(self, task: Task) -> None:
        self._system_prompt = user_system_prompt(task)

    def respond(self, conversation: list[Message]) -> UserMessage:
        if self._system_prompt is None:
            raise RuntimeError("call reset(task) before respond()")
        messages = [{"role": "system", "content": self._system_prompt}, *flip_roles(conversation)]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0].message
        return UserMessage(
            role="user",
            content=choice.content or "",
            usage=_usage_dict(getattr(response, "usage", None)),
        )


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def user_from_env(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> OpenAIUser:
    """Build an ``OpenAIUser`` from ``USER_SIM_MODEL``, ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``."""
    from openai import OpenAI

    model = model or os.environ.get("USER_SIM_MODEL")
    if not model:
        raise OSError("set USER_SIM_MODEL (or pass model=) to use a simulated-user model")
    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
    )
    return OpenAIUser(client, model, **kwargs)
