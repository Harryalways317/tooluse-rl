from __future__ import annotations

from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from tooluse_rl.env.users import STOP, MockUser, OpenAIUser, flip_roles, is_stop_message

from .conftest import requires_data


def test_mock_user_replays_then_stops():
    user = MockUser(["hello", "my email is x@y.z"])
    user.reset(task=None)
    assert user.respond([]).content == "hello"
    assert user.respond([]).content == "my email is x@y.z"
    stop = user.respond([])
    assert stop.content == STOP and is_stop_message(stop)


def test_mock_user_custom_exhausted_reply_keeps_going():
    user = MockUser([], exhausted_reply="still here")
    user.reset(task=None)
    assert not is_stop_message(user.respond([]))


@pytest.mark.parametrize("marker", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"])
def test_stop_markers(marker):
    assert is_stop_message(UserMessage(role="user", content=f"ok thanks {marker}"))
    assert not is_stop_message(UserMessage(role="user", content="ok thanks"))


def test_flip_roles_hides_tool_traffic_and_swaps_roles():
    conversation = [
        AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
        UserMessage(role="user", content="cancel my order"),
        AssistantMessage(
            role="assistant", tool_calls=[ToolCall(id="c1", name="get_order_details", arguments={})]
        ),
        ToolMessage(id="c1", role="tool", content="{}", requestor="assistant"),
        AssistantMessage(role="assistant", content="Sure, please confirm."),
    ]
    assert flip_roles(conversation) == [
        {"role": "user", "content": "Hi! How can I help you today?"},
        {"role": "assistant", "content": "cancel my order"},
        {"role": "user", "content": "Sure, please confirm."},
    ]


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )


@requires_data
def test_openai_user_uses_tau2_prompt_and_flipped_roles():
    from tooluse_rl.env.domain import task_by_id

    task = task_by_id("retail", "88")
    client = _FakeClient("I'd like to change my bookshelf order.")
    user = OpenAIUser(client, "fake-model")
    user.reset(task)
    reply = user.respond(
        [AssistantMessage(role="assistant", content="Hi! How can I help you today?")]
    )
    assert reply.content == "I'd like to change my bookshelf order."
    assert reply.usage == {"prompt_tokens": 12, "completion_tokens": 3}
    sent = client.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert (
        "<scenario>" in sent[0]["content"] and "daiki.silva6295@example.com" in sent[0]["content"]
    )
    assert STOP in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "Hi! How can I help you today?"}
    assert client.calls[0]["model"] == "fake-model"


def test_user_from_env_requires_model(monkeypatch):
    from tooluse_rl.env.users import user_from_env

    monkeypatch.delenv("USER_SIM_MODEL", raising=False)
    with pytest.raises(OSError, match="USER_SIM_MODEL"):
        user_from_env()
