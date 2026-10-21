from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from tooluse_rl.env.policies import MockPolicy, NullPolicy, OpenAIPolicy, Step, to_openai_messages


def test_step_from_dict_variants():
    assert Step.from_dict({"say": "hi"}) == Step(say="hi")
    assert Step.from_dict({"call": "f", "args": {"a": 1}}) == Step(calls=[("f", {"a": 1})])
    assert Step.from_dict({"calls": [{"name": "f"}, {"name": "g", "args": {"b": 2}}]}) == Step(
        calls=[("f", {}), ("g", {"b": 2})]
    )
    with pytest.raises(ValueError):
        Step.from_dict({"nope": 1})


def test_mock_policy_replays_then_repeats_exhausted_line():
    policy = MockPolicy(
        [{"say": "hello"}, {"call": "get_order_details", "args": {"order_id": "#W1"}}],
        exhausted="bye",
    )
    policy.reset(task=None, system_prompt="", tools=[])
    first = policy.act([])
    assert first.content == "hello" and not first.is_tool_call()
    second = policy.act([])
    assert second.is_tool_call() and second.content is None
    assert second.tool_calls[0].name == "get_order_details"
    assert second.tool_calls[0].id == "call_1"
    assert policy.act([]).content == "bye"
    assert policy.act([]).content == "bye"


def test_null_policy_never_calls_tools():
    policy = NullPolicy("nope")
    message = policy.act([])
    assert message.content == "nope" and not message.is_tool_call()


def test_to_openai_messages_shapes():
    conversation = [
        UserMessage(role="user", content="hi"),
        AssistantMessage(
            role="assistant", tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})]
        ),
        ToolMessage(id="c1", role="tool", content="ok", requestor="assistant"),
        AssistantMessage(role="assistant", content="done"),
    ]
    out = to_openai_messages(conversation)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "f", "arguments": json.dumps({"a": 1})},
        }
    ]
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert out[3] == {"role": "assistant", "content": "done"}


class _FakeClient:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content, tool_calls=self.tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )


def _tc(id_, name, arguments):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def test_openai_policy_parses_tool_calls_and_sends_tools():
    client = _FakeClient(tool_calls=[_tc("x1", "get_order_details", '{"order_id": "#W1"}')])
    policy = OpenAIPolicy(client, "m")
    policy.reset(
        task=None,
        system_prompt="SYS",
        tools=[{"type": "function", "function": {"name": "get_order_details"}}],
    )
    message = policy.act([UserMessage(role="user", content="hi")])
    assert message.is_tool_call()
    assert message.tool_calls[0] == ToolCall(
        id="x1", name="get_order_details", arguments={"order_id": "#W1"}
    )
    assert message.usage == {"prompt_tokens": 5, "completion_tokens": 2}
    sent = client.calls[0]
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["tools"][0]["function"]["name"] == "get_order_details"


def test_openai_policy_keeps_malformed_arguments_for_the_env_to_reject():
    client = _FakeClient(tool_calls=[_tc("x1", "f", "{not json")])
    policy = OpenAIPolicy(client, "m")
    policy.reset(task=None, system_prompt="", tools=[])
    message = policy.act([])
    assert message.tool_calls[0].arguments == {"__invalid_json__": "{not json"}
