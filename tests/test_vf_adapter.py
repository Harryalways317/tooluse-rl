"""Offline checks of the verifiers v1 adapter (no model, no MCP transport)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from tooluse_rl.env.domain import task_by_id
from tooluse_rl.env.reward import compute_reward
from tooluse_rl.vf import Tau2Taskset, Tau2TasksetConfig, Tau2UserSimEnv, Tau2UserSimEnvConfig
from tooluse_rl.vf.convert import episode_from_trace, vf_messages_to_tau2
from tooluse_rl.vf.toolset import tool_specs

from .conftest import fixture_paths, load_fixture, requires_data


def test_retail_tool_specs_need_no_data():
    specs = {s.name: s for s in tool_specs("retail")}
    assert len(specs) == 16
    assert [p.name for p in specs["cancel_pending_order"].params] == ["order_id", "reason"]
    assert specs["cancel_pending_order"].doc.startswith("Cancel a pending order")


@requires_data
def test_taskset_loads_tau2_splits():
    train = Tau2Taskset(Tau2TasksetConfig(domain="retail", split="train")).load()
    assert len(train) == 74
    task = next(t for t in train if t.data.task_id == "88")
    assert task.key == "retail/88" and task.data.id == "retail/88"
    assert "daiki.silva6295@example.com" in task.data.prompt_text  # the user's scenario
    assert task.data.system_prompt.startswith("<instructions>")  # the assistant's policy
    assert task.data.reward_basis == ["DB", "NL_ASSERTION"] and task.data.num_gold_actions == 1
    assert task.config.domain == "retail"
    assert [fn.__name__ for fn in task.hooks("reward")] == ["format_score", "task_success"]

    airline = Tau2Taskset(Tau2TasksetConfig(domain="airline", split="test")).load()
    assert len(airline) == 20 and airline[0].config.domain == "airline"


@requires_data
def test_toolset_registers_domain_tools_over_mcp():
    from mcp.server.mcpserver import MCPServer

    task = Tau2Taskset(Tau2TasksetConfig(domain="retail", split="train")).load()[0]
    (toolset,) = type(task).toolsets(task.config)
    assert type(toolset).__name__ == "Tau2RetailToolset"
    assert type(toolset).__module__ == "tooluse_rl.vf.servers.retail"  # launched as `python -m`
    server = MCPServer("test")
    toolset.register(server)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert len(tools) == 16
    assert set(tools["cancel_pending_order"].input_schema["required"]) == {"order_id", "reason"}
    assert (
        tools["modify_pending_order_items"].input_schema["properties"]["item_ids"]["type"]
        == "array"
    )


@requires_data
def test_toolset_serves_a_fresh_db_per_task():
    task = Tau2Taskset(Tau2TasksetConfig(domain="retail", split="train")).load()[0]
    (toolset,) = type(task).toolsets(task.config)
    asyncio.run(toolset.setup_task(SimpleNamespace(id="retail/88", task_id="88")))
    before = toolset._env.get_db_hash()
    out = toolset.cancel_pending_order(order_id="#W8835847", reason="no longer needed")
    assert json.loads(out)["status"] == "cancelled"
    assert toolset._env.get_db_hash() != before
    asyncio.run(toolset.setup_task(SimpleNamespace(id="retail/88", task_id="88")))
    assert toolset._env.get_db_hash() == before


@requires_data
def test_env_config_uses_tau2_user_prompt_and_stop_marker():
    config = Tau2UserSimEnvConfig()
    assert "{scenario}" in config.persona and "User Simulation Guidelines" in config.persona
    assert config.done_marker == "###STOP###"
    assert config.user.harness.id == "null" and config.assistant.harness.id == "null"
    assert config.user.max_turns == 30
    assert issubclass(Tau2UserSimEnv, vf.Env)


@requires_data
def test_plugin_resolution_by_taskset_id():
    from verifiers.v1.utils.loaders import environment_class, taskset_class

    assert taskset_class("tooluse-rl") is Tau2Taskset
    assert environment_class("tooluse-rl") is Tau2UserSimEnv


def _to_vf(messages: list[dict]) -> list[vf.Message]:
    out: list[vf.Message] = []
    for m in messages:
        if m["role"] == "user":
            out.append(vf.UserMessage(content=m["content"]))
        elif m["role"] == "assistant":
            calls = [
                vf.ToolCall(id=c["id"], name=c["name"], arguments=json.dumps(c["arguments"]))
                for c in m.get("tool_calls") or []
            ] or None
            out.append(vf.AssistantMessage(content=m["content"], tool_calls=calls))
        elif m["role"] == "tool":
            out.append(vf.ToolMessage(tool_call_id=m["tool_call_id"], content=m["content"]))
    return out


@requires_data
@pytest.mark.parametrize("path", fixture_paths(), ids=lambda p: p.stem)
def test_trace_messages_regrade_like_the_fixture(path):
    fixture = load_fixture(path)
    record = fixture["record"]
    trace = SimpleNamespace(
        messages=[vf.SystemMessage(content="policy"), *_to_vf(record["messages"])]
    )
    episode = episode_from_trace(trace, domain=fixture["domain"], task_id=fixture["task_id"])
    assert episode.termination_reason.value == fixture["expected"]["termination_reason"]
    assert episode.tool_errors == fixture["expected"]["tool_errors"]
    reward = compute_reward(episode, task_by_id(fixture["domain"], fixture["task_id"]))
    assert reward.task_success == pytest.approx(fixture["expected"]["task_success"])
    assert reward.format_score == pytest.approx(fixture["expected"]["format_score"])


def test_vf_messages_to_tau2_drops_system_and_flags_errors():
    messages = [
        vf.SystemMessage(content="s"),
        vf.UserMessage(content="hi"),
        vf.AssistantMessage(
            content=None, tool_calls=[vf.ToolCall(id="1", name="f", arguments="{bad")]
        ),
        vf.ToolMessage(tool_call_id="1", content="Error: nope"),
    ]
    out = vf_messages_to_tau2(messages)
    assert [type(m).__name__ for m in out] == ["UserMessage", "AssistantMessage", "ToolMessage"]
    assert out[1].tool_calls[0].arguments == {"__invalid_json__": "{bad"}
    assert out[2].error is True and out[2].requestor == "assistant"
