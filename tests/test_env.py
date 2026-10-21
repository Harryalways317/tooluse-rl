from __future__ import annotations

import pytest
from tau2.data_model.message import ToolCall

from tooluse_rl.env.domain import (
    DOMAINS,
    execute_tool_call,
    fresh_env,
    load_tasks,
    policy_system_prompt,
    task_by_id,
    task_splits,
    tool_schemas,
)

from .conftest import requires_data

pytestmark = requires_data

RETAIL_TOOLS = {
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_item_details",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}


def test_task_splits_match_the_benchmark():
    assert {k: len(v) for k, v in task_splits("retail").items()} == {
        "train": 74,
        "test": 40,
        "base": 114,
    }
    assert {k: len(v) for k, v in task_splits("airline").items()} == {
        "train": 30,
        "test": 20,
        "base": 50,
    }
    assert len(load_tasks("retail", "train")) == 74
    assert len(load_tasks("airline", None)) == 50


def test_train_and_test_are_disjoint():
    splits = task_splits("retail")
    assert not set(splits["train"]) & set(splits["test"])


def test_task_by_id():
    assert task_by_id("retail", "88").id == "88"
    with pytest.raises(KeyError):
        task_by_id("retail", "does-not-exist")


def test_unknown_domain_rejected():
    with pytest.raises(ValueError):
        fresh_env("telecom")
    assert DOMAINS == ("retail", "airline")


def test_fresh_env_is_deterministic_and_isolated():
    a, b = fresh_env("retail"), fresh_env("retail")
    pristine = a.get_db_hash()
    assert pristine == b.get_db_hash()

    result = execute_tool_call(
        a,
        ToolCall(
            id="c1",
            name="cancel_pending_order",
            arguments={"order_id": "#W8835847", "reason": "no longer needed"},
        ),
    )
    assert not result.error
    assert a.get_db_hash() != pristine, "the write must change the DB"
    assert b.get_db_hash() == pristine, "the sibling env must not see the write"
    assert fresh_env("retail").get_db_hash() == pristine, "a new env must start pristine again"


def test_tool_schemas_are_openai_function_tools():
    schemas = tool_schemas(fresh_env("retail"))
    assert {s["function"]["name"] for s in schemas} == RETAIL_TOOLS
    cancel = next(s for s in schemas if s["function"]["name"] == "cancel_pending_order")
    assert cancel["type"] == "function"
    assert set(cancel["function"]["parameters"]["required"]) == {"order_id", "reason"}


def test_policy_system_prompt_wraps_the_domain_policy():
    env = fresh_env("retail")
    prompt = policy_system_prompt(env)
    assert prompt.startswith("<instructions>")
    assert "<policy>" in prompt and env.get_policy() in prompt


def test_tool_errors_are_messages_not_exceptions():
    env = fresh_env("retail")
    unknown = execute_tool_call(env, ToolCall(id="c1", name="cancel_order", arguments={}))
    assert unknown.error and unknown.content.startswith("Error:")
    bad_args = execute_tool_call(
        env, ToolCall(id="c2", name="get_order_details", arguments={"nope": 1})
    )
    assert bad_args.error
