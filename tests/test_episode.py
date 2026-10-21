from __future__ import annotations

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason

from tooluse_rl.env.domain import AGENT_GREETING, task_by_id
from tooluse_rl.env.episode import run_episode
from tooluse_rl.env.policies import MockPolicy
from tooluse_rl.env.users import MockUser
from tooluse_rl.eval.mock_scripts import SCRIPTS

from .conftest import requires_data

pytestmark = requires_data


def test_episode_structure_mirrors_tau2():
    script = SCRIPTS["cancel_single_order_success"]
    task = task_by_id("retail", script.task_id)
    episode = run_episode(
        task, script.make_policy(), script.make_user(), max_steps=script.max_steps
    )

    assert episode.termination_reason == TerminationReason.USER_STOP
    assert isinstance(episode.messages[0], AssistantMessage)
    assert episode.messages[0].content == AGENT_GREETING
    assert isinstance(episode.messages[1], UserMessage)
    assert (
        isinstance(episode.messages[-1], UserMessage)
        and "###STOP###" in episode.messages[-1].content
    )

    # every tool call is answered by a tool message with the same id, immediately after it
    for i, message in enumerate(episode.messages):
        if isinstance(message, AssistantMessage) and message.is_tool_call():
            for j, call in enumerate(message.tool_calls):
                result = episode.messages[i + 1 + j]
                assert isinstance(result, ToolMessage) and result.id == call.id
    assert [m.turn_idx for m in episode.messages] == list(range(len(episode.messages)))
    assert (
        episode.steps == len(episode.messages) - 1
    )  # greeting is not a step; 1 tool call per batch
    assert episode.tool_errors == 0
    assert episode.final_db_hash and episode.policy_name == "mock-policy"


def test_max_steps_termination():
    task = task_by_id("retail", "88")
    episode = run_episode(
        task,
        MockPolicy([], exhausted="One moment."),
        MockUser([], exhausted_reply="ok?"),
        max_steps=6,
    )
    assert episode.termination_reason == TerminationReason.MAX_STEPS
    assert episode.steps == 6


def test_too_many_errors_termination():
    task = task_by_id("retail", "88")
    policy = MockPolicy([{"call": "not_a_tool", "args": {}}] * 10)
    episode = run_episode(
        task, policy, MockUser(["help"], exhausted_reply="still waiting"), max_errors=3
    )
    assert episode.termination_reason == TerminationReason.TOO_MANY_ERRORS
    assert episode.tool_errors == 3


def test_agent_error_on_empty_message():
    class Empty:
        name = "empty"

        def reset(self, *a, **k):
            pass

        def act(self, conversation):
            return AssistantMessage(role="assistant", content="")

    episode = run_episode(task_by_id("retail", "88"), Empty(), MockUser(["hi"]))
    assert episode.termination_reason == TerminationReason.AGENT_ERROR
