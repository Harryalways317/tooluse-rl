"""The half-duplex episode loop (a pluggable re-implementation of tau2's ``Orchestrator``).

Turn structure, termination reasons and step counting mirror tau2 so that
trajectories recorded here grade identically under tau2's evaluator:

* the agent opens with the greeting, the user replies;
* an agent turn is either text (-> user) or tool calls (-> environment -> agent);
* the episode ends when the user emits ``###STOP###`` (``user_stop``), when
  ``max_steps`` messages have been exchanged (``max_steps``) or after
  ``max_errors`` failed tool calls (``too_many_errors``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tau2.data_model.message import AssistantMessage, Message, ToolMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.utils.utils import get_now

from tooluse_rl.env.domain import (
    AGENT_GREETING,
    TRAIN_DOMAIN,
    db_hash,
    execute_tool_call,
    fresh_env,
    policy_system_prompt,
    tool_schemas,
)
from tooluse_rl.env.policies import Policy
from tooluse_rl.env.users import User, is_stop_message

#: tau2's ``Orchestrator`` default; its CLI default is 200.
DEFAULT_MAX_STEPS = 100
DEFAULT_MAX_ERRORS = 10


@dataclass
class Episode:
    domain: str
    task_id: str
    messages: list[Message]
    termination_reason: TerminationReason
    steps: int
    tool_errors: int
    trial: int = 0
    seed: int | None = None
    policy_name: str = ""
    user_name: str = ""
    system_prompt: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0
    final_db_hash: str | None = None
    policy_usage: dict[str, int] = field(default_factory=dict)
    user_usage: dict[str, int] = field(default_factory=dict)

    @property
    def assistant_messages(self) -> list[AssistantMessage]:
        return [m for m in self.messages if isinstance(m, AssistantMessage)]

    @property
    def tool_messages(self) -> list[ToolMessage]:
        return [m for m in self.messages if isinstance(m, ToolMessage)]


def _add_usage(total: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, int | float):
            total[key] = total.get(key, 0) + int(value)


def run_episode(
    task: Task,
    policy: Policy,
    user: User,
    *,
    domain: str = TRAIN_DOMAIN,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    trial: int = 0,
    seed: int | None = None,
    greeting: str = AGENT_GREETING,
) -> Episode:
    """Run one task to termination and return the full message trajectory."""
    env = fresh_env(domain, task)
    system_prompt = policy_system_prompt(env)
    policy.reset(task, system_prompt, tool_schemas(env))
    user.reset(task)

    started_at = get_now()
    t0 = time.perf_counter()
    messages: list[Message] = [AssistantMessage(role="assistant", content=greeting, turn_idx=0)]
    turn = 1
    steps = 0
    errors = 0
    policy_usage: dict[str, int] = {}
    user_usage: dict[str, int] = {}
    to_user = True  # the greeting is addressed to the user
    reason: TerminationReason

    while True:
        if steps >= max_steps:
            reason = TerminationReason.MAX_STEPS
            break
        if errors >= max_errors:
            reason = TerminationReason.TOO_MANY_ERRORS
            break
        if to_user:
            user_message = user.respond(messages)
            user_message.turn_idx = turn
            turn += 1
            messages.append(user_message)
            steps += 1
            _add_usage(user_usage, user_message.usage)
            if is_stop_message(user_message):
                reason = TerminationReason.USER_STOP
                break
            to_user = False
            continue

        agent_message = policy.act(messages)
        agent_message.turn_idx = turn
        turn += 1
        messages.append(agent_message)
        steps += 1
        _add_usage(policy_usage, agent_message.usage)
        if not (agent_message.has_content() or agent_message.is_tool_call()):
            reason = TerminationReason.AGENT_ERROR
            break
        if agent_message.is_tool_call():
            results = []
            for call in agent_message.tool_calls:
                result = execute_tool_call(env, call)
                result.turn_idx = turn
                turn += 1
                results.append(result)
                errors += int(result.error)
            messages.extend(results)
            steps += 1  # the environment's reply is one step, as in tau2
        else:
            to_user = True

    return Episode(
        domain=domain,
        task_id=task.id,
        messages=messages,
        termination_reason=reason,
        steps=steps,
        tool_errors=errors,
        trial=trial,
        seed=seed,
        policy_name=policy.name,
        user_name=user.name,
        system_prompt=system_prompt,
        started_at=started_at,
        ended_at=get_now(),
        duration_s=time.perf_counter() - t0,
        final_db_hash=db_hash(env),
        policy_usage=policy_usage,
        user_usage=user_usage,
    )
