"""Deterministic construction of tau2-bench domain environments.

tau2's own ``get_environment`` re-reads ``db.json`` on every call. That is
deterministic but slow (2.8 MB for retail, 7 MB for airline), so we load each
domain DB once and hand tau2 a deep copy per episode. The result is identical:
every episode starts from the pristine DB, then the task's ``initial_state`` is
applied exactly as tau2's orchestrator does.
"""

from __future__ import annotations

from functools import cache

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.environment.db import DB
from tau2.environment.environment import Environment
from tau2.registry import registry

DOMAINS: tuple[str, ...] = ("retail", "airline")
TRAIN_DOMAIN = "retail"
HELD_OUT_DOMAIN = "airline"

#: tau2's default opener; the agent speaks first and the simulated user replies.
AGENT_GREETING = "Hi! How can I help you today?"


def _check_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain {domain!r}; expected one of {DOMAINS}")


def load_tasks(domain: str, split: str | None = "train") -> list[Task]:
    """Tasks for ``domain``. Splits: ``train`` / ``test`` / ``base`` (all); ``None`` = all."""
    _check_domain(domain)
    return registry.get_tasks_loader(domain)(split)


def task_splits(domain: str) -> dict[str, list[str]]:
    _check_domain(domain)
    return registry.get_task_splits_loader(domain)()


def task_by_id(domain: str, task_id: str) -> Task:
    for task in load_tasks(domain, None):
        if task.id == str(task_id):
            return task
    raise KeyError(f"no task {task_id!r} in domain {domain!r}")


@cache
def _base_db(domain: str) -> DB:
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.utils import RETAIL_DB_PATH

        return RetailDB.load(RETAIL_DB_PATH)
    if domain == "airline":
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.utils import AIRLINE_DB_PATH

        return FlightDB.load(AIRLINE_DB_PATH)
    raise ValueError(domain)


def fresh_env(domain: str, task: Task | None = None) -> Environment:
    """A new environment on a pristine copy of the domain DB.

    When ``task`` is given its ``initial_state`` (data patches, initialization
    actions, seeded message history) is applied, mirroring
    ``Orchestrator.initialize``. No retail or airline task currently defines one,
    but the evaluator replays it, so we apply it the same way.
    """
    _check_domain(domain)
    env = registry.get_env_constructor(domain)(db=_base_db(domain))
    if task is not None:
        apply_initial_state(env, task)
    return env


def apply_initial_state(env: Environment, task: Task) -> None:
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init is not None else None,
        initialization_actions=init.initialization_actions if init is not None else None,
        message_history=list(init.message_history or []) if init is not None else [],
    )


def tool_schemas(env: Environment) -> list[dict]:
    """OpenAI ``tools=[...]`` entries for the agent-side tools of ``env``."""
    return [tool.openai_schema for tool in env.get_tools()]


def policy_system_prompt(env: Environment) -> str:
    """The exact system prompt tau2's reference ``LLMAgent`` uses (instructions + policy)."""
    return SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=env.get_policy())


def execute_tool_call(env: Environment, call: ToolCall) -> ToolMessage:
    """Run one agent tool call. Errors become ``ToolMessage(error=True)``, never exceptions."""
    return env.get_response(call)


def db_hash(env: Environment) -> str | None:
    return env.get_db_hash()
