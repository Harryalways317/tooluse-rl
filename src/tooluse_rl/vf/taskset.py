"""tau2 tasks as a verifiers taskset.

``TaskData.prompt`` carries the *user's* scenario (persona + structured
instructions): ``UserSimEnv`` turns it into the simulated user's system prompt
and nulls it for the assistant, which only ever sees the domain policy in
``system_prompt`` plus the conversation. Scoring reads the trace and the task's
evaluation criteria, never the prompt, exactly as the user-sim contract asks.
"""

from __future__ import annotations

from typing import Any, Literal

import verifiers.v1 as vf

from tooluse_rl.env.domain import fresh_env, load_tasks, policy_system_prompt, task_by_id
from tooluse_rl.env.reward import (
    DEFAULT_SHAPING_WEIGHT,
    DEFAULT_STEP_BUDGET,
    evaluate_task,
    format_signals,
)
from tooluse_rl.vf.convert import episode_from_trace

Domain = Literal["retail", "airline"]


class Tau2TaskData(vf.TaskData):
    domain: Domain = "retail"
    task_id: str
    split: str = "train"
    reward_basis: list[str] = []
    num_gold_actions: int = 0


class Tau2TaskConfig(vf.TaskConfig):
    domain: Domain = "retail"
    tools: vf.ToolsetConfig = vf.ToolsetConfig()
    shaping_weight: float = DEFAULT_SHAPING_WEIGHT
    step_budget: int = DEFAULT_STEP_BUDGET
    with_nl_judge: bool = False
    """Run tau2's LLM-judged NL assertions too (paid, non-deterministic; eval only)."""


class Tau2Task(vf.Task[Tau2TaskData, vf.State, Tau2TaskConfig]):
    @property
    def key(self) -> str:
        return f"{self.data.domain}/{self.data.task_id}"

    @classmethod
    def toolsets(cls, config: Tau2TaskConfig) -> list[vf.Toolset]:
        if config.domain == "airline":
            from tooluse_rl.vf.servers.airline import Tau2AirlineToolset as toolset_cls
        else:
            from tooluse_rl.vf.servers.retail import Tau2RetailToolset as toolset_cls
        return [toolset_cls(config.tools)]

    def _episode(self, trace: vf.Trace):
        return episode_from_trace(trace, domain=self.data.domain, task_id=self.data.task_id)

    @vf.reward
    async def task_success(self, trace: vf.Trace) -> float:
        """tau2's verdict: DB end state (and env assertions / communicate checks) match gold."""
        task = task_by_id(self.data.domain, self.data.task_id)
        info = evaluate_task(self._episode(trace), task, with_nl_judge=self.config.with_nl_judge)
        trace.info["tau2_reward_info"] = info.model_dump(mode="json")
        return float(info.reward)

    @vf.reward(weight=DEFAULT_SHAPING_WEIGHT)
    async def format_score(self, trace: vf.Trace) -> float:
        """Light shaping in [0, 1]; weighted by ``DEFAULT_SHAPING_WEIGHT`` (config value is informational)."""
        return format_signals(self._episode(trace), step_budget=self.config.step_budget).score

    @vf.metric
    async def tool_errors(self, trace: vf.Trace) -> float:
        return float(self._episode(trace).tool_errors)

    @vf.metric
    async def num_messages(self, trace: vf.Trace) -> float:
        return float(len(self._episode(trace).messages))


class Tau2TasksetConfig(vf.TasksetConfig):
    domain: Domain = "retail"
    split: str = "train"
    """tau2 split name: ``train`` (74 retail / 30 airline), ``test`` (40 / 20) or ``base`` (all)."""
    task: Tau2TaskConfig = Tau2TaskConfig()


class Tau2Taskset(vf.Taskset[Tau2Task, Tau2TasksetConfig]):
    def load(self) -> list[Tau2Task]:
        domain = self.config.domain
        system_prompt = policy_system_prompt(fresh_env(domain))
        task_config = self.config.task.model_copy(update={"domain": domain})
        tasks: list[Tau2Task] = []
        for task in load_tasks(domain, self.config.split):
            criteria = task.evaluation_criteria
            data: dict[str, Any] = {
                "id": f"{domain}/{task.id}",
                "name": f"tau2-{domain}-{task.id}",
                "description": (task.description.purpose if task.description else None),
                "prompt": str(task.user_scenario),
                "system_prompt": system_prompt,
                "domain": domain,
                "task_id": str(task.id),
                "split": self.config.split or "base",
                "reward_basis": [r.value for r in (criteria.reward_basis if criteria else [])],
                "num_gold_actions": len(criteria.actions or []) if criteria else 0,
            }
            tasks.append(Tau2Task(Tau2TaskData(**data), task_config))
        return tasks
