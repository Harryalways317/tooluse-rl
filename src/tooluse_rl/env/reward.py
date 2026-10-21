"""Reward = task success (tau2's evaluator) + light format shaping.

Task success is tau2's own verdict: the environment DB after replaying the
agent's tool calls must equal the DB after replaying the task's gold actions
(plus any env assertions, and for airline the substring communicate check).
It is 1.0 or 0.0.

What we deliberately do *not* run by default is tau2 2.0's LLM-judged
``NL_ASSERTION`` component, which is in the reward basis of 112/114 retail
tasks. It costs an API call per assertion and is non-deterministic, so it is
unusable as an RL reward; pass ``with_nl_judge=True`` at eval time to get the
official number. Airline (``DB`` x ``COMMUNICATE``) is fully offline.

Format shaping is a small, bounded bonus (default weight 0.1) so a well-formed
failure can never outrank a success. The signals are the ones that make tau2
episodes go wrong in practice: hallucinated / malformed tool calls, mixing text
and tool calls in one turn (the policy prompt forbids it), not ending the
conversation, and running long.
"""

from __future__ import annotations

import contextlib
import io
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tau2.data_model.message import AssistantMessage, Message, ToolMessage
from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.utils.utils import get_now

from tooluse_rl.env.domain import _base_db

__all__ = [
    "DEFAULT_SHAPING_WEIGHT",
    "DEFAULT_STEP_BUDGET",
    "EpisodeLike",
    "FormatSignals",
    "RewardBreakdown",
    "compute_reward",
    "evaluate_task",
    "format_signals",
]

DEFAULT_SHAPING_WEIGHT = 0.1
DEFAULT_STEP_BUDGET = 40

#: Offline communicate scoring (tau2 1.x behaviour). tau2 2.0 defaults to an LLM judge.
OFFLINE_COMMUNICATE = CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)


class EpisodeLike(Protocol):
    """What the reward needs; satisfied by ``Episode`` and by verifiers traces."""

    domain: str
    task_id: str
    messages: list[Message]
    termination_reason: TerminationReason
    steps: int
    tool_errors: int


def to_simulation_run(episode: EpisodeLike) -> SimulationRun:
    now = get_now()
    return SimulationRun(
        id=uuid.uuid4().hex,
        task_id=str(episode.task_id),
        start_time=getattr(episode, "started_at", "") or now,
        end_time=getattr(episode, "ended_at", "") or now,
        duration=float(getattr(episode, "duration_s", 0.0) or 0.0),
        termination_reason=episode.termination_reason,
        messages=list(episode.messages),
        trial=getattr(episode, "trial", None),
        seed=getattr(episode, "seed", None),
    )


def evaluation_type_for(task: Task, *, with_nl_judge: bool) -> EvaluationType:
    basis = set(task.evaluation_criteria.reward_basis) if task.evaluation_criteria else set()
    if with_nl_judge:
        return EvaluationType.ALL
    if RewardType.NL_ASSERTION in basis:
        return EvaluationType.ENV  # retail: DB (+ env assertions) only, judge skipped
    return EvaluationType.ALL  # airline: DB x substring communicate, offline


def evaluate_task(
    episode: EpisodeLike,
    task: Task,
    *,
    with_nl_judge: bool = False,
    strict_replay: bool = True,
) -> RewardInfo:
    """tau2's ``RewardInfo`` for the episode (``reward`` is 1.0 or 0.0)."""
    # tau2's airline tools print reservation JSON to stdout while the evaluator
    # replays gold actions; keep that noise out of CLI tables and logs.
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_simulation(
            to_simulation_run(episode),
            task,
            evaluation_type_for(task, with_nl_judge=with_nl_judge),
            solo_mode=False,
            domain=episode.domain,
            env_kwargs={"db": _base_db(episode.domain)},
            strict_replay=strict_replay,
            communicate_scoring=OFFLINE_COMMUNICATE,
        )


@dataclass
class FormatSignals:
    clean_termination: bool
    no_tool_errors: bool
    no_mixed_turns: bool
    within_step_budget: bool

    @property
    def score(self) -> float:
        values = [
            self.clean_termination,
            self.no_tool_errors,
            self.no_mixed_turns,
            self.within_step_budget,
        ]
        return sum(values) / len(values)


def format_signals(
    episode: EpisodeLike, *, step_budget: int = DEFAULT_STEP_BUDGET
) -> FormatSignals:
    assistant = [m for m in episode.messages if isinstance(m, AssistantMessage)]
    tool_results = [m for m in episode.messages if isinstance(m, ToolMessage)]
    mixed = any(m.is_tool_call() and m.has_content() for m in assistant)
    return FormatSignals(
        clean_termination=episode.termination_reason
        in {TerminationReason.USER_STOP, TerminationReason.AGENT_STOP},
        no_tool_errors=not any(m.error for m in tool_results),
        no_mixed_turns=not mixed,
        within_step_budget=episode.steps <= step_budget,
    )


@dataclass
class RewardBreakdown:
    total: float
    task_success: float
    format_score: float
    shaping_weight: float
    db_match: bool | None
    termination_reason: str
    evaluation_type: str
    reward_basis: list[str]
    signals: FormatSignals
    tau2_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["signals"] = asdict(self.signals)
        return data


def compute_reward(
    episode: EpisodeLike,
    task: Task,
    *,
    shaping_weight: float = DEFAULT_SHAPING_WEIGHT,
    step_budget: int = DEFAULT_STEP_BUDGET,
    with_nl_judge: bool = False,
    strict_replay: bool = True,
) -> RewardBreakdown:
    """``task_success + shaping_weight * format_score``."""
    info = evaluate_task(episode, task, with_nl_judge=with_nl_judge, strict_replay=strict_replay)
    signals = format_signals(episode, step_budget=step_budget)
    success = float(info.reward)
    return RewardBreakdown(
        total=success + shaping_weight * signals.score,
        task_success=success,
        format_score=signals.score,
        shaping_weight=shaping_weight,
        db_match=info.db_check.db_match if info.db_check is not None else None,
        termination_reason=episode.termination_reason.value,
        evaluation_type=evaluation_type_for(task, with_nl_judge=with_nl_judge).value,
        reward_basis=[r.value for r in (info.reward_basis or [])],
        signals=signals,
        tau2_info=info.info,
    )


def is_success(reward: float) -> bool:
    """tau2's success test (``reward == 1`` up to float noise)."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def successes(rewards: Sequence[float]) -> int:
    return sum(is_success(r) for r in rewards)
