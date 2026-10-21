"""pass^k and run summaries.

``pass_hat_k`` is tau-bench's metric (arXiv 2406.12045): the probability that
``k`` independent trials of the same task *all* succeed, estimated from
``num_trials`` trials as C(successes, k) / C(num_trials, k). We import tau2's
implementation so the number is the benchmark's number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tau2.metrics.agent_metrics import pass_hat_k

__all__ = ["EpisodeResult", "Summary", "markdown_table", "pass_hat_k", "summarize"]


@dataclass
class EpisodeResult:
    task_id: str
    trial: int
    success: bool
    reward: float
    steps: int
    tool_errors: int
    termination_reason: str
    policy_prompt_tokens: int = 0
    policy_completion_tokens: int = 0


@dataclass
class Summary:
    num_tasks: int
    num_trials: int
    num_episodes: int
    pass_hat: dict[int, float | None]
    avg_reward: float
    avg_task_success: float
    avg_steps: float
    avg_tool_errors: float
    avg_policy_tokens: float
    termination_counts: dict[str, int] = field(default_factory=dict)

    def as_row(self, **extra: Any) -> dict[str, Any]:
        row: dict[str, Any] = dict(extra)
        row.update({"tasks": self.num_tasks, "trials": self.num_trials})
        for k, value in self.pass_hat.items():
            row[f"pass^{k}"] = value
        row.update(
            {
                "avg reward": self.avg_reward,
                "avg steps": self.avg_steps,
                "tool errors/ep": self.avg_tool_errors,
                "policy tokens/ep": self.avg_policy_tokens,
            }
        )
        return row


def summarize(results: Iterable[EpisodeResult], ks: Sequence[int] = (1, 4)) -> Summary:
    results = list(results)
    by_task: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)
    if not by_task:
        return Summary(0, 0, 0, {k: None for k in ks}, 0.0, 0.0, 0.0, 0.0, 0.0)

    trials_per_task = {t: len(rs) for t, rs in by_task.items()}
    num_trials = min(trials_per_task.values())
    pass_hat: dict[int, float | None] = {}
    for k in ks:
        if num_trials < k:
            pass_hat[k] = None
            continue
        # Use the first `num_trials` trials of every task so every task weighs the same.
        pass_hat[k] = sum(
            pass_hat_k(num_trials, sum(r.success for r in rs[:num_trials]), k)
            for rs in by_task.values()
        ) / len(by_task)

    n = len(results)
    return Summary(
        num_tasks=len(by_task),
        num_trials=num_trials,
        num_episodes=n,
        pass_hat=pass_hat,
        avg_reward=sum(r.reward for r in results) / n,
        avg_task_success=sum(r.success for r in results) / n,
        avg_steps=sum(r.steps for r in results) / n,
        avg_tool_errors=sum(r.tool_errors for r in results) / n,
        avg_policy_tokens=sum(r.policy_prompt_tokens + r.policy_completion_tokens for r in results)
        / n,
        termination_counts=dict(Counter(r.termination_reason for r in results)),
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:,.1f}"
    return str(value)


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render dict rows as a GitHub-flavoured Markdown table (union of keys, first-seen order)."""
    if not rows:
        return "(no rows)"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(" --- " for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)
