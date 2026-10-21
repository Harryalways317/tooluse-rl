"""Baseline evaluation CLI.

Runs a policy (any OpenAI-compatible endpoint, e.g. vLLM) against a simulated
user (any OpenAI-compatible endpoint) on N tau2 tasks for T trials and prints
pass^1 / pass^4 as a Markdown table. ``--mock`` runs the hand-written scripts
in :mod:`tooluse_rl.eval.mock_scripts` with no network at all.

Examples::

    uv run tooluse-rl-eval --mock
    uv run tooluse-rl-eval --policy null --user mock --domain retail --split test --n-tasks 5
    POLICY_MODEL=Qwen/Qwen3-8B POLICY_BASE_URL=http://localhost:8000/v1 \\
    USER_SIM_MODEL=gpt-4.1-mini OPENAI_API_KEY=... \\
      uv run tooluse-rl-eval --domain retail --split test --n-tasks 40 --trials 4 --out outputs/base.jsonl
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from tooluse_rl.env.domain import load_tasks, task_by_id
from tooluse_rl.env.episode import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS, run_episode
from tooluse_rl.env.metrics import EpisodeResult, markdown_table, summarize
from tooluse_rl.env.policies import NullPolicy, Policy, policy_from_env
from tooluse_rl.env.reward import compute_reward, is_success
from tooluse_rl.env.users import MockUser, User, user_from_env
from tooluse_rl.eval.mock_scripts import SCRIPTS
from tooluse_rl.trajectory import episode_to_record, write_jsonl

__all__ = ["build_parser", "main", "run"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tooluse-rl-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mock", action="store_true", help="run the scripted mock episodes (offline)")
    p.add_argument("--domain", default="retail", choices=["retail", "airline"])
    p.add_argument("--split", default="test", help="train | test | base (all)")
    p.add_argument("--n-tasks", type=int, default=5, help="first N tasks of the split (0 = all)")
    p.add_argument(
        "--task-ids", nargs="*", default=None, help="explicit task ids (overrides --n-tasks)"
    )
    p.add_argument("--trials", type=int, default=4, help="trials per task (pass^4 needs >= 4)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS)
    p.add_argument(
        "--policy",
        default="openai",
        choices=["openai", "null"],
        help="openai = OpenAI-compatible endpoint",
    )
    p.add_argument("--policy-model", default=None, help="default: $POLICY_MODEL")
    p.add_argument(
        "--policy-base-url", default=None, help="default: $POLICY_BASE_URL or $OPENAI_BASE_URL"
    )
    p.add_argument("--policy-temperature", type=float, default=0.0)
    p.add_argument(
        "--user",
        default="openai",
        choices=["openai", "mock"],
        help="mock = replies ###STOP### immediately",
    )
    p.add_argument("--user-model", default=None, help="default: $USER_SIM_MODEL")
    p.add_argument("--user-base-url", default=None, help="default: $OPENAI_BASE_URL")
    p.add_argument(
        "--with-nl-judge",
        action="store_true",
        help="also run tau2's LLM-judged NL assertions (paid)",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="write one JSONL trajectory record per episode"
    )
    p.add_argument("--per-task", action="store_true", help="also print a per-task table")
    return p


def _make_policy(args: argparse.Namespace) -> Policy:
    if args.policy == "null":
        return NullPolicy()
    return policy_from_env(
        model=args.policy_model, base_url=args.policy_base_url, temperature=args.policy_temperature
    )


def _make_user(args: argparse.Namespace) -> User:
    if args.user == "mock":
        return MockUser()
    return user_from_env(model=args.user_model, base_url=args.user_base_url)


def _run_one(
    task, policy, user, args, *, domain: str, trial: int, max_steps: int, run_id: str, label: str
):
    episode = run_episode(
        task,
        policy,
        user,
        domain=domain,
        max_steps=max_steps,
        max_errors=args.max_errors,
        trial=trial,
    )
    reward = compute_reward(episode, task, with_nl_judge=args.with_nl_judge)
    result = EpisodeResult(
        task_id=label,
        trial=trial,
        success=is_success(reward.task_success),
        reward=reward.total,
        steps=episode.steps,
        tool_errors=episode.tool_errors,
        termination_reason=episode.termination_reason.value,
        policy_prompt_tokens=episode.policy_usage.get("prompt_tokens", 0),
        policy_completion_tokens=episode.policy_usage.get("completion_tokens", 0),
    )
    record = episode_to_record(episode, reward, run_id=run_id)
    return result, record


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[EpisodeResult]]:
    run_id = uuid.uuid4().hex[:8]
    results: list[EpisodeResult] = []
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    if args.mock:
        for script in SCRIPTS.values():
            task = task_by_id(script.domain, script.task_id)
            script_results: list[EpisodeResult] = []
            for trial in range(args.trials):
                result, record = _run_one(
                    task,
                    script.make_policy(),
                    script.make_user(),
                    args,
                    domain=script.domain,
                    trial=trial,
                    max_steps=script.max_steps,
                    run_id=run_id,
                    label=script.name,
                )
                script_results.append(result)
                records.append(record)
            results.extend(script_results)
            summary = summarize(script_results)
            rows.append(
                summary.as_row(script=script.name, task=f"{script.domain}/{script.task_id}")
            )
        rows.append(summarize(results).as_row(script="ALL (mock)", task="—"))
        return rows, records, results

    policy = _make_policy(args)
    user = _make_user(args)
    tasks = load_tasks(args.domain, args.split)
    if args.task_ids:
        wanted = {str(t) for t in args.task_ids}
        tasks = [t for t in tasks if t.id in wanted]
    elif args.n_tasks:
        tasks = tasks[: args.n_tasks]
    for task in tasks:
        for trial in range(args.trials):
            result, record = _run_one(
                task,
                policy,
                user,
                args,
                domain=args.domain,
                trial=trial,
                max_steps=args.max_steps,
                run_id=run_id,
                label=task.id,
            )
            results.append(result)
            records.append(record)
    summary = summarize(results)
    rows.append(
        summary.as_row(policy=policy.name, user=user.name, domain=args.domain, split=args.split)
    )
    return rows, records, results


def per_task_rows(results: list[EpisodeResult]) -> list[dict[str, Any]]:
    by_task: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    return [
        {
            "task": task_id,
            "successes": f"{sum(r.success for r in rs)}/{len(rs)}",
            "avg reward": sum(r.reward for r in rs) / len(rs),
            "avg steps": sum(r.steps for r in rs) / len(rs),
            "terminations": ", ".join(sorted({r.termination_reason for r in rs})),
        }
        for task_id, rs in by_task.items()
    ]


def _quiet_tau2_logs() -> None:
    """tau2 logs tool internals at INFO through loguru; keep only warnings and above."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _quiet_tau2_logs()
    rows, records, results = run(args)
    print(markdown_table(rows))
    if args.per_task:
        print()
        print(markdown_table(per_task_rows(results)))
    if args.out is not None:
        n = write_jsonl(args.out, records)
        print(f"\nwrote {n} trajectory records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
