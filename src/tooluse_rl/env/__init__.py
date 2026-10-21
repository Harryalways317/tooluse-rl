"""Framework-agnostic tau2-bench environment layer.

Everything the baseline evaluator, the fixture recorder and the verifiers
adapter share: deterministic environment construction, the policy/user
protocols with mock implementations, the half-duplex episode loop, the reward
and the pass^k metrics.
"""

from tooluse_rl.env.domain import (
    AGENT_GREETING,
    DOMAINS,
    HELD_OUT_DOMAIN,
    TRAIN_DOMAIN,
    execute_tool_call,
    fresh_env,
    load_tasks,
    policy_system_prompt,
    task_by_id,
    task_splits,
    tool_schemas,
)
from tooluse_rl.env.episode import Episode, run_episode
from tooluse_rl.env.metrics import EpisodeResult, Summary, markdown_table, pass_hat_k, summarize
from tooluse_rl.env.policies import MockPolicy, NullPolicy, OpenAIPolicy, Policy, policy_from_env
from tooluse_rl.env.reward import FormatSignals, RewardBreakdown, compute_reward, evaluate_task
from tooluse_rl.env.users import STOP, MockUser, OpenAIUser, User, is_stop_message, user_from_env

__all__ = [
    "AGENT_GREETING",
    "DOMAINS",
    "HELD_OUT_DOMAIN",
    "STOP",
    "TRAIN_DOMAIN",
    "Episode",
    "EpisodeResult",
    "FormatSignals",
    "MockPolicy",
    "MockUser",
    "NullPolicy",
    "OpenAIPolicy",
    "OpenAIUser",
    "Policy",
    "RewardBreakdown",
    "Summary",
    "User",
    "compute_reward",
    "evaluate_task",
    "execute_tool_call",
    "fresh_env",
    "is_stop_message",
    "load_tasks",
    "markdown_table",
    "pass_hat_k",
    "policy_from_env",
    "policy_system_prompt",
    "run_episode",
    "summarize",
    "task_by_id",
    "task_splits",
    "tool_schemas",
    "user_from_env",
]
