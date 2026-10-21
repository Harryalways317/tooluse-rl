"""Hand-written scripted episodes on real tau2 retail tasks.

Each script pairs a ``MockPolicy`` with a ``MockUser`` on a *real* task so the
whole pipeline (environment, episode loop, tau2's replay-based evaluator, the
format shaping) runs offline. They serve three purposes:

* ``tooluse-rl-eval --mock`` runs them end to end;
* ``scripts/record_fixtures.py`` records them into ``tests/fixtures``;
* ``tests/test_reward.py`` asserts the recorded and the live reward agree with
  ``expected``.

The user/order ids come from the retail DB (``data/tau2/domains/retail/db.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tooluse_rl.env.policies import MockPolicy
from tooluse_rl.env.users import STOP, MockUser

__all__ = ["SCRIPTS", "MockScript"]


@dataclass
class MockScript:
    name: str
    description: str
    task_id: str
    policy_steps: list[dict[str, Any]]
    user_lines: list[str]
    expected: dict[str, Any]
    domain: str = "retail"
    max_steps: int = 100
    policy_exhausted: str = "Is there anything else I can help you with?"
    user_exhausted: str = STOP
    tags: list[str] = field(default_factory=list)

    def make_policy(self) -> MockPolicy:
        return MockPolicy(self.policy_steps, exhausted=self.policy_exhausted)

    def make_user(self) -> MockUser:
        return MockUser(self.user_lines, exhausted_reply=self.user_exhausted)

    def script_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy_steps,
            "user": self.user_lines,
            "policy_exhausted": self.policy_exhausted,
            "user_exhausted": self.user_exhausted,
        }


# Task 88 (retail/train): Daiki Silva wants the bookshelf in pending order #W8835847
# changed to 4 ft; that variant is unavailable, so the gold action is
# cancel_pending_order(#W8835847, "ordered by mistake").
_DAIKI_EMAIL = "daiki.silva6295@example.com"
_DAIKI_USER = [
    "Hi, I'd like to change the bookshelf in my pending order to the 4 ft height, same material and color.",
    _DAIKI_EMAIL,
    "Yes, please cancel the whole order. I ordered it by mistake.",
    "Thanks, that's all.",
]
_ASK_EMAIL = {
    "say": "I can help with that. Could you share the email address on your account so I can look you up?"
}
_LOOKUP = [
    {"call": "find_user_id_by_email", "args": {"email": _DAIKI_EMAIL}},
    {"call": "get_order_details", "args": {"order_id": "#W8835847"}},
    {"call": "get_product_details", "args": {"product_id": "8600330539"}},
    {
        "say": (
            "I checked order #W8835847: the 4 ft glass/white bookshelf is not available, so I "
            "can't modify it. Would you like me to cancel the whole order instead? Please confirm "
            "(yes/no) and the reason: 'no longer needed' or 'ordered by mistake'."
        )
    },
]
_DONE = {
    "say": (
        "Done: order #W8835847 is cancelled. The refund will go back to the original payment "
        "method within 5-7 business days. Anything else?"
    )
}


def _cancel(reason: str) -> dict[str, Any]:
    return {"call": "cancel_pending_order", "args": {"order_id": "#W8835847", "reason": reason}}


SCRIPTS: dict[str, MockScript] = {
    "cancel_single_order_success": MockScript(
        name="cancel_single_order_success",
        description="Looks the user up, confirms, cancels with the right reason: full reward.",
        task_id="88",
        policy_steps=[_ASK_EMAIL, *_LOOKUP, _cancel("ordered by mistake"), _DONE],
        user_lines=_DAIKI_USER,
        expected={
            "task_success": 1.0,
            "format_score": 1.0,
            "total": 1.1,
            "termination_reason": "user_stop",
            "tool_errors": 0,
        },
        tags=["success"],
    ),
    "cancel_wrong_reason_fails": MockScript(
        name="cancel_wrong_reason_fails",
        description="Same conversation but the cancel reason is wrong: the DB differs from gold, reward 0.",
        task_id="88",
        policy_steps=[_ASK_EMAIL, *_LOOKUP, _cancel("no longer needed"), _DONE],
        user_lines=_DAIKI_USER,
        expected={
            "task_success": 0.0,
            "format_score": 1.0,
            "total": 0.1,
            "termination_reason": "user_stop",
            "tool_errors": 0,
        },
        tags=["failure", "db-mismatch"],
    ),
    "hallucinated_tool_then_recovers": MockScript(
        name="hallucinated_tool_then_recovers",
        description="Calls a non-existent tool first (tool error), then completes the task: success minus shaping.",
        task_id="88",
        policy_steps=[
            _ASK_EMAIL,
            {"call": "cancel_order", "args": {"order_id": "#W8835847"}},
            *_LOOKUP,
            _cancel("ordered by mistake"),
            _DONE,
        ],
        user_lines=_DAIKI_USER,
        expected={
            "task_success": 1.0,
            "format_score": 0.75,
            "total": 1.075,
            "termination_reason": "user_stop",
            "tool_errors": 1,
        },
        tags=["success", "format-penalty"],
    ),
    # Task 24 (retail/train): Sofia Hernandez asks to cancel a grill but changes her mind when
    # asked to confirm. Gold actions are empty: the DB must be left untouched.
    "no_change_task_success": MockScript(
        name="no_change_task_success",
        description="Read-only conversation on a task whose correct outcome is to change nothing.",
        task_id="24",
        policy_steps=[
            {"say": "Sure. Could you give me your full name and zip code?"},
            {
                "call": "find_user_id_by_name_zip",
                "args": {"first_name": "Sofia", "last_name": "Hernandez", "zip": "98193"},
            },
            {"call": "get_user_details", "args": {"user_id": "sofia_hernandez_5364"}},
            {"call": "get_order_details", "args": {"order_id": "#W3561391"}},
            {
                "say": (
                    "I found pending order #W3561391 with a portable gas grill. To confirm: cancel "
                    "it and refund to the original payment method? (yes/no)"
                )
            },
            {"say": "Understood, I'll leave order #W3561391 as it is."},
        ],
        user_lines=[
            "Hi, I want to cancel the grill I ordered.",
            "Sofia Hernandez, zip 98193. I don't remember my email.",
            "Hmm, actually no. On second thought I'd like to keep the grill, never mind.",
            "Thanks, that's all.",
        ],
        expected={
            "task_success": 1.0,
            "format_score": 1.0,
            "total": 1.1,
            "termination_reason": "user_stop",
            "tool_errors": 0,
        },
        tags=["success", "no-op"],
    ),
    "runs_out_of_steps": MockScript(
        name="runs_out_of_steps",
        description="Agent stalls and the user never hangs up: max_steps termination, reward 0.",
        task_id="88",
        policy_steps=[],
        policy_exhausted="Let me check on that for you.",
        user_lines=["I want to cancel order #W8835847."],
        user_exhausted="Any update?",
        max_steps=12,
        expected={
            "task_success": 0.0,
            "format_score": 0.75,
            "total": 0.075,
            "termination_reason": "max_steps",
            "tool_errors": 0,
        },
        tags=["failure", "timeout"],
    ),
}
