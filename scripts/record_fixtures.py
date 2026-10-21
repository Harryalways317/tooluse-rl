#!/usr/bin/env python
"""Record the scripted mock episodes into tests/fixtures/<name>.json.

Re-run after changing a script in tooluse_rl.eval.mock_scripts or bumping the
tau2-bench pin. Fails loudly if a script no longer produces its expected reward.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from tooluse_rl.env.domain import task_by_id
from tooluse_rl.env.episode import run_episode
from tooluse_rl.env.reward import compute_reward
from tooluse_rl.eval.mock_scripts import SCRIPTS
from tooluse_rl.trajectory import episode_to_record

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS.values():
        task = task_by_id(script.domain, script.task_id)
        episode = run_episode(
            task,
            script.make_policy(),
            script.make_user(),
            domain=script.domain,
            max_steps=script.max_steps,
        )
        reward = compute_reward(episode, task)
        got = {
            "task_success": reward.task_success,
            "format_score": reward.format_score,
            "total": reward.total,
            "termination_reason": reward.termination_reason,
            "tool_errors": episode.tool_errors,
        }
        for key, want in script.expected.items():
            ok = (
                math.isclose(got[key], want, abs_tol=1e-9)
                if isinstance(want, float)
                else got[key] == want
            )
            if not ok:
                raise SystemExit(f"{script.name}: {key} = {got[key]!r}, expected {want!r}")
        fixture = {
            "name": script.name,
            "description": script.description,
            "domain": script.domain,
            "task_id": script.task_id,
            "max_steps": script.max_steps,
            "tags": script.tags,
            "script": script.script_dict(),
            "expected": script.expected,
            "record": episode_to_record(episode, reward, run_id="fixture"),
        }
        path = FIXTURES / f"{script.name}.json"
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
        print(
            f"wrote {path.relative_to(FIXTURES.parents[1])}: reward {reward.total:.3f} ({reward.termination_reason})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
