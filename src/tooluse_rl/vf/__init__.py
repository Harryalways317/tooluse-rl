"""verifiers v1 adapter: tau2 retail/airline as a verifiers taskset + user-sim env.

``verifiers`` resolves the taskset id ``tooluse-rl`` to the ``tooluse_rl``
package, which re-exports these two classes lazily. ``Tau2Taskset`` loads the
tasks, ``Tau2UserSimEnv`` runs the two-agent conversation (assistant under
evaluation, simulated user) with tau2's user prompt and stop marker, and the
per-rollout tool server in :mod:`tooluse_rl.vf.servers` exposes the domain
tools over MCP.
"""

from tooluse_rl.vf.env import Tau2UserSimEnv, Tau2UserSimEnvConfig
from tooluse_rl.vf.taskset import (
    Tau2Task,
    Tau2TaskConfig,
    Tau2TaskData,
    Tau2Taskset,
    Tau2TasksetConfig,
)

__all__ = [
    "Tau2Task",
    "Tau2TaskConfig",
    "Tau2TaskData",
    "Tau2Taskset",
    "Tau2TasksetConfig",
    "Tau2UserSimEnv",
    "Tau2UserSimEnvConfig",
]
