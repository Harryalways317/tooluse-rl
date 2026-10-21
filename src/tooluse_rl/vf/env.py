"""verifiers' built-in two-agent ``user-sim`` env, configured the tau2 way.

The user agent gets tau2's text user-simulator guidelines with the task's
scenario spliced into ``<scenario>...</scenario>`` and ends the call with
``###STOP###`` (tau2's marker) instead of the default ``###DONE###``. Both
roles default to the tool-less ``null`` harness: the assistant calls the
domain tools through the per-rollout MCP server, the user has no tools.
"""

from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field
from tau2.data_model.persona import PersonaConfig
from tau2.user.guidelines import simulation_guidelines
from tau2.user.user_simulator import build_user_system_prompt
from verifiers.v1.envs.user_sim import UserSimEnv, UserSimEnvConfig

from tooluse_rl.env.users import STOP

__all__ = ["Tau2UserSimEnv", "Tau2UserSimEnvConfig", "tau2_user_persona"]

#: tau2 caps a simulation at 100 orchestrator steps; 30 user turns is the same order.
DEFAULT_USER_MAX_TURNS = 30


def tau2_user_persona() -> str:
    """tau2's user-sim system prompt with ``{scenario}`` left for ``UserSimEnv`` to fill."""
    persona = PersonaConfig()
    guidelines = simulation_guidelines(persona, mode="text", use_tools=False)
    return build_user_system_prompt(
        "{scenario}", persona_config=persona, mode="text", guidelines=guidelines
    )


class Tau2UserSimEnvConfig(UserSimEnvConfig):
    assistant: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    user: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, max_turns=DEFAULT_USER_MAX_TURNS)
    persona: str = Field(default_factory=tau2_user_persona)
    done_marker: str = STOP


class Tau2UserSimEnv(UserSimEnv, vf.Env[Tau2UserSimEnvConfig]):
    """``UserSimEnv`` control flow with tau2 defaults; see the module docstring."""
