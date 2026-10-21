"""Per-rollout MCP tool server exposing a tau2 domain's agent tools.

verifiers launches ``python -m <module of the Toolset class>`` once per
rollout, calls ``setup_task`` with the rollout's ``Tau2TaskData`` and then
serves the ``@vf.tool`` methods over MCP to the harness. The tool methods are
generated from tau2's ``@is_tool`` toolkit methods so names, parameters and
docstrings match the benchmark exactly (which the evaluator's replay relies on).
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import verifiers.v1 as vf
from tau2.data_model.message import ToolCall
from tau2.environment.toolkit import TOOL_ATTR

from tooluse_rl.env.domain import execute_tool_call, fresh_env, task_by_id

__all__ = ["Tau2ToolsetBase", "make_toolset_class", "tool_specs"]


class ToolSpec:
    def __init__(self, name: str, params: list[inspect.Parameter], doc: str) -> None:
        self.name = name
        self.params = params
        self.doc = doc


def _toolkit_class(domain: str) -> type:
    if domain == "retail":
        from tau2.domains.retail.tools import RetailTools

        return RetailTools
    if domain == "airline":
        from tau2.domains.airline.tools import AirlineTools

        return AirlineTools
    raise ValueError(domain)


def tool_specs(domain: str) -> list[ToolSpec]:
    """Name, parameters and docstring of every agent tool in ``domain`` (no data needed)."""
    specs: list[ToolSpec] = []
    for name, fn in inspect.getmembers(_toolkit_class(domain), predicate=inspect.isfunction):
        if not getattr(fn, TOOL_ATTR, False):
            continue
        params = [
            p for p in inspect.signature(fn, eval_str=True).parameters.values() if p.name != "self"
        ]
        specs.append(ToolSpec(name, params, (fn.__doc__ or "").strip()))
    return specs


class Tau2ToolsetBase(vf.Toolset[vf.ToolsetConfig]):
    DOMAIN: ClassVar[str] = ""

    async def setup_task(self, task: Any) -> None:
        task_id = str(getattr(task, "task_id", None) or task.id)
        self._env = fresh_env(self.DOMAIN, task_by_id(self.DOMAIN, task_id))
        self._calls = 0

    def _call(self, name: str, arguments: dict[str, Any]) -> str:
        self._calls += 1
        result = execute_tool_call(
            self._env, ToolCall(id=f"srv_{self._calls}", name=name, arguments=arguments)
        )
        return result.content or ""


def _make_tool_method(spec: ToolSpec, qualname_prefix: str):
    def method(self, **kwargs: Any) -> str:
        return self._call(spec.name, kwargs)

    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params += [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in spec.params]
    method.__signature__ = inspect.Signature(params, return_annotation=str)  # type: ignore[attr-defined]
    method.__annotations__ = {p.name: p.annotation for p in spec.params} | {"return": str}
    method.__name__ = spec.name
    method.__qualname__ = f"{qualname_prefix}.{spec.name}"
    method.__doc__ = spec.doc
    return vf.tool(method)


def make_toolset_class(domain: str, *, module: str) -> type[Tau2ToolsetBase]:
    """Build ``Tau2<Domain>Toolset`` with one ``@vf.tool`` method per domain tool.

    ``module`` must be the module that defines the class (verifiers launches the
    server as ``python -m <module>``), so the caller passes its ``__name__``.
    """
    class_name = f"Tau2{domain.capitalize()}Toolset"
    namespace: dict[str, Any] = {
        "DOMAIN": domain,
        "__module__": module,
        "__doc__": f"tau2 {domain} agent tools served over MCP, one fresh DB per rollout.",
    }
    for spec in tool_specs(domain):
        namespace[spec.name] = _make_tool_method(spec, class_name)
    return type(class_name, (Tau2ToolsetBase,), namespace)
