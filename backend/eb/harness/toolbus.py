from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

Effect = Literal["read", "write", "costly", "dangerous", "forbidden"]


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: str | None = None
    ref_id: str | None = None
    summary: dict[str, Any] = {}


@dataclass
class ToolSpec:
    name: str
    version: str
    effect: Effect
    handler: Callable
    approval: str | None = None
    rate_limit: str | None = None
    expose_mcp: bool = True
    compensate: str | None = None
    description: str = ""


REGISTRY: dict[str, ToolSpec] = {}
FORBIDDEN = {"deploy.release", "contract.create", "contract.modify", "gate.override"}


def tool(
    name: str,
    *,
    effect: Effect = "read",
    version: str = "1.0.0",
    approval: str | None = None,
    rate_limit: str | None = None,
    compensate: str | None = None,
    description: str = "",
) -> Callable:
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = ToolSpec(
            name=name,
            version=version,
            effect=effect,
            handler=fn,
            approval=approval,
            rate_limit=rate_limit,
            compensate=compensate,
            description=description or (fn.__doc__ or "").strip(),
        )
        return fn

    return deco


@dataclass
class BusDecision:
    allowed: bool
    reason: str = ""
    needs_approval: bool = False


class ToolBus:
    def __init__(self, actor_type: str = "agent") -> None:
        self.actor_type = actor_type
        self.prior: list[str] = field(default_factory=list) if False else []
        self.prior = []

    def check(self, name: str, args: dict[str, Any] | None = None) -> BusDecision:
        args = args or {}
        if name in FORBIDDEN or (spec := REGISTRY.get(name)) and spec.effect == "forbidden":
            return BusDecision(False, f"{name} is structurally forbidden")
        spec = REGISTRY.get(name)
        if spec is None:
            return BusDecision(False, f"unknown tool {name}")
        if spec.effect == "dangerous":
            return BusDecision(False, "dangerous tools require human confirmation", needs_approval=True)
        if spec.approval:
            # tiny expression language: "gpus > 1 or max_hours > 6"
            if _eval_approval(spec.approval, args):
                return BusDecision(True, "needs approval", needs_approval=True)
        return BusDecision(True)

    async def call(self, name: str, ctx: Any, args: dict[str, Any]) -> ToolResult:
        decision = self.check(name, args)
        if not decision.allowed:
            return ToolResult(ok=False, error=decision.reason)
        if decision.needs_approval:
            return ToolResult(ok=False, error="approval_required", data={"tool": name, "args": args})
        spec = REGISTRY[name]
        if spec.effect != "read":
            from eb.harness.budget import allow_costly

            if not allow_costly():
                return ToolResult(ok=False, error="weekly gpu budget exceeded")
        result = await spec.handler(ctx, args)
        self.prior.append(name)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(ok=True, data=result, summary=_summarize(result))

    def permission_matrix(self) -> list[dict[str, Any]]:
        rows = []
        for spec in sorted(REGISTRY.values(), key=lambda s: s.name):
            rows.append(
                {
                    "name": spec.name,
                    "effect": spec.effect,
                    "approval": spec.approval or ("permanent ban" if spec.name in FORBIDDEN else "none"),
                    "rate_limit": spec.rate_limit,
                    "description": spec.description,
                }
            )
        for name in sorted(FORBIDDEN):
            if name not in REGISTRY:
                rows.append(
                    {
                        "name": name,
                        "effect": "forbidden",
                        "approval": "permanent ban",
                        "rate_limit": None,
                        "description": "structurally forbidden",
                    }
                )
        return rows


def _summarize(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return {k: data[k] for k in list(data)[:8]}
    return {"repr": str(data)[:200]}


def _eval_approval(expr: str, args: dict[str, Any]) -> bool:
    env = {k: args.get(k) for k in args}
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))  # noqa: S307  — tiny DSL, no builtins
    except Exception:
        return False
