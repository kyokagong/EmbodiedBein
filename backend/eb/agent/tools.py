from __future__ import annotations

from typing import Any

from sqlalchemy import select

from eb.capacity import estimate_job
from eb.domain.job import JobSpec
from eb.harness.toolbus import ToolResult, tool
from eb.store.models import Job, ModelVersion


class AgentCtx:
    def __init__(self, api: Any, principal: str) -> None:
        self.api = api
        self.principal = principal


@tool("query.jobs", effect="read", rate_limit="200/h", description="检索任务")
async def query_jobs(ctx: AgentCtx, args: dict) -> ToolResult:
    session = ctx.api
    q = select(Job).order_by(Job.created_at.desc()).limit(int(args.get("limit") or 50))
    if status := args.get("status"):
        q = q.where(Job.status == status)
    rows = (await session.scalars(q)).all()
    items = [{"id": j.id, "kind": j.kind, "status": j.status, "error": j.error} for j in rows]
    return ToolResult(ok=True, data={"items": items}, ref_id=items[0]["id"] if items else None, summary={"n": len(items)})


@tool("query.models", effect="read", rate_limit="200/h", description="检索模型")
async def query_models(ctx: AgentCtx, args: dict) -> ToolResult:
    session = ctx.api
    rows = (await session.scalars(select(ModelVersion).limit(50))).all()
    items = [{"id": m.id, "name": m.name, "embodiment_env_id": m.embodiment_env_id} for m in rows]
    return ToolResult(ok=True, data={"items": items}, ref_id=items[0]["id"] if items else None, summary={"n": len(items)})


@tool("capacity.estimate", effect="read", rate_limit="200/h", description="查容量估算器")
async def capacity_estimate(ctx: AgentCtx, args: dict) -> ToolResult:
    spec = JobSpec.model_validate(args)
    est = estimate_job(spec)
    return ToolResult(ok=True, data=est.__dict__, summary={"verdict": est.verdict, "peak": est.per_gpu_peak_gb})


@tool("job.submit.train", effect="costly", approval="gpus > 1 or max_hours > 6", rate_limit="6/day")
async def submit_train(ctx: AgentCtx, args: dict) -> ToolResult:
    from eb.orchestrator.service import Orchestrator

    spec = JobSpec.model_validate(args)
    orch = Orchestrator()
    job = await orch.submit(ctx.api, spec, ctx.principal, None)
    return ToolResult(ok=True, data={"id": job.id, "status": job.status}, ref_id=job.id)


@tool("deploy.release", effect="forbidden")
async def deploy_release(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="forbidden")


@tool("embodiment_env.publish", effect="forbidden")
async def embodiment_env_publish(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="forbidden")


@tool("embodiment.scan", effect="read", rate_limit="50/h", description="只读扫描本体资产")
async def embodiment_scan(ctx: AgentCtx, args: dict) -> ToolResult:
    from pathlib import Path

    from eb.store.models import EmbodimentRow
    from eb.worktree import git_is_dirty

    eid = args.get("embodiment_id") or "emb_berkeley_humanoid_lite"
    session = ctx.api
    row = await session.get(EmbodimentRow, eid) if hasattr(session, "get") else None
    if row is None:
        return ToolResult(ok=True, data={"wrote": False, "hint": "只读扫描，不写上游仓"})
    src = row.source or {}
    root = Path(src.get("path") or "")
    dirty = git_is_dirty(root) if (root / ".git").exists() else None
    return ToolResult(
        ok=True,
        data={"wrote": False, "embodiment_id": eid, "path": str(root), "dirty": dirty},
        summary={"wrote": False},
    )


@tool("gate.override", effect="forbidden")
async def gate_override(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="forbidden")


@tool("report.write", effect="read", rate_limit="10/day")
async def report_write(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=True, data={"report": args.get("text", "daily report")})
