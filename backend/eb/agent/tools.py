from __future__ import annotations

from typing import Any

from eb.harness.toolbus import ToolResult, tool


class AgentCtx:
    def __init__(self, api: Any, principal: str) -> None:
        self.api = api
        self.principal = principal


@tool("query.jobs", effect="read", rate_limit="200/h", description="检索任务")
async def query_jobs(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=True, data=args, summary={"hint": "use REST /jobs"})


@tool("capacity.estimate", effect="read", rate_limit="200/h", description="查容量估算器")
async def capacity_estimate(ctx: AgentCtx, args: dict) -> ToolResult:
    from eb.capacity import estimate_job
    from eb.domain.job import JobSpec

    spec = JobSpec.model_validate(args)
    est = estimate_job(spec)
    return ToolResult(ok=True, data=est.__dict__, summary={"verdict": est.verdict, "peak": est.per_gpu_peak_gb})


@tool("job.submit.train", effect="costly", approval="gpus > 1 or max_hours > 6", rate_limit="6/day")
async def submit_train(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="go through POST /api/v1/jobs")


@tool("deploy.release", effect="forbidden")
async def deploy_release(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="forbidden")


@tool("contract.modify", effect="forbidden")
async def contract_modify(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="forbidden")


@tool("gate.override", effect="forbidden")
async def gate_override(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=False, error="forbidden")


@tool("report.write", effect="read", rate_limit="10/day")
async def report_write(ctx: AgentCtx, args: dict) -> ToolResult:
    return ToolResult(ok=True, data={"report": args.get("text", "daily report")})
