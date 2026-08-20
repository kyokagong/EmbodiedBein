"""Sequential pipeline DAG: submit next Job only after the previous SUCCEEDED."""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from eb.domain.fsm import JOB_TERMINAL, JobStatus
from eb.domain.job import JobSpec
from eb.orchestrator.service import Orchestrator
from eb.store.models import Job, Pipeline

Action = Literal["wait", "submit", "succeeded", "failed"]


def pipeline_next(steps: list[Any], job_ids: list[str], last_status: str | None) -> tuple[Action, int | None]:
    """Pure stepper. last_status is the terminal/non-terminal status of job_ids[-1]."""
    if not steps:
        return "succeeded", None
    if not job_ids:
        return "submit", 0
    if last_status is None:
        return "failed", None
    try:
        st = JobStatus(last_status)
    except ValueError:
        return "failed", None
    if st not in JOB_TERMINAL:
        return "wait", None
    if st != JobStatus.SUCCEEDED:
        return "failed", None
    nxt = len(job_ids)
    if nxt >= len(steps):
        return "succeeded", None
    return "submit", nxt


async def start_pipeline(session: AsyncSession, pipeline: Pipeline) -> Pipeline:
    spec = dict(pipeline.spec or {})
    if spec.get("status") in ("running", "succeeded"):
        return pipeline
    spec["status"] = "running"
    spec.setdefault("job_ids", [])
    spec.setdefault("cursor", 0)
    pipeline.spec = spec
    flag_modified(pipeline, "spec")
    await session.commit()
    await session.refresh(pipeline)
    return pipeline


async def advance_pipeline(
    session: AsyncSession,
    pipeline: Pipeline,
    orch: Orchestrator,
    principal: str,
) -> Pipeline:
    spec = dict(pipeline.spec or {})
    if spec.get("status") != "running":
        return pipeline
    steps = spec.get("steps") or []
    job_ids = list(spec.get("job_ids") or [])
    last_status = None
    last_error = None
    if job_ids:
        last = await session.get(Job, job_ids[-1])
        if last is None:
            spec["status"] = "failed"
            spec["error"] = f"missing job {job_ids[-1]}"
            pipeline.spec = spec
            flag_modified(pipeline, "spec")
            await session.commit()
            return pipeline
        last_status = last.status
        last_error = last.error
    action, idx = pipeline_next(steps, job_ids, last_status)
    if action == "wait":
        return pipeline
    if action == "succeeded":
        spec["status"] = "succeeded"
        pipeline.spec = spec
        flag_modified(pipeline, "spec")
        await session.commit()
        return pipeline
    if action == "failed":
        spec["status"] = "failed"
        spec["error"] = last_error or last_status
        pipeline.spec = spec
        flag_modified(pipeline, "spec")
        await session.commit()
        return pipeline
    assert action == "submit" and idx is not None
    job_spec = JobSpec.model_validate(steps[idx])
    job = await orch.submit(session, job_spec, principal, None)
    job_ids.append(job.id)
    spec["job_ids"] = job_ids
    spec["cursor"] = idx
    spec["status"] = "running"
    pipeline.spec = spec
    flag_modified(pipeline, "spec")
    await session.commit()
    await session.refresh(pipeline)
    return pipeline
