"""Finite sequential execution snapshots with typed checkpoint bindings."""

from __future__ import annotations

from typing import Any, Literal
from copy import deepcopy
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from eb.domain.fsm import JOB_TERMINAL, JobStatus
from eb.domain.job import JobSpec, ResourceSpec
from eb.orchestrator.service import Orchestrator
from eb.store.models import Job, Pipeline, PipelineRun, Run
from eb.ids import new_id
from eb.settings import get_settings

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


def step_spec(step: dict, index: int) -> tuple[JobSpec, dict]:
    bindings = step.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("step bindings must be an object")
    for target, binding in bindings.items():
        if (target != "inputs.base_model" or not isinstance(binding, dict)
                or binding.get("output") != "model_directory"
                or type(binding.get("from_step")) is not int
                or not 0 <= binding["from_step"] < index):
            raise ValueError("only earlier model_directory -> inputs.base_model bindings are supported")
    return JobSpec.model_validate(step.get("job", {k: v for k, v in step.items() if k != "bindings"})), bindings


async def start_pipeline(session: AsyncSession, pipeline: Pipeline, principal: str = "user:kk",
                         idem_key: str | None = None, *, resume_from: PipelineRun | None = None,
                         memory_gb: int | None = None) -> PipelineRun:
    scope = f"pipeline-resume:{resume_from.id}" if resume_from else f"pipeline:{pipeline.id}"
    key = f"{scope}:{principal}:{idem_key}" if idem_key else None
    if key:
        existing = await session.scalar(select(PipelineRun).where(PipelineRun.idem_key == key))
        if existing:
            return existing
    if resume_from and (resume_from.status != "failed" or resume_from.pipeline_id != pipeline.id):
        raise ValueError("only a failed run of this pipeline can be resumed")
    if memory_gb is not None and (resume_from is None or type(memory_gb) is not int or not 1 <= memory_gb <= 1024):
        raise ValueError("resume memory_gb must be an integer between 1 and 1024 GiB")
    spec = deepcopy((resume_from.spec if resume_from else pipeline.spec) or {})
    for key_name in ("status", "job_ids", "cursor", "error"):
        spec.pop(key_name, None)
    steps = spec.get("steps", [])
    if not isinstance(steps, list) or len(steps) > 64:
        raise ValueError("pipeline requires a finite steps list, maximum 64")
    for index, step in enumerate(steps):
        step_spec(step, index)
    job_ids = []
    if resume_from:
        for job_id in resume_from.job_ids or []:
            job = await session.get(Job, job_id)
            if not job or job.status != "succeeded":
                break
            job_ids.append(job_id)
        if len(job_ids) >= len(steps):
            raise ValueError("failed run has no remaining stage to resume")
        if memory_gb is not None:
            index = len(job_ids)
            current, _ = step_spec(steps[index], index)
            resources = ResourceSpec.model_validate({**current.resources.model_dump(), "memory_gb": memory_gb})
            if resources.shm_gb > resources.memory_gb:
                raise ValueError("resume memory_gb must be at least the stage shm_gb")
            target = steps[index].get("job", steps[index])
            target["resources"] = {**target.get("resources", {}), "memory_gb": memory_gb}
            spec["resource_adjustment"] = {"step_index": index, "source_run_id": resume_from.id,
                "memory_gb": {"before": current.resources.memory_gb, "after": memory_gb}}
        checkpoint_sources = set()
        for index, step in enumerate(steps):
            job_spec, bindings = step_spec(step, index)
            if index < len(job_ids) and job_spec.kind in {"train.sft", "train.rlft"}:
                checkpoint_sources.add(index)
            checkpoint_sources.update(binding["from_step"] for binding in bindings.values()
                                      if binding["from_step"] < len(job_ids))
        for index in sorted(checkpoint_sources):
            await bound_checkpoint(session, job_ids[index])
        spec["resumed_from"] = resume_from.id
    run = PipelineRun(id=new_id("pr"), pipeline_id=pipeline.id, spec=spec, status="running",
                      job_ids=job_ids, created_by=principal, idem_key=key)
    session.add(run)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        if key:
            existing = await session.scalar(select(PipelineRun).where(PipelineRun.idem_key == key))
            if existing:
                return existing
        raise
    await session.refresh(run)
    return run


def pipeline_step_key(run_id: str, index: int) -> str:
    return f"pipeline-run:{run_id}:step:{index}"


def checkpoint_directory(run_root: Path, run_id: str) -> str:
    from eb.checkpoints import checkpoint_model
    return checkpoint_model(run_root, run_id)["path"]


async def bound_checkpoint(session: AsyncSession, job_id: str) -> str:
    job = await session.get(Job, job_id)
    if not job or job.status != "succeeded":
        raise ValueError("binding requires a succeeded producer job")
    run = await session.scalar(select(Run).where(Run.job_id == job_id).order_by(Run.attempt.desc()))
    if not run or run.status != "finished":
        raise ValueError("binding requires a finished producer run")
    return checkpoint_directory(get_settings().data_root / "runs" / run.id, run.id)


async def advance_pipeline(session: AsyncSession, pipeline: PipelineRun, orch: Orchestrator,
                           principal: str | None = None) -> PipelineRun:
    if pipeline.status != "running":
        return pipeline
    steps, job_ids = pipeline.spec.get("steps", []), list(pipeline.job_ids or [])
    last = await session.get(Job, job_ids[-1]) if job_ids else None
    action, index = pipeline_next(steps, job_ids, last.status if last else None)
    if action == "wait":
        return pipeline
    if action in {"succeeded", "failed"}:
        pipeline.status = action
        pipeline.error = (last.error or last.status) if action == "failed" and last else None
        if action == "failed" and not last:
            pipeline.error = "missing producer job"
        await session.commit()
        return pipeline
    assert index is not None
    key = pipeline_step_key(pipeline.id, index)
    # Recover the durable Job first if its commit happened before recording job_ids.
    job = await session.scalar(select(Job).where(Job.idem_key == key))
    try:
        if job is None:
            spec, bindings = step_spec(steps[index], index)
            for target, binding in bindings.items():
                path = await bound_checkpoint(session, job_ids[binding["from_step"]])
                spec = spec.model_copy(update={"inputs": spec.inputs.model_copy(update={"base_model": path})})
            job = await orch.submit(session, spec, pipeline.created_by, key)
        pipeline.job_ids = [*job_ids, job.id]
        await session.commit()
    except IntegrityError:
        # Concurrent advance lost the unique Job idem_key race; retry on the next scheduler tick.
        await session.rollback()
        await session.refresh(pipeline)
    except (ValueError, PermissionError, OSError, KeyError) as exc:
        pipeline.status, pipeline.error = "failed", str(exc)
        await session.commit()
    return pipeline
