from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.fsm import JobStatus, RunStatus
from eb.domain.job import CONTAINERLESS_KINDS, JobKind, JobSpec
from eb.orchestrator.service import Orchestrator
from eb.pipelines import advance_pipeline
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Job, PipelineRun, Run

log = logging.getLogger(__name__)


def _classify_container_failure(run_id: str, fallback: str) -> tuple[str, str]:
    path = get_settings().data_root / "runs" / run_id / "console.log"
    if path.is_file():
        text = path.read_text(errors="replace")
        if ("take_picture" in text
                and "RuntimeError: cannot create buffer" in text):
            return ("graphics_buffer_exhausted",
                    "RoboTwin 相机图形缓冲区不足，请降低每卡并发环境数后重新提交")
    return fallback, fallback


def _memory_budget_mb(doctor: dict, reserved_mb: int = 0) -> int:
    # ponytail: conservatively reserve full running limits; per-container usage can reduce double counting later.
    return max(0, int(doctor.get("memory_available_mb") or 0) - 4096 - reserved_mb)


def _free_gpu_indices(doctor: dict, min_free_mb: float = 0) -> list[int] | None:
    """Return free GPU indices, or None if this tick must skip admission."""
    gpus = doctor.get("gpus") or []
    busy = {g["index"] for g in gpus if g.get("run_id")}
    free = [int(g["index"]) for g in gpus if g.get("index") not in busy
            and g.get("memory_free_mb", 0) >= min_free_mb]
    if gpus:
        return free
    fake = os.environ.get("EB_FAKE_GPUS", "").strip()
    if not fake:
                fake = (get_settings().fake_gpus or "").strip()
    if not fake:
        return None
    return [int(x.strip()) for x in fake.split(",") if x.strip()]


class Scheduler:
    def __init__(self, orch: Orchestrator | None = None) -> None:
        self.orch = orch or Orchestrator()
        self.ebd = EbdClient()

    async def tick(self, session: AsyncSession) -> None:
        await self.reap_running(session)
        await self.advance_pipelines(session)
        doctor = await self.ebd.doctor()
        running = (await session.scalars(select(Job).where(Job.status == JobStatus.RUNNING.value))).all()
        reserved_mb = sum(int((job.spec.get("resources") or {}).get("memory_gb", 0)) * 1024
                          for job in running if job.kind not in CONTAINERLESS_KINDS)
        memory_budget = _memory_budget_mb(doctor, reserved_mb)
        free = _free_gpu_indices(doctor)
        if free is None:
            log.error("ebd doctor reported no GPUs; skip GPU admission this tick (set EB_FAKE_GPUS for tests)")
        jobs = (
            await session.scalars(
                select(Job)
                .where(Job.status == JobStatus.QUEUED.value)
                .order_by(Job.priority.desc(), Job.created_at.asc())
            )
        ).all()
        for job in jobs:
            spec = JobSpec.model_validate({**job.spec, "kind": job.kind, "embodiment_env_id": job.embodiment_env_id})
            if spec.kind in CONTAINERLESS_KINDS:
                try:
                    await self.orch.admit_and_start(session, job, [])
                except Exception:
                    log.exception("admit_and_start failed for %s", job.id)
                continue
            required_memory = spec.resources.memory_gb * 1024
            if required_memory > memory_budget:
                continue
            gpu_kinds = {
                JobKind.TRAIN_RL,
                JobKind.TRAIN_SFT,
                JobKind.TRAIN_RLFT,
                JobKind.EVAL_SIM,
                JobKind.EVAL_SIM2SIM,
                JobKind.EXPORT_POLICY,
                JobKind.DATA_COLLECT,
                JobKind.DATA_RETARGET,
                JobKind.ASSET_VALIDATE,
                JobKind.SERVE_POLICY,
            }
            need = spec.resources.gpus
            if spec.kind in gpu_kinds:
                need = need or 1
            if not need:
                memory_budget -= required_memory
                try:
                    await self.orch.admit_and_start(session, job, [])
                except Exception:
                    log.exception("admit_and_start failed for %s", job.id)
                continue
            if free is None:
                continue
            from eb.capacity import estimate_job
            eligible = _free_gpu_indices(doctor, estimate_job(spec).per_gpu_peak_gb * 1024) or []
            available = [index for index in free if index in eligible]
            if len(available) < need:
                continue
            assigned = available[:need]
            memory_budget -= required_memory
            free = [index for index in free if index not in assigned]
            try:
                await self.orch.admit_and_start(session, job, assigned)
            except Exception:
                log.exception("admit_and_start failed for %s", job.id)

    async def advance_pipelines(self, session: AsyncSession) -> None:
        rows = (await session.scalars(select(PipelineRun).where(PipelineRun.status == "running"))).all()
        for pipe in rows:
            try:
                await advance_pipeline(session, pipe, self.orch)
            except Exception:
                log.exception("advance_pipeline failed for %s", pipe.id)

    async def reap_running(self, session: AsyncSession) -> None:
        jobs = (await session.scalars(select(Job).where(Job.status == JobStatus.RUNNING.value))).all()
        now = datetime.now(UTC)
        for job in jobs:
            # Serialize terminal writes with cancellation and refresh stale query results.
            await session.refresh(job, with_for_update=True)
            if job.status != JobStatus.RUNNING.value:
                await session.commit()
                continue
            run = await session.scalar(
                select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc())
            )
            if not run or not run.id:
                await session.commit()
                continue

            # Check timeout
            timeout_h = (job.spec or {}).get("policy", {}).get("timeout_h")
            if timeout_h and run.started_at:
                started = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=UTC)
                elapsed_hours = (now - started).total_seconds() / 3600.0
                if elapsed_hours > float(timeout_h):
                    log.warning("Job %s exceeded timeout limit %sh (ran %.2fh). Cancelling.", job.id, timeout_h, elapsed_hours)
                    try:
                        await self.ebd.cancel_run(run.id)
                    except Exception:
                        log.exception("Timeout worker cleanup failed for %s", run.id)
                        await session.commit()
                        continue
                    run.status = RunStatus.ABORTED.value
                    run.abort_reason = "timeout"
                    run.finished_at = now
                    job.status = JobStatus.FAILED.value
                    job.error = f"Job exceeded timeout limit of {timeout_h}h"
                    from eb.orchestrator.collect import archive_partial_eval
                    await archive_partial_eval(session, job, run)
                    await session.commit()
                    continue

            try:
                info = await self.ebd.get_run(run.id)
            except Exception:
                log.exception("get_run failed for %s", run.id)
                await session.commit()
                continue
            status = (info or {}).get("status")
            if info.get("oom_killed"):
                try:
                    await self.ebd.cancel_run(run.id)
                except Exception:
                    log.exception("OOM worker cleanup failed for %s", run.id)
                    await session.commit()
                    continue
                run.status = RunStatus.ABORTED.value
                run.abort_reason = "oom"
                run.finished_at = now
                job.status = JobStatus.FAILED.value
                job.error = "A worker was OOM-killed; remaining container workers were cancelled"
                from eb.orchestrator.collect import archive_partial_eval
                await archive_partial_eval(session, job, run)
                await session.commit()
                continue
            if status in (None, "running", "unknown"):
                await session.commit()
                continue
            if status == "exited":
                code = info.get("exit_code")
                if code == 0:
                    await self.orch.collect_and_finish(session, job, run)
                else:
                    reason, message = _classify_container_failure(run.id, f"exit_{code}")
                    run.status = RunStatus.ABORTED.value
                    run.abort_reason = reason
                    run.finished_at = now
                    job.status = JobStatus.FAILED.value
                    job.error = message
                    from eb.orchestrator.collect import archive_partial_eval
                    await archive_partial_eval(session, job, run)
                    await session.commit()
            else:
                await session.commit()

    async def reconcile(self, session: AsyncSession) -> None:
        """On api start: RUNNING jobs whose container is dead become infra_failed."""
        ps = await self.ebd.ps()
        live = {
            item.get("run_id")
            for item in (ps.get("items") or [])
            if item.get("status") == "running"
        }
        jobs = (await session.scalars(select(Job).where(Job.status == JobStatus.RUNNING.value))).all()
        for job in jobs:
            run = await session.scalar(
                select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc())
            )
            if not run:
                continue
            if run.id in live:
                continue
            try:
                info = await self.ebd.get_run(run.id)
            except Exception:
                info = {}
            if info.get("status") == "exited" and info.get("exit_code") == 0:
                await self.orch.collect_and_finish(session, job, run)
                continue
            if info.get("status") == "exited":
                reason, message = _classify_container_failure(
                    run.id, f"exit_{info.get('exit_code')}")
                run.status = RunStatus.ABORTED.value
                run.abort_reason = reason
                job.status = JobStatus.FAILED.value
                job.error = message
            else:
                run.status = RunStatus.ABORTED.value
                run.abort_reason = "container_missing"
                job.status = JobStatus.INFRA_FAILED.value
                job.error = "RUNNING job has no live container after restart"
            run.finished_at = datetime.now(UTC)
            from eb.orchestrator.collect import archive_partial_eval
            await archive_partial_eval(session, job, run)
            await session.commit()
