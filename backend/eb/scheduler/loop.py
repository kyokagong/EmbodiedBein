from __future__ import annotations

import logging
import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.fsm import JobStatus, RunStatus
from eb.domain.job import JobSpec
from eb.orchestrator.service import Orchestrator
from eb.store.ebd import EbdClient
from eb.store.models import Job, Run

log = logging.getLogger(__name__)


def _free_gpu_indices(doctor: dict) -> list[int] | None:
    """Return free GPU indices, or None if this tick must skip admission."""
    gpus = doctor.get("gpus") or []
    busy = {g["index"] for g in gpus if g.get("run_id")}
    free = [int(g["index"]) for g in gpus if g.get("index") not in busy]
    if gpus:
        return free
    fake = os.environ.get("EB_FAKE_GPUS", "").strip()
    if not fake:
        from eb.settings import get_settings

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
        doctor = await self.ebd.doctor()
        free = _free_gpu_indices(doctor)
        if free is None:
            log.error("ebd doctor reported no GPUs; skip admission this tick (set EB_FAKE_GPUS for tests)")
            return
        jobs = (
            await session.scalars(
                select(Job)
                .where(Job.status == JobStatus.QUEUED.value)
                .order_by(Job.priority.desc(), Job.created_at)
            )
        ).all()
        for job in jobs:
            spec = JobSpec.model_validate({**job.spec, "kind": job.kind, "contract_id": job.contract_id})
            need = spec.resources.gpus
            if need and len(free) < need:
                continue
            assigned = free[:need] if need else []
            free = free[need:]
            try:
                await self.orch.admit_and_start(session, job, assigned)
            except Exception:
                log.exception("admit_and_start failed for %s", job.id)

    async def reap_running(self, session: AsyncSession) -> None:
        jobs = (await session.scalars(select(Job).where(Job.status == JobStatus.RUNNING.value))).all()
        for job in jobs:
            run = await session.scalar(
                select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc())
            )
            if not run or not run.id:
                continue
            try:
                info = await self.ebd.get_run(run.id)
            except Exception:
                log.exception("get_run failed for %s", run.id)
                continue
            status = (info or {}).get("status")
            if status in (None, "running", "unknown"):
                continue
            if status == "exited":
                code = info.get("exit_code")
                if code == 0:
                    await self.orch.collect_and_finish(session, job, run)
                else:
                    run.status = RunStatus.ABORTED.value
                    run.abort_reason = f"exit_{code}"
                    job.status = JobStatus.FAILED.value
                    job.error = run.abort_reason
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
                run.status = RunStatus.ABORTED.value
                run.abort_reason = f"exit_{info.get('exit_code')}"
                job.status = JobStatus.FAILED.value
                job.error = run.abort_reason
            else:
                run.status = RunStatus.ABORTED.value
                run.abort_reason = "container_missing"
                job.status = JobStatus.INFRA_FAILED.value
                job.error = "RUNNING job has no live container after restart"
            await session.commit()
