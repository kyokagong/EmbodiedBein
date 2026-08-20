from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.fsm import JobStatus
from eb.domain.job import JobSpec
from eb.orchestrator.service import Orchestrator
from eb.store.ebd import EbdClient
from eb.store.models import Job

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, orch: Orchestrator | None = None) -> None:
        self.orch = orch or Orchestrator()
        self.ebd = EbdClient()

    async def tick(self, session: AsyncSession) -> None:
        doctor = await self.ebd.doctor()
        gpus = doctor.get("gpus") or []
        busy = {g["index"] for g in gpus if g.get("run_id")}
        free = [g["index"] for g in gpus if g.get("index") not in busy]
        if not gpus:
            # no NVML: allow a virtual 4-GPU pool tracked only by running jobs
            running = (
                await session.scalars(select(Job).where(Job.status == JobStatus.RUNNING.value))
            ).all()
            used = sum(int((j.spec.get("resources") or {}).get("gpus") or 0) for j in running)
            free = list(range(max(0, 4 - used)))
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
