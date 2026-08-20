from __future__ import annotations

import logging
import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.fsm import JobStatus
from eb.domain.job import JobSpec
from eb.orchestrator.service import Orchestrator
from eb.store.ebd import EbdClient
from eb.store.models import Job

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
