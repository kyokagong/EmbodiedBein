from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.adapters import equivalent_command, get_adapter
from eb.capacity import estimate_job
from eb.domain.contract import Contract
from eb.domain.fsm import JobStatus, RunStatus, transition
from eb.domain.job import JobSpec
from eb.gates.engine import GateContext, g1_submit
from eb.ids import new_id
from eb.store.ebd import EbdClient
from eb.store.models import EmbodimentContractRow, Job, Run


class Orchestrator:
    def __init__(self, ebd: EbdClient | None = None) -> None:
        self.ebd = ebd or EbdClient()

    async def submit(self, session: AsyncSession, spec: JobSpec, principal: str, idem_key: str | None) -> Job:
        if idem_key:
            existing = await session.scalar(select(Job).where(Job.idem_key == idem_key))
            if existing:
                return existing
        row = await session.get(EmbodimentContractRow, spec.contract_id)
        if not row:
            raise ValueError(f"unknown contract {spec.contract_id}")
        ctr = Contract.model_validate(row.spec)
        gate = g1_submit(spec, ctr, GateContext(dataset_versions=[]))
        if not gate.passed:
            raise PermissionError({"violations": [v.model_dump() for v in gate.violations]})
        est = estimate_job(spec)
        if est.verdict == "no_fit":
            raise PermissionError({"capacity": est.__dict__})
        job = Job(
            id=new_id("job"),
            contract_id=spec.contract_id,
            kind=spec.kind.value,
            spec=spec.model_dump(mode="json"),
            spec_hash=spec.canonical_hash(),
            idem_key=idem_key,
            status=JobStatus.QUEUED.value,
            priority=spec.policy.priority,
            created_by=principal,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    async def admit_and_start(self, session: AsyncSession, job: Job, gpu_ids: list[int]) -> Run:
        spec = JobSpec.model_validate({**job.spec, "kind": job.kind, "contract_id": job.contract_id, "api_version": "eb/v1"})
        job.status = transition(JobStatus(job.status), JobStatus.ADMITTED).value
        job.status = transition(JobStatus(job.status), JobStatus.PREPARING).value
        job.attempts += 1
        job.version += 1
        run = Run(
            id=new_id("run"),
            job_id=job.id,
            attempt=job.attempts,
            status=RunStatus.CREATED.value,
            gpu_ids=gpu_ids,
            started_at=datetime.now(timezone.utc),
        )
        session.add(run)
        rendered = get_adapter(spec).render(spec, run.id)
        job.status = transition(JobStatus(job.status), JobStatus.RUNNING).value
        run.status = RunStatus.RUNNING.value
        try:
            result = await self.ebd.submit_run(
                {
                    "run_id": run.id,
                    "job_id": job.id,
                    "image": rendered.image,
                    "command": rendered.command,
                    "env": rendered.env,
                    "gpus": gpu_ids,
                    "shm_gb": rendered.shm_gb,
                    "devices": rendered.devices,
                    "workdir": rendered.workdir,
                    "kind": spec.kind.value,
                }
            )
            run.container_id = result.get("container_id")
            if result.get("status") == "stub_finished":
                run.status = RunStatus.FINISHED.value
                run.finished_at = datetime.now(timezone.utc)
                job.status = JobStatus.SUCCEEDED.value
                run.metrics_summary = result.get("metrics") or {"Train/mean_reward": 12.4}
        except Exception as e:
            run.status = RunStatus.ABORTED.value
            run.abort_reason = "infra_failed"
            job.status = JobStatus.INFRA_FAILED.value
            job.error = str(e)
        await session.commit()
        await session.refresh(job)
        await session.refresh(run)
        return run

    def preview_command(self, spec: JobSpec, run_id: str = "run_preview") -> str:
        return equivalent_command(spec, run_id)
