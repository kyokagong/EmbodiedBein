"""Operator-only preview/submission for the fixed A1 reference candidate."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.job import JobSpec
from eb.ids import new_id
from eb.settings import get_settings
from eb.store.models import Artifact, Dataset, Job, LineageEdge
from eb.ufo_grounded_reference import OPERATION, build_plan, preview_hash, spec_for
from eb.ufo_native import ENV_ID

router = APIRouter()


class GroundedA1In(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="RPO 58 + grounded A1 reference candidate", min_length=1, max_length=200)
    preview_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")


def operator(principal):
    if principal.actor_type == "agent":
        raise HTTPException(403, "A1 候选数据准备仅供操作者使用")


@router.post("/datasets/{parent_id}/grounded-a1/preview")
async def preview(parent_id: str, body: GroundedA1In, session: SessionDep, principal: PrincipalDep):
    operator(principal)
    try:
        plan = await build_plan(session, parent_id)
        spec = spec_for(plan, "ds_preview", body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "plan": plan,
        "preview_hash": preview_hash(spec),
        "resources": spec.resources.model_dump(),
        "new_experiment_required": True,
        "reference_pd_only": True,
        "qualification": "Candidate remains not_training_ready unless the full diagnostic passes.",
    }


@router.post("/datasets/{parent_id}/grounded-a1")
async def submit(
    parent_id: str,
    body: GroundedA1In,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    operator(principal)
    if not 1 <= len(idem) <= 200:
        raise HTTPException(400, "Idempotency-Key 长度必须为 1..200")
    try:
        plan = await build_plan(session, parent_id)
        expected = preview_hash(spec_for(plan, "ds_preview", body.name))
        if body.preview_hash != expected:
            raise HTTPException(409, "A1 预览已过期，请重新检查固定源、协议和镜像")
        key = "ufo-a1:" + hashlib.sha256((principal.id + ":" + idem).encode()).hexdigest()
        lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
        await session.execute(select(func.pg_advisory_xact_lock(lock_id)))
        previous = await session.scalar(select(Job).where(Job.idem_key == key))
        if previous:
            if (
                preview_hash(JobSpec.model_validate(previous.spec)) != expected
                or previous.created_by != principal.id
            ):
                raise HTTPException(409, "此幂等键已用于不同的候选准备")
            row = await session.get(Dataset, previous.spec["params"]["dataset_id"])
            if row is None:
                raise HTTPException(409, "既有候选登记缺失")
            return {"id": previous.id, "status": previous.status, "dataset_id": row.id, "reused": True}
        from eb.domain.disk import disk_preflight
        from eb.orchestrator.service import Orchestrator

        check = disk_preflight(str(get_settings().data_root), needed_bytes=512 * 1024**2)
        if not check.ok:
            raise ValueError(check.reason)
        dataset_id = new_id("ds")
        row = Dataset(
            id=dataset_id,
            name=body.name.strip(),
            source_kind="ufo_native_motion",
            source_uri="ufo-grounded-a1:" + parent_id,
            dest_path="",
            format="unknown",
            status="queued",
            total_bytes=0,
            embodiment_env_id=ENV_ID,
            meta={
                "usage": "train",
                "operation": OPERATION,
                "parent_dataset_id": parent_id,
                "parent_version_id": plan["parent_version_id"],
                "plan_hash": plan["hash"],
            },
            summary_stats={"validation": {"status": "not_validated", "training_ready": False}},
        )
        session.add(row)
        await session.flush()
        spec = spec_for(plan, dataset_id, body.name)
        job = await Orchestrator().submit(session, spec, principal.id, key, commit=False)
        row.meta = {**row.meta, "validation_job_id": job.id}
        session.add(
            LineageEdge(
                src_type="dataset",
                src_id=parent_id,
                dst_type="dataset",
                dst_id=row.id,
                relation="derived",
                run_id=None,
            )
        )
        await session.commit()
        return {"id": job.id, "status": job.status, "dataset_id": row.id, "reused": False}
    except (ValueError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}/grounded-a1/files/{path:path}")
async def output_file(
    dataset_id: str, path: str, request: Request, session: SessionDep, principal: PrincipalDep
):
    row = await session.get(Dataset, dataset_id)
    if row is None or (row.meta or {}).get("operation") != OPERATION:
        raise HTTPException(404)
    run_id = (row.meta or {}).get("validation_run_id")
    artifact = await session.scalar(
        select(Artifact).where(
            Artifact.run_id == run_id, Artifact.kind == "ufo_grounded_reference_diagnostic"
        )
    )
    receipt = (artifact.meta or {}).get("receipt", {}) if artifact else {}
    if receipt.get("dataset_id") != dataset_id or path not in {
        "candidate-receipt.json",
        *(f["path"] for f in receipt.get("files", [])),
    }:
        raise HTTPException(404, "没有经过收集校验的候选产物")
    from eb.api.routes_ufo_native import stream_output

    return await stream_output(request, run_id, path)
