"""Operator-only native-launch comparison; never creates a TrainJob/model."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.job import JobSpec
from eb.store.models import Artifact, Job, Run
from eb.ufo_native_baseline import (
    ARTIFACT_KIND,
    OPERATION,
    SCOPED_ARTIFACT_KIND,
    SCOPED_SCHEMA,
    VERIFICATION_SCOPE,
)
from eb.ufo_native_baseline_service import build_plan, preview_hash, spec_for

router = APIRouter(prefix="/ufo-native/baselines")


class BaselineIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="UFO v2 native launch 112-update baseline", min_length=1, max_length=120)
    preview_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")


def operator(principal):
    if principal.actor_type == "agent":
        raise HTTPException(403, "Native training comparison requires an operator")


@router.post("/{source_run_id}/preview")
async def preview(source_run_id: str, body: BaselineIn, session: SessionDep, principal: PrincipalDep):
    operator(principal)
    try:
        plan = await build_plan(session, source_run_id)
        spec = spec_for(plan, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "plan": plan,
        "preview_hash": preview_hash(spec),
        "resources": spec.resources.model_dump(),
        "actual_optimizer_updates": 112,
        "world_size": 4,
        "diagnostic_only": True,
        "qualification": "Native training comparison; no TrainJob or ModelVersion publication.",
    }


@router.post("/{source_run_id}")
async def submit(
    source_run_id: str,
    body: BaselineIn,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    operator(principal)
    if not 1 <= len(idem) <= 200:
        raise HTTPException(400, "Idempotency-Key must have 1..200 characters")
    try:
        plan = await build_plan(session, source_run_id)
        spec = spec_for(plan, body.name)
        expected = preview_hash(spec)
        if body.preview_hash != expected:
            raise HTTPException(409, "Baseline preview changed; review the frozen source and image again")
        key = "ufo-baseline:" + hashlib.sha256((principal.id + ":" + idem).encode()).hexdigest()
        lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
        await session.execute(select(func.pg_advisory_xact_lock(lock_id)))
        previous = await session.scalar(select(Job).where(Job.idem_key == key))
        if previous:
            if (
                previous.created_by != principal.id
                or preview_hash(JobSpec.model_validate(previous.spec)) != expected
            ):
                raise HTTPException(409, "Idempotency-Key already binds a different baseline")
            return {"id": previous.id, "status": previous.status, "reused": True}
        from eb.domain.disk import disk_preflight
        from eb.orchestrator.service import Orchestrator
        from eb.settings import get_settings

        check = disk_preflight(str(get_settings().data_root), needed_bytes=8 * 1024**3)
        if not check.ok:
            raise ValueError(check.reason)
        job = await Orchestrator().submit(session, spec, principal.id, key)
        return {"id": job.id, "status": job.status, "reused": False}
    except (ValueError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs/{run_id}/verification-scope")
async def verification_scope(run_id: str, session: SessionDep, principal: PrincipalDep):
    from eb.ufo_native import plan_hash

    run = await session.get(Run, run_id)
    job = await session.get(Job, run.job_id) if run else None
    if not job or job.status != "succeeded" or job.spec.get("params", {}).get("operation") != OPERATION:
        raise HTTPException(404)
    raw = await session.scalar(
        select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == ARTIFACT_KIND)
    )
    scoped = await session.scalar(
        select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == SCOPED_ARTIFACT_KIND)
    )
    payload = (scoped.meta or {}).get("verification") if scoped else None
    if (
        not raw
        or not scoped
        or not isinstance(payload, dict)
        or plan_hash(payload) != scoped.sha256
        or payload.get("schema_version") != SCOPED_SCHEMA
        or payload.get("run_id") != run_id
        or payload.get("verification_scope") != VERIFICATION_SCOPE
        or payload.get("legacy_rank_optimizer_claims_accepted") is not False
        or payload.get("raw_artifact") != {"id": raw.id, "sha256": raw.sha256}
    ):
        raise HTTPException(404, "No immutable scoped verification has been published")
    return {"artifact_id": scoped.id, "sha256": scoped.sha256, "verification": payload}


@router.post("/runs/{run_id}/verification-scope")
async def publish_scope(run_id: str, session: SessionDep, principal: PrincipalDep):
    operator(principal)
    from eb.ufo_native_baseline_service import backfill_scoped_verification

    run = await session.get(Run, run_id)
    job = await session.get(Job, run.job_id) if run else None
    if not job:
        raise HTTPException(404)
    try:
        await backfill_scoped_verification(session, job, run)
        await session.commit()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return await verification_scope(run_id, session, principal)


@router.get("/runs/{run_id}/files/{path:path}")
async def output_file(run_id: str, path: str, request: Request, session: SessionDep, principal: PrincipalDep):
    from eb.settings import get_settings
    from eb.ufo_native import safe_file, sha256_file
    from eb.ufo_native_baseline_service import read_receipt

    run = await session.get(Run, run_id)
    job = await session.get(Job, run.job_id) if run else None
    if not job or job.spec.get("params", {}).get("operation") != OPERATION:
        raise HTTPException(404)
    artifact = await session.scalar(
        select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == ARTIFACT_KIND)
    )
    if not artifact or job.status != "succeeded":
        raise HTTPException(404, "Baseline has no validated diagnostic Artifact")
    outputs = get_settings().data_root / "runs" / run_id / "outputs"
    try:
        receipt = read_receipt(outputs, job.spec["params"]["baseline_plan"], run_id)
        from eb.ufo_native import plan_hash

        if plan_hash(receipt) != artifact.sha256:
            raise ValueError("Baseline immutable receipt changed")
        public = {
            "baseline-report.json",
            "metrics-differences.json",
            "metrics-index.json",
            "native/train_log.txt",
            "native/ufo_tracking_eval.csv",
        }
        row = next((row for row in receipt["files"] if row["path"] == path), None)
        if path == "baseline-receipt.json":
            pass
        elif not row or (path not in public and not path.startswith("analysis/")):
            raise ValueError("Only indexed baseline diagnostic evidence is downloadable")
        elif sha256_file(safe_file(outputs, path)) != row["sha256"]:
            raise ValueError("Baseline evidence hash changed")
    except (ValueError, OSError) as exc:
        raise HTTPException(404, str(exc)) from exc
    from eb.api.routes_ufo_native import stream_output

    response = await stream_output(request, run_id, path)
    response.headers["Link"] = (
        f'</api/v1/ufo-native/baselines/runs/{run_id}/verification-scope>; rel="describedby"'
    )
    response.headers["X-EB-Verification-Scope"] = "shared-rank0-checkpoint-and-all-rank-model-sync"
    response.headers["X-EB-Live-Rank-Optimizer-Verification"] = "not_observed"
    response.headers["X-EB-Legacy-Rank-Optimizer-Claims-Accepted"] = "false"
    return response
