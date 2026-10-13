from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.job import JobKind, JobSpec
from eb.ids import new_id
from eb.motion_preview import FORMATS, PREVIEW_IMAGE, PREVIEW_SCHEMA, read_motion_manifest
from eb.orchestrator.service import Orchestrator
from eb.paths import validate_allowed_path
from eb.store.ebd import EbdClient
from eb.store.models import Dataset, Job, Run

router = APIRouter()


async def motion_dataset(did, session):
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404, "数据集不存在")
    if row.format not in FORMATS:
        raise HTTPException(400, "此数据格式尚不支持参考动作回放")
    validation = (row.summary_stats or {}).get("validation") or {}
    if row.status != "validated" or not validation.get("manifest_sha256"):
        raise HTTPException(409, "参考动作尚未完成结构校验或冻结索引缺失")
    try:
        root = validate_allowed_path(row.dest_path)
        manifest, digest = read_motion_manifest(root, validation["manifest_sha256"])
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return row, root, manifest, digest


async def preview_job(did, digest, session):
    return await session.scalar(select(Job).where(
        Job.kind == JobKind.DATA_CONVERT.value,
        Job.spec["params"]["operation"].astext == "preview_rpo_motion",
        Job.spec["params"]["dataset_id"].astext == did,
        Job.spec["params"]["manifest_sha256"].astext == digest,
        Job.spec["params"]["preview_schema"].astext == PREVIEW_SCHEMA,
    ).order_by(Job.created_at.desc()).limit(1))


async def preview_status(job, session):
    if not job:
        return {"status": "not_generated", "job_id": None, "run_id": None}
    run = await session.scalar(select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc()).limit(1))
    return {"status": job.status, "job_id": job.id, "run_id": run.id if run else None, "error": job.error}


@router.get("/datasets/{did}/motions")
async def dataset_motions(did: str, session: SessionDep, principal: PrincipalDep):
    row, _, manifest, digest = await motion_dataset(did, session)
    clips = manifest["clips"]
    return {"items": clips, "total": len(clips), "frames": sum(c["frames"] for c in clips),
            "seconds": sum(c["frames"] / c["fps"] for c in clips),
            "joint_order": manifest["joint_order"], "manifest_sha256": digest,
            "source_refs": manifest.get("source_refs", []),
            "validation": (row.summary_stats or {}).get("validation", {}),
            "preview": await preview_status(await preview_job(did, digest, session), session)}


@router.post("/datasets/{did}/motion-preview")
async def create_motion_preview(did: str, session: SessionDep, principal: PrincipalDep,
                                rebuild: bool = False):
    row, root, _, digest = await motion_dataset(did, session)
    existing = await preview_job(did, digest, session)
    if existing and (existing.status not in {"succeeded", "failed", "infra_failed", "cancelled"}
                     or (existing.status == "succeeded" and not rebuild)):
        return await preview_status(existing, session)
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id=row.embodiment_env_id,
        name=f"参考动作预览 · {row.name}", image=PREVIEW_IMAGE,
        params={"operation": "preview_rpo_motion", "dataset_id": did,
                "manifest_sha256": digest, "preview_schema": PREVIEW_SCHEMA},
        inputs={"datasets": [{"id": did, "path": str(root)}]},
        runtime={"network": "none"}, resources={"gpus": 0, "cpus": 1, "memory_gb": 2, "shm_gb": 1},
        policy={"timeout_h": .25, "max_retries": 0})
    key = f"motion-preview:{did}:{digest}:{PREVIEW_SCHEMA}"
    if existing:
        key += f":{new_id('attempt')}"
    try:
        job = await Orchestrator().submit(session, spec, principal.id, key)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await preview_status(job, session)


@router.get("/datasets/{did}/motions/{clip_id}")
async def motion_clip(did: str, clip_id: str, session: SessionDep, principal: PrincipalDep):
    _, _, manifest, digest = await motion_dataset(did, session)
    clip = next((c for c in manifest["clips"] if c["id"] == clip_id), None)
    if not clip:
        raise HTTPException(404, "参考动作片段不存在")
    status = await preview_status(await preview_job(did, digest, session), session)
    if status["status"] != "succeeded" or not status["run_id"]:
        raise HTTPException(409, "回放预览尚未生成，请先生成或重试预览")
    try:
        payload = await EbdClient().motion_preview(status["run_id"], clip_id)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(410 if exc.response.status_code == 404 else 502, "回放产物缺失或不可用") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "数据面暂不可用") from exc
    if payload.get("manifest_sha256") != digest or payload.get("clip_sha256") != clip["sha256"]:
        raise HTTPException(409, "回放预览与当前冻结数据不一致")
    return payload
