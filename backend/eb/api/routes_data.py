from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.disk import disk_preflight
from eb.domain.data_view import compact_dataset_stats, dataset_episode_page
from eb.ids import new_id
from eb.paths import PathNotAllowed, is_under, validate_allowed_path
from eb.settings import get_settings
from eb.store.models import Dataset

router = APIRouter()
LIBERO_REPO = "RLinf/LIBERO-assets"
LIBERO_REVISION = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"
LIBERO_ENV = "ee_libero_franka_delta_v1"


class DatasetIn(BaseModel):
    name: str = ""
    source_kind: str = "path"
    source_uri: str
    dest_path: str | None = None
    hf_token: str | None = None
    revision: str | None = None
    embodiment_env_id: str | None = None
    usage: Literal["train", "eval"] = "train"
    purpose: Literal["motion_prior"] | None = None


class DatasetUsageIn(BaseModel):
    usage: Literal["train", "eval"]


class OfficialRpoAmpIn(BaseModel):
    name: str = "RoboParty RPO AMP stand/walk probe"


class RoboTwinDatasetIn(BaseModel):
    name: str = ""
    profile_id: str | None = None
    task: str | None = None
    scenario: Literal["clean", "randomized"] | None = None
    embodiment: str | None = None

    @model_validator(mode="after")
    def complete_selection(self):
        raw = (self.task, self.scenario, self.embodiment)
        if bool(self.profile_id) == all(value is not None for value in raw):
            raise ValueError("选择一个标准转换包，或完整选择 task/scenario/embodiment")
        return self


def _ds_out(row: Dataset) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "source_kind": row.source_kind,
        "source_uri": row.source_uri,
        "dest_path": row.dest_path,
        "format": row.format,
        "status": row.status,
        "bytes": row.total_bytes,
        "total_bytes": row.total_bytes,
        "download_progress": row.download_progress or {},
        "summary_stats": compact_dataset_stats(row.summary_stats),
        "error_message": row.error_message,
        "layer": "L1",
        "episodes": (row.summary_stats or {}).get("episodes_count"),
        "license": row.license,
        "embodiment_env_id": row.embodiment_env_id,
        "usage": (row.meta or {}).get("usage", "train"),
        "purpose": (row.meta or {}).get("purpose"),
        "g2": "—",
        "content_hash": None,
        "validation": (row.summary_stats or {}).get("validation", {"status": "not_validated", "training_ready": False}),
        "source_refs": (row.summary_stats or {}).get("source_refs", []),
        **({"candidate_evidence": {
            "receipt_url": f"/api/v1/datasets/{row.id}/grounded-a1/files/candidate-receipt.json",
            "video_url": f"/api/v1/datasets/{row.id}/grounded-a1/files/reference/replay.mp4",
            "quality_url": f"/api/v1/datasets/{row.id}/grounded-a1/files/reference/quality.json",
            "reference_pd_only": True,
        }} if (row.meta or {}).get("operation") == "prepare_ufo_rpo_grounded_a1_v1"
              and (row.meta or {}).get("candidate_evidence_verified") is True else {}),
    }


def _ufo_motion_source(value: str) -> Path:
    root = get_settings().ufo_native_root / "runs"
    source = Path(value)
    if (not source.is_absolute() or source != source.resolve() or not source.is_relative_to(root.resolve())
            or source == root or not source.is_dir()
            or any(p.is_symlink() for p in (source, *source.parents))):
        raise ValueError("UFO motion source must be an existing unlinked directory under UFO/runs")
    required = ("train_clipped.pkl", "validation.pkl", "qualification.json", "physics_audit/mask_feasibility.json")
    for relative in required:
        path = source / relative
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"UFO motion source is missing an unlinked {relative}")
    return source


async def ufo_data_spec(row: Dataset):
    import asyncio

    from eb.domain.job import JobSpec
    from eb.store.ebd import EbdClient
    from eb.ufo_native import ENV_ID
    from eb.ufo_training import DATA_OPERATION, IMAGE

    if (row.source_kind != "ufo_native_motion" or row.embodiment_env_id != ENV_ID
            or (row.meta or {}).get("usage", "train") != "train"
            or not re.fullmatch(r"ds_[A-Za-z0-9]+", row.id)):
        raise ValueError("UFO native motion registration is inconsistent")
    source = _ufo_motion_source(row.source_uri)
    settings = get_settings()
    provenance = settings.ufo_native_root / "runs/rpo_reference_v2/import_report.json"
    if not provenance.is_file() or any(p.is_symlink() for p in (provenance, *provenance.parents)):
        raise ValueError("UFO original import provenance is missing")
    client = EbdClient()
    image = await client.image_info(IMAGE)
    if image.get("ok") is not True or not re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("digest", "")):
        raise ValueError("UFO native training image is unavailable")
    snapshots = await asyncio.gather(client.snapshot(str(settings.ufo_native_root)),
        client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")))
    ufo, assets = ({"kind": "content_snapshot", "path": value["path"], "commit": value["content_hash"]}
                   for value in snapshots)
    return JobSpec(kind="data.convert", embodiment_env_id=ENV_ID, image=image["digest"],
        params={"operation": DATA_OPERATION, "src": str(source), "dataset_id": row.id,
                "provenance_source": str(provenance), "ufo_source_ref": ufo, "asset_source_ref": assets},
        resources={"gpus": 0, "cpus": 4, "memory_gb": 16, "shm_gb": 2},
        runtime={"network": "none"}, policy={"timeout_h": 0.5, "max_retries": 0, "retry_on": []})


@router.post("/datasets/rpo-amp-official")
async def create_official_rpo_amp_dataset(body: OfficialRpoAmpIn, session: SessionDep,
                                          principal: PrincipalDep) -> dict:
    from eb.domain.job import JobKind, JobSpec
    from eb.orchestrator.service import Orchestrator
    from eb.rpo_wbc import AMP_MOTION_CONVERTER_REVISION
    revision = "6b1c3d9988497c8961dcba77892de32edc1770e1"

    rows = (await session.scalars(select(Dataset).where(
        Dataset.source_kind == "rpo_amp_official"))).all()
    existing = next((row for row in rows if (row.meta or {}).get("converter_revision")
                     == AMP_MOTION_CONVERTER_REVISION), None)
    if existing:
        await reconcile_dataset(existing, session)
        if existing.status in {"invalid", "failed"} and not Path(existing.dest_path).exists():
            spec = JobSpec(kind=JobKind.DATA_CONVERT,
                           embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                           params={"operation":"convert_rpo_amp_official", "dataset_id":existing.id},
                           runtime={"network":"none"},
                           resources={"gpus":0, "cpus":2, "memory_gb":4, "shm_gb":1})
            job = await Orchestrator().submit(session, spec, principal.id,
                                              f"rpo-amp-official:{existing.id}:{new_id('attempt')}", commit=False)
            existing.meta = {**(existing.meta or {}), "validation_job_id":job.id}
            existing.status = "queued"
            existing.error_message = None
            await session.commit()
        return _ds_out(existing)
    for legacy in rows:
        if (legacy.meta or {}).get("converter_revision") == "grounded-retimed-v3":
            # Keep the immutable diagnostic input and its active PPO control provenance.
            continue
        stats = dict(legacy.summary_stats or {})
        validation = dict(stats.get("validation") or {})
        validation.update(status="invalid", training_ready=False,
                          errors=["superseded by corrected RPO AMP motion selection"])
        legacy.summary_stats = {**stats, "validation":validation}
        legacy.status = "invalid"
        legacy.error_message = "superseded by corrected RPO AMP motion selection"
    did = new_id("ds")
    destination = get_settings().datasets_root / did / "content"
    destination.parent.mkdir(parents=True, exist_ok=False)
    check = disk_preflight(str(destination.parent), needed_bytes=0, archive=True)
    if not check.ok:
        raise HTTPException(400, check.reason)
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                   params={"operation":"convert_rpo_amp_official", "dataset_id":did},
                   runtime={"network":"none"},
                   resources={"gpus":0, "cpus":2, "memory_gb":4, "shm_gb":1})
    job = await Orchestrator().submit(session, spec, principal.id, f"rpo-amp-official:{did}", commit=False)
    row = Dataset(id=did, name=body.name, format="unknown", license="BSD-3-Clause",
                  source_kind="rpo_amp_official", source_uri=f"https://github.com/Roboparty/robolab/tree/{revision}/data/motions/rpo_lab",
                  dest_path=str(destination), status="queued", total_bytes=0,
                  embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                  meta={"usage":"train", "purpose":"motion_prior", "source_revision":revision,
                        "converter_revision":AMP_MOTION_CONVERTER_REVISION,
                        "license_scope":"repository; upstream motion provenance not independently verified",
                        "validation_job_id":job.id},
                  summary_stats={"validation":{"status":"not_validated", "training_ready":False}})
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _ds_out(row)


@router.get("/dataset-sources/robotwin")
async def robotwin_catalog(principal: PrincipalDep) -> dict:
    from eb.robotwin import (DOWNLOAD_PROFILES, ROBOTWIN_EMBODIMENTS, ROBOTWIN_HF_REPO,
                             ROBOTWIN_HF_REVISION, ROBOTWIN_MODELS, ROBOTWIN_TASKS)
    return {"repo_id":ROBOTWIN_HF_REPO, "revision":ROBOTWIN_HF_REVISION,
            "tasks":list(ROBOTWIN_TASKS),
            "embodiments":[item.__dict__ for item in ROBOTWIN_EMBODIMENTS.values()],
            "download_profiles":[item.public() for item in DOWNLOAD_PROFILES.values()],
            "models":[item.public() for item in ROBOTWIN_MODELS.values()]}


@router.post("/datasets/robotwin")
async def create_robotwin_dataset(body: RoboTwinDatasetIn, session: SessionDep,
                                   principal: PrincipalDep) -> dict:
    from eb.robotwin import ROBOTWIN_EMBODIMENTS, get_download_profile, raw_archive_profile
    try:
        profile = (get_download_profile(body.profile_id) if body.profile_id else
                   raw_archive_profile(str(body.task), str(body.scenario), str(body.embodiment)))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.embodiment and not ROBOTWIN_EMBODIMENTS[body.embodiment].download_ready:
        raise HTTPException(409, ROBOTWIN_EMBODIMENTS[body.embodiment].blocked_reason)
    rows = (await session.scalars(select(Dataset).where(Dataset.source_kind == "robotwin"))).all()
    for existing in rows:
        meta = existing.meta or {}
        if meta.get("robotwin_profile_id") == profile.id and meta.get("source_revision") == profile.revision:
            await reconcile_dataset(existing, session)
            return _ds_out(existing)
    did = new_id("ds")
    destination = get_settings().datasets_root / did / "content"
    destination.parent.mkdir(parents=True, exist_ok=True)
    check = disk_preflight(str(destination), needed_bytes=0, archive=True)
    if not check.ok:
        raise HTTPException(400, check.reason)
    row = Dataset(id=did, name=body.name or profile.label, format=profile.format, license="mit",
        source_kind="robotwin", source_uri=profile.repo_id, dest_path=str(destination),
        status="registered", total_bytes=0, embodiment_env_id=profile.embodiment_env_id,
        meta={"source_revision":profile.revision, "usage":"train",
              "robotwin_profile_id":profile.id, "native_embodiment":profile.native_embodiment,
              "scenario":profile.scenario, "task":profile.task,
              "compatibility":"native_direct", "requires_probe":profile.requires_probe},
        summary_stats={"validation":{"status":"not_validated", "training_ready":False}})
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _ds_out(row)


@router.post("/datasets")
async def create_dataset(body: DatasetIn, session: SessionDep, principal: PrincipalDep) -> dict:
    did = new_id("ds")
    kind = body.source_kind
    if kind not in {"path", "huggingface", "http", "libero_benchmark", "ufo_native_motion"}:
        raise HTTPException(400, "未知数据来源类型")
    if kind == "http":
        raise HTTPException(400, "HTTP acquisition is unavailable until bounded source validation is implemented")
    if body.purpose == "motion_prior" and (
            kind != "path" or body.usage != "train"
            or body.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1"):
        raise HTTPException(400, "AMP motion_prior requires a local train path and matching RPO interface")
    if body.hf_token:
        raise HTTPException(400, "Inline HF tokens are not accepted; public datasets only")
    if kind == "ufo_native_motion":
        from eb.ufo_native import ENV_ID
        if body.usage != "train" or body.embodiment_env_id != ENV_ID or body.dest_path or body.purpose:
            raise HTTPException(400, "UFO 原生 motion 需要 RPO 接口、train 用途和独立的冻结输出目录")
    if kind == "libero_benchmark":
        if (body.source_uri != LIBERO_REPO or body.revision != LIBERO_REVISION
                or body.dest_path or body.usage != "eval"
                or body.embodiment_env_id != LIBERO_ENV):
            raise HTTPException(400, "LIBERO 评估资产必须使用固定来源、版本和本体环境")
        existing = await session.scalar(select(Dataset).where(
            Dataset.source_kind == kind, Dataset.source_uri == LIBERO_REPO))
        if existing:
            await reconcile_dataset(existing, session)
            return _ds_out(existing)
    if kind == "huggingface" and not re.fullmatch(r"[0-9a-f]{40}", body.revision or ""):
        raise HTTPException(400, "HF requires a pinned 40-character commit SHA")
    dest = body.dest_path
    if kind == "libero_benchmark":
        dest = str(get_settings().datasets_root / did / "bundle")
    elif kind == "ufo_native_motion":
        try:
            dest = str(_ufo_motion_source(body.source_uri))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    elif kind == "path":
        try:
            dest_p = validate_allowed_path(body.source_uri)
        except PathNotAllowed as e:
            raise HTTPException(400, str(e)) from e
        if not dest_p.is_dir():
            raise HTTPException(400, "source path must be an existing directory")
        dest = str(dest_p)
    else:
        dest = dest or str(get_settings().datasets_root / did)
        try:
            dest_p = validate_allowed_path(dest)
        except PathNotAllowed as e:
            raise HTTPException(400, str(e)) from e
        if not is_under(get_settings().datasets_root, dest_p):
            raise HTTPException(400, "downloads must be stored under the managed datasets root")
        if dest_p.exists() and (not dest_p.is_dir() or any(dest_p.iterdir())):
            raise HTTPException(400, "download destination must be a new or empty directory")
        dest = str(dest_p)
    check = disk_preflight(dest, needed_bytes=0, archive=kind == "http")
    if not check.ok:
        raise HTTPException(400, check.reason)
    row = Dataset(
        id=did,
        name=body.name or ("LIBERO benchmark (4 suites)" if kind == "libero_benchmark" else Path(dest).name),
        format="libero_benchmark" if kind == "libero_benchmark" else "unknown",
        license=None,
        source_kind=kind,
        source_uri=body.source_uri,
        dest_path=dest,
        status="registered",
        total_bytes=0,
        embodiment_env_id=body.embodiment_env_id,
        meta={"source_revision": body.revision, "usage": body.usage, "purpose": body.purpose},
        summary_stats={"validation": {"status": "not_validated", "training_ready": False}},
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _ds_out(row)


@router.patch("/datasets/{did}/usage")
async def set_dataset_usage(did: str, body: DatasetUsageIn, session: SessionDep,
                            principal: PrincipalDep) -> dict:
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    if row.source_kind == "libero_benchmark" and body.usage != "eval":
        raise HTTPException(400, "LIBERO benchmark 不能用作训练示教数据")
    row.meta = {**(row.meta or {}), "usage": body.usage}
    await session.commit()
    return _ds_out(row)


@router.post("/datasets/{did}/validate")
async def validate_dataset(did: str, session: SessionDep, principal: PrincipalDep, reindex: bool = False) -> dict:
    from eb.domain.job import JobKind, JobSpec
    from eb.orchestrator.service import Orchestrator
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    from eb.ufo_grounded_reference import OPERATION as GROUNDED_OPERATION
    if (row.meta or {}).get("operation") == GROUNDED_OPERATION:
        if reindex:
            raise HTTPException(400, "A1 候选不可原地重跑；请从冻结父版本创建新候选")
        await reconcile_dataset(row, session)
        return _ds_out(row)
    if not row.embodiment_env_id:
        raise HTTPException(400, "embodiment_env_id is required to submit a data task")
    retry = False
    if (row.meta or {}).get("validation_job_id"):
        await reconcile_dataset(row, session)
        if row.source_kind in {"libero_benchmark", "ufo_native_motion"} and row.status == "validated":
            if row.source_kind == "ufo_native_motion" and reindex:
                raise HTTPException(400, "冻结的 UFO 数据不可原地重索引；请登记新数据版本")
            return _ds_out(row)
        retry = row.source_kind in {"robotwin", "ufo_native_motion"} and row.status in {"invalid", "failed"}
        if not retry and (not reindex or row.status not in {"validated", "invalid", "failed"}):
            return _ds_out(row)
    if row.source_kind == "ufo_native_motion":
        try:
            spec = await ufo_data_spec(row)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        params = spec.params
    elif row.source_kind == "robotwin":
        if reindex:
            raise HTTPException(400, "RoboTwin 本地重新索引尚未开放；请重试失败的固定下载")
        meta = row.meta or {}
        params = {"operation":"acquire_robotwin", "dataset_id":row.id, "dest":row.dest_path,
                  "profile_id":meta.get("robotwin_profile_id")}
        if str(meta.get("robotwin_profile_id", "")).startswith("raw:"):
            params.update(task=meta.get("task"), scenario=meta.get("scenario"),
                          embodiment=meta.get("native_embodiment"))
        Path(row.dest_path).parent.mkdir(parents=True, exist_ok=True)
    elif row.source_kind == "libero_benchmark":
        if (row.source_uri != LIBERO_REPO or (row.meta or {}).get("source_revision") != LIBERO_REVISION
                or row.dest_path != str(get_settings().datasets_root / row.id / "bundle")):
            raise HTTPException(400, "LIBERO benchmark 登记信息不匹配")
        Path(row.dest_path).parent.mkdir(parents=True, exist_ok=True)
        params = {"operation": "acquire_libero_benchmark", "dataset_id": row.id,
                  "dest": row.dest_path, "repo_id": LIBERO_REPO, "revision": LIBERO_REVISION}
    elif (row.meta or {}).get("purpose") == "motion_prior":
        if (row.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1"
                or row.source_kind not in {"path", "rpo_amp_official"}):
            raise HTTPException(400, "AMP motion dataset registration is inconsistent")
        params = {"operation": "validate_rpo_amp_motion", "src": row.dest_path, "dataset_id": row.id}
    else:
        params = {"operation": "validate_lerobot_v3", "src": row.dest_path, "dataset_id": row.id}
    if row.source_kind == "huggingface":
        params.update(repo_id=row.source_uri, revision=(row.meta or {}).get("source_revision"), local_only=reindex)
        Path(row.dest_path).mkdir(parents=True, exist_ok=True)
    if row.source_kind != "ufo_native_motion":
        spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id=row.embodiment_env_id,
                       params=params,
                       runtime={"network":"none"} if params["operation"] == "validate_rpo_amp_motion" else {})
    try:
        attempt = f":{new_id('attempt')}" if reindex or retry else ""
        job = await Orchestrator().submit(session, spec, principal.id, f"dataset-validation:{did}{attempt}")
    except (ValueError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc
    row.meta = {**(row.meta or {}), "validation_job_id": job.id}
    row.status = "queued"
    row.error_message = None
    if row.source_kind == "libero_benchmark":
        row.summary_stats = {"validation": {"status": "not_validated", "training_ready": False,
                                             "evaluation_ready": False}}
    await session.commit()
    return _ds_out(row)


async def reconcile_dataset(row: Dataset, session) -> None:
    from eb.store.models import Job, Run
    from eb.settings import get_settings
    jid = (row.meta or {}).get("validation_job_id")
    if not jid:
        return
    job = await session.get(Job, jid)
    run = await session.scalar(select(Run).where(Run.job_id == jid).order_by(Run.attempt.desc()))
    if not job or not run:
        return
    if ((row.meta or {}).get("validation_run_id") == run.id and row.status in {"validated", "invalid"}
            and (row.source_kind != "robotwin" or (row.summary_stats or {}).get("fps"))):
        if row.source_kind == "ufo_native_motion":
            from eb.ufo_training_dataset import publish_native_dataset_version
            if await publish_native_dataset_version(session, row, job, run):
                await session.commit()
        return
    report_path = get_settings().data_root / "runs" / run.id / "outputs" / "dataset-index.json"
    if job.status in {"succeeded", "failed", "infra_failed"} and report_path.is_file():
        if report_path.stat().st_size > 64 * 1024 * 1024:
            row.status, row.error_message = "failed", "metadata report exceeds 64 MiB"
        else:
            stats = json.loads(report_path.read_text())
            from eb.ufo_grounded_reference import OPERATION as GROUNDED_OPERATION
            if (row.meta or {}).get("operation") == GROUNDED_OPERATION:
                from eb.ufo_grounded_reference import verified_report
                try:
                    _, stats = await verified_report(report_path.parent,
                        job.spec["params"]["grounded_plan"], run.id, row.id)
                    row.meta = {**row.meta, "candidate_evidence_verified": True}
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    row.meta = {**row.meta, "candidate_evidence_verified": False}
                    stats = {"format": "ufo_native_motion", "validation": {
                        "scope": "ufo_native_motion_v1", "status": "invalid", "training_ready": False,
                        "errors": [f"A1 candidate evidence is incomplete or invalid: {exc}"], "unchecked": []}}
            if row.source_kind == "robotwin" and not stats.get("source_refs"):
                stats["source_refs"] = [{"kind":"huggingface_dataset", "repo_id":row.source_uri,
                    "revision":(row.meta or {}).get("source_revision"),
                    "files":stats.get("archives", [])}]
            if row.source_kind == "robotwin" and not stats.get("fps"):
                stats["fps"] = (((stats.get("features") or {}).get("observation.images.cam_high") or {})
                                .get("info", {}).get("video.fps"))
            validation = stats.get("validation", {})
            row.summary_stats = stats
            row.meta = {**(row.meta or {}), "validation_run_id": run.id}
            row.status = "validated" if validation.get("status") == "validated" and job.status == "succeeded" else "invalid"
            row.error_message = "; ".join(validation.get("errors", [])) or None
            row.format = (stats.get("format") or ("libero_benchmark" if row.source_kind == "libero_benchmark" else "lerobot")) if row.status == "validated" else "unknown"
            row.total_bytes = int(stats.get("total_bytes") or 0)
            if row.source_kind == "ufo_native_motion" and row.status == "validated":
                expected = get_settings().data_root / "runs" / run.id / "outputs/data"
                if (validation.get("scope") != "ufo_native_motion_v1"
                        or validation.get("training_ready") is not True
                        or stats.get("format") != "ufo_native_motion"
                        or stats.get("storage_ref") != str(expected)
                        or validation.get("storage_ref") != str(expected)):
                    row.status, row.format = "invalid", "unknown"
                    row.error_message = "UFO frozen data report has an inconsistent scope or storage path"
                    row.summary_stats = {**stats, "validation": {**validation, "status": "invalid",
                        "training_ready": False,
                        "errors": [row.error_message]}}
                else:
                    row.dest_path = str(expected)
    elif job.status in {"failed", "infra_failed", "cancelled"}:
        row.status, row.error_message = "failed", job.error or job.status
    elif row.status not in {"validated", "invalid"}:
        row.status = job.status
    if row.source_kind == "ufo_native_motion":
        from eb.ufo_training_dataset import publish_native_dataset_version
        await publish_native_dataset_version(session, row, job, run)
    await session.commit()


@router.get("/datasets/{did}/episodes")
async def dataset_episodes(did: str, session: SessionDep, principal: PrincipalDep,
                           cursor: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)) -> dict:
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    return dataset_episode_page(row.summary_stats, cursor, limit)


@router.get("/datasets/{did}/tree")
async def dataset_tree(
    did: str,
    session: SessionDep,
    principal: PrincipalDep,
    path: str = "",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    root = Path(row.dest_path or ((row.meta or {}).get("root") if row.meta else "") or "")
    if not root.exists():
        raise HTTPException(404, "dest_path 不存在")
    rel = path.strip("/")
    cur = (root / rel).resolve() if rel else root.resolve()
    if not is_under(root, cur) and cur != root.resolve():
        raise HTTPException(403, "path 越权")
    if not cur.is_dir():
        raise HTTPException(400, "不是目录")
    kids = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    start = max(page - 1, 0) * page_size
    chunk = kids[start : start + page_size]
    items = []
    for p in chunk:
        items.append(
            {
                "name": p.name,
                "path": str(p.relative_to(root)).replace("\\", "/"),
                "is_dir": p.is_dir(),
                "bytes": p.stat().st_size if p.is_file() else None,
                "mtime": int(p.stat().st_mtime),
            }
        )
    return {"path": rel, "page": page, "page_size": page_size, "total": len(kids), "items": items}


@router.get("/datasets/{did}/media")
async def dataset_media(did: str, session: SessionDep, principal: PrincipalDep, path: str = Query(...)):
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    root = Path(row.dest_path or ((row.meta or {}).get("root") if row.meta else "") or "")
    dest = (root / path).resolve()
    if not is_under(root, dest) or not dest.is_file():
        raise HTTPException(404)
    return FileResponse(dest)


from eb.api.routes_ufo_grounded_reference import router as grounded_reference_router

router.include_router(grounded_reference_router)
