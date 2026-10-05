from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.ids import new_id
from eb.paths import PathNotAllowed, validate_allowed_path
from eb.store.models import LineageEdge, WeightModel, Job, Run
from eb.workbench import dir_size
from eb.model_sources import MODEL_SOURCES, source_by_repo

router = APIRouter()


class BaseModelIn(BaseModel):
    name: str
    path: str
    embodiment_id: str | None = None
    framework: str = "pytorch"


def _out(row: WeightModel) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "path": row.path,
        "framework": row.framework,
        "size_bytes": row.size_bytes,
        "embodiment_id": row.embodiment_id,
        "train_job_id": row.train_job_id,
        "parent_model_id": row.parent_model_id,
        "metrics_at_checkpoint": row.metrics_at_checkpoint or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


class DownloadModelIn(BaseModel):
    name: str = "RLinf π0.5 LIBERO SFT"
    repo_id: str = "RLinf/RLinf-Pi05-LIBERO-SFT"
    revision: str
    embodiment_id: str = "emb_libero_franka"


@router.get("/weights/sources")
async def model_sources(principal: PrincipalDep) -> dict:
    return {"items":[{"source_id":source_id, **source} for source_id, source in MODEL_SOURCES.items()]}


@router.get("/weights/hub-revision")
async def hub_revision(principal: PrincipalDep, repo_id: str = "RLinf/RLinf-Pi05-LIBERO-SFT") -> dict:
    import httpx
    import re
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise HTTPException(400, "非法模型仓库")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            from eb.settings import get_settings
            endpoint = get_settings().hf_endpoint.rstrip("/")
            response = await client.get(f"{endpoint}/api/models/{repo_id}")
            response.raise_for_status()
            info = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, "无法读取公开模型版本") from exc
    return {"repo_id":repo_id, "revision":info["sha"]}


@router.post("/weights/download")
async def download_model(body: DownloadModelIn, session: SessionDep, principal: PrincipalDep) -> dict:
    import re
    from eb.domain.job import JobSpec, JobKind
    from eb.orchestrator.service import Orchestrator
    if not re.fullmatch(r"[0-9a-f]{40}", body.revision) or not re.fullmatch(r"[\w.-]+/[\w.-]+", body.repo_id):
        raise HTTPException(400, "下载必须固定公开仓库的完整 revision")
    try:
        source_id, source = source_by_repo(body.repo_id)
    except KeyError:
        source_id, source = None, None
    if not source or body.embodiment_id != source["embodiment_id"]:
        raise HTTPException(400, "模型仓库与本体接口不匹配")
    mid = new_id("mdl")
    from eb.settings import get_settings
    model_path = str(get_settings().models_root / mid)
    spec = JobSpec(kind=JobKind.DATA_CONVERT, name=body.name, embodiment_env_id=source["embodiment_env_id"],
        params={"operation":"download_model", "source_id":source_id, "repo_id":body.repo_id, "revision":body.revision, "model_id":mid},
        resources={"gpus":0,"cpus":2,"memory_gb":4,"shm_gb":1})
    job = await Orchestrator().submit(session, spec, principal.id, f"download:{mid}", commit=False)
    row = WeightModel(id=mid, name=body.name, kind="downloading", path="", framework=source["framework"],
        embodiment_id=body.embodiment_id, metrics_at_checkpoint={"download":{"job_id":job.id,
        "repo_id":body.repo_id,"revision":body.revision,"model_path":model_path,
        "source_id":source_id,"required_files":source["required_files"],"status":"queued"}})
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _out(row)


@router.post("/weights/{mid}/retry-download")
async def retry_model_download(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.domain.job import JobKind, JobSpec
    from eb.orchestrator.service import Orchestrator
    from eb.settings import get_settings
    row = await session.get(WeightModel, mid)
    if not row:
        raise HTTPException(404, "模型不存在")
    await reconcile_model(row, session)
    download = dict((row.metrics_at_checkpoint or {}).get("download") or {})
    source = MODEL_SOURCES.get(download.get("source_id"))
    if (row.kind != "download_failed" or not source or row.embodiment_id != source["embodiment_id"]
            or not re.fullmatch(r"[0-9a-f]{40}", str(download.get("revision", "")))):
        raise HTTPException(409, "该模型下载不能重试")
    spec = JobSpec(kind=JobKind.DATA_CONVERT, name=row.name, embodiment_env_id=source["embodiment_env_id"],
        params={"operation":"download_model", "repo_id":download["repo_id"],
                "revision":download["revision"], "model_id":row.id, "source_id":download["source_id"]},
        resources={"gpus":0,"cpus":2,"memory_gb":4,"shm_gb":1})
    job = await Orchestrator().submit(session, spec, principal.id,
        f"download-retry:{row.id}:{new_id('req')}", commit=False)
    download.update(job_id=job.id, status="queued", error=None,
                    model_path=str(get_settings().models_root / row.id),
                    required_files=source["required_files"])
    row.kind = "downloading"
    row.metrics_at_checkpoint = {**(row.metrics_at_checkpoint or {}), "download":download}
    await session.commit()
    await session.refresh(row)
    return _out(row)


async def reconcile_model(row, session):
    import json
    from eb.settings import get_settings
    metadata = dict(row.metrics_at_checkpoint or {})
    download = dict(metadata.get("download") or {})
    if not download.get("source_id"):
        try:
            download["source_id"] = source_by_repo(download.get("repo_id", ""))[0]
        except KeyError:
            pass
    if row.kind != "downloading" or not download.get("job_id"):
        return
    job = await session.get(Job, download["job_id"])
    if not job:
        return
    download.update(status=job.status, error=job.error)
    if job.status == "succeeded":
        run = await session.scalar(select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc()).limit(1))
        try:
            if run is None:
                raise ValueError("下载缺少实际运行记录")
            root = get_settings().data_root / "runs" / run.id / "outputs"
            manifest = root / "model-download.json"
            if manifest.is_symlink() or manifest.stat().st_size > 1024 * 1024:
                raise ValueError("下载凭证过大")
            receipt = json.loads(manifest.read_text())
            if (not isinstance(receipt, dict) or receipt.get("status") != "complete"
                    or receipt.get("revision") != download["revision"] or receipt.get("repo_id") != download["repo_id"]
                    or (receipt.get("source_id") is not None and receipt.get("source_id") != download.get("source_id"))):
                raise ValueError("下载凭证不匹配")
            expected_model_path = download.get("model_path")
            model = Path(expected_model_path) if expected_model_path else root / "model"
            expected_receipt_path = expected_model_path or "model"
            if (type(receipt.get("bytes")) is not int or receipt["bytes"] <= 0
                    or receipt.get("model_path") != expected_receipt_path or model.is_symlink() or not model.is_dir()
                    or not isinstance(receipt.get("files"), list) or not receipt["files"]):
                raise ValueError("下载缺少完整模型目录")
            total, seen = 0, set()
            for item in receipt["files"]:
                relative = item.get("path") if isinstance(item, dict) else None
                if not isinstance(relative, str) or not relative or "\\" in relative:
                    raise ValueError("下载文件路径无效")
                parts = PurePosixPath(relative)
                if parts.is_absolute() or ".." in parts.parts or not parts.parts:
                    raise ValueError("下载文件路径越界")
                path = model
                for component in parts.parts:
                    path = path / component
                    if path.is_symlink():
                        raise ValueError("下载文件不能使用符号链接")
                if (not path.resolve().is_relative_to(model.resolve()) or path in seen or not path.is_file()
                        or type(item.get("bytes")) is not int or item["bytes"] < 0
                        or path.stat().st_size != item["bytes"]
                        or not isinstance(item.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                    raise ValueError("下载文件缺失、重复或大小/摘要凭证无效")
                total += item["bytes"]
                seen.add(path)
            source = MODEL_SOURCES.get(download.get("source_id"))
            required = {model / name for name in (source or {}).get("required_files", ())}
            if total != receipt["bytes"] or not required or not required.issubset(seen) or any(path.stat().st_size == 0 for path in required):
                raise ValueError("下载缺少该模型必需资产或总大小不匹配")
            row.path, row.kind = str(model), "base"
            row.size_bytes = receipt["bytes"]
            metadata["source"] = receipt
        except (OSError, ValueError, KeyError) as exc:
            row.kind = "download_failed"
            download["error"] = str(exc)
    elif job.status in {"failed", "infra_failed", "cancelled"}:
        row.kind = "download_failed"
    metadata["download"] = download
    row.metrics_at_checkpoint = metadata
    await session.commit()


@router.get("/weights")
async def list_weights(session: SessionDep, principal: PrincipalDep, kind: str | None = None) -> dict:
    rows = (await session.scalars(select(WeightModel))).all()
    for row in rows:
        await reconcile_model(row, session)
    items = [_out(r) for r in rows]
    if kind:
        items = [i for i in items if i["kind"] == kind]
    return {"items": items}


@router.post("/weights")
async def register_base(body: BaseModelIn, session: SessionDep, principal: PrincipalDep) -> dict:
    try:
        path = validate_allowed_path(body.path)
    except PathNotAllowed as e:
        raise HTTPException(400, str(e)) from e
    if not path.exists():
        raise HTTPException(400, "路径不存在")
    row = WeightModel(
        id=new_id("mdl"),
        name=body.name or path.name,
        kind="base",
        path=str(path),
        framework=body.framework,
        size_bytes=dir_size(path),
        embodiment_id=body.embodiment_id,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _out(row)


@router.get("/weights/{mid}")
async def get_weight(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(WeightModel, mid)
    if not row:
        raise HTTPException(404)
    await reconcile_model(row, session)
    edges = (
        await session.scalars(
            select(LineageEdge).where((LineageEdge.src_id == mid) | (LineageEdge.dst_id == mid))
        )
    ).all()
    return {
        **_out(row),
        "edges": [
            {"src_type": e.src_type, "src_id": e.src_id, "dst_type": e.dst_type, "dst_id": e.dst_id, "relation": e.relation}
            for e in edges
        ],
    }


@router.get("/weights/{mid}/lineage")
async def weight_lineage(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await get_weight(mid, session, principal)
