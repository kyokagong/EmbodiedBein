from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.job import JobKind, JobSpec
from eb.domain.rlinf_form import TEMPLATES, build_overrides, get_template, ray_port_for
from eb.ids import new_id
from eb.orchestrator.service import Orchestrator
from eb.settings import get_settings
from eb.store.models import EmbodimentRow, Job, LineageEdge, TrainJob, WeightModel
from eb.workbench import dir_size, embodiment_root

router = APIRouter()
orch = Orchestrator()

_KIND = {"berkeley_reach_sft": JobKind.TRAIN_SFT}


class TrainIn(BaseModel):
    name: str
    template_id: str
    embodiment_id: str
    image: str | None = None
    gpus: int = 2
    dataset_ids: list[str] = []
    base_model_id: str | None = None
    max_epochs: int | None = None
    total_num_envs: int | None = None
    lr: str | None = None
    extra_overrides: list[str] = []
    auto_register_latest: bool = False
    gym_id: str | None = None


def _job_out(row: TrainJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "status": row.status,
        "template_id": row.template_id,
        "image_digest": row.image_digest,
        "embodiment_id": row.embodiment_id,
        "dataset_ids": row.dataset_ids or [],
        "base_model_id": row.base_model_id,
        "gpus_allocated": row.gpus_allocated or [],
        "ray_port": row.ray_port,
        "params": row.params or {},
        "hydra_overrides": row.hydra_overrides or [],
        "run_dir": row.run_dir,
        "auto_register_latest": row.auto_register_latest,
        "metrics_summary": row.metrics_summary or {},
        "error_message": row.error_message,
        "job_id": row.job_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/train/templates")
async def train_templates(principal: PrincipalDep) -> dict:
    return {
        "items": [
            {
                "id": t.id,
                "label": t.label,
                "kind": t.kind,
                "config_path": t.config_path,
                "config_name": t.config_name,
                "script": t.script,
            }
            for t in TEMPLATES
        ]
    }


@router.get("/train")
async def list_train(session: SessionDep, principal: PrincipalDep, status: str | None = None) -> dict:
    q = select(TrainJob).order_by(TrainJob.created_at.desc())
    rows = (await session.scalars(q)).all()
    items = [_job_out(r) for r in rows]
    if status:
        items = [i for i in items if i["status"] == status]
    return {"items": items}


@router.post("/train/preview")
async def preview_train(body: TrainIn, session: SessionDep, principal: PrincipalDep) -> dict:
    settings = get_settings()
    try:
        tpl = get_template(body.template_id)
    except KeyError as e:
        raise HTTPException(400, f"未知模板 {body.template_id}") from e
    extra = list(body.extra_overrides)
    if body.max_epochs is not None:
        extra.append(f"runner.max_epochs={body.max_epochs}")
    if body.total_num_envs is not None:
        extra.append(f"env.train.total_num_envs={body.total_num_envs}")
    if body.lr:
        extra.append(f"actor.optim.lr={body.lr}")
    try:
        ov = build_overrides(body.template_id, body.gpus, body.name, extra)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    emb = await session.get(EmbodimentRow, body.embodiment_id)
    emb_path = str(embodiment_root(emb)) if emb else ""
    mounts = [
        f"{emb_path} → /workspace/berkeley ro" if emb_path else None,
        f"{settings.rlinf_repo} → /workspace/rlinf ro",
        "/var/lib/eb/runs/<id>/output → /workspace/output rw",
    ]
    return {
        "template": tpl.id,
        "overrides": ov,
        "mounts": [m for m in mounts if m],
        "image": body.image or "eb/rlinf:0.4.0-emb",
    }


@router.post("/train")
async def submit_train(body: TrainIn, session: SessionDep, principal: PrincipalDep) -> dict:
    settings = get_settings()
    try:
        tpl = get_template(body.template_id)
    except KeyError as e:
        raise HTTPException(400, f"未知模板 {body.template_id}") from e
    emb = await session.get(EmbodimentRow, body.embodiment_id)
    if not emb:
        raise HTTPException(400, "本体不存在")
    tid = new_id("trn")
    extra = list(body.extra_overrides)
    if body.max_epochs is not None:
        extra.append(f"runner.max_epochs={body.max_epochs}")
    if body.total_num_envs is not None:
        extra.append(f"env.train.total_num_envs={body.total_num_envs}")
    if body.lr:
        extra.append(f"actor.optim.lr={body.lr}")
    try:
        ov = build_overrides(body.template_id, body.gpus, body.name, extra)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    port = ray_port_for(tid)
    run_dir = str(settings.data_root / "runs" / tid)
    Path(run_dir, "output").mkdir(parents=True, exist_ok=True)
    Path(run_dir, "tb").mkdir(parents=True, exist_ok=True)
    image = body.image or "eb/rlinf:0.4.0-emb"
    kind = _KIND.get(body.template_id, JobKind.TRAIN_RLFT)
    spec = JobSpec(
        kind=kind,
        embodiment_env_id=tpl.embodiment_env_id,
        task_id=tpl.task_id,
        name=body.name,
        image=image,
        params={
            "template_id": tpl.id,
            "config_name": tpl.config_name,
            "config_path": tpl.config_path,
            "script": tpl.script,
            "stack": "rlinf",
            "ray_port": port,
            "task": body.gym_id or "",
        },
        overrides=ov,
        resources={"gpus": body.gpus, "shm_gb": 32},
        code_refs={
            "embodiment": {"kind": "git_external", "path": str(embodiment_root(emb))},
            "stack": {"kind": "git_external", "path": str(settings.rlinf_repo)},
        },
        runtime={"ray": {"port": port}},
    )
    try:
        job = await orch.submit(session, spec, principal.id, None)
    except (ValueError, PermissionError) as e:
        raise HTTPException(400, str(e)) from e
    row = TrainJob(
        id=tid,
        name=body.name,
        status=job.status,
        template_id=tpl.id,
        image_digest=image,
        embodiment_id=body.embodiment_id,
        dataset_ids=body.dataset_ids,
        base_model_id=body.base_model_id,
        gpus_allocated=[],
        ray_port=port,
        params=spec.params,
        hydra_overrides=ov,
        run_dir=run_dir,
        auto_register_latest=body.auto_register_latest,
        job_id=job.id,
    )
    session.add(row)
    session.add(
        LineageEdge(
            src_type="embodiment",
            src_id=body.embodiment_id,
            dst_type="train_job",
            dst_id=tid,
            relation="used_embodiment",
        )
    )
    for dsid in body.dataset_ids:
        session.add(
            LineageEdge(
                src_type="dataset",
                src_id=dsid,
                dst_type="train_job",
                dst_id=tid,
                relation="used_dataset",
            )
        )
    await session.commit()
    await session.refresh(row)
    return _job_out(row)


@router.get("/train/{tid}")
async def get_train(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TrainJob, tid)
    if not row:
        raise HTTPException(404)
    if row.job_id:
        job = await session.get(Job, row.job_id)
        if job:
            row.status = job.status
            row.error_message = job.error
    await session.commit()
    return _job_out(row)


@router.get("/train/{tid}/checkpoints")
async def list_checkpoints(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TrainJob, tid)
    if not row:
        raise HTTPException(404)
    root = Path(row.run_dir) / "output" / "checkpoints"
    items = []
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".pt", ".pth", ".onnx", ".safetensors"}:
                items.append({"path": str(p), "name": p.name, "bytes": p.stat().st_size})
    return {"items": items}


class RegisterCkptIn(BaseModel):
    path: str
    name: str | None = None


@router.post("/train/{tid}/register-model")
async def register_ckpt(tid: str, body: RegisterCkptIn, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.paths import PathNotAllowed, validate_allowed_path

    row = await session.get(TrainJob, tid)
    if not row:
        raise HTTPException(404)
    try:
        path = validate_allowed_path(body.path)
    except PathNotAllowed as e:
        raise HTTPException(400, str(e)) from e
    mid = new_id("mdl")
    model = WeightModel(
        id=mid,
        name=body.name or path.name,
        kind="trained",
        path=str(path),
        size_bytes=dir_size(path),
        embodiment_id=row.embodiment_id,
        train_job_id=tid,
        parent_model_id=row.base_model_id,
    )
    session.add(model)
    session.add(
        LineageEdge(
            src_type="train_job",
            src_id=tid,
            dst_type="model",
            dst_id=mid,
            relation="produced_by",
        )
    )
    if row.base_model_id:
        session.add(
            LineageEdge(
                src_type="model",
                src_id=row.base_model_id,
                dst_type="model",
                dst_id=mid,
                relation="trained_from",
            )
        )
    await session.commit()
    return {"id": mid, "name": model.name, "path": model.path}
