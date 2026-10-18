from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import HTTPException
from sqlalchemy import delete, select, text

from eb.domain.fsm import JOB_TERMINAL
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import (
    Artifact,
    AuditLog,
    Bundle,
    Dataset,
    DatasetVersion,
    EvalReport,
    Job,
    LineageEdge,
    ModelVersion,
    Pipeline,
    PipelineRun,
    Run,
    Share,
    TrainJob,
    WeightModel,
)

# Submissions share this lock; a deletion reserves its inputs before touching files.
STORAGE_LOCK = 730142
TABLES = (
    Job,
    Run,
    TrainJob,
    WeightModel,
    ModelVersion,
    Artifact,
    EvalReport,
    Bundle,
    Dataset,
    DatasetVersion,
    Pipeline,
    PipelineRun,
    Share,
)


def snapshot(row) -> dict:
    return {
        column.key: (value.isoformat() if isinstance(value, datetime) else value)
        for column in row.__table__.columns
        for value in [getattr(row, row.__mapper__.get_property_by_column(column).key)]
    }


def references(value, ids: set[str], paths: list[str]) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_receipt" and isinstance(item, dict):
                # Source-build citations never load a parent checkpoint; tombstones keep their history.
                item = {
                    k: v for k, v in item.items() if k not in {"guardfix_parent_run", "extension_parent_run"}
                }
            if references(item, ids, paths):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(references(item, ids, paths) for item in value)
    if not isinstance(value, str):
        return False
    if not any(key in value for key in ids) and not any(
        path in value or (value.startswith("/") and path.startswith(value.rstrip("/") + "/"))
        for path in paths
    ):
        return False
    if value.startswith(("{", "[")):
        try:
            return references(json.loads(value), ids, paths)
        except ValueError:
            pass
    tokens = set(re.findall(r"[A-Za-z0-9_-]+", value))
    if ids & tokens:
        return True
    for path in paths:
        # Covers frozen JSON paths and path-valued Hydra overrides.
        if re.search(r"(?<![\w/.-])" + re.escape(path.rstrip("/")) + r"(?=$|[/\s'\",;)])", value):
            return True
        if value.startswith("/") and "=" not in value:
            candidate = Path(value)
            target = Path(path)
            if candidate == target or candidate in target.parents:
                return True
    return False


async def guard_storage_inputs(session, value) -> None:
    await session.execute(text(f"SELECT pg_advisory_xact_lock_shared({STORAGE_LOCK})"))
    receipts = (await session.scalars(select(AuditLog).where(AuditLog.action == "storage.delete"))).all()
    for receipt in receipts:
        plan = receipt.before or {}
        if references(value, set(plan.get("ids", [])), plan.get("paths", [])):
            raise ValueError(f"输入已删除或正在清理：{receipt.subject}")


def managed_model_path(path: str) -> bool:
    if not path or not Path(path).is_absolute() or ".." in Path(path).parts:
        return False
    settings = get_settings()
    for root in (settings.models_root, settings.data_root / "runs"):
        try:
            parts = Path(path).relative_to(root).parts
        except ValueError:
            continue
        if root == settings.models_root:
            if len(parts) >= 2 and parts[0] == "runs" and re.fullmatch(r"run_[\w-]+", parts[1]):
                return True
            if len(parts) >= 1 and re.fullmatch(r"mdl_[\w-]+", parts[0]):
                return True
        elif len(parts) >= 2 and re.fullmatch(r"run_[\w-]+", parts[0]):
            return True
    return False


async def deletion_plan(session, kind: str, object_id: str) -> dict:
    records = {table.__tablename__: list((await session.scalars(select(table))).all()) for table in TABLES}
    removed: dict[str, list] = {}
    paths, preserved = [], []
    run_ids = []
    if kind == "model":
        model = next(
            (
                r
                for table in (WeightModel, ModelVersion)
                for r in records[table.__tablename__]
                if r.id == object_id
            ),
            None,
        )
        if model is None:
            raise HTTPException(404, "模型不存在")
        removed[model.__tablename__] = [model]
        if isinstance(model, WeightModel):
            download = (model.metrics_at_checkpoint or {}).get("download") or {}
            producer = next((j for j in records["jobs"] if j.id == download.get("job_id")), None)
            if producer and producer.status not in JOB_TERMINAL:
                raise HTTPException(409, "下载仍在运行，请先取消任务")
            paths = [model.path or download.get("model_path", "")]
        else:
            artifacts = [a for a in records["artifacts"] if a.id == model.artifact_id]
            removed["artifacts"] = artifacts
            paths = [a.storage for a in artifacts]
    else:
        train = (
            next((r for r in records["train_jobs"] if r.id == object_id), None) if kind == "train" else None
        )
        if kind == "train" and train is None:
            raise HTTPException(404, "训练任务不存在")
        job_id = train.job_id if train else object_id
        job = next((r for r in records["jobs"] if r.id == job_id), None)
        expected = (
            {"train.sft", "train.rl", "train.rlft"}
            if kind == "train"
            else ({"asset.validate"} if kind == "job" else {"eval.sim", "eval.sim2sim"})
        )
        failed_job = job and job.status in {"failed", "infra_failed", "timeout"}
        if job and job.kind not in expected and not (kind == "job" and failed_job):
            raise HTTPException(409, "任务类型不匹配")
        if not job and not train:
            raise HTTPException(404, "评估任务不存在")
        if (job.status if job else train.status) not in JOB_TERMINAL:
            raise HTTPException(409, "只能删除已结束的任务，请先取消活动任务")
        runs = [r for r in records["runs"] if r.job_id == job_id]
        if any(r.status not in {"finished", "aborted"} for r in runs):
            raise HTTPException(409, "Run 尚未完成收集，请稍后重试")
        run_ids = [r.id for r in runs]
        removed["jobs"] = [job] if job else []
        removed["train_jobs"] = [r for r in records["train_jobs"] if r.job_id == job_id or r is train]
        removed["runs"] = runs
        removed["artifacts"] = [a for a in records["artifacts"] if a.run_id in run_ids]
        removed["eval_reports"] = [r for r in records["eval_reports"] if r.run_id in run_ids]
        settings = get_settings()
        paths = [
            str(root / rid)
            for rid in run_ids
            for root in (
                settings.data_root / "runs",
                settings.data_root / "tsdb",
                settings.models_root / "runs",
            )
        ]
        if failed_job and job.spec.get("params", {}).get("operation") == "download_model":
            for weight in records["weight_models"]:
                download = (weight.metrics_at_checkpoint or {}).get("download") or {}
                if (
                    weight.kind == "download_failed"
                    and download.get("job_id") == job.id
                    and job.spec["params"].get("model_id") == weight.id
                ):
                    removed.setdefault("weight_models", []).append(weight)
                    path = weight.path or download.get("model_path", "")
                    if path:
                        (paths if managed_model_path(path) else preserved).append(path)
        for projection in removed["train_jobs"]:
            if projection.run_dir and projection.run_dir not in paths:
                # Legacy trn_* directories are not Run-owned; never infer ownership from a path.
                preserved.append(projection.run_dir)

    paths = list(dict.fromkeys(p for p in paths if p))
    if kind == "model":
        preserved = [p for p in paths if not managed_model_path(p)]
        paths = [p for p in paths if managed_model_path(p)]
    ids = {r.id for rows in removed.values() for r in rows}
    suppressed_runs = []
    if kind == "model":
        if isinstance(model, ModelVersion) and (model.card or {}).get("run_id"):
            suppressed_runs.append(model.card["run_id"])
        suppressed_runs.extend(a.run_id for a in removed.get("artifacts", []) if a.run_id)
        if isinstance(model, WeightModel) and model.train_job_id:
            producers = {r.job_id for r in records["train_jobs"] if r.id == model.train_job_id}
            suppressed_runs.extend(r.id for r in records["runs"] if r.job_id in producers)
    producer_jobs = {r.job_id for r in records["runs"] if r.id in suppressed_runs}
    blockers = []
    for table in TABLES:
        for row in records[table.__tablename__]:
            if row in removed.get(table.__tablename__, []):
                continue
            data = snapshot(row)
            data.pop("id", None)
            if (isinstance(row, Run) and row.id in suppressed_runs) or (
                isinstance(row, TrainJob) and row.job_id in producer_jobs
            ):
                # A producer's output index is historical evidence, not a model input.
                summary = dict(data.get("metrics_summary") or {})
                summary.pop("checkpoint", None)
                if summary.get("model_version_id") == object_id:
                    summary.pop("model_version_id")
                    data["metrics_summary"] = summary
            if references(data, ids, paths):
                blockers.append(
                    {
                        "type": table.__tablename__,
                        "id": getattr(row, "id", getattr(row, "token", None)),
                        "name": getattr(row, "name", None),
                    }
                )
    for artifact in removed.get("artifacts", []):
        if artifact.refcount or (artifact.pinned and kind != "model"):
            blockers.append({"type": "pinned_artifact", "id": artifact.id})
    live_ids = {getattr(r, "id", None) for rows in records.values() for r in rows} - ids
    edges = (await session.scalars(select(LineageEdge).where(LineageEdge.src_id.in_(ids)))).all()
    for edge in edges:
        if edge.dst_id in live_ids and not any(b["id"] == edge.dst_id for b in blockers):
            blockers.append({"type": edge.dst_type, "id": edge.dst_id})
    return {
        "kind": kind,
        "id": object_id,
        "ids": sorted(ids),
        "run_ids": run_ids,
        "paths": paths,
        "suppressed_model_runs": sorted(set(suppressed_runs)),
        "preserved_paths": preserved,
        "blockers": blockers,
        "records": {table: [snapshot(r) for r in rows] for table, rows in removed.items()},
    }


async def delete_object(session, principal, kind: str, object_id: str, *, preview: bool = False) -> dict:
    if principal.actor_type != "user":
        raise HTTPException(403, "删除仅允许用户操作，平台 Agent 不可执行")
    subject = f"{kind}:{object_id}"
    await session.execute(text(f"SELECT pg_advisory_xact_lock({STORAGE_LOCK})"))
    receipt = await session.scalar(
        select(AuditLog)
        .where(AuditLog.action == "storage.delete", AuditLog.subject == subject)
        .with_for_update()
    )
    if receipt:
        if (receipt.after or {}).get("state") == "completed":
            return {"ok": True, "already_deleted": True, **receipt.after}
        plan = receipt.before
    else:
        plan = await deletion_plan(session, kind, object_id)
    if preview:
        return {k: v for k, v in plan.items() if k != "records"}
    if plan["blockers"]:
        raise HTTPException(
            409, {"message": "仍被其他对象引用，请先处理引用后再删除", "blockers": plan["blockers"]}
        )
    if receipt is None:
        receipt = AuditLog(
            principal=principal.id,
            action="storage.delete",
            subject=subject,
            before=plan,
            after={"state": "pending"},
        )
        session.add(receipt)
        await session.commit()  # Durable reservation blocks new consumers, even after a crash.
        await session.execute(text(f"SELECT pg_advisory_xact_lock({STORAGE_LOCK})"))
        await session.refresh(receipt, with_for_update=True)
        if (receipt.after or {}).get("state") == "completed":
            return {"ok": True, "already_deleted": True, **receipt.after}
    try:
        result = await EbdClient().delete_storage(plan["paths"], plan["run_ids"])
    except httpx.HTTPStatusError as exc:
        await session.rollback()
        try:
            reason = exc.response.json().get("error") or "ebd 文件清理失败"
        except ValueError:
            reason = "ebd 文件清理失败"
        raise HTTPException(409, f"{reason}；清理记录已保留，处理后重试删除") from exc
    except Exception as exc:
        await session.rollback()
        raise HTTPException(503, "文件清理未完成，删除记录已保留；重试同一删除请求可继续") from exc
    # Foreign keys require children first. Bloodline edges and the audit tombstone stay.
    for table in (EvalReport, Artifact, WeightModel, ModelVersion, TrainJob, Run, Job):
        keys = [r["id"] for r in plan["records"].get(table.__tablename__, [])]
        if keys:
            await session.execute(delete(table).where(table.id.in_(keys)))
    receipt.after = {
        "state": "completed",
        "id": object_id,
        "files": result,
        "preserved_paths": plan["preserved_paths"],
    }
    await session.commit()
    return {"ok": True, **receipt.after}
