from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from passlib.hash import pbkdf2_sha256
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from eb.agent.playbooks import catalog as playbook_catalog
from eb.agent.playbooks import dispatch as playbook_dispatch
from eb.agent.tools import AgentCtx  # noqa: F401  — register tools
from eb.api.deps import PrincipalDep, SessionDep
from eb.capacity import estimate_job
from eb.domain.contract import EmbodimentEnv
from eb.domain.data_view import (
    compact_dataset_stats,
    dataset_overview,
    episode_media,
    license_gate,
    normalize_funnel,
    quality_summary,
    slice_preview,
)
from eb.domain.fsm import JOB_TERMINAL, JobStatus
from eb.domain.job import JobSpec
from eb.gates.engine import evaluate_gate_dsl, g3_model
from eb.harness.toolbus import ToolBus
from eb.ids import new_id
from eb.orchestrator.service import Orchestrator
from eb.pipelines import start_pipeline
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import (
    AgentSession,
    AgentStep,
    Artifact,
    Bundle,
    Dataset,
    DatasetVersion,
    EmbodimentEnvRow,
    EmbodimentRow,
    EvalReport,
    EvalSuite,
    Experiment,
    Job,
    LineageEdge,
    ModelFamily,
    ModelVersion,
    WeightModel,
    Pipeline,
    Robot,
    Run,
    Share,
    TrainJob,
    User,
)

health_router = APIRouter()
api_v1 = APIRouter()
ws_router = APIRouter()
media_router = APIRouter()
spa_router = APIRouter()
orch = Orchestrator()
bus = ToolBus()


class LoginBody(BaseModel):
    username: str
    password: str


class JobBody(BaseModel):
    spec: dict[str, Any]


@health_router.get("/healthz")
async def healthz() -> dict:
    ebd = await EbdClient().health()
    return {"ok": True, "ebd": ebd}


@api_v1.post("/auth/login")
async def login(body: LoginBody, request: Request, session: SessionDep) -> dict:
    user = await session.scalar(select(User).where(User.username == body.username))
    if not user or not pbkdf2_sha256.verify(body.password, user.password_hash):
        raise HTTPException(401, "invalid credentials")
    request.session["user"] = user.username
    return {"username": user.username, "actor_type": user.actor_type}


@api_v1.post("/auth/logout")
async def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@api_v1.get("/auth/me")
async def me(principal: PrincipalDep) -> dict:
    return {"principal": principal.id, "actor_type": principal.actor_type}


@api_v1.get("/datasets")
async def list_datasets(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Dataset))).all()
    items = []
    for r in rows:
        if (r.meta or {}).get("validation_job_id"):
            from eb.api.routes_data import reconcile_dataset
            await reconcile_dataset(r, session)
        q = select(DatasetVersion).where(DatasetVersion.dataset_id == r.id).order_by(DatasetVersion.created_at)
        vers = (await session.scalars(q)).all()
        vdicts = [
            _row(v, ["id", "version", "stats", "funnel", "compatible_envs", "content_hash"])
            for v in vers
        ]
        environment = await session.get(EmbodimentEnvRow, r.embodiment_env_id) if r.embodiment_env_id else None
        embodiment = await session.get(EmbodimentRow, environment.embodiment_id) if environment else None
        overview = dataset_overview(
            name=r.name, fmt=r.format, license=r.license, embodiment_env_id=r.embodiment_env_id, versions=vdicts,
            summary_stats=r.summary_stats, environment=environment.spec if environment else None,
        )
        items.append(
            {
                **_row(r, ["id", "name", "format", "license", "embodiment_env_id", "meta"]),
                "embodiment_id": environment.embodiment_id if environment else None,
                "embodiment_name": embodiment.display_name if embodiment else None,
                "usage": (r.meta or {}).get("usage", "train"),
                **overview,
                "summary_stats": compact_dataset_stats(r.summary_stats),
                "validation": (r.summary_stats or {}).get("validation", {"status": "not_validated", "training_ready": False}),
                "versions": vdicts,
                "source_kind": getattr(r, "source_kind", "path"),
                "source_uri": getattr(r, "source_uri", "") or "",
                "dest_path": getattr(r, "dest_path", "") or "",
                "status": getattr(r, "status", "ready"),
                "bytes": getattr(r, "total_bytes", 0) or 0,
                "total_bytes": getattr(r, "total_bytes", 0) or 0,
                "download_progress": getattr(r, "download_progress", None) or {},
                "error_message": getattr(r, "error_message", None),
            }
        )
    return {"items": items}


@api_v1.get("/datasets/{did}")
async def get_dataset(did: str, session: SessionDep, principal: PrincipalDep) -> dict:
    ds = await session.get(Dataset, did)
    if not ds:
        raise HTTPException(404)
    from eb.api.routes_data import reconcile_dataset
    await reconcile_dataset(ds, session)
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == did).order_by(DatasetVersion.created_at)
    vers = (await session.scalars(q)).all()
    fields = ["id", "version", "stats", "funnel", "compatible_envs", "content_hash"]
    vdicts = [_row(v, fields) for v in vers]
    environment = await session.get(EmbodimentEnvRow, ds.embodiment_env_id) if ds.embodiment_env_id else None
    embodiment = await session.get(EmbodimentRow, environment.embodiment_id) if environment else None
    overview = dataset_overview(
        name=ds.name, fmt=ds.format, license=ds.license, embodiment_env_id=ds.embodiment_env_id, versions=vdicts,
        summary_stats=ds.summary_stats, environment=environment.spec if environment else None,
    )
    latest = vdicts[-1] if vdicts else {}
    quality = quality_summary(
        overview["funnel"],
        license=ds.license,
        compatible_envs=(latest.get("compatible_envs") if latest else None),
        stats={**(latest.get("stats") or {}), **compact_dataset_stats(ds.summary_stats),
               "interface_compatibility": overview["interface_compatibility"]},
    )
    dest = getattr(ds, "dest_path", None) or ((ds.meta or {}).get("root") if ds.meta else None)
    return {
        **_row(ds, ["id", "name", "format", "license", "embodiment_env_id", "meta"]),
        "embodiment_id": environment.embodiment_id if environment else None,
        "embodiment_name": embodiment.display_name if embodiment else None,
        "usage": (ds.meta or {}).get("usage", "train"),
        "summary_stats": compact_dataset_stats(ds.summary_stats),
        "validation": (ds.summary_stats or {}).get("validation", {"status": "not_validated", "training_ready": False}),
        "versions": vdicts,
        "overview": overview,
        "quality": quality,
        "license_gate": license_gate(ds.license),
        "source_kind": getattr(ds, "source_kind", None) or "path",
        "source_uri": getattr(ds, "source_uri", None) or "",
        "dest_path": dest,
        "status": getattr(ds, "status", None) or "ready",
        "error_message": getattr(ds, "error_message", None),
        "total_bytes": getattr(ds, "total_bytes", None) or 0,
    }


@api_v1.get("/datasets/{did}/versions/{vid}/funnel")
async def dataset_funnel(did: str, vid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    v = await session.get(DatasetVersion, vid)
    if not v or v.dataset_id != did:
        raise HTTPException(404)
    ds = await session.get(Dataset, did)
    raw = dict(v.funnel or {})
    meta = (ds.meta if ds else {}) or {}
    if meta.get("funnel_kind"):
        raw["kind"] = meta["funnel_kind"]
    out = normalize_funnel(raw)
    if out["empty"]:
        out["empty_reason"] = (
            "尚无编译报告" if out["kind"] == "compile" else "尚无扫描结果或各层均为 0"
        )
    return out


@api_v1.get("/datasets/{did}/versions/{vid}/episodes/{ep}")
async def episode_preview(did: str, vid: str, ep: str, session: SessionDep, principal: PrincipalDep) -> dict:
    v = await session.get(DatasetVersion, vid)
    if not v or v.dataset_id != did:
        raise HTTPException(404)
    ds = await session.get(Dataset, did)
    from urllib.parse import quote
    root = Path((ds.dest_path if ds else "") or ((ds.meta if ds else {}) or {}).get("root") or "")
    payload = episode_media(root, ep, ((ds.summary_stats if ds else {}) or {}).get("episodes", []))
    if payload.get("path"):
        relative = str(Path(payload["path"]).relative_to(root))
        payload["media_url"] = f"/api/v1/datasets/{did}/media?path={quote(relative, safe='')}"
    return payload


@api_v1.post("/datasets/{did}/slice/preview")
async def dataset_slice_preview(did: str, body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    ds = await session.get(Dataset, did)
    if not ds:
        raise HTTPException(404)
    ver = await session.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == did))
    stats = (ver.stats if ver else {}) or {}
    return slice_preview(stats, body or {})


@api_v1.get("/models")
async def list_models(session: SessionDep, principal: PrincipalDep) -> dict:
    fams = (await session.scalars(select(ModelFamily))).all()
    vers = (await session.scalars(select(ModelVersion))).all()
    trains = (await session.scalars(select(TrainJob))).all()
    train_names = {t.id: t.name for t in trains}
    job_train_names = {t.job_id: t.name for t in trains if t.job_id}
    job_train_ids = {t.job_id: t.id for t in trains if t.job_id}
    from eb.store.models import WeightModel

    weights = (await session.scalars(select(WeightModel))).all()
    from eb.pbfm_workbench import model_links
    diagnostic_links = await model_links(session)
    return {
        "families": [_row(f, ["id", "name", "family"]) for f in fams],
        "versions": [{**_row(v, ["id", "family_id", "name", "embodiment_env_id", "provenance", "tags", "io_signature", "card", "created_at"]),
                      "train_job_name": job_train_names.get((v.card or {}).get("train_job_id")),
                      "train_page_id": job_train_ids.get((v.card or {}).get("train_job_id")),
                      **diagnostic_links.get(v.id, {})} for v in vers],
        "items": [
            {
                "id": w.id,
                "name": w.name,
                "kind": w.kind,
                "path": w.path,
                "embodiment_id": w.embodiment_id,
                "train_job_id": w.train_job_id,
                "train_job_name": train_names.get(w.train_job_id),
                "parent_model_id": w.parent_model_id,
            }
            for w in weights
        ],
    }


@api_v1.get("/models/{mid}")
async def get_model(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    v = await session.get(ModelVersion, mid)
    if v:
        out = _row(v, ["id", "family_id", "name", "embodiment_env_id", "io_signature", "mem_profile", "card", "tags", "provenance", "created_at"])
        train_job_id = (v.card or {}).get("train_job_id")
        producer = await session.scalar(select(TrainJob).where(TrainJob.job_id == train_job_id)) if train_job_id else None
        if producer:
            out["train_job_name"] = producer.name
            out["train_page_id"] = producer.id
        elif (v.card or {}).get("schema_version") == "eb/rpo-pbfm-trained-model/v1":
            from eb.pbfm_workbench import model_links
            out.update((await model_links(session)).get(v.id, {}))
        if v.artifact_id:
            artifact = await session.get(Artifact, v.artifact_id)
            if artifact:
                out["artifact"] = _row(artifact, ["id", "kind", "sha256", "size_bytes", "storage", "meta"])
        if (v.card or {}).get("schema_version") == "eb/rpo-wbc-model/v1" and not (v.card or {}).get("training"):
            source_run = await session.get(Run, (v.card or {}).get("run_id"))
            producer = await session.get(Job, source_run.job_id) if source_run else None
            if producer:
                out["card"] = {**(v.card or {}), "training":{
                    "kind":producer.kind, "image":(producer.spec or {}).get("image"),
                    "resources":(producer.spec or {}).get("resources") or {},
                    "overrides":(producer.spec or {}).get("overrides") or []}}
        reports = (await session.scalars(select(EvalReport).where(EvalReport.model_version_id == v.id)
                                         .order_by(EvalReport.created_at.desc()).limit(20))).all()
        out["eval_reports"] = [_row(report, ["id", "suite_id", "run_id", "metrics", "gate_results", "created_at"])
                               for report in reports]
        if v.embodiment_env_id:
            crow = await session.get(EmbodimentEnvRow, v.embodiment_env_id)
            if crow:
                ctr = EmbodimentEnv.model_validate(crow.spec)
                io = v.io_signature or {}
                out["env_check"] = g3_model(io.get("obs"), io.get("act"), ctr).model_dump()
                out["env"] = {"obs_dim": ctr.observation.dim, "action_dim": ctr.action.dim, "policy_hz": ctr.control.policy_hz}
        return out
    w = await session.get(WeightModel, mid)
    if not w:
        raise HTTPException(404)
    producer = await session.get(TrainJob, w.train_job_id) if w.train_job_id else None
    edges = (
        await session.scalars(select(LineageEdge).where((LineageEdge.src_id == mid) | (LineageEdge.dst_id == mid)))
    ).all()
    return {
        "id": w.id,
        "name": w.name,
        "kind": w.kind,
        "path": w.path,
        "framework": w.framework,
        "size_bytes": w.size_bytes,
        "embodiment_id": w.embodiment_id,
        "train_job_id": w.train_job_id,
        "train_job_name": producer.name if producer else None,
        "parent_model_id": w.parent_model_id,
        "metrics_at_checkpoint": w.metrics_at_checkpoint or {},
        "created_at": w.created_at.isoformat() if w.created_at else None,
        "edges": [_row(e, ["src_type", "src_id", "dst_type", "dst_id", "relation"]) for e in edges],
        "is_weight": True,
    }


@api_v1.get("/models/{mid}/lineage")
async def model_lineage(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    edges = (await session.scalars(select(LineageEdge).where((LineageEdge.src_id == mid) | (LineageEdge.dst_id == mid)))).all()
    model = await session.get(WeightModel, mid) or await session.get(ModelVersion, mid)
    if not model:
        raise HTTPException(404)
    endpoints = {(e.src_type, e.src_id) for e in edges} | {(e.dst_type, e.dst_id) for e in edges}
    endpoints.add(("model" if isinstance(model, WeightModel) else "model_version", mid))
    nodes = []
    for kind, node_id in sorted(endpoints):
        name, href = node_id, None
        if kind in {"model", "model_version"}:
            target = await session.get(WeightModel, node_id) if kind == "model" else await session.get(ModelVersion, node_id)
            if not target and kind == "model":  # Older Run edges use "model" for a ModelVersion.
                target = await session.get(ModelVersion, node_id)
            if target:
                name, href = target.name, f"/models/{node_id}"
        elif kind == "train_job":
            target = await session.get(TrainJob, node_id)
            if target:
                name, href = target.name, f"/train/{node_id}"
        elif kind in {"job", "run"}:
            run = await session.get(Run, node_id) if kind == "run" else None
            job = await session.get(Job, run.job_id if run else node_id)
            if job:
                train = await session.scalar(select(TrainJob).where(TrainJob.job_id == job.id))
                name = train.name if train else (job.spec or {}).get("name") or node_id
                href = f"/train/{train.id}" if train else (f"/train/runs/{node_id}" if run else f"/eval/jobs/{node_id}")
        elif kind == "dataset":
            target = await session.get(Dataset, node_id)
            if target:
                name, href = target.name, f"/data/{node_id}"
        elif kind == "embodiment":
            target = await session.get(EmbodimentRow, node_id)
            if target:
                name, href = target.display_name, f"/embodiments/{node_id}"
        nodes.append({"type": kind, "id": node_id, "name": name, "href": href})
    return {"edges": [_row(e, ["src_type", "src_id", "dst_type", "dst_id", "relation"]) for e in edges],
            "nodes": nodes}


@api_v1.get("/experiments")
async def list_experiments(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Experiment))).all()
    return {"items": [_row(r, ["id", "name", "embodiment_env_id"]) for r in rows]}


@api_v1.post("/jobs")
async def submit_job(body: JobBody, request: Request, session: SessionDep, principal: PrincipalDep) -> dict:
    spec = JobSpec.model_validate(body.spec)
    try:
        job = await orch.submit(session, spec, principal.id, request.headers.get("idempotency-key"))
    except PermissionError as e:
        raise HTTPException(400, detail=e.args[0]) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return _job_out(job)


@api_v1.get("/jobs")
async def list_jobs(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Job).order_by(Job.created_at.desc()).limit(100))).all()
    return {"items": [_job_out(j) for j in rows]}


@api_v1.get("/jobs/{jid}")
async def get_job(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    job = await session.get(Job, jid)
    if not job:
        raise HTTPException(404)
    runs = (await session.scalars(select(Run).where(Run.job_id == jid))).all()
    return {**_job_out(job), "runs": [_row(r, ["id", "attempt", "status", "abort_reason", "metrics_summary"]) for r in runs]}


@api_v1.get("/jobs/{jid}/equivalent-command")
async def eq_cmd(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    job = await session.get(Job, jid)
    if not job:
        raise HTTPException(404)
    spec = JobSpec.model_validate({**job.spec, "kind": job.kind, "embodiment_env_id": job.embodiment_env_id})
    return {"command": orch.preview_command(spec, "run_preview")}


@api_v1.post("/jobs/preview-command")
async def preview_job_command(body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.adapters import equivalent_command
    spec_dict = dict(body.get("spec") or body)
    if spec_dict.get("kind", "").startswith(("train.", "eval.")) and not spec_dict.get("task_id"):
        spec_dict["task_id"] = "tsk_bhl_velocity_biped"
    try:
        spec = JobSpec.model_validate(spec_dict)
        cmd = equivalent_command(spec, "run_preview")
        return {"command": cmd, "ok": True}
    except Exception as e:
        return {"command": f"# spec 待完善: {e}", "ok": False, "error": str(e)}


@api_v1.post("/jobs/{jid}/cancel")
async def cancel_job(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from datetime import UTC, datetime
    job = await session.get(Job, jid, with_for_update=True, populate_existing=True)
    if not job:
        raise HTTPException(404)
    if JobStatus(job.status) in JOB_TERMINAL:
        raise HTTPException(409, "terminal jobs cannot be cancelled")
    run = await session.scalar(select(Run).where(Run.job_id == jid).order_by(Run.attempt.desc()))
    if run and run.id:
        try:
            await EbdClient().cancel_run(run.id)
        except Exception as exc:
            raise HTTPException(503, "container cancellation failed; job remains active") from exc
        run.status = "aborted"
        run.abort_reason = "cancelled"
        run.finished_at = datetime.now(UTC)
    job.status = JobStatus.CANCELLED.value
    if run:
        from eb.orchestrator.collect import archive_partial_eval
        try:
            await archive_partial_eval(session, job, run)
        except (ValueError, OSError, KeyError, TypeError, httpx.HTTPError) as exc:
            operation = ((getattr(job, "spec", None) or {}).get("params") or {}).get("operation")
            if operation != "ufo_native_live":
                raise
            # Confirmed removal must remain durable even if diagnostic files are damaged.
            run.metrics_summary = {**(run.metrics_summary or {}), "diagnostic_only": True,
                                   "complete": False, "evidence_errors": [str(exc)[:500]]}
            job.error = "交互环境已关闭；部分诊断记录收集失败。"
    await session.commit()
    return _job_out(job)


@api_v1.get("/runs/compare")
async def compare_runs(ids: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.stats import significant, spec_diff

    parts = [x.strip() for x in ids.split(",") if x.strip()]
    if len(parts) < 2:
        raise HTTPException(400, "need two run ids")
    a = await session.get(Run, parts[0])
    b = await session.get(Run, parts[1])
    if not a or not b:
        raise HTTPException(404)
    ja = await session.get(Job, a.job_id)
    jb = await session.get(Job, b.job_id)
    diff = spec_diff(ja.spec if ja else {}, jb.spec if jb else {})
    series_a = {}
    series_b = {}
    try:
        series_a = (await EbdClient().tsdb_query(a.id)).get("series") or {}
        series_b = (await EbdClient().tsdb_query(b.id)).get("series") or {}
    except Exception:
        pass
    tag = "Train/mean_reward"
    pa = [p["value"] for p in (series_a.get(tag) or [])[-50:]]
    pb = [p["value"] for p in (series_b.get(tag) or [])[-50:]]
    sig = significant(pa, pb) if pa and pb else False
    return {
        "ids": parts[:2],
        "diff": diff,
        "series": {parts[0]: series_a, parts[1]: series_b},
        "significant": sig,
        "label": "差异显著" if sig else "差异不显著",
    }


@api_v1.get("/runs/{rid}")
async def get_run(rid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    run = await session.get(Run, rid)
    if not run:
        raise HTTPException(404)
    arts = (await session.scalars(select(Artifact).where(Artifact.run_id == rid))).all()
    return {
        **_row(run, ["id", "job_id", "attempt", "status", "abort_reason", "metrics_summary", "gpu_ids", "provenance"]),
        "artifacts": [_row(a, ["id", "kind", "sha256", "size_bytes", "storage"]) for a in arts],
    }


@api_v1.get("/runs/{rid}/metrics")
async def run_metrics(rid: str, session: SessionDep, principal: PrincipalDep, tag: str | None = None) -> dict:
    run = await session.get(Run, rid)
    if not run:
        raise HTTPException(404)
    try:
        return await EbdClient().tsdb_query(rid, tag)
    except Exception:
        summary = run.metrics_summary or {}
        series = {k: v for k, v in summary.items() if isinstance(v, list)}
        return {"run_id": rid, "series": series, "provenance": run.provenance}


@api_v1.get("/runs/{rid}/logs")
async def run_logs(rid: str, session: SessionDep, principal: PrincipalDep, tail: int = 200) -> dict:
    run = await session.get(Run, rid)
    if not run:
        raise HTTPException(404)
    try:
        return await EbdClient().logs(rid, tail=tail)
    except Exception as e:
        return {"run_id": rid, "lines": [], "error": str(e)}


@api_v1.get("/evals/suites")
async def list_suites(session: SessionDep, principal: PrincipalDep, benchmark: str | None = None) -> dict:
    rows = (await session.scalars(select(EvalSuite))).all()
    if benchmark:
        rows = [row for row in rows if (row.spec or {}).get("benchmark", "libero") == benchmark]
    return {"items": [_row(r, ["id", "name", "embodiment_env_id", "seeds", "frozen", "spec"]) for r in rows]}


@api_v1.get("/evals/reports")
async def list_reports(session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.eval.wilson import report_interval
    rows = (await session.scalars(select(EvalReport))).all()
    items = []
    for r in rows:
        item = _row(r, ["id", "suite_id", "model_version_id", "run_id", "metrics", "gate_results"])
        metrics = item.get("metrics") or {}
        item["wilson_ci"] = report_interval(metrics)
        items.append(item)
    return {"items": items}


@api_v1.get("/evals/reports/{eid}")
async def get_report(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.eval.wilson import report_interval
    row = await session.get(EvalReport, eid)
    if not row:
        raise HTTPException(404)
    item = _row(row, ["id", "suite_id", "model_version_id", "run_id", "metrics", "gate_results"])
    metrics = item.get("metrics") or {}
    item["wilson_ci"] = report_interval(metrics)
    return item


@api_v1.get("/evals/reports/{eid}/episodes")
async def get_report_episodes(eid: str, session: SessionDep, principal: PrincipalDep,
                              task_id: int | None = None, failures_only: bool = False) -> dict:
    """Serve bounded case evidence from a validated report, never a client path."""
    import json
    import re
    row = await session.get(EvalReport, eid)
    if not row:
        raise HTTPException(404)
    if not row.run_id or not re.fullmatch(r"run_[A-Za-z0-9_-]+", row.run_id):
        return {"items": [], "available": False}
    from eb.settings import get_settings
    runs_root = get_settings().data_root / "runs"
    run_root = runs_root / row.run_id
    root = run_root / "output"
    if any(part.is_symlink() for part in (runs_root, run_root, root)) or not root.is_dir():
        return {"items": [], "available": False, "total": 0}
    items = []
    evidence_files = sorted(root.glob("eval-episodes-rank*.jsonl"))
    if not evidence_files and (row.suite_id or "").startswith(("suite_rpo_v2_wbc_", "suite_rpo_v2_amp_", "suite_rpo_v2_flat_")):
        evidence_files = [root / "eval-cases.jsonl"] if (root / "eval-cases.jsonl").is_file() else []
    for path in evidence_files:
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise HTTPException(409, "逐案例证据不可用")
        try:
            items.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "逐案例证据损坏") from exc
    if (row.gate_results or {}).get("validity") == "incomplete":
        run = await session.get(Run, row.run_id)
        job = await session.get(Job, run.job_id) if run else None
        plan = (((job.spec or {}).get("params") or {}).get("eval_plan") or {}) if job else {}
        seen = {item.get("case_id") for item in items if isinstance(item, dict)}
        items.extend({**case, "status": "not_started", "success": None} for case in plan.get("cases", [])
                     if case.get("case_id") not in seen)
    for item in items:
        relative = item.get("video_rel") if isinstance(item, dict) else None
        if relative:
            if not isinstance(relative, str) or not re.fullmatch(r"seed_\d+/\d+\.mp4", relative):
                item.pop("video_rel", None)
                item["video_status"] = "unavailable"
                continue
            video = root / "video/eval" / relative
            if (any(part.is_symlink() for part in (root / "video", root / "video/eval", video.parent, video))
                    or not video.is_file()):
                item.pop("video_rel", None)
                item["video_status"] = "unavailable"
            else:
                item["video_status"] = "available"
    items = [item for item in items if (task_id is None or item.get("task_id") == task_id)
             and (not failures_only or item.get("status") in {"task_failure", "timeout", "invalid"})]
    return {"items": items[:1000], "available": bool(evidence_files), "total": len(items)}


@api_v1.get("/evals/reports/{eid}/media/{seed}/{index}")
async def get_eval_video(eid: str, seed: int, index: int, session: SessionDep,
                         principal: PrincipalDep):
    import re
    if seed < 0 or index < 0 or seed > 1_000_000 or index > 10_000:
        raise HTTPException(404)
    row = await session.get(EvalReport, eid)
    if not row or not row.run_id or not re.fullmatch(r"run_[A-Za-z0-9_-]+", row.run_id):
        raise HTTPException(404)
    from eb.settings import get_settings
    directory = get_settings().data_root / "runs" / row.run_id / "output/video/eval" / f"seed_{seed}"
    path = directory / f"{index}.mp4"
    output = get_settings().data_root / "runs" / row.run_id / "output"
    runs_root = get_settings().data_root / "runs"
    if (any(part.is_symlink() for part in (runs_root, runs_root / row.run_id, output,
                                           output / "video", output / "video/eval", directory, path))
            or not path.is_file() or not path.resolve().is_relative_to(output.resolve())):
        raise HTTPException(404)
    # FileResponse serves bounded chunks/Range without parsing video in the API.
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@api_v1.get("/evals/reports/{eid}/cross-domain-drift")
async def get_cross_domain_drift(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.eval.cross_domain import compute_observation_drift
    row = await session.get(EvalReport, eid)
    if not row:
        raise HTTPException(404)
    suite = await session.get(EvalSuite, row.suite_id) if row.suite_id else None
    ee_id = suite.embodiment_env_id if suite else ""
    env_row = await session.get(EmbodimentEnvRow, ee_id) if ee_id else None
    layout = ((env_row.spec or {}).get("observation") or {}).get("layout", []) if env_row else []
    stats = (row.metrics or {}).get("observation_stats") or {}
    return compute_observation_drift(ee_id, layout, stats.get("isaac"), stats.get("mujoco"), stats.get("real"))



@api_v1.get("/evals/leaderboard")
async def leaderboard(session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.eval.wilson import report_interval
    rows = (await session.scalars(select(EvalReport))).all()
    items = []
    for r in rows:
        if (r.gate_results or {}).get("validity") != "complete" or (r.metrics or {}).get("profile") != "standard":
            continue
        run = await session.get(Run, r.run_id) if r.run_id else None
        job = await session.get(Job, run.job_id) if run else None
        if not run or run.status != "finished" or not job or job.status != "succeeded":
            continue
        if ((job.spec or {}).get("params") or {}).get("rpo_eval_plan", {}).get("diagnostic_environment"):
            continue
        item = _row(r, ["id", "suite_id", "model_version_id", "run_id", "metrics", "gate_results"])
        metrics = item.get("metrics") or {}
        item["wilson_ci"] = report_interval(metrics)
        items.append(item)
    items = sorted(
        items,
        key=lambda it: float((it.get("metrics") or {}).get("success_rate") or (it.get("metrics") or {}).get("mean_reward") or 0),
        reverse=True,
    )
    return {"items": items}


@api_v1.get("/assets")
async def list_assets(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EmbodimentEnvRow))).all()
    items = []
    for r in rows:
        spec = r.spec or {}
        bind = spec.get("sim_bindings") or {}
        items.append(
            {
                "embodiment_env_id": r.id,
                "mjcf": bind.get("mjcf"),
                "usd": bind.get("usd"),
                "isaac_task": bind.get("isaac_task"),
            }
        )
    return {"items": items}


@api_v1.post("/bundles/build")
async def build_bundle(body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.bundle.pack import build_ebp
    from eb.domain.contract import EmbodimentEnv as Ctr

    onnx = Path(body.get("onnx") or "/data/kk/Berkeley-Humanoid-Lite/checkpoints/policy_biped_25hz_a.onnx")
    cid = body.get("embodiment_env_id") or "ee_bhl_biped_12dof_v1"
    if cid.startswith('ee_rpo_v2_ufo_native_'):
        raise HTTPException(403, '原生 UFO 模型必须使用绑定仿真证据的专用交付入口')
    row = await session.get(EmbodimentEnvRow, cid)
    if not row:
        raise HTTPException(404, "embodiment env")
    ctr = Ctr.model_validate(row.spec)
    deploy = body.get("deploy") or {
        "num_observations": ctr.observation.dim,
        "num_actions": ctr.action.dim,
        "action_scale": ctr.action.scale,
    }
    dest, manifest, can_sign = build_ebp(
        embodiment_env_id=cid,
        onnx_path=onnx,
        deploy_yaml=deploy,
        gates_passed=body.get("gates_passed") or [],
        provenance=body.get("provenance") or {},
        ctr=ctr,
    )
    b = Bundle(
        id=new_id("bnd"),
        embodiment_env_id=cid,
        stage="S0",
        signed=False,
        manifest={**manifest, "cas": str(dest), "can_sign": can_sign},
    )
    session.add(b)
    await session.commit()
    return _row(b, ["id", "embodiment_env_id", "stage", "signed", "manifest"])


@api_v1.get("/bundles")
async def list_bundles(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Bundle))).all()
    return {"items": [_row(r, ["id", "embodiment_env_id", "stage", "signed", "manifest"]) for r in rows]}


@api_v1.get("/bundles/next")
async def next_bundle(session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.scalar(select(Bundle).where(Bundle.signed.is_(True)))
    if not row:
        row = await session.scalar(select(Bundle))
    if not row:
        raise HTTPException(404, "no bundle")
    return _row(row, ["id", "embodiment_env_id", "stage", "signed", "manifest"])


@api_v1.post("/bundles/{bid}/sign")
async def sign_bundle(bid: str, session: SessionDep, principal: PrincipalDep, request: Request) -> dict:
    if request.headers.get("x-confirm") != bid:
        raise HTTPException(400, "X-Confirm header must match bundle id")
    b = await session.get(Bundle, bid)
    if not b:
        raise HTTPException(404)
    gates = [str(g) for g in (b.manifest or {}).get("gates_passed") or []]
    if b.embodiment_env_id.startswith('ee_rpo_v2_ufo_native_') or (b.manifest or {}).get('simulation_only'):
        raise HTTPException(403, '仿真交付包不支持此签名或真机资格提升')
    has_g4 = any(g.upper() == "G4" or g.lower() == "g4" for g in gates)
    has_s2s = any("sim2sim" in g.lower() for g in gates)
    if not (has_g4 and has_s2s):
        raise HTTPException(403, "G4+sim2sim required before sign")
    b.signed = True
    await session.commit()
    return _row(b, ["id", "signed", "stage"])


@api_v1.get("/robots")
async def list_robots(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Robot))).all()
    return {"items": [_row(r, ["id", "name", "embodiment_env_id", "calibration"]) for r in rows]}


@api_v1.post("/releases/{rid}/promote")
async def promote(rid: str, request: Request, principal: PrincipalDep) -> dict:
    if request.headers.get("x-confirm") != rid:
        raise HTTPException(400, "X-Confirm required")
    raise HTTPException(403, "S0/S1/S2 promotion is human-only and requires physical confirmation")


@api_v1.get("/resources/gpus")
async def gpus(principal: PrincipalDep) -> dict:
    return await EbdClient().doctor()


@api_v1.get("/resources/queue")
async def queue(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Job).where(Job.status.in_(["queued", "running", "preparing"])))).all()
    return {"items": [_job_out(j) for j in rows]}


@api_v1.post("/capacity/estimate")
async def cap_est(body: JobBody, principal: PrincipalDep) -> dict:
    spec = JobSpec.model_validate(body.spec)
    est = estimate_job(spec)
    return {
        "verdict": est.verdict,
        "per_gpu_peak_gb": est.per_gpu_peak_gb,
        "headroom_gb": est.headroom_gb,
        "confidence": est.confidence,
        "basis": est.basis,
        "verify": est.verify,
        "alternatives": est.alternatives,
    }


@api_v1.get("/agent/tools")
async def agent_tools(principal: PrincipalDep) -> dict:
    return {"items": bus.permission_matrix()}


@api_v1.get("/agent/playbooks")
async def agent_playbooks(principal: PrincipalDep) -> dict:
    return {"items": playbook_catalog()}


@api_v1.get("/agent/mcp")
async def agent_mcp(principal: PrincipalDep) -> dict:
    return {
        "name": "embodiedbein",
        "tools": [
            {"name": r["name"], "effect": r["effect"], "description": r["description"]}
            for r in bus.permission_matrix()
            if r["effect"] != "forbidden"
        ],
    }


@api_v1.post("/agent/sessions")
async def new_session(session: SessionDep, principal: PrincipalDep) -> dict:
    s = AgentSession(id=new_id("ags"))
    session.add(s)
    await session.commit()
    return {"id": s.id, "autonomy": s.autonomy}


@api_v1.post("/agent/sessions/{sid}/messages")
async def agent_msg(sid: str, body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    s = await session.get(AgentSession, sid)
    if not s:
        raise HTTPException(404)
    text = body.get("text", "")
    session.add(AgentStep(session_id=sid, seq=0, role="user", payload={"text": text}))
    result = await playbook_dispatch(session, text, bus)
    session.add(
        AgentStep(
            session_id=sid,
            seq=1,
            role="assistant",
            payload={"text": result.reply, "playbook": result.name, "llm_calls": result.llm_calls},
            guardrail_result=result.guardrail,
        )
    )
    await session.commit()
    return {"reply": result.reply, "guardrail": result.guardrail, "playbook": result.name, "llm_calls": result.llm_calls}


@api_v1.get("/agent/audit")
async def agent_audit(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(AgentStep).order_by(AgentStep.id.desc()).limit(200))).all()
    return {"items": [_row(r, ["session_id", "role", "tool", "guardrail_result", "payload"]) for r in rows]}


@api_v1.get("/shares/{token}")
async def get_share(token: str, session: SessionDep) -> dict:
    s = await session.get(Share, token)
    if not s or s.revoked:
        raise HTTPException(404)
    return {"title": s.title, "snapshot": s.snapshot, "expires_at": s.expires_at}


@api_v1.post("/shares")
async def create_share(body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    n_envs = await session.scalar(select(func.count()).select_from(EmbodimentEnvRow)) or 0
    n_models = await session.scalar(select(func.count()).select_from(ModelVersion)) or 0
    n_jobs = await session.scalar(select(func.count()).select_from(Job)) or 0
    snapshot = body.get("snapshot") or {
        "kpis": [
            {"label": "本体环境", "value": str(n_envs), "foot": "embodiment envs"},
            {"label": "模型", "value": str(n_models), "foot": "model versions"},
            {"label": "任务", "value": str(n_jobs), "foot": "jobs"},
        ]
    }
    s = Share(token=new_id("sh"), title=body.get("title", "share"), snapshot=snapshot)
    session.add(s)
    await session.commit()
    return {"token": s.token}


@api_v1.get("/system/version")
async def version(principal: PrincipalDep) -> dict:
    return {"version": "0.1.0", "api": "eb/v1"}


@api_v1.get("/system/images")
async def images(principal: PrincipalDep) -> dict:
    return {"items": ["eb/isaac-lab:2.3.2", "eb/rlinf:0.4.0-emb", "eb/rlinf:libero-pi05", "eb/mujoco:latest", "eb/tools:latest"]}


@api_v1.get("/system/quotas")
async def get_system_quotas(session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.harness.budget import weekly_limit, used_hours
    w_limit = weekly_limit()
    u_hours = used_hours()
    return {
        "disk": {
            "data_root": str(get_settings().data_root),
            "state_root": str(get_settings().data_root),
            "models_root": str(get_settings().models_root),
            "datasets_root": str(get_settings().datasets_root),
            "watermark_warn_pct": 80,
            "watermark_block_pct": 90,
            "cas_retention_days": 30,
            "auto_gc": True,
        },
        "agent_gpu_hours": {
            "weekly_limit": w_limit,
            "used": u_hours,
            "remaining": max(0.0, w_limit - u_hours),
        },
        "concurrency": {
            "max_concurrent_runs": 4,
            "priority_enabled": True,
        },
    }


@api_v1.get("/system/settings")
async def get_system_settings(session: SessionDep, principal: PrincipalDep) -> dict:
    return {
        "version": "0.1.0",
        "api": "eb/v1",
        "timezone": "Asia/Shanghai",
        "redis_port": 6380,
        "images": [
            {
                "name": "eb/isaac-lab:2.3.2",
                "digest": "sha256:eb1saaclab232fake",
                "supported_kinds": ["train.rl", "eval.sim", "export.policy"],
                "model_family": "lowlevel",
                "status": "ready",
            },
            {
                "name": "eb/rlinf:0.4.0-emb",
                "digest": "sha256:rlinf040embfake",
                "supported_kinds": ["train.sft", "train.rlft", "serve.policy"],
                "model_family": "vla",
                "status": "ready",
            },
            {
                "name": "eb/mujoco:latest",
                "digest": "sha256:mujocolatestfake",
                "supported_kinds": ["eval.sim2sim"],
                "model_family": "all",
                "status": "ready",
            },
            {
                "name": "eb/tools:latest",
                "digest": "sha256:toolslatestfake",
                "supported_kinds": ["data.collect", "data.convert", "data.slice", "bundle.build"],
                "model_family": "all",
                "status": "ready",
            },
        ],
        "scanner": {
            "roots": ["/data/kk/Berkeley-Humanoid-Lite", "/data/kk/RLinf", "/data/kk/berkeley_ego_data"],
            "timeout_seconds": 30,
            "exclude_patterns": [".git", "__pycache__", "*.tmp", "node_modules"],
        },
        "notifications": {
            "enabled": False,
            "webhook_url": "",
            "events": ["run.failed", "bundle.promoted", "gate.failed"],
        },
    }


@api_v1.post("/sources/scan")
async def scan_sources(body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    path = body.get("path") or "/data/kk/berkeley_ego_data"
    result = await EbdClient().scan(path)
    ds = await session.get(Dataset, "ds_berkeley_ego")
    if ds:
        ver = await session.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id))
        if ver:
            kinds = result.get("kinds") or {}
            files = int(result.get("files") or 0)
            rejected = ((result.get("funnel") or {}).get("raw") or {}).get("rejected") or []
            ver.content_hash = result.get("content_hash")
            ver.stats = {
                "files": files,
                "bytes": result.get("bytes"),
                "kinds": kinds,
                "samples": result.get("samples") or [],
            }
            ver.funnel = {
                "source": {"n": files, "rejected": []},
                "raw": {"n": files, "rejected": rejected},
                "valid_episode": {"n": int(kinds.get("ego_jsonl") or 0), "rejected": []},
                "retargeted": {"n": 0, "rejected": []},
                "canonical": {"n": 0, "rejected": []},
                "slice": {"n": 0, "rejected": []},
            }
            await session.commit()
    return result


@api_v1.get("/search")
async def search(q: str, session: SessionDep, principal: PrincipalDep) -> dict:
    like = f"%{q}%"
    jobs = (await session.scalars(select(Job).where(Job.id.ilike(like) | Job.kind.ilike(like)).limit(20))).all()
    models = (
        await session.scalars(select(ModelVersion).where(ModelVersion.id.ilike(like) | ModelVersion.name.ilike(like)).limit(20))
    ).all()
    envs = (
        await session.scalars(select(EmbodimentEnvRow).where(EmbodimentEnvRow.id.ilike(like)).limit(20))
    ).all()
    items = [{"type": "job", "id": j.id, "label": j.kind} for j in jobs]
    items += [{"type": "model", "id": m.id, "label": m.name} for m in models]
    items += [{"type": "embodiment_env", "id": c.id, "label": c.id} for c in envs]
    return {"items": items}


def _pipeline_run_out(run) -> dict:
    return {**_row(run, ["id", "pipeline_id", "status", "job_ids", "error", "created_by", "created_at"]),
            "resumed_from": (run.spec or {}).get("resumed_from")}


@api_v1.get("/pipelines")
async def pipelines(session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.store.models import PipelineRun
    rows = (await session.scalars(select(Pipeline))).all()
    runs = (await session.scalars(select(PipelineRun).order_by(PipelineRun.created_at.desc()))).all()
    items = []
    for row in rows:
        history = [_pipeline_run_out(run) for run in runs if run.pipeline_id == row.id]
        latest = history[0] if history else {}
        items.append({"id": row.id, "name": row.name, "spec": row.spec, "runs": history,
                      "status": latest.get("status", "not_started"), "job_ids": latest.get("job_ids", [])})
    return {"items": items}


@api_v1.post("/pipelines/{pid}/start")
async def pipeline_start(pid: str, request: Request, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(Pipeline, pid)
    if not row:
        raise HTTPException(404)
    try:
        run = await start_pipeline(session, row, principal.id, request.headers.get("idempotency-key"))
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _pipeline_run_out(run)


@api_v1.get("/pipelines/{pid}/runs")
async def pipeline_runs(pid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.store.models import PipelineRun
    runs = (await session.scalars(select(PipelineRun).where(PipelineRun.pipeline_id == pid)
                                  .order_by(PipelineRun.created_at.desc()).limit(100))).all()
    return {"items": [_pipeline_run_out(run) for run in runs]}


@api_v1.get("/pipeline-runs/{rid}")
async def pipeline_run_detail(rid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.store.models import PipelineRun
    run = await session.get(PipelineRun, rid)
    if not run:
        raise HTTPException(404)
    return {**_pipeline_run_out(run), "spec": run.spec}


class PipelineResumeIn(BaseModel):
    memory_gb: int | None = Field(default=None, ge=1, le=1024, strict=True)


@api_v1.post("/pipeline-runs/{rid}/resume")
async def pipeline_run_resume(rid: str, request: Request, session: SessionDep, principal: PrincipalDep,
                              body: PipelineResumeIn | None = None) -> dict:
    from eb.store.models import PipelineRun
    previous = await session.get(PipelineRun, rid)
    if not previous:
        raise HTTPException(404)
    pipeline = await session.get(Pipeline, previous.pipeline_id)
    if not pipeline:
        raise HTTPException(404, "pipeline template missing")
    try:
        run = await start_pipeline(session, pipeline, principal.id, request.headers.get("idempotency-key"),
                                   resume_from=previous, memory_gb=body.memory_gb if body else None)
    except (ValueError, TypeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _pipeline_run_out(run)


@api_v1.post("/gates/{gid}/evaluations")
async def eval_gate(gid: str, body: dict, principal: PrincipalDep) -> dict:
    return evaluate_gate_dsl(body.get("assertions") or [], body.get("actuals") or {}).model_dump()


@api_v1.post("/deployments/{did}/telemetry")
async def robot_telemetry(did: str, body: dict) -> dict:
    root = get_settings().data_root
    points = body.get("points") or []
    n = 0
    for i, p in enumerate(points):
        tag = p.get("tag") or "robot.event"
        step = int(p.get("step") or i)
        value = float(p.get("value") or 0)
        # write JSONL beside cas so API can serve even if ebd helper isn't importable
        dest = root / "tsdb" / did
        dest.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in tag)
        with (dest / f"{safe}.jsonl").open("a") as f:
            f.write(f'{{"step":{step},"value":{value}}}\n')
        n += 1
    return {"ok": True, "robot": did, "n": n}


@ws_router.websocket("/ws/queue")
async def ws_queue(ws: WebSocket) -> None:
    await ws.accept()
    seq = 0
    try:
        while True:
            payload: dict[str, Any] = {"items": []}
            try:
                from eb.store.db import SessionLocal

                if SessionLocal:
                    async with SessionLocal() as session:
                        rows = (
                            await session.scalars(select(Job).where(Job.status.in_(["queued", "running", "preparing"])))
                        ).all()
                        payload = {"items": [_job_out(j) for j in rows]}
            except Exception:
                pass
            await ws.send_json({"v": 1, "type": "queue", "ts": seq, "seq": seq, "payload": payload})
            seq += 1
            await asyncio.sleep(2)
    except Exception:
        pass


@ws_router.websocket("/ws/runs/{rid}")
async def ws_run(ws: WebSocket, rid: str) -> None:
    await ws.accept()
    seq = 0
    try:
        while True:
            metrics: dict[str, Any] = {}
            lines: list[str] = []
            try:
                metrics = await EbdClient().tsdb_query(rid)
            except Exception:
                pass
            try:
                logs = await EbdClient().logs(rid, tail=30)
                lines = logs.get("lines") or []
            except Exception:
                pass
            await ws.send_json(
                {"v": 1, "type": "metrics", "ts": seq, "seq": seq, "payload": {"run_id": rid, **metrics}}
            )
            seq += 1
            await ws.send_json(
                {"v": 1, "type": "log", "ts": seq, "seq": seq, "payload": {"run_id": rid, "lines": lines}}
            )
            seq += 1
            await asyncio.sleep(2)
    except Exception:
        pass


@media_router.api_route("/media/{path:path}", methods=["GET", "HEAD"])
async def media_proxy(path: str, request: Request):
    settings = get_settings()
    url = f"http://ebd/media/{path}"
    headers = {}
    if rng := request.headers.get("range"):
        headers["range"] = rng
    try:
        client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=settings.ebd_media_uds), timeout=60.0)
        r = await client.send(client.build_request("GET", url, headers=headers), stream=True)
        return StreamingResponse(r.aiter_bytes(), status_code=r.status_code, headers=dict(r.headers))
    except Exception as e:
        raise HTTPException(502, f"ebd media: {e}") from e


@spa_router.get("/showcase/{token}")
async def showcase_page(token: str, request: Request):
    return await spa_fallback(token, request)


@spa_router.get("/{full_path:path}")
async def spa_fallback(full_path: str, request: Request):
    if full_path.startswith(("api/", "ws/", "media/", "artifacts/")):
        raise HTTPException(404)
    dist: Path = request.app.state.web_dist
    candidate = dist / full_path
    if full_path and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    index = dist / "index.html"
    if index.exists():
        return FileResponse(index, headers={"Cache-Control": "no-store"})
    return {"hint": "frontend not built", "path": full_path}


def _row(obj: Any, fields: list[str]) -> dict:
    out = {}
    for f in fields:
        val = getattr(obj, f)
        out[f] = val
    return out


def _job_out(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "embodiment_env_id": job.embodiment_env_id,
        "priority": job.priority,
        "created_by": job.created_by,
        "spec": job.spec,
        "error": job.error,
    }
