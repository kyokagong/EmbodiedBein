from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from passlib.hash import pbkdf2_sha256
from pydantic import BaseModel
from sqlalchemy import func, select

from eb.agent.playbooks import catalog as playbook_catalog
from eb.agent.playbooks import dispatch as playbook_dispatch
from eb.agent.tools import AgentCtx  # noqa: F401  — register tools
from eb.api.deps import PrincipalDep, SessionDep
from eb.capacity import estimate_job
from eb.domain.contract import Contract
from eb.domain.fsm import JobStatus
from eb.domain.job import JobSpec
from eb.gates.engine import GateContext, evaluate_gate_dsl, g1_submit, g3_model
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
    EmbodimentContractRow,
    EvalReport,
    EvalSuite,
    Experiment,
    Job,
    LineageEdge,
    ModelFamily,
    ModelVersion,
    Pipeline,
    Robot,
    Run,
    Share,
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


@api_v1.get("/contracts")
async def list_contracts(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EmbodimentContractRow))).all()
    return {"items": [_contract_out(r) for r in rows]}


@api_v1.get("/contracts/{cid}")
async def get_contract(cid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentContractRow, cid)
    if not row:
        raise HTTPException(404)
    ctr = Contract.model_validate(row.spec)
    return {**_contract_out(row), "self_check": ctr.validate_self().model_dump()}


@api_v1.post("/contracts/{cid}/validate")
async def validate_contract(cid: str, body: JobBody, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentContractRow, cid)
    if not row:
        raise HTTPException(404)
    spec = JobSpec.model_validate({**body.spec, "contract_id": cid})
    result = g1_submit(spec, Contract.model_validate(row.spec), GateContext(dataset_versions=[]))
    return result.model_dump()


@api_v1.get("/datasets")
async def list_datasets(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Dataset))).all()
    return {"items": [_row(r, ["id", "name", "format", "license", "contract_id"]) for r in rows]}


@api_v1.get("/datasets/{did}")
async def get_dataset(did: str, session: SessionDep, principal: PrincipalDep) -> dict:
    ds = await session.get(Dataset, did)
    if not ds:
        raise HTTPException(404)
    vers = (await session.scalars(select(DatasetVersion).where(DatasetVersion.dataset_id == did))).all()
    return {
        **_row(ds, ["id", "name", "format", "license", "contract_id", "meta"]),
        "versions": [_row(v, ["id", "version", "stats", "funnel", "compatible_contracts"]) for v in vers],
    }


@api_v1.get("/datasets/{did}/versions/{vid}/funnel")
async def dataset_funnel(did: str, vid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    v = await session.get(DatasetVersion, vid)
    if not v or v.dataset_id != did:
        raise HTTPException(404)
    return v.funnel or {}


@api_v1.get("/datasets/{did}/versions/{vid}/episodes/{ep}")
async def episode_preview(did: str, vid: str, ep: str, session: SessionDep, principal: PrincipalDep) -> dict:
    v = await session.get(DatasetVersion, vid)
    if not v or v.dataset_id != did:
        raise HTTPException(404)
    root = Path("/data/kk/berkeley_ego_data")
    videos = list(root.rglob("*.mp4")) if root.exists() else []
    video = None
    for p in videos:
        if ep in p.as_posix() or ep in p.stem:
            video = p
            break
    if video is None and videos:
        try:
            idx = int(ep)
            video = videos[idx % len(videos)]
        except ValueError:
            video = videos[0]
    if video is None:
        return {"id": ep, "video_cas": None, "action_series": [], "missing": "缺媒体"}
    from eb.onnx_io import sha256_file

    digest = sha256_file(video)
    cas = get_settings().data_root / "cas" / digest[:2] / digest
    cas.parent.mkdir(parents=True, exist_ok=True)
    if not cas.exists():
        cas.write_bytes(video.read_bytes()[: 8 * 1024 * 1024])  # first 8MB for preview
    return {
        "id": ep,
        "path": str(video),
        "video_cas": digest,
        "media_url": f"/media/{digest[:2]}/{digest}",
        "action_series": [{"step": i, "value": 0.0} for i in range(20)],
        "note": "动作曲线需对应 jsonl；无并列轨迹时显示占位零线并标明缺动作",
    }


@api_v1.get("/models")
async def list_models(session: SessionDep, principal: PrincipalDep) -> dict:
    fams = (await session.scalars(select(ModelFamily))).all()
    vers = (await session.scalars(select(ModelVersion))).all()
    return {
        "families": [_row(f, ["id", "name", "family"]) for f in fams],
        "versions": [_row(v, ["id", "family_id", "name", "contract_id", "provenance", "tags", "io_signature"]) for v in vers],
    }


@api_v1.get("/models/{mid}")
async def get_model(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    v = await session.get(ModelVersion, mid)
    if not v:
        raise HTTPException(404)
    out = _row(v, ["id", "family_id", "name", "contract_id", "io_signature", "mem_profile", "card", "tags", "provenance"])
    if v.contract_id:
        crow = await session.get(EmbodimentContractRow, v.contract_id)
        if crow:
            ctr = Contract.model_validate(crow.spec)
            io = v.io_signature or {}
            out["contract_check"] = g3_model(io.get("obs"), io.get("act"), ctr).model_dump()
            out["contract"] = {"obs_dim": ctr.observation.dim, "action_dim": ctr.action.dim, "policy_hz": ctr.control.policy_hz}
    return out


@api_v1.get("/models/{mid}/lineage")
async def model_lineage(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    edges = (await session.scalars(select(LineageEdge).where((LineageEdge.src_id == mid) | (LineageEdge.dst_id == mid)))).all()
    return {"edges": [_row(e, ["src_type", "src_id", "dst_type", "dst_id", "relation"]) for e in edges]}


@api_v1.get("/experiments")
async def list_experiments(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Experiment))).all()
    return {"items": [_row(r, ["id", "name", "contract_id"]) for r in rows]}


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
    spec = JobSpec.model_validate({**job.spec, "kind": job.kind, "contract_id": job.contract_id})
    return {"command": orch.preview_command(spec, "run_preview")}


@api_v1.post("/jobs/{jid}/cancel")
async def cancel_job(jid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    job = await session.get(Job, jid)
    if not job:
        raise HTTPException(404)
    job.status = JobStatus.CANCELLED.value
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
async def list_suites(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EvalSuite))).all()
    return {"items": [_row(r, ["id", "name", "contract_id", "seeds", "frozen"]) for r in rows]}


@api_v1.get("/evals/reports")
async def list_reports(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EvalReport))).all()
    return {"items": [_row(r, ["id", "suite_id", "model_version_id", "metrics", "gate_results"]) for r in rows]}


@api_v1.get("/evals/reports/{eid}")
async def get_report(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EvalReport, eid)
    if not row:
        raise HTTPException(404)
    return _row(row, ["id", "suite_id", "model_version_id", "metrics", "gate_results"])


@api_v1.get("/evals/leaderboard")
async def leaderboard(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EvalReport))).all()
    items = sorted(
        rows,
        key=lambda r: float((r.metrics or {}).get("success_rate") or (r.metrics or {}).get("mean_reward") or 0),
        reverse=True,
    )
    return {"items": [_row(r, ["id", "suite_id", "model_version_id", "metrics", "gate_results"]) for r in items]}


@api_v1.get("/assets")
async def list_assets(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EmbodimentContractRow))).all()
    items = []
    for r in rows:
        spec = r.spec or {}
        bind = spec.get("sim_bindings") or {}
        items.append(
            {
                "contract_id": r.id,
                "mjcf": bind.get("mjcf"),
                "usd": bind.get("usd"),
                "isaac_task": bind.get("isaac_task"),
            }
        )
    return {"items": items}


@api_v1.post("/bundles/build")
async def build_bundle(body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.bundle.pack import build_ebp
    from eb.domain.contract import Contract as Ctr

    onnx = Path(body.get("onnx") or "/data/kk/Berkeley-Humanoid-Lite/checkpoints/policy_biped_25hz_a.onnx")
    cid = body.get("contract_id") or "ec_bhl_biped_12dof_v1"
    row = await session.get(EmbodimentContractRow, cid)
    if not row:
        raise HTTPException(404, "contract")
    ctr = Ctr.model_validate(row.spec)
    deploy = body.get("deploy") or {
        "num_observations": ctr.observation.dim,
        "num_actions": ctr.action.dim,
        "action_scale": ctr.action.scale,
    }
    dest, manifest, can_sign = build_ebp(
        contract_id=cid,
        onnx_path=onnx,
        deploy_yaml=deploy,
        gates_passed=body.get("gates_passed") or [],
        provenance=body.get("provenance") or {},
        ctr=ctr,
    )
    b = Bundle(
        id=new_id("bnd"),
        contract_id=cid,
        stage="S0",
        signed=False,
        manifest={**manifest, "cas": str(dest), "can_sign": can_sign},
    )
    session.add(b)
    await session.commit()
    return _row(b, ["id", "contract_id", "stage", "signed", "manifest"])


@api_v1.get("/bundles")
async def list_bundles(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Bundle))).all()
    return {"items": [_row(r, ["id", "contract_id", "stage", "signed", "manifest"]) for r in rows]}


@api_v1.get("/bundles/next")
async def next_bundle(session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.scalar(select(Bundle).where(Bundle.signed.is_(True)))
    if not row:
        row = await session.scalar(select(Bundle))
    if not row:
        raise HTTPException(404, "no bundle")
    return _row(row, ["id", "contract_id", "stage", "signed", "manifest"])


@api_v1.post("/bundles/{bid}/sign")
async def sign_bundle(bid: str, session: SessionDep, principal: PrincipalDep, request: Request) -> dict:
    if request.headers.get("x-confirm") != bid:
        raise HTTPException(400, "X-Confirm header must match bundle id")
    b = await session.get(Bundle, bid)
    if not b:
        raise HTTPException(404)
    gates = [str(g) for g in (b.manifest or {}).get("gates_passed") or []]
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
    return {"items": [_row(r, ["id", "name", "contract_id", "calibration"]) for r in rows]}


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
    n_contracts = await session.scalar(select(func.count()).select_from(EmbodimentContractRow)) or 0
    n_models = await session.scalar(select(func.count()).select_from(ModelVersion)) or 0
    n_jobs = await session.scalar(select(func.count()).select_from(Job)) or 0
    snapshot = body.get("snapshot") or {
        "kpis": [
            {"label": "契约", "value": str(n_contracts), "foot": "embodiment contracts"},
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
    return {"items": ["eb/isaac-lab:2.3.2", "eb/rlinf:0.4.0-emb", "eb/mujoco:latest", "eb/tools:latest"]}


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
            ver.stats = {"files": files, "bytes": result.get("bytes"), "kinds": kinds}
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
    contracts = (
        await session.scalars(select(EmbodimentContractRow).where(EmbodimentContractRow.id.ilike(like)).limit(20))
    ).all()
    items = [{"type": "job", "id": j.id, "label": j.kind} for j in jobs]
    items += [{"type": "model", "id": m.id, "label": m.name} for m in models]
    items += [{"type": "contract", "id": c.id, "label": c.id} for c in contracts]
    return {"items": items}


@api_v1.get("/pipelines")
async def pipelines(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Pipeline))).all()
    return {
        "items": [
            {
                "id": r.id,
                "name": r.name,
                "status": (r.spec or {}).get("status"),
                "job_ids": (r.spec or {}).get("job_ids") or [],
                "spec": r.spec,
            }
            for r in rows
        ]
    }


@api_v1.post("/pipelines/{pid}/start")
async def pipeline_start(pid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(Pipeline, pid)
    if not row:
        raise HTTPException(404)
    await start_pipeline(session, row)
    return {"id": row.id, "status": (row.spec or {}).get("status")}


@api_v1.post("/gates/{gid}/evaluations")
async def eval_gate(gid: str, body: dict, principal: PrincipalDep) -> dict:
    return evaluate_gate_dsl(body.get("assertions") or [], body.get("actuals") or {}).model_dump()


@api_v1.post("/deployments/{did}/telemetry")
async def robot_telemetry(did: str, body: dict) -> dict:
    root = get_settings().data_root
    points = body.get("points") or []
    try:
        import eb_tsdb  # type: ignore
    except Exception:
        eb_tsdb = None
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


def _contract_out(row: EmbodimentContractRow) -> dict:
    return {
        "id": row.id,
        "status": row.status,
        "obs_dim": row.obs_dim,
        "action_dim": row.action_dim,
        "policy_hz": row.policy_hz,
        "spec": row.spec,
    }


def _job_out(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "contract_id": job.contract_id,
        "priority": job.priority,
        "created_by": job.created_by,
        "spec": job.spec,
        "error": job.error,
    }
