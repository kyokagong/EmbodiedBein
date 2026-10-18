"""Operator-owned live native simulation; ebd owns control and media transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from websockets.asyncio.client import unix_connect
from websockets.exceptions import WebSocketException

from eb.api.deps import PrincipalDep, SessionDep, current_principal
from eb.api.routes_ufo_native import approved_image, native_model, stream_output, submit
from eb.domain.environment_profile import compatible_profile
from eb.domain.fsm import JOB_TERMINAL
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import AuditLog, EnvironmentProfile, EvalSuite, Job, ModelVersion, Run
from eb.ufo_live import OPERATION, TASK_IDS, environment_spec, live_suite_spec, validate_live_plan
from eb.ufo_native import ENV_ID, EXPORT_SCHEMA, IMAGE, TASK_ID, TASKS, plan_hash, safe_file

router = APIRouter(prefix="/ufo-native/live")


class LiveIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    environment_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    initial_task: Literal["move-ego-0-0", "move-ego-0-0.3", "move-ego-0-0.7"] = "move-ego-0-0"
    duration_seconds: int = Field(default=300, ge=30, le=600)
    seed: int = Field(default=71000, ge=0, le=2**31 - 1)
    preview_hash: str | None = None


class ControlIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    action: Literal["task", "pause", "resume", "reset", "stop"]
    task: Literal["move-ego-0-0", "move-ego-0-0.3", "move-ego-0-0.7"] | None = None

    @model_validator(mode="after")
    def task_matches_action(self):
        if (
            self.action == "task"
            and self.task is None
            or self.action != "task"
            and "task" in self.model_fields_set
        ):
            raise ValueError("只有 task 命令必须填写冻结任务")
        return self


class PendingCloseIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


def require_operator(principal):
    if principal.actor_type != "user":
        raise HTTPException(403, "实时仿真仅供操作者使用，平台 Agent 不可启动或控制")


async def lock_submission(session, principal, key):
    name = f"eb-ufo-live-session\0{principal.id}\0{key}".encode()
    lock_id = int.from_bytes(hashlib.sha256(name).digest()[:8], "big", signed=True)
    await session.execute(select(func.pg_advisory_xact_lock(lock_id)))


async def pending_close_marker(session, principal, key):
    return await session.scalar(select(AuditLog.id).where(
        AuditLog.principal == principal.id, AuditLog.action == "ufo_live.close_pending",
        AuditLog.subject == key,
    ).limit(1))


async def build_plan(session, body: LiveIn):
    model, artifact, payload = await native_model(session, body.model_id, exported=True)
    source, native_artifact, _ = await native_model(session, model.card["source_model_id"], exported=False)
    if (
        payload.get("source_model_id") != source.id
        or payload.get("source_artifact_sha256") != native_artifact.sha256
    ):
        raise ValueError("导出模型与冻结原生父模型身份不一致")
    environment = await session.get(EnvironmentProfile, body.environment_id)
    backend = next(
        (
            backend
            for backend in ("mujoco", "mjlab")
            if body.environment_id == environment_spec(backend)["id"]
        ),
        None,
    )
    if not environment or not backend:
        raise ValueError("环境不支持此原生导出模型")
    expected = environment_spec(backend)
    compatible_profile(environment, "eval", expected["suite_id"], ENV_ID, TASK_ID)
    if environment.content_hash != expected["content_hash"]:
        raise ValueError("实时环境定义已变化，必须使用发布版本")
    suite = await session.get(EvalSuite, environment.suite_id)
    if (
        not suite
        or not suite.frozen
        or suite.embodiment_env_id != ENV_ID
        or suite.spec != live_suite_spec(backend)
    ):
        raise ValueError("实时诊断套件不存在或已变化")
    plan = {
        "operation": OPERATION,
        "name": model.name,
        "model_id": model.id,
        "source_model_id": source.id,
        "input_path": artifact.storage,
        "source_artifact_sha256": artifact.sha256,
        "native_source_path": native_artifact.storage,
        "native_source_sha256": native_artifact.sha256,
        "image_digest": await approved_image(),
        "backend": backend,
        "environment_id": environment.id,
        "environment_hash": environment.content_hash,
        "suite_id": suite.id,
        "suite_spec": suite.spec,
        "tasks": list(TASK_IDS),
        "initial_task": body.initial_task,
        "duration_seconds": body.duration_seconds,
        "seed": body.seed,
    }
    plan["hash"] = plan_hash(plan)
    return validate_live_plan(plan)


async def session_job(session, principal, job_id, *, lock=False):
    require_operator(principal)
    job = (
        await session.get(Job, job_id, with_for_update=True, populate_existing=True)
        if lock else await session.get(Job, job_id)
    )
    if (
        not job
        or job.created_by != principal.id
        or (job.spec.get("params") or {}).get("operation") != OPERATION
    ):
        raise HTTPException(404, "实时仿真会话不存在")
    return job


async def owned_job(session, principal, job_id):
    job = await session_job(session, principal, job_id)
    try:
        plan = validate_live_plan(job.spec["params"]["native_plan"])
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, "实时仿真计划无效") from exc
    return job, plan


async def owned_run(session, principal, rid, *, mutation=False):
    require_operator(principal)
    if not re.fullmatch(r"run_[a-f0-9]{20}", rid):
        raise HTTPException(400, "Run 标识无效")
    run = await session.get(Run, rid)
    if not run:
        raise HTTPException(404, "实时仿真运行不存在")
    job, plan = await owned_job(session, principal, run.job_id)
    if mutation and (run.status != "running" or job.status != "running"):
        raise HTTPException(409, "实时仿真运行已停止")
    return run, job, plan


@router.get("/capabilities")
async def capabilities(session: SessionDep, principal: PrincipalDep):
    require_operator(principal)
    rows = (
        await session.scalars(
            select(ModelVersion)
            .where(
                ModelVersion.embodiment_env_id == ENV_ID,
                ModelVersion.card["schema_version"].astext == EXPORT_SCHEMA,
            )
            .order_by(ModelVersion.created_at.desc())
        )
    ).all()
    models = []
    for model in rows:
        try:
            _, artifact, _ = await native_model(session, model.id, exported=True)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        models.append(
            {
                "id": model.id,
                "name": model.name,
                "embodiment_env_id": ENV_ID,
                "payload_sha256": artifact.sha256,
            }
        )
    environments = []
    for backend in ("mujoco", "mjlab"):
        expected = environment_spec(backend)
        row = await session.get(EnvironmentProfile, expected["id"])
        if row and row.status == "active" and row.content_hash == expected["content_hash"]:
            environments.append(
                {
                    key: expected[key]
                    for key in (
                        "id",
                        "name",
                        "backend",
                        "content_hash",
                        "embodiment_env_id",
                    )
                }
            )
    image = await EbdClient().image_info(IMAGE)
    return {
        "models": models,
        "environments": environments,
        "tasks": TASKS,
        "image_ready": image.get("ok", False),
    }


@router.post("/sessions/preview")
async def preview(body: LiveIn, session: SessionDep, principal: PrincipalDep):
    from eb.api.routes_ufo_native import spec_for

    require_operator(principal)
    try:
        plan = await build_plan(session, body)
        return {"plan": plan, "resources": spec_for(plan).resources.model_dump()}
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/sessions")
async def start(
    body: LiveIn,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    require_operator(principal)
    try:
        PendingCloseIn(idempotency_key=idem)
        await lock_submission(session, principal, idem)
        if await pending_close_marker(session, principal, idem):
            raise HTTPException(409, "此次启动已随页面关闭而取消，请重新检查配置后启动。")
        plan = await build_plan(session, body)
        previous = await session.scalar(select(Job).where(Job.idem_key == idem))
        if previous and previous.created_by != principal.id:
            raise HTTPException(409, "此幂等键已被使用")
        return await submit(plan, body.preview_hash, session, principal, idem)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


def session_summary(job, run=None):
    plan = (job.spec.get("params") or {}).get("native_plan") or {}
    if not isinstance(plan, dict):
        plan = {}
    return {
        "id": job.id,
        "status": job.status,
        "error": job.error,
        "run_id": run.id if run else None,
        "run_status": run.status if run else None,
        "plan_hash": plan.get("hash"),
        "model_id": plan.get("model_id"),
        "payload_sha256": plan.get("source_artifact_sha256"),
        "environment_id": plan.get("environment_id"),
        "backend": plan.get("backend"),
        "files": (run.metrics_summary or {}).get("files", []) if run else [],
        "recording": (run.metrics_summary or {}).get("recording") if run else None,
        "metrics_summary": run.metrics_summary if run else {},
    }


@router.get("/sessions")
async def active_sessions(session: SessionDep, principal: PrincipalDep):
    require_operator(principal)
    jobs = (await session.scalars(select(Job).where(
        Job.created_by == principal.id,
        Job.spec["params"]["operation"].astext == OPERATION,
        Job.status.not_in([str(status) for status in JOB_TERMINAL]),
    ).order_by(Job.created_at.desc()).limit(50))).all()
    return {"items": [session_summary(job) for job in jobs]}


@router.post("/sessions/{jid}/close")
async def close_session(jid: str, request: Request, session: SessionDep, principal: PrincipalDep):
    from eb.api.routes import cancel_job

    if not same_origin(request):
        raise HTTPException(403, "关闭交互环境必须同源")
    # Shutdown is valid even before a Run/state/socket exists or the plan becomes unreadable.
    job = await session_job(session, principal, jid, lock=True)
    if job.status not in JOB_TERMINAL:
        session.add(AuditLog(principal=principal.id, action="ufo_live.close", subject=jid,
                             after={"job_id": jid, "action": "cancel_and_release"}))
        await cancel_job(jid, session, principal)
    run = await session.scalar(select(Run).where(Run.job_id == jid).order_by(Run.attempt.desc()).limit(1))
    return session_summary(job, run)


@router.post("/sessions/close-pending")
async def close_pending(body: PendingCloseIn, request: Request, session: SessionDep, principal: PrincipalDep):
    require_operator(principal)
    if not same_origin(request):
        raise HTTPException(403, "关闭交互环境必须同源")
    key = body.idempotency_key
    await lock_submission(session, principal, key)
    previous = await session.scalar(select(Job).where(Job.idem_key == key))
    if previous and (
        previous.created_by != principal.id
        or (previous.spec.get("params") or {}).get("operation") != OPERATION
    ):
        raise HTTPException(404, "实时仿真会话不存在")
    if not await pending_close_marker(session, principal, key):
        # A late create request must observe this marker, even if no Job exists yet.
        session.add(AuditLog(principal=principal.id, action="ufo_live.close_pending", subject=key,
                             after={"action": "cancel_pending_submission"}))
    result = await close_session(previous.id, request, session, principal) if previous else None
    await session.commit()
    return {"closed": True, "job": result}


@router.get("/sessions/{jid}")
async def status(jid: str, session: SessionDep, principal: PrincipalDep):
    job, _ = await owned_job(session, principal, jid)
    run = await session.scalar(select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc()).limit(1))
    return session_summary(job, run)


async def transport(call):
    try:
        return await call
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, "实时仿真通道尚未就绪或命令冲突") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "实时仿真数据通道不可用") from exc


@router.post("/runs/{rid}/control")
async def control(rid: str, body: ControlIn, session: SessionDep, principal: PrincipalDep):
    _, job, _ = await owned_run(session, principal, rid, mutation=True)
    result = await transport(EbdClient().live_control(rid, body.model_dump(exclude_none=True)))
    session.add(
        AuditLog(
            principal=principal.id,
            action="ufo_live.control",
            subject=rid,
            after={"job_id": job.id, "run_id": rid, **body.model_dump(exclude_none=True)},
        )
    )
    await session.commit()
    return result


@router.post("/runs/{rid}/heartbeat")
async def heartbeat(rid: str, session: SessionDep, principal: PrincipalDep):
    await owned_run(session, principal, rid, mutation=True)
    return await transport(EbdClient().live_heartbeat(rid))


def validate_socket_message(value):
    """Validate only control metadata; image bytes never enter a Python decoder."""
    if not isinstance(value, dict):
        raise ValueError("消息必须为对象")
    kind = value.get("type")
    fields = {
        "heartbeat": {"type"}, "frame_ack": {"type", "seq"},
        "control": {"type", "command"},
        "camera": {"type", "seq", "azimuth", "elevation", "distance"},
    }
    if not isinstance(kind, str) or kind not in fields or set(value) != fields[kind]:
        raise ValueError("不支持的实时消息")
    if kind == "control":
        ControlIn.model_validate(value["command"])
    if kind in ("camera", "frame_ack"):
        if type(value["seq"]) is not int or not 0 <= value["seq"] <= (
            2**31 - 1 if kind == "camera" else 2**53 - 1
        ):
            raise ValueError("序号无效")
    if kind == "camera":
        for key, low, high in (("azimuth", -360, 360), ("elevation", -85, -5), ("distance", 1, 8)):
            number = value[key]
            if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
                raise ValueError("相机参数无效")
    return value


def same_origin(websocket):
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    try:
        source = urlsplit(origin)
        target = urlsplit(str(websocket.url))
        expected_scheme = "https" if target.scheme in ("https", "wss") else "http"
        port = 443 if expected_scheme == "https" else 80
        return (
            source.scheme == expected_scheme and source.hostname == target.hostname
            and (source.port or port) == (target.port or port)
            and not source.username and not source.password and source.path in ("", "/")
            and not source.query and not source.fragment
        )
    except ValueError:
        return False


@router.websocket("/runs/{rid}/stream")
async def stream(rid: str, websocket: WebSocket, session: SessionDep):
    try:
        if not same_origin(websocket):
            raise HTTPException(403, "实时通道必须同源")
        principal = await current_principal(websocket, session)
        _, job, plan = await owned_run(session, principal, rid, mutation=True)
        if (plan.get("suite_spec", {}).get("stream") or {}).get("transport") != "websocket-jpeg":
            raise HTTPException(409, "此历史会话未启用原生画面推流，请创建新会话")
        job_id = job.id
    except HTTPException as exc:
        await websocket.close(code=4400 + exc.status_code - 400, reason=str(exc.detail))
        return
    # Do not hold a database transaction while a browser watches the stream.
    await session.rollback()
    accepted = False
    try:
        async with unix_connect(
            get_settings().ebd_media_uds, uri=f"ws://ebd/v1/runs/{rid}/live/stream",
            max_size=2 * 1024**2, max_queue=1, write_limit=4096,
            open_timeout=3, close_timeout=1, ping_interval=None, compression=None,
        ) as upstream:
            await websocket.accept(headers=[(b"cache-control", b"private, no-store")])
            accepted = True
            pending = {}

            async def to_ebd():
                window, count = time.monotonic(), 0
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    raw = message.get("text")
                    if raw is None or len(raw.encode()) > 4096:
                        raise ValueError("实时输入过大或类型无效")
                    now = time.monotonic()
                    if now - window >= 1:
                        window, count = now, 0
                    count += 1
                    if count > 120:
                        raise ValueError("实时输入过快")
                    value = validate_socket_message(json.loads(raw))
                    if value["type"] == "control":
                        if pending:
                            raise ValueError("请等待上一命令接受结果")
                        pending[value["command"]["id"]] = value["command"]
                    await asyncio.wait_for(upstream.send(raw), timeout=1)

            async def to_browser():
                while True:
                    value = await upstream.recv()
                    if isinstance(value, bytes):
                        if len(value) > 2 * 1024**2:
                            raise ValueError("实时帧过大")
                        await asyncio.wait_for(websocket.send_bytes(value), timeout=1)
                        continue
                    if len(value.encode()) > 256 * 1024:
                        raise ValueError("实时状态过大")
                    metadata = json.loads(value)
                    if metadata.get("type") == "control_result":
                        command = pending.pop(metadata.get("id"), None)
                        if command is None:
                            raise ValueError("命令回执不匹配")
                        if metadata.get("ok") is True:
                            session.add(AuditLog(
                                principal=principal.id, action="ufo_live.control", subject=rid,
                                after={"job_id": job_id, "run_id": rid, **command},
                            ))
                            await session.commit()
                    await asyncio.wait_for(websocket.send_text(value), timeout=1)

            tasks = [asyncio.create_task(to_ebd()), asyncio.create_task(to_browser())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except (WebSocketDisconnect, WebSocketException, OSError, TimeoutError):
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1011 if accepted else 4502, reason="实时仿真数据通道已断开")
    except ValueError as exc:
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1008, reason=str(exc)[:100])
    else:
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()


@router.get("/runs/{rid}/{kind}")
async def snapshot(
    rid: str, kind: Literal["state", "scene"], request: Request, session: SessionDep, principal: PrincipalDep
):
    await owned_run(session, principal, rid)
    return await stream_output(request, rid, "", live_kind=kind)


@router.api_route("/models/{mid}/assets/{path:path}", methods=["GET", "HEAD"])
async def asset(mid: str, path: str, request: Request, session: SessionDep, principal: PrincipalDep):
    require_operator(principal)
    try:
        _, artifact, manifest = await native_model(session, mid, exported=True)
        item = next((item for item in manifest["files"] if item["path"] == path), None)
        if (
            not item
            or not path.startswith("assets/meshes/")
            or not path.lower().endswith(".stl")
            or not 0 < item["bytes"] <= 64 * 1024**2
        ):
            raise ValueError("不是冻结模型网格")
        safe_file(Path(artifact.storage), path)
        relative = Path(artifact.storage).relative_to(get_settings().data_root / "runs")
        run_id, outputs, *package = relative.parts
        if not re.fullmatch(r"run_[a-f0-9]{20}", run_id) or outputs != "outputs" or package != ["payload"]:
            raise ValueError("模型产物路径不属于受管 Run")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(404, "冻结模型网格不可用") from exc
    return await stream_output(request, run_id, "payload/" + path)
