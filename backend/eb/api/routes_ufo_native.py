"""Human workbench for frozen native UFO inference jobs."""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.job import JobSpec
from eb.orchestrator.service import Orchestrator
from eb.paths import validate_allowed_path
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Artifact, Bundle, EvalReport, EvalSuite, Job, ModelVersion, Run
from eb.ufo_native import (
    ENV_ID,
    EXPORT_SCHEMA,
    IMAGE,
    MODEL_SCHEMA,
    OPERATIONS,
    SOURCE_FILES,
    TASK_ID,
    TASKS,
    eval_cases,
    plan_hash,
    read_json,
    safe_file,
    sha256_file,
    suite_spec,
    validate_bundle_source,
    validate_config,
)

router = APIRouter(prefix="/ufo-native")


class ImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    path: str
    preview_hash: str | None = None


class ActionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_hash: str | None = None


class EvalIn(ActionIn):
    backend: Literal["mjlab", "mujoco"] = "mjlab"
    profile: Literal["smoke", "acceptance"] = "smoke"


async def approved_image():
    image = await EbdClient().image_info(IMAGE)
    if not image.get("ok") or not re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("digest", "")):
        raise ValueError("原生 UFO 推理任务镜像尚未构建")
    return image["digest"]


async def scanned_file(path):
    path = validate_allowed_path(str(path))
    info = await EbdClient().scan(str(path))
    if info.get("files") != 1 or len(info.get("samples", [])) != 1:
        raise ValueError("文件无法通过 ebd 冻结")
    item = info["samples"][0]
    if item.get("size") != path.stat().st_size or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
        raise ValueError("文件冻结摘要无效")
    return {"source": str(path), "bytes": item["size"], "sha256": item["sha256"]}


async def import_plan(body: ImportIn):
    settings = get_settings()
    provided = Path(body.path).expanduser()
    if any(p.is_symlink() for p in (provided, *provided.parents)):
        raise ValueError("原生模型来源路径不能包含符号链接")
    root = validate_allowed_path(body.path)
    repo = settings.ufo_native_root.resolve()
    if not root.is_relative_to(repo / "runs") or root.is_symlink() or (root / "checkpoint/.saving").exists():
        raise ValueError("只接受 UFO runs 下完整、未在保存中的模型目录")
    for name in SOURCE_FILES:
        safe_file(root, name)
    codec = validate_config(
        read_json(root / "config.json"),
        read_json(root / "effective_config.json"),
        read_json(root / "checkpoint/model/config.json"),
        read_json(root / "checkpoint/model/init_kwargs.json"),
    )
    env = read_json(root / "config.json")["env"]
    scan = await EbdClient().scan(str(root / "checkpoint/model"))
    expected = {str(root / name) for name in SOURCE_FILES if name.startswith("checkpoint/model/")}
    samples = scan.get("samples", [])
    if scan.get("files") != 3 or {item["path"] for item in samples} != expected:
        raise ValueError("原生 checkpoint/model 必须只有完整权重、配置和 init_kwargs")
    files = [
        {
            "path": Path(item["path"]).relative_to(root).as_posix(),
            "bytes": item["size"],
            "sha256": item["sha256"],
        }
        for item in samples
    ]
    files += [
        {"path": name, "bytes": safe_file(root, name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in SOURCE_FILES
        if not name.startswith("checkpoint/model/")
    ]
    xml = validate_allowed_path(env["mjcf_path"])
    robot_config = validate_allowed_path(env["robot_config_path"])
    if not xml.is_relative_to(repo) or not robot_config.is_relative_to(repo):
        raise ValueError("机器人配置必须来自只读 UFO 来源")
    tree = ET.parse(safe_file(repo, xml.relative_to(repo).as_posix()))
    if tree.findall(".//include"):
        raise ValueError("当前原生导入只支持无外部 include 的已冻结 RPO MJCF")
    if any("file" in node.attrib and node.tag != "mesh" for node in tree.iter()):
        raise ValueError("原生 MJCF 含未冻结的外部资产引用")
    meshdir = tree.getroot().find("compiler").get("meshdir", "")
    meshroot = (xml.parent / meshdir).resolve()
    meshroot = validate_allowed_path(str(meshroot))
    motion = env["motion_data_path"]
    if not isinstance(motion, str):
        raise ValueError("首期原生导入要求单个辅助 motion 来源")
    motion = Path(motion)
    motion = motion if motion.is_absolute() else repo / motion
    if not motion.resolve().is_relative_to(repo):
        raise ValueError("辅助 motion 超出 UFO 来源")
    extra_sources = [
        (xml, "assets/rpo.xml"),
        (robot_config, "assets/robot.yaml"),
        (motion, "sim-data/motion.pkl"),
        (repo / "LICENSE", "licenses/UFO.txt"),
    ]
    for mesh in tree.findall(".//asset/mesh"):
        filename = mesh.get("file", "")
        extra_sources.append((safe_file(meshroot, filename), "assets/meshes/" + filename))
    extras = await asyncio.gather(*(scanned_file(path) for path, _ in extra_sources))
    for item, (_, destination) in zip(extras, extra_sources):
        item["path"] = destination
    if sum(f["bytes"] for f in files + extras) > 2 * 1024**3:
        raise ValueError("原生推理输入超过 2 GiB 导入预算")
    plan = {
        "operation": "ufo_native_import",
        "name": body.name.strip(),
        "image_digest": await approved_image(),
        "source": {
            "path": str(root),
            "files": sorted(files, key=lambda f: f["path"]),
            "extras": extras,
            "codec": codec,
        },
    }
    plan["hash"] = plan_hash(plan)
    return plan


async def native_model(session, mid, *, exported=None):
    model = await session.get(ModelVersion, mid)
    if (
        not model
        or model.embodiment_env_id != ENV_ID
        or model.card.get("schema_version") not in (MODEL_SCHEMA, EXPORT_SCHEMA)
    ):
        raise ValueError("原生 UFO 模型版本不存在")
    if exported is not None and (model.card["schema_version"] == EXPORT_SCHEMA) != exported:
        raise ValueError("此操作要求导出后的推理版本" if exported else "此操作要求原生 checkpoint 版本")
    artifact = await session.get(Artifact, model.artifact_id)
    if not artifact or artifact.kind not in ("ufo_native_model", "ufo_native_export"):
        raise ValueError("原生模型产物不存在")
    root = validate_allowed_path(artifact.storage)
    if not root.is_relative_to(get_settings().data_root.resolve() / "runs"):
        raise ValueError("原生模型不在受管存储中")
    manifest = read_json(safe_file(root, "manifest.json"))
    if plan_hash(manifest) != artifact.sha256:
        raise ValueError("原生模型 manifest 已变化")
    return model, artifact, manifest


async def action_plan(session, mid, operation, body=None):
    exported = operation != "ufo_native_export"
    model, artifact, payload_manifest = await native_model(session, mid, exported=exported)
    plan = {
        "operation": operation,
        "name": model.name,
        "model_id": model.id,
        "input_path": artifact.storage,
        "source_artifact_sha256": artifact.sha256,
        "image_digest": await approved_image(),
    }
    if operation == "ufo_native_eval":
        _, native_artifact, _ = await native_model(session, model.card["source_model_id"], exported=False)
        suite_id = f"suite_rpo_ufo_native_{body.backend}_{body.profile}_v2"
        suite = await session.get(EvalSuite, suite_id)
        expected = suite_spec(body.backend, body.profile)
        if not suite or not suite.frozen or suite.spec != expected or suite.embodiment_env_id != ENV_ID:
            raise ValueError("原生评估套件未冻结或已变化")
        plan.update(
            backend=body.backend,
            profile=body.profile,
            cases=eval_cases(body.profile),
            suite_id=suite_id,
            suite_spec=expected,
            native_source_path=native_artifact.storage,
            native_source_sha256=native_artifact.sha256,
        )
    if operation == "ufo_native_bundle":
        source_model, _, source_manifest = await native_model(
            session, model.card["source_model_id"], exported=False
        )
        plan.update(source_model_id=source_model.id, source_import_manifest=source_manifest)
        validate_bundle_source(payload_manifest, plan)
        reports = (
            await session.scalars(
                select(EvalReport)
                .where(EvalReport.model_version_id == mid)
                .order_by(EvalReport.created_at.desc())
            )
        ).all()
        selected = {}
        for report in reports:
            gate = report.gate_results or {}
            backend = gate.get("backend")
            if (
                backend in ("mjlab", "mujoco")
                and backend not in selected
                and gate.get("validity") == "complete"
            ):
                path = get_settings().data_root / "runs" / report.run_id / "outputs"
                source_receipt = read_json(safe_file(path, "receipt.json"))
                if source_receipt.get("payload_sha256") != artifact.sha256:
                    raise ValueError("评估报告绑定其他推理包")
                selected[backend] = {
                    "id": report.id,
                    "path": str(path),
                    "backend": backend,
                    "plan_hash": gate["plan_hash"],
                    "qualification": gate["qualification"],
                    "receipt_sha256": sha256_file(path / "receipt.json"),
                }
        if set(selected) != {"mjlab", "mujoco"}:
            raise ValueError("交付包需要两个后端的完整报告；任务失败可保留为仿真诊断包")
        plan["reports"] = [selected[key] for key in ("mjlab", "mujoco")]
    plan["hash"] = plan_hash(plan)
    return plan


def spec_for(plan):
    operation = plan["operation"]
    gpu = int(operation in ("ufo_native_eval", "ufo_native_live") and (
        plan["backend"] == "mjlab" or bool((plan.get("suite_spec") or {}).get("stream"))))
    kind = (
        "eval.sim2sim"
        if operation == "ufo_native_eval" and plan["backend"] == "mujoco"
        else OPERATIONS[operation]
    )
    return JobSpec(
        kind=kind,
        name=plan["name"] + " · " + operation.removeprefix("ufo_native_"),
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID if operation in ("ufo_native_eval", "ufo_native_live") else None,
        params={"operation": operation, "native_plan": plan},
        image=plan["image_digest"],
        inputs={"base_model": plan.get("input_path")},
        runtime={"network": "none"},
        resources={"gpus": gpu, "cpus": 4, "memory_gb": 16, "shm_gb": 2},
        policy={"timeout_h": 0.5, "max_retries": 0},
    )


async def submit(plan, expected, session, principal, key):
    if principal.actor_type == "agent":
        raise HTTPException(403, "原生部署入口仅供操作者使用；平台 Agent 仍须经过 ToolBus")
    if expected != plan["hash"]:
        raise HTTPException(409, "预览已过期，请重新检查")
    previous = await session.scalar(select(Job).where(Job.idem_key == key))
    if previous:
        if (previous.spec.get("params") or {}).get("native_plan", {}).get("hash") != plan["hash"]:
            raise HTTPException(409, "此幂等键已用于另一个冻结计划")
        return {"id": previous.id, "status": previous.status}
    if plan["operation"] == "ufo_native_import":
        fingerprint = plan_hash({"source": plan["source"], "image_digest": plan["image_digest"]})
        model = await session.scalar(
            select(ModelVersion).where(
                ModelVersion.embodiment_env_id == ENV_ID,
                ModelVersion.card["source_fingerprint"].astext == fingerprint,
            )
        )
        if model:
            run = await session.get(Run, model.card["run_id"])
            job = await session.get(Job, run.job_id)
            return {"id": job.id, "status": job.status, "model_version_id": model.id, "reused": True}
    job = await Orchestrator().submit(session, spec_for(plan), principal.id, key)
    return {"id": job.id, "status": job.status}


@router.get("/capabilities")
async def capabilities(principal: PrincipalDep):
    image = await EbdClient().image_info(IMAGE)
    return {
        "source_default": str(get_settings().ufo_native_root / "runs/rpo_experiment_20261016/final_model"),
        "interface_id": ENV_ID,
        "tasks": TASKS,
        "image_ready": image.get("ok", False),
    }


@router.post("/imports/preview")
async def preview_import(body: ImportIn, principal: PrincipalDep):
    try:
        plan = await import_plan(body)
        return {"plan": plan, "resources": spec_for(plan).resources.model_dump()}
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/imports")
async def start_import(
    body: ImportIn,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    try:
        plan = await import_plan(body)
        return await submit(plan, body.preview_hash, session, principal, idem)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


async def model_action(mid, action, body, session, principal, key=None):
    try:
        plan = await action_plan(session, mid, "ufo_native_" + action, body)
        if key is None:
            return {"plan": plan, "resources": spec_for(plan).resources.model_dump()}
        return await submit(plan, body.preview_hash, session, principal, key)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/models/{mid}/export/preview")
async def preview_export(mid: str, body: ActionIn, session: SessionDep, principal: PrincipalDep):
    return await model_action(mid, "export", body, session, principal)


@router.post("/models/{mid}/export")
async def start_export(
    mid: str,
    body: ActionIn,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    return await model_action(mid, "export", body, session, principal, idem)


@router.post("/models/{mid}/evals/preview")
async def preview_eval(mid: str, body: EvalIn, session: SessionDep, principal: PrincipalDep):
    return await model_action(mid, "eval", body, session, principal)


@router.post("/models/{mid}/evals")
async def start_eval(
    mid: str,
    body: EvalIn,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    return await model_action(mid, "eval", body, session, principal, idem)


@router.post("/models/{mid}/bundles/preview")
async def preview_bundle(mid: str, body: ActionIn, session: SessionDep, principal: PrincipalDep):
    return await model_action(mid, "bundle", body, session, principal)


@router.post("/models/{mid}/bundles")
async def start_bundle(
    mid: str,
    body: ActionIn,
    session: SessionDep,
    principal: PrincipalDep,
    idem: str = Header(..., alias="Idempotency-Key"),
):
    return await model_action(mid, "bundle", body, session, principal, idem)


def file_url(run_id, path):
    return f"/api/v1/ufo-native/runs/{run_id}/files/{quote(path, safe='/')}"


@router.get("/models/{mid}/status")
async def model_status(mid: str, session: SessionDep, principal: PrincipalDep):
    try:
        model, _, _ = await native_model(session, mid)
    except (ValueError, OSError) as exc:
        raise HTTPException(404, str(exc)) from exc
    exports = (
        await session.scalars(select(ModelVersion).where(ModelVersion.card["source_model_id"].astext == mid))
    ).all()
    jobs = (
        await session.scalars(
            select(Job)
            .where(Job.spec["params"]["native_plan"]["model_id"].astext == mid)
            .order_by(Job.created_at.desc())
            .limit(30)
        )
    ).all()
    job_rows = []
    for job in jobs:
        run = await session.scalar(
            select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc()).limit(1)
        )
        job_rows.append(
            {
                "id": job.id,
                "name": job.spec.get("name"),
                "kind": job.kind,
                "status": job.status,
                "error": job.error,
                "run_id": run.id if run else None,
                "metrics_summary": run.metrics_summary if run else {},
            }
        )
    reports = (
        await session.scalars(
            select(EvalReport)
            .where(EvalReport.model_version_id == mid)
            .order_by(EvalReport.created_at.desc())
        )
    ).all()
    report_rows = []
    for report in reports:
        metrics = dict(report.metrics)
        metrics["cases"] = [
            {**row, "video_url": file_url(report.run_id, row["video"])} for row in metrics.get("cases", [])
        ]
        names = (
            ["receipt.json", "summary.svg", "timeseries.svg", "cases.jsonl"]
            if report.gate_results.get("validity") == "complete"
            else [item["path"] for item in report.gate_results.get("partial_files", [])]
        )
        report_rows.append(
            {
                "id": report.id,
                "run_id": report.run_id,
                "backend": report.gate_results.get("backend"),
                "profile": report.gate_results.get("profile"),
                "qualification": report.gate_results.get("qualification"),
                "metrics": metrics,
                "files": [{"path": name, "url": file_url(report.run_id, name)} for name in names],
            }
        )
    bundles = (await session.scalars(select(Bundle).where(Bundle.model_version_id == mid))).all()
    return {
        "model": {"id": model.id, "name": model.name, "card": model.card},
        "exports": [{"id": m.id, "name": m.name, "card": m.card} for m in exports],
        "jobs": job_rows,
        "reports": report_rows,
        "bundles": [
            {"id": b.id, "manifest": b.manifest, "download_url": b.manifest.get("download_url")}
            for b in bundles
        ],
    }


async def stream_output(request, run_id, path, *, live_kind=None):
    if live_kind is not None and live_kind not in ("state", "scene"):
        raise HTTPException(400, "Unsupported live snapshot")
    settings = get_settings()
    # The private UDS uses plain HTTP; avoid rebuilding an unused TLS context at 10Hz.
    client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=settings.ebd_media_uds, verify=False), timeout=60.0
    )
    forwarded = {key: request.headers[key] for key in ("range",) if key in request.headers}
    try:
        response = await client.send(
            client.build_request(
                request.method,
                (f"http://ebd/v1/runs/{run_id}/live/{live_kind}" if live_kind is not None
                 else f"http://ebd/v1/runs/{run_id}/outputs/{quote(path, safe='/')}"),
                headers=forwarded,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(502, "ebd 文件通道不可用") from exc

    async def chunks():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    headers = {
        key: value
        for key, value in response.headers.items()
        if key in ("content-type", "content-length", "content-range", "accept-ranges", "cache-control")
    }
    if path.endswith(".ebp"):
        headers["Content-Disposition"] = 'attachment; filename="rpo-ufo-native.ebp"'
    return StreamingResponse(chunks(), status_code=response.status_code, headers=headers)


@router.api_route("/runs/{rid}/files/{path:path}", methods=["GET", "HEAD"])
async def output_file(rid: str, path: str, request: Request, session: SessionDep, principal: PrincipalDep):
    run = await session.get(Run, rid)
    if not run:
        raise HTTPException(404, "没有完整的原生产物")
    job = await session.get(Job, run.job_id)
    if not job or (job.spec.get("params") or {}).get("operation") not in OPERATIONS:
        raise HTTPException(404)
    root = get_settings().data_root / "runs" / rid / "outputs"
    if run.status == "finished":
        value = read_json(root / "receipt.json")
        allowed = {"receipt.json"} | {item["path"] for item in value.get("files", [])}
    elif (job.spec.get("params") or {}).get("operation") == "ufo_native_live":
        artifact = await session.scalar(select(Artifact).where(
            Artifact.run_id == rid, Artifact.kind == "ufo_native_live_diagnostic"))
        allowed = {item["path"] for item in (artifact.meta if artifact else {}).get("partial_files", [])}
    else:
        report = await session.scalar(select(EvalReport).where(EvalReport.run_id == rid))
        allowed = {item["path"] for item in (report.gate_results if report else {}).get("partial_files", [])}
    if path not in allowed:
        raise HTTPException(404)
    return await stream_output(request, rid, path)


@router.api_route("/bundles/{bid}/download", methods=["GET", "HEAD"])
async def bundle_download(bid: str, request: Request, session: SessionDep, principal: PrincipalDep):
    bundle = await session.get(Bundle, bid)
    if not bundle or bundle.manifest.get("schema_version") != "eb/rpo-ufo-native-bundle/v1":
        raise HTTPException(404)
    return await stream_output(request, bundle.manifest["run_id"], "policy.ebp")
