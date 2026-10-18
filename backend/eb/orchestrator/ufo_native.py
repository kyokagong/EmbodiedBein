"""Registration and submission checks for native inference artifacts."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from sqlalchemy import select

from eb.ids import new_id
from eb.store.ebd import EbdClient
from eb.store.models import Artifact, Bundle, EvalReport, LineageEdge, ModelFamily, ModelVersion
from eb.ufo_native import (
    ENV_ID,
    EXPORT_SCHEMA,
    MODEL_SCHEMA,
    plan_hash,
    read_json,
    safe_file,
    suite_spec,
    validate_bundle_source,
    validate_cases,
    validate_job,
)


async def validate_submission(session, spec):
    from eb.api.routes_ufo_native import ActionIn, EvalIn, ImportIn, action_plan, import_plan, spec_for

    plan = validate_job(spec)
    if plan["operation"] == "ufo_native_import":
        expected = await import_plan(ImportIn(name=plan["name"], path=plan["source"]["path"]))
    elif plan["operation"] == "ufo_native_live":
        from eb.api.routes_ufo_live import LiveIn, build_plan
        expected = await build_plan(session, LiveIn(
            model_id=plan["model_id"], environment_id=plan["environment_id"],
            initial_task=plan["initial_task"], duration_seconds=plan["duration_seconds"], seed=plan["seed"],
        ))
    else:
        body = (
            EvalIn(backend=plan["backend"], profile=plan["profile"])
            if plan["operation"] == "ufo_native_eval"
            else ActionIn()
        )
        expected = await action_plan(session, plan["model_id"], plan["operation"], body)
    canonical = spec_for(expected)
    if (
        plan != expected
        or spec.inputs != canonical.inputs
        or spec.task_id != canonical.task_id
        or spec.fanout
        or spec.outputs
        or spec.resources.devices
    ):
        raise ValueError("Native submission differs from registered inputs or frozen suite")


async def verify_index(root, files, *, max_files=200):
    if not files or len(files) > max_files or len({f["path"] for f in files}) != len(files):
        raise ValueError("Native output index is incomplete or duplicate")

    async def verify(item):
        path = safe_file(root, item["path"])
        if (
            type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= 2 * 1024**3
            or path.stat().st_size != item["bytes"]
            or not re.fullmatch("[a-f0-9]{64}", item["sha256"])
        ):
            raise ValueError("Native output file metadata differs")
        scan = await EbdClient().scan(str(path))
        if scan.get("files") != 1 or scan["samples"][0]["sha256"] != item["sha256"]:
            raise ValueError(f"Native output hash differs: {item['path']}")

    limiter = asyncio.Semaphore(16)

    async def bounded_verify(item):
        async with limiter:
            await verify(item)

    await asyncio.gather(*(bounded_verify(item) for item in files))


async def collect_native(session, job, run, outputs):
    from eb.domain.job import JobSpec

    plan = validate_job(JobSpec.model_validate(job.spec))
    value = read_json(safe_file(outputs, "receipt.json"))
    if (
        value.get("schema_version") != "eb/ufo-native-receipt/v1"
        or value.get("run_id") != run.id
        or value.get("plan_hash") != plan["hash"]
        or value.get("operation") != plan["operation"]
        or value.get("image_digest") != plan["image_digest"]
    ):
        raise ValueError("Native output receipt identity differs")
    operation = plan["operation"]
    if operation in ("ufo_native_import", "ufo_native_export"):
        package = "model" if operation == "ufo_native_import" else "payload"
        root = outputs / package
        manifest = read_json(safe_file(root, "manifest.json"))
        if value.get("manifest") != manifest or value.get("package") != package:
            raise ValueError("Native package receipt differs")
        await verify_index(root, manifest["files"])
        if operation == "ufo_native_import":
            if manifest.get("source") != plan["source"] or manifest.get("loader_verified") is not True:
                raise ValueError("Native import lacks loader/source evidence")
        else:
            verification = manifest.get("verification") or {}
            if (
                manifest.get("source_artifact_sha256") != plan["source_artifact_sha256"]
                or verification.get("verified") is not True
                or verification.get("samples", 0) < 96
                or verification.get("max_target_error_rad", float("inf")) > 1e-4
                or manifest.get("input_dim") != 631
                or manifest.get("output_dim") != 23
            ):
                raise ValueError("Native export verification is incomplete")
        kind = "ufo_native_model" if operation == "ufo_native_import" else "ufo_native_export"
        artifact = await session.scalar(
            select(Artifact).where(Artifact.sha256 == plan_hash(manifest), Artifact.kind == kind)
        )
        if not artifact:
            artifact = Artifact(
                id=new_id("art"),
                kind=kind,
                sha256=plan_hash(manifest),
                size_bytes=sum(f["bytes"] for f in manifest["files"]),
                storage=str(root),
                run_id=run.id,
                pinned=True,
                meta={"manifest": manifest, "simulation_only": True},
            )
            session.add(artifact)
            await session.flush()
        model = await session.scalar(select(ModelVersion).where(ModelVersion.artifact_id == artifact.id))
        if not model:
            family_id = "mf_rpo_ufo_native"
            if not await session.get(ModelFamily, family_id):
                session.add(ModelFamily(id=family_id, name="RPO native UFO FB-CPR", family="mlp"))
                await session.flush()
            model = ModelVersion(
                id=new_id("mv"),
                family_id=family_id,
                name=plan["name"] + (" · ONNX" if package == "payload" else ""),
                embodiment_env_id=ENV_ID,
                artifact_id=artifact.id,
                provenance="partial",
                io_signature={"obs": 375, "latent": 256, "actor_obs": 631, "act": 23},
                tags=[
                    "rpo-v2",
                    "ufo-native",
                    "simulation-only",
                    "onnx" if package == "payload" else "safetensors",
                ],
                card={
                    "schema_version": MODEL_SCHEMA if package == "model" else EXPORT_SCHEMA,
                    "package_path": str(root),
                    "run_id": run.id,
                    "simulation_only": True,
                    "readiness": "export_verified" if package == "payload" else "imported",
                    "source_model_id": plan.get("model_id"),
                    "training_provenance": "external_declared",
                    "prompt_derivation": "external_unverified",
                    "codec": manifest["codec"],
                    "source_fingerprint": plan_hash(
                        {"source": plan["source"], "image_digest": plan["image_digest"]}
                    )
                    if package == "model"
                    else None,
                },
            )
            session.add(model)
            session.add(
                LineageEdge(
                    src_type="run",
                    src_id=run.id,
                    dst_type="model_version",
                    dst_id=model.id,
                    relation="produced",
                    run_id=run.id,
                )
            )
            if plan.get("model_id"):
                session.add(
                    LineageEdge(
                        src_type="model_version",
                        src_id=plan["model_id"],
                        dst_type="model_version",
                        dst_id=model.id,
                        relation="exported_from",
                        run_id=run.id,
                    )
                )
        run.metrics_summary = {
            "model_version_id": model.id,
            "artifact_id": artifact.id,
            "native_operation": operation,
            "package_bytes": artifact.size_bytes,
            **({"export_verification": manifest["verification"]} if package == "payload" else {}),
        }
    elif operation == "ufo_native_eval":
        if value.get("payload_sha256") != plan["source_artifact_sha256"]:
            raise ValueError("Native evaluation used another inference payload")
        metrics = validate_cases(plan, value["cases"])
        await verify_index(outputs, value["files"])
        filenames = {f["path"] for f in value["files"]}
        if not {"summary.svg", "timeseries.svg", "cases.jsonl"}.issubset(filenames):
            raise ValueError("Native evaluation plots/raw cases are missing")
        for case in value["cases"]:
            media = case.get("media") or {}
            if (
                case["video"] not in filenames
                or f"traces/{case['case_id']}.npz" not in filenames
                or not media.get("decodable")
                or media.get("frames", 0) < 1
                or media.get("fps") != 25
                or not 0 < media.get("seconds", 0) <= 21
            ):
                raise ValueError("Native evaluation case video is unavailable")
        report = await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id))
        if not report:
            report = EvalReport(
                id=new_id("ev"),
                suite_id=plan["suite_id"],
                model_version_id=plan["model_id"],
                run_id=run.id,
                metrics=metrics,
                gate_results={
                    "qualification": metrics["qualification"],
                    "validity": "complete",
                    "backend": plan["backend"],
                    "profile": plan["profile"],
                    "plan_hash": plan["hash"],
                    "receipt": value,
                    "scope": "simulation_only",
                    "physics_family": "MuJoCo",
                },
            )
            session.add(report)
            session.add(
                LineageEdge(
                    src_type="model_version",
                    src_id=plan["model_id"],
                    dst_type="eval_report",
                    dst_id=report.id,
                    relation="evaluated",
                    run_id=run.id,
                )
            )
        run.metrics_summary = {
            "eval_report_id": report.id,
            "qualification": metrics["qualification"],
            "success_rate": metrics["success_rate"],
            "planned": metrics["planned"],
            "completed": metrics["completed"],
        }
    elif operation == "ufo_native_live":
        from eb.orchestrator.ufo_live import collect_live

        await collect_live(session, run, outputs, plan, value)
    else:
        await verify_index(outputs, value["files"])
        manifest = value["manifest"]
        validate_bundle_source(manifest.get("payload_manifest") or {}, plan)
        if (
            manifest.get("source_import_manifest") != plan["source_import_manifest"]
            or manifest.get("source_model_id") != plan["source_model_id"]
            or manifest.get("payload_sha256") != plan["source_artifact_sha256"]
            or manifest.get("simulation_only") is not True
            or manifest.get("signed") is not False
            or set(manifest.get("coverage", {})) != {"mjlab", "mujoco"}
            or manifest.get("model_id") != plan["model_id"]
            or manifest.get("image_digest") != plan["image_digest"]
            or manifest.get("coverage")
            != {
                r["backend"]: {"report_id": r["id"], "qualification": r["qualification"]}
                for r in plan["reports"]
            }
        ):
            raise ValueError("Native bundle claims unsupported scope")
        evidence = manifest.get("evidence") or []
        if len(evidence) != len(plan["reports"]):
            raise ValueError("Native bundle report evidence is incomplete")
        for actual, expected in zip(evidence, plan["reports"]):
            if (
                actual.get("report_id") != expected["id"]
                or actual.get("backend") != expected["backend"]
                or actual.get("qualification") != expected["qualification"]
                or actual.get("receipt_sha256") != expected["receipt_sha256"]
                or (actual.get("receipt") or {}).get("plan_hash") != expected["plan_hash"]
                or plan_hash(actual["receipt"])
                != plan_hash(read_json(Path(expected["path"]) / "receipt.json"))
            ):
                raise ValueError("Native bundle report evidence changed")
        bundle = await session.scalar(select(Bundle).where(Bundle.manifest["run_id"].astext == run.id))
        if not bundle:
            bid = new_id("bnd")
            manifest = {
                **manifest,
                "run_id": run.id,
                "download_url": f"/api/v1/ufo-native/bundles/{bid}/download",
                "file": value["files"][0],
            }
            bundle = Bundle(
                id=bid,
                model_version_id=plan["model_id"],
                embodiment_env_id=ENV_ID,
                stage="simulation_only",
                signed=False,
                manifest=manifest,
            )
            session.add(bundle)
            session.add(
                Artifact(
                    id=new_id("art"),
                    kind="ufo_native_bundle",
                    run_id=run.id,
                    sha256=value["files"][0]["sha256"],
                    size_bytes=value["files"][0]["bytes"],
                    storage=str(outputs / "policy.ebp"),
                    pinned=True,
                    meta={"bundle_id": bid},
                )
            )
            session.add(
                LineageEdge(
                    src_type="model_version",
                    src_id=plan["model_id"],
                    dst_type="bundle",
                    dst_id=bid,
                    relation="packaged",
                    run_id=run.id,
                )
            )
        run.metrics_summary = {
            "bundle_id": bundle.id,
            "simulation_only": True,
            "download_url": bundle.manifest["download_url"],
        }


async def seed_native(session):
    from eb.domain.environment_profile import profile_hash
    from eb.domain.task import TaskSpec
    from eb.store.models import EmbodimentEnvRow, EnvironmentProfile, EvalSuite, TaskRow
    from eb.ufo_native import TASK_ID

    env = await session.get(EmbodimentEnvRow, ENV_ID)
    if not env or env.status != "active":
        return
    task = TaskSpec(id=TASK_ID, compatible_env_ids=[ENV_ID], status="active", stack_bindings={})
    if not await session.get(TaskRow, TASK_ID):
        session.add(TaskRow(id=TASK_ID, spec=task.model_dump(), spec_hash=task.spec_hash(), status="active"))
        await session.flush()
    for backend in ("mjlab", "mujoco"):
        for profile in ("smoke", "acceptance"):
            sid = f"suite_rpo_ufo_native_{backend}_{profile}_v2"
            spec = suite_spec(backend, profile)
            if not await session.get(EvalSuite, sid):
                session.add(
                    EvalSuite(
                        id=sid,
                        name=f"UFO/RPO {backend} {profile}",
                        embodiment_env_id=ENV_ID,
                        seeds=[c["seed"] for c in spec["cases"]],
                        frozen=True,
                        spec=spec,
                    )
                )
            pid = f"ep_rpo_ufo_native_{backend}_{profile}_v2"
            if not await session.get(EnvironmentProfile, pid):
                row = EnvironmentProfile(
                    id=pid,
                    name=f"UFO/RPO {backend} {profile}",
                    usage="eval",
                    embodiment_env_id=ENV_ID,
                    task_id=TASK_ID,
                    suite_id=sid,
                    template_id=None,
                    status="active",
                    definition={
                        "task": "冻结原生 UFO 站立/前行任务；ONNX 闭环，50 Hz/400 Hz。",
                        "reward": "不计算训练奖励。",
                        "failure": "高度低于 0.5 m 或倾角超过 0.8 rad 独立判倒；缺媒体为证据不完整。",
                        "success": "逐例按冻结 Suite 判定；smoke 只用于诊断，真机资格未验证。",
                        "settings": {},
                    },
                    content_hash="",
                )
                row.content_hash = profile_hash(row)
                session.add(row)
    await session.flush()
    from eb.ufo_live import environment_spec, live_suite_spec
    for backend in ("mujoco", "mjlab"):
        expected = environment_spec(backend)
        sid = expected["suite_id"]
        if not await session.get(EvalSuite, sid):
            session.add(EvalSuite(
                id=sid, name=expected["name"], embodiment_env_id=ENV_ID,
                seeds=[71000], frozen=True, spec=live_suite_spec(backend),
            ))
        if not await session.get(EnvironmentProfile, expected["id"]):
            session.add(EnvironmentProfile(
                **{key: value for key, value in expected.items() if key != "backend"}, status="active",
            ))
    await session.flush()
