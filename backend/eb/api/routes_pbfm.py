"""Operator entrypoints for frozen PBFM data, export and independent terrain Jobs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.ids import new_id
from eb.orchestrator.service import Orchestrator
from eb.pbfm_data import ENV_ID, EXPORT_OPERATION, IMAGE, OPERATION, PLAN_SCHEMA, validate_source_manifest
from eb.pbfm_evaluation import EVAL_OPERATION, SUITE_ID, build_eval_plan, suite_spec
from eb.pbfm_io import fingerprint, hash_value, read_json, safe_file, sha256
from eb.pbfm_runtime_probe import ProbeIn
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Artifact, Dataset, DatasetVersion, EvalReport, EvalSuite, Job, ModelVersion, Run

router = APIRouter(prefix="/pbfm")


class DataIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    parent_version_id: str
    source_manifest: str
    preview_hash: str | None = None


class ActionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motion_id: int = Field(default=0, ge=0, le=100000)
    preview_hash: str | None = None


class UfoDataIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    parent_dataset_id: str
    split_groups: dict[str, list[str]]
    excluded_groups: list[str]
    preview_hash: str | None = None


class FreezeSourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest_relative_path: str = "pbfm-source.json"


class RegisterPeriodicIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=112, lt=100000)
    preview_hash: str | None = None


@router.post("/training/{jid}/checkpoints/preview")
async def preview_periodic(jid: str, body: RegisterPeriodicIn, session: SessionDep, principal: PrincipalDep):
    operator(principal)
    from eb.pbfm_periodic import periodic_context

    try:
        _, _, _, checkpoint, binding = await periodic_context(session, jid, body.step)
        return {"hash": fingerprint(binding), "source": binding, "counts": checkpoint["counts"],
                "bytes": checkpoint["bytes"], "training_completed": False}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/training/{jid}/checkpoints/register")
async def register_periodic_checkpoint(jid: str, body: RegisterPeriodicIn,
        session: SessionDep, principal: PrincipalDep):
    operator(principal)
    from eb.api.routes_train import _lock_native_train_submission
    from eb.pbfm_periodic import periodic_context, register_periodic

    try:
        await _lock_native_train_submission(session, f"pbfm-periodic:{jid}:{body.step}")
        _, _, _, _, binding = await periodic_context(session, jid, body.step)
        if body.preview_hash != fingerprint(binding):
            raise HTTPException(409, "P-BFM periodic checkpoint preview changed")
        model_id = await register_periodic(session, jid, body.step)
        await session.commit()
        return {"model_version_id": model_id, "source": binding, "training_completed": False}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/datasets/{did}/sources/freeze")
async def freeze_source(did: str, body: FreezeSourceIn, session: SessionDep, principal: PrincipalDep):
    """Register source byte identity; numerical qualification remains a data.convert Job."""
    operator(principal)
    parent = await session.get(Dataset, did)
    if (
        not parent
        or parent.source_kind not in {"path", "rpo_robot_state", "ufo_native_motion"}
        or not (parent.embodiment_env_id or "").startswith("ee_rpo_v2_")
    ):
        raise HTTPException(422, "register an RPO local source Dataset before freezing its source version")
    try:
        from eb.paths import validate_allowed_path

        root = validate_allowed_path(parent.dest_path)
        manifest_path = safe_file(root, body.manifest_relative_path)
        manifest = validate_source_manifest(read_json(manifest_path))
        digest = sha256(manifest_path)
        for split in manifest["splits"].values():
            for row in split:
                path = safe_file(root, row["path"])
                scan = await EbdClient().scan(str(path))
                files = scan.get("samples") or []
                if (
                    scan.get("files") != 1
                    or len(files) != 1
                    or files[0].get("sha256") != row["sha256"]
                    or files[0].get("size") != row["bytes"]
                ):
                    raise ValueError("source bytes differ from the declared manifest")
        label = "pbfm-source-" + digest[:20]
        version = await session.scalar(
            select(DatasetVersion).where(DatasetVersion.dataset_id == did, DatasetVersion.version == label)
        )
        if version is None:
            version = DatasetVersion(
                id=new_id("dsv"),
                dataset_id=did,
                version=label,
                compatible_envs=[ENV_ID],
                content_hash="sha256:" + digest,
                stats={
                    "source_manifest_sha256": digest,
                    "pbfm_source_schema": manifest["schema_version"],
                    "source_files": manifest["splits"],
                    "numerically_qualified": False,
                },
            )
            session.add(version)
            await session.commit()
        return {
            "dataset_id": did,
            "version_id": version.id,
            "content_hash": version.content_hash,
            "source_manifest": str(manifest_path),
            "numerically_qualified": False,
        }
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


def operator(principal):
    if principal.actor_type == "agent":
        raise HTTPException(403, "PBFM workbench requires an operator; platform Agents use ToolBus")


async def image_digest():
    info = await EbdClient().image_info(IMAGE)
    if info.get("ok") is not True or not hash_value(info.get("digest"), prefixed=True):
        raise ValueError("PBFM isolated runtime image is unavailable")
    return info["digest"]


async def build_data_plan(body: DataIn, session):
    version = await session.get(DatasetVersion, body.parent_version_id)
    parent = await session.get(Dataset, version.dataset_id) if version else None
    if (
        not parent
        or not hash_value(version.content_hash, prefixed=True)
        or parent.status in {"failed", "invalid"}
        or parent.source_kind not in {"path", "ufo_native_motion", "rpo_robot_state", "rpo_amp_motion"}
        or not parent.embodiment_env_id
        or not parent.embodiment_env_id.startswith("ee_rpo_v2_")
    ):
        raise ValueError("PBFM preparation requires a frozen registered RPO RobotState/UFO DatasetVersion")
    source_root = Path(parent.dest_path)
    manifest_path = Path(body.source_manifest)
    if not manifest_path.is_absolute() or not manifest_path.is_relative_to(source_root):
        raise ValueError("motion source manifest must be inside the registered parent content root")
    manifest_path = safe_file(source_root, manifest_path.relative_to(source_root).as_posix())
    manifest = validate_source_manifest(read_json(manifest_path))
    if version.content_hash != "sha256:" + sha256(manifest_path) and (version.stats or {}).get(
        "source_manifest_sha256"
    ) != sha256(manifest_path):
        raise ValueError("source manifest is not frozen by the registered parent DatasetVersion")
    client = EbdClient()
    for split in manifest["splits"].values():
        for row in split:
            path = safe_file(source_root, row["path"])
            scan = await client.scan(str(path))
            samples = scan.get("samples") or []
            if (
                scan.get("files") != 1
                or len(samples) != 1
                or samples[0].get("sha256") != row["sha256"]
                or samples[0].get("size") != row["bytes"]
            ):
                raise ValueError("source file does not match its immutable hash/size")
    settings = get_settings()
    snapshots = await asyncio.gather(
        client.snapshot(str(settings.pbfm_root)),
        client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")),
    )
    refs = {
        name: {
            "kind": "content_snapshot",
            "path": snap["path"],
            "snapshot_hash": snap["content_hash"],
            "manifest_path": str(Path(snap["path"]).parent / "manifest.json"),
        }
        for name, snap in zip(("pbfm", "asset"), snapshots)
    }
    plan = {
        "schema_version": PLAN_SCHEMA,
        "operation": OPERATION,
        "parent_version_id": version.id,
        "parent_dataset_id": parent.id,
        "parent_manifest_sha256": version.content_hash.removeprefix("sha256:"),
        "source_root": str(source_root),
        "source_manifest": manifest,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_refs": refs,
        "image_digest": await image_digest(),
    }
    plan["hash"] = fingerprint(plan)
    return plan


async def build_action_plan(mid: str, operation: str, body: ActionIn, session):
    model = await session.get(ModelVersion, mid)
    artifact = await session.get(Artifact, model.artifact_id) if model else None
    if (
        not model
        or model.family_id != "mf_rpo_pbfm_depth"
        or model.embodiment_env_id != ENV_ID
        or (model.card or {}).get("schema_version") != "eb/rpo-pbfm-trained-model/v1"
        or not artifact
    ):
        raise ValueError("model is not a verified PBFM RPO candidate")
    run = await session.get(Run, artifact.run_id)
    job = await session.get(Job, run.job_id) if run else None
    periodic = (model.card or {}).get("periodic_checkpoint")
    if not run or not job or not periodic and (run.status != "finished" or job.status != "succeeded"):
        raise ValueError("model has no successful immutable producer")
    periodic_source = None
    if periodic:
        from eb.pbfm_periodic import periodic_model_source

        periodic_source = await periodic_model_source(session, model, job, run)
    producer_image = job.spec.get("image")
    available = await EbdClient().image_info(producer_image)
    if (
        not hash_value(producer_image, prefixed=True)
        or available.get("ok") is not True
        or available.get("digest") != producer_image
    ):
        raise ValueError("the model producer's frozen PBFM image is unavailable")
    train = job.spec["params"]["native_train_plan"]
    identity = train["identity"]
    dataset = await session.get(Dataset, identity["dataset_id"])
    from eb.pbfm_data import require_pbfm_dataset

    data = require_pbfm_dataset(dataset)
    if data["manifest_sha256"] != identity["dataset_manifest"]:
        raise ValueError("training Dataset manifest changed after the model producer was frozen")
    data_root = Path(data["storage_ref"])
    validation_root = data_root.parent / "holdout"
    validation = read_json(safe_file(validation_root, "manifest.json"))
    if validation.get("split") != "validation" or validation.get("parent_version_id") != data.get(
        "parent_version_id"
    ):
        raise ValueError("model-bound validation prompt data differs")
    if sha256(validation_root / "manifest.json") != (data.get("holdout_manifests") or {}).get("validation"):
        raise ValueError("validation prompt manifest differs from the model's frozen training Dataset")
    native = (Path(periodic_source["native_path"]) if periodic_source else
              get_settings().data_root / "runs" / run.id / "outputs/native")
    resume = await session.get(Artifact, (model.card or {}).get("resume_artifact_id"))
    network_manifest = (artifact.meta or {}).get("manifest") or {}
    resume_manifest = (resume.meta or {}).get("manifest") if resume else None
    if (
        artifact.kind != "pbfm_native_trained_model"
        or artifact.sha256 != "sha256:" + fingerprint(network_manifest)
        or not resume
        or resume.kind != "pbfm_native_resume"
        or resume.run_id != run.id
        or resume.sha256 != "sha256:" + fingerprint(resume_manifest or {})
        or resume.storage != str(native)
        or periodic_source is not None and resume_manifest != periodic_source["manifest"]
    ):
        raise ValueError("published PBFM model/checkpoint manifest identity differs")
    frozen_files = {row["path"]: row for row in resume_manifest["files"]}
    files = []
    for name in (
        "config.json",
        "effective_config.json",
        "source-adaptation.json",
        "checkpoint/model/config.json",
        "checkpoint/model/init_kwargs.json",
        "checkpoint/model/model.safetensors",
    ):
        path = safe_file(native, name)
        info = await EbdClient().scan(str(path))
        rows = info.get("samples") or []
        if info.get("files") != 1 or len(rows) != 1 or not hash_value(rows[0].get("sha256")):
            raise ValueError("model bytes cannot be frozen by ebd")
        entry = {"path": name, "bytes": rows[0]["size"], "sha256": rows[0]["sha256"]}
        if frozen_files.get(name) != entry:
            raise ValueError("published model or inference configuration changed after collection")
        files.append(entry)
    refs = {}
    for name in ("pbfm", "asset"):
        content_hash = identity[name + "_snapshot"]
        root = get_settings().data_root / "snapshots" / content_hash.removeprefix("sha256:") / "code"
        refs[name] = {
            "kind": "content_snapshot",
            "path": str(root),
            "snapshot_hash": content_hash,
            "manifest_path": str(root.parent / "manifest.json"),
        }
    base = {
        "operation": operation,
        "model_version_id": model.id,
        "model_hash": artifact.sha256.removeprefix("sha256:"),
        "model_path": str(native),
        "model_files": files,
        "source_refs": refs,
        "camera_contract": identity["camera_contract"],
        "terrain_contract": identity["terrain_contract"],
        "dataset_manifest": data["manifest_sha256"],
        "prompt_path": str(validation_root),
        "prompt_manifest_sha256": sha256(validation_root / "manifest.json"),
        "prompt_file_sha256": validation["file_sha256"],
        "motion_id": body.motion_id,
        "image_digest": producer_image,
        "robot_xml_sha256": data["robot_xml_sha256"],
        "pbfm_patch_sha256": data["pbfm_patch_sha256"],
        "evaluation_mode": {
            "domain_randomization": False,
            "observation_noise": False,
            "camera_installation_randomization": False,
        },
        "paired_rollout": {
            "seed": 71000,
            "steps": 32,
            "terrain": "flat",
            "difficulty_row": 0,
            "heading_world_xy": [1.0, 0.0],
            "velocity_world_xy": [0.3, 0.0],
            "comparison": "PyTorch and actual ONNX each feed their own successor observations",
        },
    }
    if operation == EXPORT_OPERATION:
        base["schema_version"] = "eb/pbfm-export-plan/v1"
        base["hash"] = fingerprint(base)
        return base
    exports = (
        await session.scalars(
            select(Artifact)
            .where(Artifact.kind == "pbfm_policy", Artifact.meta["model_version_id"].astext == mid)
            .order_by(Artifact.created_at.desc())
        )
    ).all()
    if not exports:
        raise ValueError(
            "export and verify the current model's complete direct-depth policy and tracking prompt first"
        )
    exported = exports[0]
    export_plan = exported.meta.get("plan") or {}
    if export_plan.get("model_hash") != base["model_hash"] or export_plan.get("motion_id") != body.motion_id:
        raise ValueError("export prompt is bound to another model or motion")
    geometry = {
        "robot": "rpo",
        "foot_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "xml_sha256": data["robot_xml_sha256"],
        "method": "native collision geom projection",
    }
    bindings = {
        k: v
        for k, v in base.items()
        if k
        not in {
            "operation",
            "model_version_id",
            "model_hash",
            "dataset_manifest",
            "image_digest",
            "source_refs",
            "camera_contract",
        }
    }
    return build_eval_plan(
        **{
            k: base[k]
            for k in (
                "model_version_id",
                "model_hash",
                "dataset_manifest",
                "image_digest",
                "source_refs",
                "camera_contract",
            )
        },
        export_hash=exported.sha256,
        robot_geometry=geometry,
        export_path=exported.storage,
        export_artifact_id=exported.id,
        **bindings,
    )


def action_spec(plan):
    from eb.pbfm_evaluation import action_spec

    return action_spec(plan)


async def submit(plan, spec, expected, key, principal, session):
    operator(principal)
    if expected != plan["hash"]:
        raise HTTPException(409, "PBFM preview is stale; preview the frozen inputs again")
    previous = await session.scalar(select(Job).where(Job.idem_key == key))
    if previous:
        if previous.spec.get("params", {}).get("pbfm_plan", {}).get("hash") != plan["hash"]:
            raise HTTPException(409, "idempotency key belongs to another PBFM plan")
        return {"id": previous.id, "status": previous.status}
    job = await Orchestrator().submit(session, spec, principal.id, key)
    return {"id": job.id, "status": job.status}


@router.post("/datasets/prepare/preview")
async def preview_data(body: DataIn, session: SessionDep, principal: PrincipalDep):
    operator(principal)
    try:
        from eb.pbfm_data import spec_for

        plan = await build_data_plan(body, session)
        return {
            "hash": plan["hash"],
            "plan": plan,
            "spec": spec_for(plan, "ds_preview").model_dump(mode="json"),
        }
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/datasets/from-ufo/preview")
async def preview_ufo_data(body: UfoDataIn, session: SessionDep, principal: PrincipalDep):
    from eb.pbfm_data import spec_for
    from eb.pbfm_ufo_bridge import build_registered_ufo_plan

    operator(principal)
    try:
        plan = await build_registered_ufo_plan(body, session)
        return {
            "hash": plan["hash"],
            "plan": plan,
            "spec": spec_for(plan, "ds_preview").model_dump(mode="json"),
        }
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/datasets/from-ufo")
async def prepare_ufo_data(
    body: UfoDataIn,
    session: SessionDep,
    principal: PrincipalDep,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    from eb.pbfm_data import spec_for
    from eb.pbfm_ufo_bridge import build_registered_ufo_plan

    operator(principal)
    try:
        plan = await build_registered_ufo_plan(body, session)
        if body.preview_hash != plan["hash"]:
            raise HTTPException(409, "PBFM data preview is stale")
        previous = await session.scalar(select(Job).where(Job.idem_key == idempotency_key))
        if previous:
            did = previous.spec["params"]["dataset_id"]
            result = await submit(
                plan, spec_for(plan, did), body.preview_hash, idempotency_key, principal, session
            )
            return {**result, "dataset_id": did}
        dataset = Dataset(
            id=new_id("ds"),
            name=body.name,
            format="pbfm_rpo_motion",
            source_kind="pbfm_rpo_motion",
            source_uri="dataset-version:" + plan["parent_version_id"],
            status="registered",
            embodiment_env_id=ENV_ID,
            meta={"data_plan": plan, "usage": "train", "operation": OPERATION},
        )
        session.add(dataset)
        await session.flush()
        result = await submit(
            plan, spec_for(plan, dataset.id), body.preview_hash, idempotency_key, principal, session
        )
        dataset.meta = {**dataset.meta, "validation_job_id": result["id"]}
        await session.commit()
        return {**result, "dataset_id": dataset.id}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/datasets/prepare")
async def prepare_data(
    body: DataIn,
    session: SessionDep,
    principal: PrincipalDep,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    operator(principal)
    try:
        from eb.pbfm_data import spec_for

        plan = await build_data_plan(body, session)
        if body.preview_hash != plan["hash"]:
            raise HTTPException(409, "PBFM data preview is stale")
        previous = await session.scalar(select(Job).where(Job.idem_key == idempotency_key))
        if previous:
            result = await submit(
                plan,
                spec_for(plan, previous.spec["params"]["dataset_id"]),
                body.preview_hash,
                idempotency_key,
                principal,
                session,
            )
            return {**result, "dataset_id": previous.spec["params"]["dataset_id"]}
        dataset = Dataset(
            id=new_id("ds"),
            name=body.name,
            format="pbfm_rpo_motion",
            source_kind="pbfm_rpo_motion",
            source_uri="dataset-version:" + body.parent_version_id,
            status="registered",
            embodiment_env_id=ENV_ID,
            meta={"data_plan": plan, "usage": "train", "operation": OPERATION},
        )
        session.add(dataset)
        await session.flush()
        result = await submit(
            plan, spec_for(plan, dataset.id), body.preview_hash, idempotency_key, principal, session
        )
        dataset.meta = {**dataset.meta, "validation_job_id": result["id"]}
        await session.commit()
        return {**result, "dataset_id": dataset.id}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


async def model_action(mid, operation, body, session, principal, key=None):
    operator(principal)
    try:
        plan = await build_action_plan(mid, operation, body, session)
        spec = action_spec(plan)
        if key is None:
            return {"hash": plan["hash"], "plan": plan, "spec": spec.model_dump(mode="json")}
        if operation == EVAL_OPERATION:
            suite = await session.get(EvalSuite, SUITE_ID)
            if suite is None:
                session.add(
                    EvalSuite(
                        id=SUITE_ID,
                        name="PBFM RPO independent terrain validation v1",
                        embodiment_env_id=ENV_ID,
                        seeds=[71000, 71001, 71010, 71011],
                        frozen=True,
                        spec=suite_spec(),
                    )
                )
                await session.flush()
            elif not suite.frozen or suite.spec != suite_spec():
                raise ValueError("published PBFM Suite differs; derive a new suite")
        result = await submit(plan, spec, body.preview_hash, key, principal, session)
        await session.commit()
        return result
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/models/{mid}/export/preview")
async def preview_export(mid: str, body: ActionIn, session: SessionDep, principal: PrincipalDep):
    return await model_action(mid, EXPORT_OPERATION, body, session, principal)


@router.post("/models/{mid}/export")
async def start_export(
    mid: str,
    body: ActionIn,
    session: SessionDep,
    principal: PrincipalDep,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    return await model_action(mid, EXPORT_OPERATION, body, session, principal, idempotency_key)


@router.post("/models/{mid}/evals/preview")
async def preview_eval(mid: str, body: ActionIn, session: SessionDep, principal: PrincipalDep):
    return await model_action(mid, EVAL_OPERATION, body, session, principal)


@router.post("/models/{mid}/evals")
async def start_eval(
    mid: str,
    body: ActionIn,
    session: SessionDep,
    principal: PrincipalDep,
    idempotency_key: str = Header(alias="Idempotency-Key"),
):
    return await model_action(mid, EVAL_OPERATION, body, session, principal, idempotency_key)


@router.get("/models/{mid}")
async def model_status(mid: str, session: SessionDep, principal: PrincipalDep):
    model = await session.get(ModelVersion, mid)
    if not model or model.family_id != "mf_rpo_pbfm_depth":
        raise HTTPException(404)
    exports = (
        await session.scalars(
            select(Artifact)
            .where(Artifact.kind == "pbfm_policy", Artifact.meta["model_version_id"].astext == mid)
            .order_by(Artifact.created_at.desc())
        )
    ).all()
    reports = (await session.scalars(select(EvalReport).where(EvalReport.model_version_id == mid))).all()
    report_rows = []
    for report in reports:
        base = f"/api/v1/pbfm/runs/{report.run_id}/files/"
        root = get_settings().data_root / "runs" / report.run_id / "outputs"
        gates = report.gate_results or {}
        indexed = {
            row["path"]: row
            for row in [*gates.get("receipt", {}).get("files", []), *gates.get("partial_files", [])]
        }
        cases = []
        try:
            case_path = safe_file(root, "cases.json", maximum=16 * 1024**2)
            if sha256(case_path) != indexed.get("cases.json", {}).get("sha256"):
                raise ValueError("case index changed")
            cases = [
                {
                    "case_id": row["case_id"],
                    "depth_condition": row["condition"],
                    "success": row["success"],
                    "video_url": base + row["video"],
                    "depth_url": base + row["depth_video"],
                    "depth_data_url": base + row["depth"],
                    "trace_url": base + row["trace"],
                }
                for row in read_json(case_path)["cases"]
            ]
        except (ValueError, OSError, KeyError):
            partial_cases = (report.metrics or {}).get("cases")
            if isinstance(partial_cases, list):
                cases = [
                    {
                        "case_id": row["case_id"],
                        "depth_condition": row["condition"],
                        "success": row["success"],
                        "video_url": base + row["video"],
                        "depth_url": base + row["depth_video"],
                        "depth_data_url": base + row["depth"],
                        "trace_url": base + row["trace"],
                    }
                    for row in partial_cases
                ]
        report_rows.append(
            {
                "id": report.id,
                "run_id": report.run_id,
                "metrics": report.metrics,
                "gate_results": report.gate_results,
                "qualification": gates.get("qualification", "incomplete"),
                "cases": cases,
                "cases_url": base + "cases.json"
                if "cases.json" in indexed
                else base + "cases.jsonl"
                if "cases.jsonl" in indexed
                else None,
                "media_index_url": base + "media-index.json" if "media-index.json" in indexed else None,
                "plot_url": base + "analysis/terrain.svg" if "analysis/terrain.svg" in indexed else None,
            }
        )
    export_base = f"/api/v1/pbfm/runs/{exports[0].run_id}/files/policy/" if exports else None
    return {
        "model_version_id": mid,
        "schema_version": model.card.get("schema_version"),
        "export_ready": bool(exports),
        "export_artifact_id": exports[0].id if exports else None,
        "export_url": f"/api/v1/pbfm/runs/{exports[0].run_id}/files/policy.zip" if exports else None,
        "onnx_url": export_base + "policy.onnx" if export_base else None,
        "export_manifest_url": export_base + "manifest.json" if export_base else None,
        "eval_reports": report_rows,
    }


@router.api_route("/runs/{rid}/files/{path:path}", methods=["GET", "HEAD"])
async def output_file(rid: str, path: str, request: Request, session: SessionDep, principal: PrincipalDep):
    from eb.api.routes_ufo_native import stream_output
    from eb.pbfm_runtime_probe import TRAINING_MODES

    run = await session.get(Run, rid)
    job = await session.get(Job, run.job_id) if run else None
    if not job or job.spec.get("embodiment_env_id") != ENV_ID:
        raise HTTPException(404, "PBFM evidence is unavailable")
    root = get_settings().data_root / "runs" / rid / "outputs"
    try:
        from eb.pbfm_evaluation import verify_output_index

        report = await session.scalar(select(EvalReport).where(EvalReport.run_id == rid))
        gates = (report.gate_results or {}) if report else {}
        if run.status != "finished" or gates.get("partial_files"):
            if (
                (
                    run.status not in {"finished", "failed", "aborted", "cancelled", "canceled"}
                    or job.status not in {"failed", "cancelled", "canceled", "timeout", "infra_failed"}
                )
                or not report
                or gates.get("plan_hash") != job.spec.get("params", {}).get("pbfm_plan", {}).get("hash")
                or gates.get("qualification") != "incomplete"
            ):
                raise ValueError("partial media requires an incomplete failed/cancelled PBFM report")
            selected = [row for row in gates.get("partial_files", []) if row.get("path") == path]
            await verify_output_index(root, selected, {path})
            return await stream_output(request, rid, path)

        params = job.spec.get("params") or {}
        if params.get("template_id") == "rpo_pbfm_fb_depth_native" or (
            params.get("operation") == "probe_pbfm_runtime"
            and params.get("probe_mode") in TRAINING_MODES | {"segment112"}
        ):
            evidence = (run.metrics_summary or {}).get("training_evidence") or {}
            if evidence.get("run_id") != rid or not path.startswith("analysis/"):
                raise ValueError("not indexed training analysis")
            files = evidence.get("files") or []
        else:
            if job.spec["params"]["operation"] == "probe_pbfm_runtime":
                from eb.pbfm_training import training_fingerprint

                receipt = read_json(root / "probe-receipt.json")
                if (
                    receipt.get("run_id") != rid
                    or receipt.get("plan_hash")
                    != training_fingerprint({**job.spec["params"]["native_train_plan"], "run_id": rid})
                    or receipt != (run.metrics_summary or {}).get("runtime_probe")
                ):
                    raise ValueError("probe evidence differs from collected receipt")
                selected = [row for row in receipt["files"] if row.get("path") == path]
                await verify_output_index(root, selected, {path})
                return await stream_output(request, rid, path)
            receipt_name = {
                EXPORT_OPERATION: "export-receipt.json",
                OPERATION: "data-receipt.json",
                EVAL_OPERATION: "eval-receipt.json",
            }[job.spec["params"]["operation"]]
            receipt = read_json(root / receipt_name)
            if params["operation"] == EVAL_OPERATION:
                saved = gates.get("receipt")
            else:
                artifact_id = (run.metrics_summary or {}).get("artifact_id")
                artifact = await session.get(Artifact, artifact_id) if artifact_id else None
                kind = "pbfm_motion" if params["operation"] == OPERATION else "pbfm_policy"
                if not artifact or artifact.run_id != rid or artifact.kind != kind:
                    raise ValueError("collected PBFM artifact differs")
                saved = (artifact.meta or {}).get("receipt")
                if artifact.sha256 != fingerprint(saved):
                    raise ValueError("collected PBFM receipt digest differs")
            if (
                receipt.get("run_id") != rid
                or receipt.get("plan_hash") != job.spec["params"]["pbfm_plan"]["hash"]
                or receipt != saved
            ):
                raise ValueError("receipt differs")
            files = receipt["files"]
        selected = [row for row in files if row.get("path") == path]
        await verify_output_index(root, selected, {path})
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(404, "file lacks complete indexed integrity evidence") from exc
    return await stream_output(request, rid, path)


@router.post("/runtime-probe/preview")
async def preview_runtime_probe(body: ProbeIn, session: SessionDep, principal: PrincipalDep):
    from eb.pbfm_runtime_probe import build_spec, render

    operator(principal)
    try:
        spec = await build_spec(body, session)
        rendered = render(spec, "run_preview")
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "preview_hash": "sha256:" + spec.canonical_hash(),
        "diagnostic_only": True,
        "training_ready": False,
        "expected_failure": body.failure_rank is not None,
        "diagnostic_plan": spec.params["native_train_plan"],
        "image": rendered.image,
        "argv": rendered.command,
        "resources": spec.resources.model_dump(),
        "inputs": spec.inputs.model_dump(),
        "mounts": [f"{src} → {dst} {mode}" for src, dst, mode in rendered.mounts],
    }


@router.post("/runtime-probe")
async def submit_runtime_probe(
    body: ProbeIn,
    session: SessionDep,
    principal: PrincipalDep,
    idempotency_key: str = Header(min_length=1, max_length=120, alias="Idempotency-Key"),
):
    from eb.api.routes_train import _lock_native_train_submission
    from eb.pbfm_runtime_probe import build_spec

    operator(principal)
    try:
        spec = await build_spec(body, session)
        if body.preview_hash != "sha256:" + spec.canonical_hash():
            raise HTTPException(409, "P-BFM runtime probe preview changed; create a new preview")
        idem = f"pbfm-probe:{principal.id}:{idempotency_key}"
        await _lock_native_train_submission(session, idem)
        existing = await session.scalar(select(Job).where(Job.idem_key == idem))
        if existing is not None:
            from eb.domain.job import JobSpec

            if JobSpec.model_validate(existing.spec).canonical_hash() != spec.canonical_hash():
                raise HTTPException(409, "Idempotency-Key belongs to another runtime probe")
            return {
                "job_id": existing.id,
                "status": existing.status,
                "diagnostic_only": True,
                "expected_failure": body.failure_rank is not None,
            }
        job = await Orchestrator().submit(session, spec, principal.id, idem)
    except (ValueError, OSError, KeyError, PermissionError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "job_id": job.id,
        "status": job.status,
        "diagnostic_only": True,
        "expected_failure": body.failure_rank is not None,
    }
