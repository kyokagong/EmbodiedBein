"""Fixed A1 reference candidate protocol; computation runs only in a CPU Job.

This operation derives a new Dataset. It never mutates the 58-clip experiment
or makes a policy-quality claim from a reference PD diagnostic.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

OPERATION = "prepare_ufo_rpo_grounded_a1_v1"
IMAGE = "eb/ufo-native:rpo-ground-a1-v1"
BASE_IMAGE = "sha256:e122bb0c96f28c3d2006dd0c466bf392f2f4dc8f0e09fba0e5f2e36f6cefdfe6"
PLAN_SCHEMA = "eb/ufo-grounded-a1-plan/v1"
RECEIPT_SCHEMA = "eb/ufo-grounded-a1-receipt/v1"
PARENT_ID = "ds_6541f726d7804007a93a"
PARENT_MANIFEST = "242fc13c3dd2f24b8d6aafcb688c35e010745eaba27efbe68f195c2185843f1b"
TRAIN_SHA = "73174cdba1f8c99b11646ae2dee28a0a1ae154895ce9f6f4f10d9e599fa82dee"
VALIDATION_SHA = "29d63fc411480a0be41c36c676006aa3eca6280ffc08a89032686a9dbe8ebc16"
A1_PATH = "robolab/data/motions/rpo_lab/A1-_Stand_stageii.pkl"
A1_SHA = "189eabf1622f468ed1264c938e8941b2ae181eb295c62e73722e70db40039c5f"
ROOT_Z_SHIFT = 0.04917634611991695
SOURCE_KEY = "official_A1-_Stand_stageii"
PROTOCOL = {
    "source_path": A1_PATH,
    "source_sha256": A1_SHA,
    "source_frames": 457,
    "source_fps": 120.0,
    "duration_s": 3.8,
    "root_z_shift_m": ROOT_Z_SHIFT,
    "grounding_method": "constant median sole-depth correction from all 457 original frames",
    "source_arrays_unchanged": False,
    "preserved": ["root_xy", "root_quaternion", "joint_trajectory", "fps", "all_original_frames"],
    "seed": 71000,
    "physics_dt": 0.0025,
    "decimation": 8,
    "control_hz": 50,
    "steps": 190,
    "video_fps": 25,
    "root_initializations": 1,
    "initial_velocity": "native mj_differentiatePos on the first source interval",
    "target_interpolation": "linear original 120Hz joints at native 50Hz control ticks",
    "plane_friction": [1.0, 0.005, 0.0001],
    "domain_randomization_applied": False,
    "observation_noise_applied": False,
    "reference_pd_only": True,
    "policy_evaluation": False,
}


def quality_errors(metrics: dict) -> list[str]:
    """Fail closed on missing/nonfinite evidence; no failed frames are removed."""
    errors = []
    thresholds = {
        "min_height_m": (0.5, "min"),
        "max_tilt_rad": (0.8, "max"),
        "drift_m": (0.5, "max"),
        "floor_max_m": (0.005, "max"),
        "self_max_m": (0.02, "max"),
        "grounding_delta_error_m": (1e-6, "max"),
        "source_joint_excess_rad": (1e-5, "max"),
        "source_velocity_excess_rad_s": (0.001, "max"),
        "max_joint_excess_rad": (0.001, "max"),
        "max_velocity_excess_rad_s": (0.001, "max"),
        "max_torque_excess_nm": (0.001, "max"),
    }
    for name, (limit, direction) in thresholds.items():
        value = metrics.get(name)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (direction == "max" and value < 0)
            or (value < limit if direction == "min" else value > limit)
        ):
            errors.append(f"reference diagnostic failed {name}")
    if (
        metrics.get("finite") is not True
        or metrics.get("completed_steps") != 190
        or metrics.get("expected_steps") != 190
    ):
        errors.append("reference diagnostic is nonfinite or did not complete all 3.8 seconds")
    physics_seconds = metrics.get("physics_seconds")
    if (
        type(metrics.get("mujoco_warning_count")) is not int
        or metrics.get("mujoco_warning_count") != 0
        or metrics.get("simulation_time_monotonic") is not True
        or type(physics_seconds) not in (float, int)
        or not math.isfinite(physics_seconds)
        or not math.isclose(physics_seconds, 3.8, abs_tol=1e-8, rel_tol=0)
    ):
        errors.append("reference diagnostic physics warned, reset or changed the physical horizon")
    if type(metrics.get("ground_contact_frames")) is not int or metrics["ground_contact_frames"] < 1:
        errors.append("reference diagnostic has no observed foot-plane contact")
    video = metrics.get("video") or {}
    if (
        video.get("decodable") is not True
        or video.get("frames") != 96
        or video.get("fps") != 25.0
        or video.get("seconds") != 3.84
    ):
        errors.append("full-horizon diagnostic video is incomplete or undecodable")
    return errors


def spec_for(plan: dict, dataset_id: str, name: str = ""):
    from eb.domain.job import JobSpec
    from eb.ufo_native import ENV_ID

    return JobSpec(
        kind="data.convert",
        embodiment_env_id=ENV_ID,
        name=name,
        image=plan["image_digest"],
        params={"operation": OPERATION, "dataset_id": dataset_id, "grounded_plan": plan},
        resources={"gpus": 0, "cpus": 4, "memory_gb": 16, "shm_gb": 2},
        runtime={"network": "none"},
        policy={"timeout_h": 0.5, "max_retries": 0, "retry_on": []},
    )


def preview_hash(spec) -> str:
    from eb.ufo_native import plan_hash

    value = spec.model_dump(mode="json", exclude={"name"})
    value["params"]["dataset_id"] = "ds_preview"
    return plan_hash(value)


def snapshot_root(ref: dict, settings) -> Path:
    commit = ref.get("commit", "")
    root = settings.data_root / "snapshots" / commit.removeprefix("sha256:") / "code"
    if (
        ref.get("kind") != "content_snapshot"
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", commit)
        or ref.get("path") != str(root)
        or not root.is_dir()
        or any(p.is_symlink() for p in (root, *root.parents))
    ):
        raise ValueError("A1 candidate requires owned frozen native sources")
    return root


async def build_plan(session, parent_id: str) -> dict:
    import asyncio

    from sqlalchemy import select

    from eb.api.routes_ufo_native import scanned_file
    from eb.settings import get_settings
    from eb.store.ebd import EbdClient
    from eb.store.models import Dataset, DatasetVersion, Job, Run
    from eb.ufo_native import plan_hash, read_json, safe_file
    from eb.ufo_training import DATA_OPERATION, require_native_dataset

    if parent_id != PARENT_ID:
        raise ValueError("This bounded protocol accepts only the frozen 58-clip reference Dataset")
    parent = await session.get(Dataset, parent_id)
    if parent is None or require_native_dataset(parent)["manifest_sha256"] != PARENT_MANIFEST:
        raise ValueError("The approved parent Dataset manifest differs")
    meta = parent.meta or {}
    producer = await session.get(Job, meta.get("validation_job_id"))
    run = await session.get(Run, meta.get("validation_run_id"))
    version = await session.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == parent_id, DatasetVersion.version == "native-v1"
        )
    )
    if (
        not producer
        or not run
        or not version
        or producer.kind != "data.convert"
        or producer.status != "succeeded"
        or run.status != "finished"
        or run.job_id != producer.id
        or version.content_hash != "sha256:" + PARENT_MANIFEST
        or (version.stats or {}).get("producer_run") != run.id
        or (producer.spec.get("params") or {}).get("operation") != DATA_OPERATION
    ):
        raise ValueError("The parent lacks immutable successful native preparation lineage")
    settings = get_settings()
    outputs = settings.data_root / "runs" / run.id / "outputs"
    if parent.dest_path != str(outputs / "data"):
        raise ValueError("Parent storage is outside its successful producer Run")
    refs = {name: producer.spec["params"][name + "_source_ref"] for name in ("ufo", "asset")}
    roots = {name: snapshot_root(ref, settings) for name, ref in refs.items()}
    originals = [
        ("train.pkl", "data/train.pkl"),
        ("manifest.json", "data/manifest.json"),
        ("validation.pkl", "holdout/validation.pkl"),
        ("validation-manifest.json", "holdout/manifest.json"),
        *[
            (name, "provenance/" + name)
            for name in ("qualification.json", "import_report.json", "mask_feasibility.json")
        ],
    ]
    paths = [(alias, safe_file(outputs, path)) for alias, path in originals]
    paths.extend([("a1-source", safe_file(roots["asset"], A1_PATH))])
    files = await asyncio.gather(*(scanned_file(path) for _, path in paths))
    indexed = {alias: {**item, "path": alias} for (alias, _), item in zip(paths, files)}
    for alias, expected in (
        ("train.pkl", TRAIN_SHA),
        ("validation.pkl", VALIDATION_SHA),
        ("manifest.json", PARENT_MANIFEST),
        ("a1-source", A1_SHA),
    ):
        if indexed[alias]["sha256"] != expected:
            raise ValueError(f"Approved parent or A1 bytes changed: {alias}")
    manifest = read_json(outputs / "data/manifest.json", limit=4 * 1024**2)
    holdout = read_json(outputs / "holdout/manifest.json", limit=4 * 1024**2)
    if manifest.get("episodes_count") != 58 or holdout.get("episodes_count") != 15:
        raise ValueError("Approved train/validation clip counts differ")
    image = await EbdClient().image_info(IMAGE)
    if image.get("ok") is not True or not re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("digest", "")):
        raise ValueError("The separate A1 data task image is unavailable")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "operation": OPERATION,
        "parent_dataset_id": parent.id,
        "parent_version_id": version.id,
        "parent_run_id": run.id,
        "parent_manifest_sha256": PARENT_MANIFEST,
        "robot_config_sha256": manifest["robot_config_sha256"],
        "robot_xml_sha256": manifest["robot_xml_sha256"],
        "image_digest": image["digest"],
        "base_training_image": BASE_IMAGE,
        "ufo_source_ref": refs["ufo"],
        "asset_source_ref": refs["asset"],
        "asset_mount": str(settings.assets_root / "roboparty/rpo-v2"),
        "parent_files": [indexed[alias] for alias, _ in originals],
        "source": indexed["a1-source"],
        "protocol": PROTOCOL,
    }
    plan["hash"] = plan_hash(plan)
    return plan


async def validate_submission(session, spec) -> None:
    from eb.store.models import Dataset

    row = await session.get(Dataset, spec.params.get("dataset_id"))
    proposed = spec.params.get("grounded_plan") or {}
    if (
        not row
        or row.source_kind != "ufo_native_motion"
        or (row.meta or {}).get("operation") != OPERATION
        or (row.meta or {}).get("parent_dataset_id") != proposed.get("parent_dataset_id")
        or (row.meta or {}).get("plan_hash") != proposed.get("hash")
        or (row.meta or {}).get("validation_job_id") is not None
        or row.source_uri != "ufo-grounded-a1:" + proposed.get("parent_dataset_id", "")
    ):
        raise ValueError("A1 candidate must be a separately registered derived Dataset")
    plan = await build_plan(session, proposed["parent_dataset_id"])
    expected = spec_for(plan, row.id)
    if spec.model_dump(mode="json", exclude={"name"}) != expected.model_dump(mode="json", exclude={"name"}):
        raise ValueError("A1 data Job differs from the fixed source, protocol or CPU runtime")


def render(spec, run_id):
    import json

    from eb.adapters.base import RenderedJob
    from eb.settings import get_settings

    settings = get_settings()
    plan = spec.params["grounded_plan"]
    output = settings.data_root / "runs" / run_id / "outputs"
    mounts = [(item["source"], "/parent/" + item["path"], "ro") for item in plan["parent_files"]]
    for name, target in (("ufo", "/sources/ufo"), ("asset", str(settings.assets_root / "roboparty/rpo-v2"))):
        root = Path(plan[name + "_source_ref"]["path"])
        mounts += [
            (str(root), target, "ro"),
            (str(root.parent / "manifest.json"), f"/sources/{name}-manifest.json", "ro"),
        ]
    mounts.append((str(output), "/outputs", "rw"))
    return RenderedJob(
        image=spec.image,
        command=[
            "python",
            "-m",
            "eb.runners.ufo_grounded_reference",
            "--plan-json",
            json.dumps(plan, sort_keys=True),
            "--dataset-id",
            spec.params["dataset_id"],
            "--storage-ref",
            str(output / "data"),
        ],
        env={
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "EB_RUN_ID": run_id,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        },
        mounts=mounts,
        writable_paths=[],
        workdir="/outputs",
        gpus=0,
        shm_gb=2,
    )


async def verified_report(outputs, plan, run_id, dataset_id):
    from eb.orchestrator.ufo_native import verify_index
    from eb.ufo_native import finite_tree, read_json, safe_file

    receipt = read_json(safe_file(outputs, "candidate-receipt.json"), limit=4 * 1024**2)
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("run_id") != run_id
        or receipt.get("plan_hash") != plan["hash"]
        or receipt.get("dataset_id") != dataset_id
        or receipt.get("image_digest") != plan["image_digest"]
        or receipt.get("operation") != OPERATION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("reference_pd_only") is not True
        or not finite_tree(receipt)
    ):
        raise ValueError("A1 diagnostic receipt identity differs")
    files = receipt.get("files") or []
    if (
        not files
        or len(files) > 200
        or any(type(f.get("bytes")) is not int or f["bytes"] <= 0 for f in files)
        or sum(f["bytes"] for f in files) > 512 * 1024**2
    ):
        raise ValueError("A1 diagnostic output index exceeds the bounded file/byte budget")
    paths = {f["path"] for f in files}
    required = {
        "dataset-index.json",
        "reference/trace.npz",
        "reference/replay.mp4",
        "reference/quality.json",
        "robot-state/A1-grounded.npz",
        "native-motion.yaml",
        "native-config-receipt.json",
        "diagnostic-input/codec.json",
        "diagnostic-input/physics.json",
        "diagnostic-input/effective_config.json",
        "diagnostic-input/assets/rpo.xml",
        "geometry.npz",
        "provenance/qualification.json",
        "provenance/import_report.json",
        "provenance/mask_feasibility.json",
    }
    if not required <= paths:
        raise ValueError("A1 diagnostic lacks complete raw reference/media evidence")
    await verify_index(outputs, files)
    native = read_json(safe_file(outputs, "native-config-receipt.json"))
    mapping = native.get("actuator_mapping") or {}
    if (
        not finite_tree(native)
        or mapping.get("verified") is not True
        or len(mapping.get("joints", [])) != 23
        or mapping.get("dc_motor_grid_samples") != 1127
        or type(mapping.get("max_clipping_error_nm")) not in (float, int)
        or not 0 <= mapping["max_clipping_error_nm"] <= 1e-10
        or native.get("robot_source_sha256") != plan["robot_config_sha256"]
        or native.get("native_robot_xml_sha256") != plan["robot_xml_sha256"]
    ):
        raise ValueError("A1 diagnostic native robot/physics mapping evidence differs")
    quality = read_json(safe_file(outputs, "reference/quality.json"))
    if not finite_tree(quality):
        raise ValueError("A1 diagnostic quality metadata is nonfinite")
    errors = quality_errors(quality.get("metrics") or {})
    if quality.get("errors") != errors or receipt.get("training_ready") is not (not errors):
        raise ValueError("A1 readiness differs from physical/media evidence")
    report = read_json(safe_file(outputs, "dataset-index.json"), limit=4 * 1024**2)
    validation = report.get("validation") or {}
    if (
        report.get("dataset_id") != dataset_id
        or report.get("format") != "ufo_native_motion"
        or report.get("storage_ref") != str(outputs / "data")
        or validation.get("scope") != "ufo_native_motion_v1"
        or validation.get("storage_ref") != str(outputs / "data")
        or not finite_tree(report)
        or validation.get("training_ready") is not (not errors)
        or validation.get("errors") != errors
        or validation.get("status") != ("invalid" if errors else "validated")
    ):
        raise ValueError("A1 Dataset report differs from its diagnostic qualification")
    if (
        not errors
        and not {"data/train.pkl", "data/manifest.json", "holdout/validation.pkl", "holdout/manifest.json"}
        <= paths
    ):
        raise ValueError("A1 candidate cannot publish without both separately frozen splits")
    if not errors:
        indexed = {item["path"]: item for item in files}
        train = read_json(safe_file(outputs, "data/manifest.json"), limit=4 * 1024**2)
        holdout = read_json(safe_file(outputs, "holdout/manifest.json"), limit=4 * 1024**2)
        if (
            train.get("schema_version") != "eb/ufo-native-motion/v1"
            or train.get("dataset_id") != dataset_id
            or train.get("split") != "train"
            or train.get("episodes_count") != 59
            or holdout.get("episodes_count") != 15
            or train.get("parent_version_id") != plan["parent_version_id"]
            or train.get("source_transform") != PROTOCOL
            or validation.get("manifest_sha256") != indexed["data/manifest.json"]["sha256"]
            or train.get("file_sha256") != indexed["data/train.pkl"]["sha256"]
            or indexed["holdout/validation.pkl"]["sha256"] != VALIDATION_SHA
            or holdout.get("file_sha256") != VALIDATION_SHA
            or train.get("native_loader", {}).get("all_motions_loaded") is not True
            or holdout.get("native_loader", {}).get("all_motions_loaded") is not True
        ):
            raise ValueError("A1 candidate split hashes, native loader or derivation differ")
    return receipt, report


async def collect_candidate(session, job, run, outputs):
    from sqlalchemy import select

    from eb.ids import new_id
    from eb.store.models import Artifact, LineageEdge
    from eb.ufo_native import plan_hash

    plan = job.spec["params"]["grounded_plan"]
    receipt, _ = await verified_report(outputs, plan, run.id, job.spec["params"]["dataset_id"])
    digest = plan_hash(receipt)
    artifact = await session.scalar(
        select(Artifact).where(
            Artifact.run_id == run.id, Artifact.kind == "ufo_grounded_reference_diagnostic"
        )
    )
    if artifact is None:
        artifact = Artifact(
            id=new_id("art"),
            kind="ufo_grounded_reference_diagnostic",
            sha256=digest,
            size_bytes=sum(f["bytes"] for f in receipt["files"]),
            storage=str(outputs),
            run_id=run.id,
            pinned=True,
            meta={"receipt": receipt, "reference_pd_only": True},
        )
        session.add(artifact)
        await session.flush()
        session.add(
            LineageEdge(
                src_type="run",
                src_id=run.id,
                dst_type="artifact",
                dst_id=artifact.id,
                relation="produced",
                run_id=run.id,
            )
        )
        session.add(
            LineageEdge(
                src_type="dataset_version",
                src_id=plan["parent_version_id"],
                dst_type="artifact",
                dst_id=artifact.id,
                relation="derived",
                run_id=run.id,
            )
        )
    elif artifact.sha256 != digest:
        raise ValueError("Immutable A1 diagnostic changed")
    run.metrics_summary = {
        **(run.metrics_summary or {}),
        "grounded_reference": {
            "artifact_id": artifact.id,
            "training_ready": receipt["training_ready"],
            "reference_pd_only": True,
            "receipt": "candidate-receipt.json",
        },
    }
