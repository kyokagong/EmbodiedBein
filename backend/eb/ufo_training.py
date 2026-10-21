"""Operator plan for RLinf-hosted UFO; numerical defaults belong to UFO."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from eb.ufo_native import ENV_ID, TASK_ID, plan_hash, read_json

TEMPLATE = "rpo_ufo_fb_native"
IMAGE = "eb/ufo-native:rpo-train-v3"
RECONFIGURE_IMAGE = "eb/ufo-native:rpo-train-v5"
SCHEMA = "eb/ufo-native-train-plan/v1"
DATA_OPERATION = "prepare_ufo_native_motion"
MAX_UPDATES = 1_000_000


def training_fingerprint(identity: dict) -> str:
    return plan_hash(identity)


def segment_endpoint(
    target_updates: int, world_size: int, *, start_updates: int = 0, start_local_time: int = 0,
    num_envs: int = 1024, fresh_replay: bool = False,
) -> int:
    """Native checker boundaries with G=16; budget is cumulative global transitions."""
    if (
        type(target_updates) is not int
        or not 112 <= target_updates <= MAX_UPDATES
        or target_updates <= start_updates
        or target_updates % 16
        or start_updates % 16
        or world_size not in (1, 2, 3, 4)
        or num_envs not in (1024, 2048)
        or start_local_time % num_envs
    ):
        raise ValueError("agent 更新目标必须递增且按原生 16 次批次对齐")
    blocks = (target_updates - start_updates) // 16
    if fresh_replay or not start_local_time:
        first = start_local_time + (10240 // num_envs + 1) * num_envs
    else:
        first = max(start_local_time + num_envs, (10240 // num_envs + 1) * num_envs)
    return (first + blocks * num_envs) * world_size


def native_arguments(plan: dict) -> list[str]:
    knobs = plan["identity"]
    endpoint = segment_endpoint(
        plan["target_updates"],
        plan["world_size"],
        start_updates=plan.get("start_updates", 0),
        start_local_time=plan.get("start_local_time", 0),
        num_envs=knobs.get("num_envs_per_rank", 1024),
        fresh_replay=(plan.get("resume") or {}).get("replay_policy") == "fresh",
    )
    pairs = {
        "robot-config": "/sources/ufo/configs/robots/rpo_23dof.yaml",
        "data-path": "/training-data/train.pkl",
        "work-dir": "/run/eb-output/outputs/native",
        "num-envs": knobs.get("num_envs_per_rank", 1024),
        "num-env-steps": endpoint,
        "buffer-size": knobs["buffer_size"],
        "seed": knobs["seed"],
        "checkpoint-every-steps": endpoint,
        "buffer-every-steps": endpoint,
    }
    return [str(item) for key, value in pairs.items() for item in (f"--{key}", value)]


def require_native_dataset(row) -> dict:
    report = (row.summary_stats or {}).get("validation") or {}
    root = Path(row.dest_path)
    manifest = root / "manifest.json"
    if (
        row.status != "validated"
        or row.embodiment_env_id != ENV_ID
        or (row.meta or {}).get("usage", "train") != "train"
        or report.get("scope") != "ufo_native_motion_v1"
        or report.get("training_ready") is not True
        or report.get("storage_ref") != str(root)
        or manifest.is_symlink()
        or not manifest.is_file()
        or root.is_symlink()
        or manifest.stat().st_size > 4 * 1024 * 1024
    ):
        raise ValueError("UFO requires a frozen training-ready RPO native motion DatasetVersion")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != report.get("manifest_sha256"):
        raise ValueError("UFO data manifest changed after validation")
    value = read_json(manifest)
    if value.get("train_file") != "train.pkl" or not (root / "train.pkl").is_file():
        raise ValueError("UFO training split is incomplete")
    return report


def validate_train_job(spec) -> dict:
    plan = spec.params.get("native_train_plan")
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA:
        raise ValueError("Native training plan is missing or unsupported")
    if (
        spec.kind != "train.rl"
        or spec.embodiment_env_id != ENV_ID
        or spec.task_id != TASK_ID
        or spec.params.get("template_id") != TEMPLATE
        or spec.params.get("stack") != "rlinf"
        or spec.overrides
        or spec.inputs.base_model
        or len(spec.inputs.datasets) != 1
        or spec.runtime != {"network": "none", "hostname": "localhost"}
        or spec.fanout
        or spec.resources.devices
        or spec.resources.gpu_mode != "exclusive"
        or spec.outputs
        or spec.policy.max_retries != 0
        or not 0 < spec.policy.timeout_h <= 6
    ):
        raise ValueError("Native training inputs, execution boundary or topology differs")
    identity = plan.get("identity", {})
    if plan.get("cuda_allocator") not in (None, "expandable_segments"):
        raise ValueError("Unsupported native CUDA allocator")
    goal = plan.get("experiment_target_updates")
    if goal is not None and (type(goal) is not int or not plan["target_updates"] <= goal <= 1000000
                             or goal % 16):
        raise ValueError("Invalid native experiment target")
    if (
        not re.fullmatch(r"sha256:[a-f0-9]{64}", spec.image or "")
        or plan.get("image_digest") != spec.image
        or identity.get("image_digest") != spec.image
        or plan.get("world_size") != spec.resources.gpus
        or identity.get("world_size") != spec.resources.gpus
        or type(identity.get("buffer_size")) is not int
        or not 32768 <= identity["buffer_size"] <= 1048576
        or identity["buffer_size"] % 1024
        or type(identity.get("num_envs_per_rank", 1024)) is not int
        or identity.get("num_envs_per_rank", 1024) not in (1024, 2048)
        or identity["buffer_size"] % identity.get("num_envs_per_rank", 1024)
        or type(identity.get("seed")) is not int
        or not 0 <= identity["seed"] <= 1000000
        or plan.get("fingerprint") != training_fingerprint(identity)
        or plan.get("native_args") != native_arguments(plan)
        or plan.get("log_every_steps") != 1024 * spec.resources.gpus
    ):
        raise ValueError("Native training plan identity or native arguments differs")
    if (
        spec.resources.cpus != 8 * spec.resources.gpus
        or spec.resources.memory_gb != 16 + 16 * spec.resources.gpus
        or spec.resources.shm_gb != 8
    ):
        raise ValueError("Native training CPU/RAM/shm profile differs")
    expected_resume = bool(plan.get("resume"))
    replay_policy = (plan.get("resume") or {}).get("replay_policy", "restore")
    if replay_policy not in ("restore", "fresh") or replay_policy == "fresh" and not expected_resume:
        raise ValueError("Invalid native Replay restoration policy")
    if bool(spec.inputs.resume_from) != expected_resume:
        raise ValueError("Native resume binding differs")
    if not expected_resume and (plan.get("start_updates", 0) != 0 or plan.get("start_local_time", 0) != 0):
        raise ValueError("Fresh native training must start from zero counters")
    refs = spec.code_refs
    ufo = spec.params.get("ufo_source_ref") or {}
    assets = spec.params.get("asset_source_ref") or {}
    if (
        not refs
        or not refs.stack
        or refs.embodiment is not None
        or refs.stack.kind != "content_snapshot"
        or not refs.stack.commit
        or not ufo.get("commit")
        or not assets.get("commit")
    ):
        raise ValueError("Native training requires both frozen sources and assets")
    for name, commit in (("ufo", ufo["commit"]), ("rlinf", refs.stack.commit)):
        expected = {
            "path": f"/sources/{name}",
            "snapshot_hash": commit,
            "manifest_path": f"/sources/{name}-manifest.json",
        }
        if plan.get("source_refs", {}).get(name) != expected or identity.get(f"{name}_snapshot") != commit:
            raise ValueError("Native training source identity differs")
    if identity.get("asset_snapshot") != assets["commit"]:
        raise ValueError("Native robot asset identity differs")
    ds = spec.inputs.datasets[0]
    if identity.get("dataset_id") != ds.get("id") or identity.get("dataset_manifest") != (
        ds.get("validation") or {}
    ).get("manifest_sha256"):
        raise ValueError("Native dataset identity differs")
    return plan


def render_training(spec, run_id):
    from eb.adapters.base import RenderedJob
    from eb.settings import get_settings

    plan = {**validate_train_job(spec), "run_id": run_id}
    plan["hash"] = plan_hash({k: v for k, v in plan.items() if k != "hash"})
    settings = get_settings()
    root = settings.data_root / "runs" / run_id
    stack = spec.code_refs.stack
    ufo = spec.params["ufo_source_ref"]
    assets = spec.params["asset_source_ref"]
    mounts = [(str(root), "/run/eb-output", "rw"), (str(root / "ufo-train-plan.json"), "/plan.json", "ro")]
    for name, source in (("ufo", ufo), ("rlinf", stack.model_dump())):
        path = Path(source["path"])
        mounts.extend(
            (
                (str(path), f"/sources/{name}", "ro"),
                (str(path.parent / "manifest.json"), f"/sources/{name}-manifest.json", "ro"),
            )
        )
    mesh_suffix = Path("rpo_description/meshes")
    asset_destination = str(settings.assets_root / "roboparty/rpo-v2" / mesh_suffix)
    mounts.append((str(Path(assets["path"]) / mesh_suffix), asset_destination, "ro"))
    data = Path(spec.inputs.datasets[0]["path"])
    mounts.extend(
        (
            (str(data / "train.pkl"), "/training-data/train.pkl", "ro"),
            (str(data / "manifest.json"), "/training-data/manifest.json", "ro"),
        )
    )
    if plan.get("resume"):
        mounts.append((plan["resume"]["host_path"], "/resume-native", "ro"))
    if run_id not in ("run_preview", "run_validate"):
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("Native run path contains a symlink")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "ufo-train-plan.json"
        encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
        if path.is_symlink() or path.exists() and path.read_text() != encoded:
            raise ValueError("Native frozen Run plan differs")
        path.write_text(encoded)
    return RenderedJob(
        image=spec.image,
        command=["python", "-m", "eb.runners.rlinf_ufo_entry", "--plan", "/plan.json"],
        env={
            **({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
               if plan.get("cuda_allocator") == "expandable_segments" else {}),
            "EB_RUN_ID": run_id,
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "OMP_NUM_THREADS": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        },
        mounts=mounts,
        writable_paths=["outputs", "runtime", "inputs"],
        workdir="/run/eb-output",
        outputs=["outputs"],
        metrics_glob="outputs/native/train_log.txt",
        gpus=spec.resources.gpus,
        shm_gb=spec.resources.shm_gb,
    )
