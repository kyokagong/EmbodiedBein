"""Frozen control-plane plan for P-BFM's native FB-depth recipe."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from eb.runners.pbfm_source import CAMERA_CONTRACT, TERRAIN_CONTRACT

TEMPLATE = "rpo_pbfm_fb_depth_native"
ENV_ID = "ee_rpo_v2_pbfm_depth_23dof_v1"
TASK_ID = "tsk_rpo_pbfm_terrain_locomotion_v1"
IMAGE = "eb/rlinf-pbfm:rpo-depth-v1"
SCHEMA = "eb/pbfm-native-train-plan/v1"
DATA_OPERATION = "prepare_pbfm_rpo_motion"
IDENTITY_FIELDS = {
    "image_digest",
    "world_size",
    "num_envs_per_rank",
    "buffer_size",
    "seed",
    "dataset_id",
    "dataset_manifest",
    "contract_hash",
    "environment_profile",
    "pbfm_snapshot",
    "rlinf_snapshot",
    "asset_snapshot",
    "camera_contract",
    "terrain_contract",
}


def training_fingerprint(identity: dict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def segment_endpoint(
    target_updates: int,
    world_size: int,
    *,
    start_updates: int = 0,
    start_local_time: int = 0,
    num_envs: int = 1024,
) -> int:
    """P-BFM's local update checker adds a collection round on every resume."""
    if (
        type(target_updates) is not int
        or not 112 <= target_updates <= 100000
        or target_updates % 16
        or type(start_updates) is not int
        or start_updates % 16
        or not 0 <= start_updates < target_updates
        or type(world_size) is not int
        or world_size not in (1, 2, 4)
        or type(num_envs) is not int
        or num_envs not in (1024, 2048)
        or type(start_local_time) is not int
        or start_local_time < 0
        or start_local_time % num_envs
        or bool(start_updates) != bool(start_local_time)
    ):
        raise ValueError("P-BFM requires increasing native 16-update boundaries, E=1024/2048 and W=1/2/4")
    first = max(start_local_time + num_envs, (10240 // num_envs + 1) * num_envs)
    return (first + ((target_updates - start_updates) // 16) * num_envs) * world_size


def native_arguments(plan: dict) -> list[str]:
    identity = plan["identity"]
    endpoint = segment_endpoint(
        plan["target_updates"],
        plan["world_size"],
        start_updates=plan.get("start_updates", 0),
        start_local_time=plan.get("start_local_time", 0),
        num_envs=identity["num_envs_per_rank"],
    )
    pairs = {
        "robot-config": "/run/eb-output/runtime/pbfm/pbfm/configs/robots/rpo_23dof.yaml",
        "data-path": "/training-data/train.pkl",
        "work-dir": "/run/eb-output/runtime/native-staging",
        "num-envs": identity["num_envs_per_rank"],
        "num-env-steps": endpoint,
        "buffer-size": identity["buffer_size"],
        "seed": identity["seed"],
        "checkpoint-every-steps": (
            plan["long_exploration"]["checkpoint_interval_global_steps"]
            if plan.get("long_exploration") else endpoint
        ),
    }
    return [str(item) for key, value in pairs.items() for item in ("--" + key, value)]


def validate_frozen_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA:
        raise ValueError("P-BFM training plan is missing or unsupported")
    identity = plan.get("identity")
    if not isinstance(identity, dict) or not IDENTITY_FIELDS <= identity.keys():
        raise ValueError("P-BFM frozen training identity is incomplete")
    world = plan.get("world_size")
    segment_endpoint(
        plan.get("target_updates"),
        world,
        start_updates=plan.get("start_updates", 0),
        start_local_time=plan.get("start_local_time", 0),
        num_envs=identity["num_envs_per_rank"],
    )
    if (
        not re.fullmatch(r"sha256:[a-f0-9]{64}", str(plan.get("image_digest", "")))
        or identity["image_digest"] != plan["image_digest"]
        or identity["world_size"] != world
        or type(identity["buffer_size"]) is not int
        or not 32768 <= identity["buffer_size"] <= 5120000
        or identity["buffer_size"] % identity["num_envs_per_rank"]
        or (
            identity["num_envs_per_rank"] == 2048
            and identity["buffer_size"] // 2048 <= max(CAMERA_CONTRACT["sampled_ages"])
        )
        or type(identity["seed"]) is not int
        or not 0 <= identity["seed"] <= 1000000
        or identity.get("prior_plane_envs", 0) != 0
        or plan.get("prior_plane_envs", 0) != 0
        or identity["camera_contract"] != CAMERA_CONTRACT
        or identity["terrain_contract"] != TERRAIN_CONTRACT
        or plan.get("fingerprint") != training_fingerprint(identity)
        or plan.get("native_args") != native_arguments(plan)
        or plan.get("log_every_steps") != 1024 * world
        or plan.get("full_state_finite_required") is not True
    ):
        raise ValueError("P-BFM identity, QV topology, frozen configuration or arguments differ")
    if not re.fullmatch(r"[a-f0-9]{64}", str(identity["dataset_manifest"])):
        raise ValueError("P-BFM dataset manifest digest is missing")
    if not re.fullmatch(r"[a-f0-9]{64}", str(identity["contract_hash"])):
        raise ValueError("P-BFM contract hash is missing")
    for key in ("pbfm_snapshot", "rlinf_snapshot", "asset_snapshot"):
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(identity[key])):
            raise ValueError("P-BFM frozen " + key + " is missing")
    if not identity["dataset_id"] or not isinstance(identity["environment_profile"], dict):
        raise ValueError("P-BFM dataset/profile identity is incomplete")
    expected = {
        name: {
            "path": f"/sources/{name}",
            "snapshot_hash": identity[name + "_snapshot"],
            "manifest_path": f"/sources/{name}-manifest.json",
        }
        for name in ("pbfm", "rlinf")
    }
    expected["asset"] = {
        "path": "/robot-assets",
        "snapshot_hash": identity["asset_snapshot"],
        "manifest_path": "/sources/asset-manifest.json",
    }
    if plan.get("source_refs") != expected:
        raise ValueError("P-BFM frozen source references differ")
    resume = plan.get("resume")
    if not resume:
        if plan.get("start_updates", 0) or plan.get("start_local_time", 0):
            raise ValueError("P-BFM scratch must begin at zero")
    elif (
        not isinstance(resume, dict)
        or resume.get("mode", "pbfm_native_state") != "pbfm_native_state"
        or resume.get("replay_policy", "restore") != "restore"
        or plan.get("start_updates", 0) == 0
    ):
        raise ValueError("P-BFM resume requires complete native state and restored Replay")
    long = plan.get("long_exploration")
    if long:
        from eb.pbfm_longrun import validate_long_plan

        validate_long_plan(plan)
    elif plan["target_updates"] > 10000:
        raise ValueError("P-BFM targets above10000 require the explicit long exploration")
    for key, maximum in (("communication_timeout_seconds", 1800),
                         ("segment_timeout_seconds", 108000 if long else 21600)):
        value = plan.get(key, 300 if key.startswith("communication") else 21600)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("P-BFM timeout is invalid")
    return plan


def validate_train_job(spec) -> dict:
    plan = validate_frozen_plan(spec.params.get("native_train_plan"))
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
        or not 0 < spec.policy.timeout_h <= (30 if plan.get("long_exploration") else 6)
        or spec.image != plan["image_digest"]
        or spec.resources.gpus != plan["world_size"]
        or spec.resources.cpus != 8 * plan["world_size"]
        or spec.resources.memory_gb != 16 + 16 * plan["world_size"]
        or spec.resources.shm_gb != 8
        or bool(spec.inputs.resume_from) != bool(plan.get("resume"))
    ):
        raise ValueError("P-BFM Job inputs, execution boundary or resource profile differs")
    refs = spec.code_refs
    source = spec.params.get("pbfm_source_ref") or {}
    assets = spec.params.get("asset_source_ref") or {}
    identity = plan["identity"]
    if (
        not refs
        or not refs.stack
        or refs.stack.kind != "content_snapshot"
        or refs.embodiment is not None
        or refs.stack.commit != identity["rlinf_snapshot"]
        or source.get("commit") != identity["pbfm_snapshot"]
        or assets.get("commit") != identity["asset_snapshot"]
    ):
        raise ValueError("P-BFM requires frozen P-BFM, RLinf and robot assets")
    dataset = spec.inputs.datasets[0]
    if (
        dataset.get("id") != identity["dataset_id"]
        or (dataset.get("validation") or {}).get("manifest_sha256") != identity["dataset_manifest"]
    ):
        raise ValueError("P-BFM dataset identity differs")
    return plan


def render_training(spec, run_id):
    from eb.adapters.base import RenderedJob
    from eb.settings import get_settings

    plan = {**validate_train_job(spec), "run_id": run_id}
    plan["hash"] = training_fingerprint({k: v for k, v in plan.items() if k != "hash"})
    root = get_settings().data_root / "runs" / run_id
    mounts = [(str(root), "/run/eb-output", "rw"), (str(root / "pbfm-train-plan.json"), "/plan.json", "ro")]
    for name, ref in (("pbfm", spec.params["pbfm_source_ref"]), ("rlinf", spec.code_refs.stack.model_dump())):
        path = Path(ref["path"])
        mounts += [
            (str(path), f"/sources/{name}", "ro"),
            (str(path.parent / "manifest.json"), f"/sources/{name}-manifest.json", "ro"),
        ]
    mounts.append((spec.params["asset_source_ref"]["path"], "/robot-assets", "ro"))
    mounts.append(
        (
            str(Path(spec.params["asset_source_ref"]["path"]).parent / "manifest.json"),
            "/sources/asset-manifest.json",
            "ro",
        )
    )
    data = Path(spec.inputs.datasets[0]["path"])
    mounts += [(str(data / name), "/training-data/" + name, "ro") for name in ("train.pkl", "manifest.json")]
    if plan.get("resume"):
        mounts.append((plan["resume"]["host_path"], "/resume-native", "ro"))
        mounts.append(
            (str(Path(plan["resume"]["host_path"]).parent / "receipt.json"), "/resume-receipt.json", "ro")
        )
    if run_id not in ("run_preview", "run_validate"):
        if not re.fullmatch(r"run_[a-zA-Z0-9_-]+", run_id) or any(
            p.is_symlink() for p in (root, *root.parents)
        ):
            raise ValueError("P-BFM Run path is invalid")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "pbfm-train-plan.json"
        encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
        if path.is_symlink() or path.exists() and path.read_text() != encoded:
            raise ValueError("P-BFM frozen Run plan differs")
        path.write_text(encoded)
    return RenderedJob(
        image=spec.image,
        command=["python", "-m", "eb.runners.rlinf_pbfm_entry", "--plan", "/plan.json"],
        env={
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
        metrics_glob="runtime/native-staging/train_log.txt",
        gpus=spec.resources.gpus,
        shm_gb=spec.resources.shm_gb,
    )
