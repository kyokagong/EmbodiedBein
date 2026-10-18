"""Frozen metadata for native UFO inference; numerical work stays in task runners."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath

ENV_ID = "ee_rpo_v2_ufo_native_23dof_v2"
TASK_ID = "tsk_rpo_ufo_native_stand_walk_v2"
MODEL_SCHEMA = "eb/rpo-ufo-native-model/v1"
EXPORT_SCHEMA = "eb/rpo-ufo-native-export/v1"
IMAGE = "eb/ufo-native:rpo-deploy-v1"
OPERATIONS = {
    "ufo_native_import": "data.convert",
    "ufo_native_export": "export.policy",
    "ufo_native_eval": "eval.sim",
    "ufo_native_bundle": "bundle.build",
    "ufo_native_live": "serve.policy",
}
JOINTS = (
    tuple(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("thigh_yaw", "thigh_roll", "thigh_pitch", "knee", "ankle_pitch", "ankle_roll")
    )
    + ("torso_joint",)
    + tuple(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("arm_pitch", "arm_roll", "arm_yaw", "elbow_pitch", "elbow_yaw")
    )
)
TASKS = (
    {"id": "move-ego-0-0", "label": "原地站立", "speed": 0.0},
    {"id": "move-ego-0-0.3", "label": "0.3 m/s 前行", "speed": 0.3},
    {"id": "move-ego-0-0.7", "label": "0.7 m/s 前行（诊断）", "speed": 0.7},
)
SOURCE_FILES = (
    "config.json",
    "config.yaml",
    "effective_config.json",
    "checkpoint/train_status.json",
    "checkpoint/model/config.json",
    "checkpoint/model/init_kwargs.json",
    "checkpoint/model/model.safetensors",
    "reward_inference/reward_locomotion.pkl",
)


def plan_hash(value: dict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {k: v for k, v in value.items() if k != "hash"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle_source(payload: dict, plan: dict) -> dict:
    source = plan.get("source_import_manifest")
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != "eb/ufo-native-import/v1"
        or not plan.get("source_model_id")
        or payload.get("source_model_id") != plan["source_model_id"]
        or plan_hash(source) != payload.get("source_artifact_sha256")
        or plan_hash(payload) != plan.get("source_artifact_sha256")
    ):
        raise ValueError("Offline bundle source identity differs from frozen payload")
    return source


def read_json(path: Path, limit: int = 2 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f"Invalid bounded JSON: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def safe_file(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(p in ("", ".", "..") for p in relative.split("/"))
        or not parts
    ):
        raise ValueError("Unsafe relative file path")
    path = root
    if root.is_symlink():
        raise ValueError("Symlink root is forbidden")
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Symlink file is forbidden")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe file: {relative}")
    return path


def validate_config(saved: dict, effective: dict, model: dict, kwargs: dict) -> dict:
    env, task = saved["env"], effective["task"]
    robot = env["robot_training"]
    names = robot["robot"]["control_joint_names"]
    if names != list(JOINTS) or task["robot"]["dof_names"] != names:
        raise ValueError("Native joint order mismatch")
    obs = task["obs"]
    history = {"base_ang_vel": 4, "projected_gravity": 4, "dof_pos": 4, "dof_vel": 4, "actions": 4}
    if obs["history_oldest_first"] is not False or obs["obs_auxiliary"]["history_actor"] != history:
        raise ValueError("Native history layout mismatch")
    dims = {"state": 52, "last_action": 23, "history_actor": 300, "privileged_state": 358}
    if (
        {k: v["shape"][0] for k, v in kwargs["obs_space"]["spaces"].items()} != dims
        or kwargs["action_dim"] != 23
        or model["name"] != "FBcprAuxModel"
        or model["archi"]["z_dim"] != 256
        or model["archi"]["actor"]["input_filter"]["key"] != ["state", "last_action", "history_actor"]
        or set(model["obs_normalizer"]["normalizers"]) != set(dims)
    ):
        raise ValueError("Native model/normalizer dimensions mismatch")
    if any(obs["obs_scales"][key] != 1.0 for key in (*history, "history_actor")):
        raise ValueError("Unsupported native observation scale")
    control = task["robot"]["control"]
    if (
        control["action_scale"] != 0.6
        or control["action_rescale"] is not True
        or control["normalize_action"] is not True
        or control["normalize_action_from"] != 1.0
        or control["normalize_action_to"] != 5.0
        or control["action_clip_value"] != 5.0
        or robot["action_scale"] != 0.6
        or robot["normalize_action_to"] != 5.0
        or task["simulator"]["config"]["sim"] != {"fps": 400, "control_decimation": 8}
    ):
        raise ValueError("Native control configuration mismatch")
    for key in ("stiffness", "damping"):
        if task["robot"]["control"][key] != robot[key]:
            raise ValueError(f"Saved native {key} mismatch")
    default = [robot["default_joint_angles"][n] for n in names]
    kp, kd = ([robot[k][n] for n in names] for k in ("stiffness", "damping"))
    effort, velocity = robot["effort_limits"], robot["velocity_limits"]
    lower, upper = zip(*(robot["robot"]["joint_ranges"][n] for n in names))
    values = [*default, *kp, *kd, *effort, *velocity, *lower, *upper]
    if not all(math.isfinite(x) for x in values) or any(k <= 0 for k in kp):
        raise ValueError("Non-finite native control parameters")
    return {
        "schema_version": "eb/ufo-native-codec/v1",
        "joint_order": names,
        "actor_dim": 631,
        "state_dim": 52,
        "history_order": sorted(history),
        "history_oldest_first": False,
        "history_includes_current": False,
        "history_length": 4,
        "physics_dt": 0.0025,
        "decimation": 8,
        "policy_hz": 50,
        "latent_dim": 256,
        "default_pos": default,
        "kp": kp,
        "kd": kd,
        "effort": effort,
        "velocity": velocity,
        "lower": list(lower),
        "upper": list(upper),
        "action_clip": 1.0,
        "feedback_action_scale": 5.0,
        "action_gain": [0.6 * e / k * 5 for e, k in zip(effort, kp)],
        "actuator": robot["actuator"],
        "root_quaternion": "xyzw",
        "qpos_quaternion": "wxyz",
    }


def inference_overrides(effective):
    task = deepcopy(effective["task"])
    # UFO reconstructs these from the frozen robot snapshot after schema merge.
    task.pop("robot", None)
    task["obs"].pop("obs_dims", None)
    return task


def eval_cases(profile: str) -> list[dict]:
    if profile == "smoke":
        return [
            {"case_id": f"smoke-{i}", "seed": 71000 + i, "task": task["id"], "speed": task["speed"]}
            for i, task in enumerate(TASKS)
        ]
    if profile != "acceptance":
        raise ValueError("Unknown native evaluation profile")
    return [
        {
            "case_id": f"acceptance-{i}",
            "seed": 72000 + i,
            "task": TASKS[i % 2]["id"],
            "speed": TASKS[i % 2]["speed"],
        }
        for i in range(12)
    ]


def suite_spec(backend: str, profile: str) -> dict:
    if backend not in ("mjlab", "mujoco"):
        raise ValueError("Unknown native simulator")
    return {
        "benchmark": "rpo_ufo_native",
        "backend": backend,
        "profile": profile,
        "policy": "onnx",
        "seconds": 20,
        "steps": 1000,
        "cases": eval_cases(profile),
        "min_height_m": 0.5,
        "max_tilt_rad": 0.8,
        "standing_drift_m": 0.5,
        "standing_speed_rmse_m_s": 0.15,
        "walking_displacement_m": 4.0,
        "walking_speed_rmse_m_s": 0.2,
        "record_all_cases": True,
        "scope": "simulation_only",
        "physics_family": "MuJoCo",
    }


def finite_tree(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def validate_cases(plan: dict, rows: list) -> dict:
    expected = {case["case_id"]: case for case in plan["cases"]}
    if len(rows) != len(expected) or len({row["case_id"] for row in rows}) != len(expected):
        raise ValueError("Missing or duplicate native evaluation cases")
    for row in rows:
        if (
            row["case_id"] not in expected
            or any(row.get(k) != v for k, v in expected[row["case_id"]].items())
            or row["status"] not in ("success", "task_failure", "timeout")
            or type(row["steps"]) is not int
            or not 1 <= row["steps"] <= 1000
            or row.get("video") != f"videos/{row['case_id']}.mp4"
            or not finite_tree(row)
        ):
            raise ValueError("Invalid native evaluation case evidence")
        if row["status"] == "success":
            metrics = row.get("metrics") or {}
            required = {
                "min_height_m",
                "max_tilt_rad",
                "speed_rmse_m_s",
                "drift_m",
                "forward_displacement_m",
                "joint_violation_fraction",
                "velocity_violation_fraction",
                "torque_violation_fraction",
            }
            if (
                row["steps"] != 1000
                or row.get("reason") != "completed"
                or not required <= set(metrics)
                or metrics["min_height_m"] < 0.5
                or metrics["max_tilt_rad"] > 0.8
                or metrics["speed_rmse_m_s"] > (0.15 if row["speed"] == 0 else 0.2)
                or (
                    metrics["drift_m"] > 0.5 if row["speed"] == 0 else metrics["forward_displacement_m"] < 4.0
                )
                or any(
                    metrics[k] != 0
                    for k in (
                        "joint_violation_fraction",
                        "velocity_violation_fraction",
                        "torque_violation_fraction",
                    )
                )
            ):
                raise ValueError("Claimed native success lacks full-horizon task metrics")
    n = len(rows)
    passed = sum(row["status"] == "success" for row in rows)
    p, z = passed / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "planned": n,
        "completed": n,
        "valid": n,
        "invalid": 0,
        "infra_failed": 0,
        "cancelled": 0,
        "successes": passed,
        "success_rate": p,
        "wilson_95": [center - margin, center + margin],
        "qualification": "diagnostic" if plan["profile"] == "smoke" else "pass" if passed == n else "fail",
        "cases": rows,
    }


def validate_job(spec) -> dict:
    params = spec.params
    plan = params.get("native_plan") or {}
    operation = params.get("operation")
    expected_kind = (
        "eval.sim2sim"
        if operation == "ufo_native_eval" and plan.get("backend") == "mujoco"
        else OPERATIONS.get(operation)
    )
    gpu = int(operation in ("ufo_native_eval", "ufo_native_live") and (
        plan.get("backend") == "mjlab" or bool((plan.get("suite_spec") or {}).get("stream"))))
    if (
        set(params) != {"operation", "native_plan"}
        or spec.embodiment_env_id != ENV_ID
        or spec.kind != expected_kind
        or plan.get("operation") != operation
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", plan.get("image_digest", ""))
        or spec.image != plan["image_digest"]
        or plan.get("hash") != plan_hash(plan)
        or spec.overrides
        or spec.inputs.datasets
        or spec.inputs.resume_from
        or spec.runtime != {"network": "none"}
        or spec.code_refs
        and (spec.code_refs.stack or spec.code_refs.embodiment)
        or spec.resources.gpus != gpu
        or spec.resources.cpus > 8
        or spec.resources.memory_gb > 16
        or spec.policy.max_retries != 0
        or not 0 < spec.policy.timeout_h <= 0.5
    ):
        raise ValueError("Native job differs from frozen bounded offline plan")
    if operation == "ufo_native_eval" and (
        spec.task_id != TASK_ID
        or plan.get("suite_spec") != suite_spec(plan["backend"], plan["profile"])
        or plan.get("cases") != eval_cases(plan["profile"])
    ):
        raise ValueError("Native evaluation suite changed")
    if operation == "ufo_native_live":
        from eb.ufo_live import validate_live_plan
        validate_live_plan(plan)
        if (
            spec.task_id != TASK_ID
            or spec.inputs.base_model != plan["input_path"]
            or spec.resources.cpus != 4 or spec.resources.memory_gb != 16
            or spec.resources.shm_gb != 2 or spec.resources.devices
            or spec.policy.timeout_h != 0.5 or spec.fanout or spec.outputs
        ):
            raise ValueError("Live native job differs from frozen resources and model")
    return plan
