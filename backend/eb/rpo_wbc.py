from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ASSET_BUNDLE = "roboparty/rpo-v2"
ASSET_SCHEMA = "eb/rpo-v2-assets/v1"
ACTOR_FRAME_DIM = 89
CRITIC_FRAME_DIM = 150
HISTORY_LENGTH = 10
ACTOR_DIM = ACTOR_FRAME_DIM * HISTORY_LENGTH
CRITIC_DIM = CRITIC_FRAME_DIM * HISTORY_LENGTH

JOINT_ORDER = (
    "left_thigh_yaw_joint", "left_thigh_roll_joint", "left_thigh_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_thigh_yaw_joint", "right_thigh_roll_joint", "right_thigh_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint", "left_arm_roll_joint", "left_arm_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
)
RPO_LAB_JOINT_ORDER = (
    "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
    "left_thigh_roll_joint", "right_thigh_roll_joint",
    "left_arm_pitch_joint", "right_arm_pitch_joint",
    "left_thigh_pitch_joint", "right_thigh_pitch_joint",
    "left_arm_roll_joint", "right_arm_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_arm_yaw_joint", "right_arm_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
)
AMP_MOTION_CONVERTER_REVISION = "grounded-retimed-v4"
UPPER_BODY_JOINTS = JOINT_ORDER[12:]
AMP_STAND_POSE_ID = "official_a1_stand_v1"
# Canonical joints from frame 95 of the frozen A1-_Stand_stageii 50 Hz dataset.
AMP_STAND_Q = (
    -.189796, -.036899, -.034211, 0., -.151957, 0.,
    .186229, .045729, -.095739, .017555, -.080579, 0.,
    .013409, -.143103, .240643, .633730, 1.194961, 0.,
    -.077857, -.160675, -.356338, 1.156749, 0.,
)
AMP_STAND_HEIGHT_M = .710682
_RPO_DEFAULT_Q = (0., 0., -.1, .3, -.2, 0., 0., 0., -.1, .3, -.2, 0.,
                  0., .18, .06, 0., .78, 0., .18, -.06, 0., .78, 0.)
AMP_STAND_ACTION = tuple((target - default) / .25
                         for target, default in zip(AMP_STAND_Q, _RPO_DEFAULT_Q))

REQUIRED_ASSETS = (
    "robolab/robolab/tasks/direct/base/base_env.py",
    "rpo_description/urdf/rpo.urdf",
    "rpo_description/mjcf/rpo.xml",
    "rpo_hardware/V2.0/roboto_origin_mechanic/03_URDF/urdf/roboto_origin.urdf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def urdf_joints(path: Path) -> list[str]:
    return [node.attrib["name"] for node in ET.parse(path).getroot().findall("joint")
            if node.attrib.get("type") != "fixed"]


def _urdf_profile(path: Path, search_root: Path) -> dict:
    root = ET.parse(path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        limits[joint.attrib["name"]] = dict(sorted((limit.attrib if limit is not None else {}).items()))
    meshes = sorted({node.attrib["filename"] for node in root.findall(".//mesh") if "filename" in node.attrib})
    available = {path.name for path in search_root.rglob("*") if path.is_file()}
    return {"limits": limits, "mesh_references": meshes,
            "missing_meshes": [name for name in meshes if Path(name).name not in available]}


def verify_asset_bundle(root: Path, revisions: dict[str, str]) -> dict:
    missing = [relative for relative in REQUIRED_ASSETS if not (root / relative).is_file()]
    training_urdf = root / "rpo_description/urdf/rpo.urdf"
    hardware_urdf = root / "rpo_hardware/V2.0/roboto_origin_mechanic/03_URDF/urdf/roboto_origin.urdf"
    training_joints = urdf_joints(training_urdf) if training_urdf.is_file() else []
    hardware_joints = urdf_joints(hardware_urdf) if hardware_urdf.is_file() else []
    training_profile = _urdf_profile(training_urdf, root / "rpo_description") if training_urdf.is_file() else {}
    hardware_profile = _urdf_profile(hardware_urdf, root / "rpo_hardware/V2.0") if hardware_urdf.is_file() else {}
    blockers = []
    if training_joints != list(JOINT_ORDER):
        blockers.append("training_joint_order_mismatch")
    if hardware_joints and hardware_joints != list(JOINT_ORDER):
        blockers.append("hardware_joint_order_mismatch")
    if training_profile.get("missing_meshes") or hardware_profile.get("missing_meshes"):
        blockers.append("mesh_dependency_missing")
    if missing:
        blockers.append("asset_files_missing")
    files = [{"path": relative, "sha256": _sha256(root / relative)}
             for relative in REQUIRED_ASSETS if (root / relative).is_file()]
    return {
        "schema_version": ASSET_SCHEMA,
        "status": "verified" if not blockers else "blocked",
        "hardware_revision": "RPO V2.0",
        "revisions": revisions,
        "joint_order": training_joints,
        "hardware_joint_order": hardware_joints,
        "asset_diff": {
            "joint_limits": {
                joint: {"training": training_profile.get("limits", {}).get(joint),
                        "hardware": hardware_profile.get("limits", {}).get(joint)}
                for joint in JOINT_ORDER
                if training_profile.get("limits", {}).get(joint) != hardware_profile.get("limits", {}).get(joint)
            },
            "training_missing_meshes": training_profile.get("missing_meshes", []),
            "hardware_missing_meshes": hardware_profile.get("missing_meshes", []),
            "training_mesh_count": len(training_profile.get("mesh_references", [])),
            "hardware_mesh_count": len(hardware_profile.get("mesh_references", [])),
            "actuator_override_source": "robolab/robolab/assets/robots/roboparty.py",
            "closed_chain_calibration": "hardware_unverified",
        },
        "files": files,
        "missing": missing,
        "blockers": blockers,
        "hardware_verified": False,
    }


def write_asset_manifest(root: Path, revisions: dict[str, str]) -> Path:
    receipt = verify_asset_bundle(root, revisions)
    path = root / "asset-manifest.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return path


def validate_eval_receipt(receipt: dict, *, criterion: str = "tracking", amp: bool = False) -> list[str]:
    errors: list[str] = []
    onnx = receipt.get("onnx") or {}
    cases = receipt.get("cases") or {}
    metrics = receipt.get("metrics") or {}
    if receipt.get("schema_version") != ("eb/rpo-amp-eval/v1" if amp else "eb/rpo-wbc-eval/v1"):
        errors.append("schema_version")
    if receipt.get("criterion", "tracking") != criterion:
        errors.append("criterion")
    if criterion in {"stability", "stand_walk"} and receipt.get("required_steps") != 1000:
        errors.append("required_steps")
    if onnx.get("obs") != (780 if amp else ACTOR_DIM):
        errors.append("onnx.obs")
    if onnx.get("act") != len(JOINT_ORDER):
        errors.append("onnx.act")
    if not re.fullmatch(r"[0-9a-f]{64}", str(onnx.get("sha256", ""))):
        errors.append("onnx.sha256")
    export_error = onnx.get("max_abs_error")
    if (type(export_error) not in (int, float) or not math.isfinite(export_error)
            or not 0 <= export_error <= 1e-4):
        errors.append("onnx.max_abs_error")
    for key, value in metrics.items():
        if type(value) in (int, float) and not math.isfinite(value):
            errors.append(f"metrics.{key}")
    if cases.get("completed") != cases.get("planned"):
        errors.append("cases.completed")
    if cases.get("valid") != cases.get("planned") or cases.get("invalid") or cases.get("infra_failed"):
        errors.append("cases.valid")
    failure_counts = ("fall_count", "hard_limit_count", "nan_count") if amp or criterion == "tracking" else ("fall_count", "nan_count")
    for key in failure_counts:
        if metrics.get(key) != 0:
            errors.append(f"metrics.{key}")
    if criterion in {"stability", "stand_walk"}:
        if metrics.get("successes") != metrics.get("n_trials") or metrics.get("n_trials") != cases.get("planned"):
            errors.append("metrics.successes")
    else:
        for key, limit in (("velocity_rmse", 0.15), ("upper_rmse", 0.1)):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or value > limit:
                errors.append(f"metrics.{key}")
    return errors
