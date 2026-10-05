"""Container-only conversion of pinned RoboParty motions into an AMP DatasetVersion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
from pathlib import Path

import numpy as np

from eb.data_validation import validate_amp_motion
from eb.rpo_wbc import AMP_MOTION_CONVERTER_REVISION, JOINT_ORDER, RPO_LAB_JOINT_ORDER
from eb.runners.rpo_amp_motion import encode_amp_frame


REVISION = "6b1c3d9988497c8961dcba77892de32edc1770e1"
OFFICIAL = {
    "A1-_Stand_stageii": ("stand", "189eabf1622f468ed1264c938e8941b2ae181eb295c62e73722e70db40039c5f", 1),
    "36_11": ("walk", "44d04c28be7ffbe516ffa23875cdbcd3767c04972c9fcdbe04aa345f3b80c4b7", 3),
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _normalize(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 0.99) or np.any(norm > 1.01):
        raise ValueError("source quaternion norm is invalid")
    return q / norm


def _rotate_inverse(q, v):
    vector = -q[..., 1:]
    t = 2 * np.cross(vector, v)
    return v + q[..., :1] * t + np.cross(vector, t)


def _slerp(q, old_t, new_t):
    index = np.searchsorted(old_t, new_t, side="right")
    left = np.clip(index - 1, 0, len(old_t) - 2)
    right = left + 1
    ratio = ((new_t - old_t[left]) / (old_t[right] - old_t[left]))[:, None]
    a, b = q[left], q[right].copy()
    dot = np.sum(a * b, axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1, 1))
    sine = np.sin(theta)
    small = sine < 1e-6
    first = np.where(small, 1 - ratio, np.sin((1 - ratio) * theta) / np.maximum(sine, 1e-6))
    second = np.where(small, ratio, np.sin(ratio * theta) / np.maximum(sine, 1e-6))
    result = first * a + second * b
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _interpolate(value, old_t, new_t):
    flattened = value.reshape(len(value), -1)
    result = np.stack([np.interp(new_t, old_t, flattened[:, col])
                       for col in range(flattened.shape[1])], axis=1)
    return result.reshape((len(new_t), *value.shape[1:]))


def _source_features(raw: dict) -> np.ndarray:
    keys = {"fps", "root_pos", "root_rot", "dof_pos", "key_body_pos"}
    if not isinstance(raw, dict) or not keys.issubset(raw):
        raise ValueError("official motion lacks required state fields")
    fps = float(raw["fps"])
    if not math.isfinite(fps) or not 1 <= fps <= 500:
        raise ValueError("motion fps is invalid")
    pos = np.asarray(raw["root_pos"], dtype=np.float64)
    quat = np.asarray(raw["root_rot"], dtype=np.float64)
    joint = np.asarray(raw["dof_pos"], dtype=np.float64)
    key_pos = np.asarray(raw["key_body_pos"], dtype=np.float64)
    count = len(pos)
    if (not 3 <= count <= 100_000 or pos.shape != (count, 3)
            or quat.shape != (count, 4) or joint.shape != (count, 23)
            or key_pos.shape != (count, 6, 3)
            or any(not np.isfinite(value).all() for value in (pos, quat, joint, key_pos))):
        raise ValueError("motion dimensions or finite values are invalid")
    joint = joint[:, [RPO_LAB_JOINT_ORDER.index(name) for name in JOINT_ORDER]]
    quat = _normalize(quat)
    old_t = np.arange(count) / fps
    new_t = np.arange(round(old_t[-1] * 50) + 1) / 50
    new_t = new_t[new_t <= old_t[-1] + 1e-9]
    if len(new_t) < 3:
        raise ValueError("motion has fewer than three 50 Hz frames")
    pos, joint, key_pos = (_interpolate(value, old_t, new_t) for value in (pos, joint, key_pos))
    quat = _slerp(quat, old_t, new_t)
    lin_w = np.gradient(pos, 0.02, axis=0)
    dq = np.gradient(joint, 0.02, axis=0)
    q0, q1 = quat[:-1], quat[1:]
    relative = np.empty_like(q0)
    relative[:, 0] = np.sum(q0 * q1, axis=1)
    relative[:, 1:] = q0[:, :1] * q1[:, 1:] - q1[:, :1] * q0[:, 1:] - np.cross(q0[:, 1:], q1[:, 1:])
    relative = np.where(relative[:, :1] < 0, -relative, relative)
    angle = 2 * np.arctan2(np.linalg.norm(relative[:, 1:], axis=1), relative[:, 0])
    axis = relative[:, 1:] / np.maximum(np.linalg.norm(relative[:, 1:], axis=1, keepdims=True), 1e-12)
    ang_b = np.vstack((axis * (angle / 0.02)[:, None], np.zeros((1, 3))))
    ang_b[-1] = ang_b[-2]
    gravity = _rotate_inverse(quat, np.broadcast_to([0.0, 0.0, -1.0], pos.shape))
    key_b = _rotate_inverse(np.broadcast_to(quat[:, None], (len(quat), 6, 4)), key_pos - pos[:, None])
    lin_b = _rotate_inverse(quat, lin_w)
    return encode_amp_frame(root_ang_vel_b=ang_b, q_abs=joint, dq=dq,
                            key_body_pos_b=key_b, projected_gravity_b=gravity,
                            root_height=pos[:, 2:3], root_lin_vel_b=lin_b)


def _walk_mask(features: np.ndarray) -> np.ndarray:
    """Require forward motion and a foot near the ground, not airborne slow drift."""
    kernel = np.ones(25) / 25  # 0.5 s at 50 Hz filters gait-cycle sway.

    def smoothed(values):
        return np.convolve(np.pad(values, (12, 12), mode="edge"), kernel, mode="valid")

    forward = smoothed(features[:, 71])
    lateral = smoothed(np.abs(features[:, 72]))
    yaw_rate = smoothed(np.abs(features[:, 2]))
    feet = features[:, 49:55].reshape(-1, 2, 3)
    feet_z = features[:, 70, None] - np.sum(feet * features[:, None, 67:70], axis=-1)
    contact = np.min(feet_z, axis=1)
    return ((forward >= 0.15) & (forward <= 0.65)
            & (lateral <= 0.20) & (yaw_rate <= 0.50)
            & (contact >= -0.05) & (contact <= 0.08))


def _straight_walk_segments(features: np.ndarray) -> list[np.ndarray]:
    """Keep sustained grounded low-speed intervals; never join across turns."""
    valid = _walk_mask(features)
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    segments = []
    for first, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        start, net_yaw = int(first), 0.0
        for index in range(start, int(end)):
            next_yaw = net_yaw + float(features[index, 2]) * 0.02
            if abs(next_yaw) > 0.25:
                if index - start >= 50:
                    segments.append(features[start:index])
                start, net_yaw = index, 0.0
            else:
                net_yaw = next_yaw
        if end - start >= 50:
            segments.append(features[start:end])
    if sum(len(segment) for segment in segments) < 200:
        raise ValueError("official walk clip has fewer than four seconds of low-speed straight gait")
    return segments


def audit(source: Path) -> dict:
    """Bounded, isolated inspection of official candidate motions before selection."""
    def summary(features):
        valid = _walk_mask(features)
        changes = np.diff(np.r_[False, valid, False].astype(np.int8))
        lengths = sorted((int(end - start) for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))), reverse=True)
        feet = features[:, 49:55].reshape(-1, 2, 3)
        feet_z = features[:, 70, None] - np.sum(feet * features[:, None, 67:70], axis=-1)
        return {"frames":len(features), "duration_s":round(len(features) / 50, 2),
                "median_root_height_m":round(float(np.median(features[:, 70])), 3),
                "median_forward_mps":round(float(np.median(features[:, 71])), 3),
                "airborne_fraction":round(float(np.mean(np.min(feet_z, axis=1) > 0.1)), 3),
                "grounded_forward_frames":int(valid.sum()),
                "grounded_forward_segments_1s":sum(length >= 50 for length in lengths),
                "grounded_forward_frames_1s":sum(length for length in lengths if length >= 50),
                "longest_grounded_forward_s":round((lengths[0] if lengths else 0) / 50, 2)}

    candidates = sorted(source.glob("*.pkl"))
    if not 1 <= len(candidates) <= 40:
        raise ValueError("official AMP motion candidate count is invalid")
    rows = []
    for path in candidates:
        if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"invalid audit candidate: {path.name}")
        row = {"name":path.name, "sha256":_digest(path)}
        try:
            with path.open("rb") as stream:
                raw = pickle.load(stream)
            row.update(summary(_source_features(raw)))
            row["time_scales"] = {str(scale):summary(_source_features(
                {**raw, "fps":float(raw["fps"]) / scale})) for scale in (2, 3)}
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return {"schema_version":"eb/rpo-amp-motion-audit/v1", "source_revision":REVISION,
            "candidates":rows}


def convert(source: Path, destination: Path, asset_manifest: Path) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("AMP DatasetVersion destination already exists")
    asset_sha = _digest(asset_manifest)
    staging = destination.with_name(destination.name + ".tmp")
    if staging.exists():
        raise FileExistsError("AMP conversion staging directory already exists")
    staging.mkdir(parents=True)
    clips, refs = [], []
    try:
        for clip_id, (behavior, expected_sha, time_scale) in OFFICIAL.items():
            path = source / f"{clip_id}.pkl"
            if path.is_symlink() or not path.is_file() or _digest(path) != expected_sha:
                raise ValueError(f"official motion hash changed: {clip_id}")
            # This process runs only in an isolated tools task container with pinned inputs.
            with path.open("rb") as stream:
                raw = pickle.load(stream)
            source_fps = float(raw["fps"])
            features = _source_features({**raw, "fps":source_fps / time_scale})
            if behavior == "stand" and np.percentile(np.linalg.norm(features[:, 71:74], axis=1), 90) > 0.15:
                raise ValueError("official stand clip moves too quickly")
            segments = [features] if behavior == "stand" else _straight_walk_segments(features)
            for index, segment in enumerate(segments):
                member_id = clip_id if behavior == "stand" else f"{clip_id}-straight-{index:02d}"
                file = staging / f"{member_id}.npy"
                np.save(file, segment)
                clips.append({"id":member_id, "source_group":clip_id,
                              "source_license":f"Roboparty/robolab@{REVISION}:BSD-3-Clause",
                              "behavior":behavior, "split":"train", "file":file.name,
                              "sha256":_digest(file), "fps":50,
                              "source_fps":source_fps, "time_scale":time_scale})
            refs.append({"kind":"github", "uri":f"https://github.com/Roboparty/robolab/blob/{REVISION}/data/motions/rpo_lab/{clip_id}.pkl",
                         "revision":REVISION, "sha256":expected_sha, "time_scale":time_scale})
        manifest = {"schema_version":"eb/rpo-amp-motion/v1", "usage":"train", "purpose":"motion_prior",
                    "qualification":"diagnostic",
                    "feature_schema":"rpo_amp_features_v1", "asset_sha256":asset_sha,
                    "selection":{"walk_forward_mps":[0.15, 0.65], "walk_lateral_abs_mps_max":0.20,
                                 "walk_yaw_abs_radps_max":0.50, "smoothing_frames":25,
                                 "walk_max_net_yaw_rad":0.25, "max_foot_clearance_min_m":0.08,
                                 "max_ground_penetration_m":0.05,
                                 "min_segment_frames":50, "min_total_frames_per_source":200},
                    "joint_order":list(JOINT_ORDER),
                    "source_joint_order":list(RPO_LAB_JOINT_ORDER),
                    "converter_revision":AMP_MOTION_CONVERTER_REVISION,
                    "source_refs":refs, "clips":clips}
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        report = validate_amp_motion(staging, asset_manifest, refs)
        staging.rename(destination)
        report["storage_ref"] = str(destination)
        report["validation"]["storage_ref"] = str(destination)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.audit:
        result = audit(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True))
        return
    if args.destination is None or args.asset_manifest is None:
        parser.error("conversion requires --destination and --asset-manifest")
    try:
        result = convert(args.source, args.destination, args.asset_manifest)
    except Exception as exc:
        result = {"validation":{"status":"invalid", "scope":"rpo_amp_motion_v1",
                                "training_ready":False, "errors":[str(exc)]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    if result["validation"]["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
