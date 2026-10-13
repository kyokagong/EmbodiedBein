"""UFO aggregate motion → hhtools single-robot trajectory (no retargeting here)."""

from __future__ import annotations

import numpy as np


def joint_trajectory(clip: dict, source_joint_names: list[str], *, limit_frames: int = 100) -> dict:
    if (len(source_joint_names) != 29 or len(set(source_joint_names)) != 29
            or any(not isinstance(n, str) or not n for n in source_joint_names)
            or not 9 <= limit_frames <= 500):
        raise ValueError("G1 bridge requires a frozen 29-joint source order and a bounded clip")
    if "dof_names" in clip and list(clip["dof_names"]) != source_joint_names:
        raise ValueError("G1 clip joint order differs from the bound source model")
    # shortcut: require explicit DOFs; add hinge-axis recovery only after verifying a real pose-only source.
    if "dof_pos" not in clip:
        raise ValueError("G1 source lacks dof_pos; pose_aa must not be flattened into joint angles")
    q = np.asarray(clip["dof_pos"], dtype=np.float64)
    pos = np.asarray(clip.get("root_trans_offset"), dtype=np.float64)
    pose = np.asarray(clip.get("pose_aa"), dtype=np.float64)
    fps = float(clip.get("fps", 0))
    if (q.ndim != 2 or q.shape[1] != 29 or not 9 <= len(q) <= 30000
            or pos.shape != (len(q), 3) or pose.ndim != 3
            or pose.shape[0] != len(q) or pose.shape[1] < 1 or pose.shape[2] != 3
            or not 10 <= fps <= 500 or not all(np.isfinite(a).all() for a in (q, pos, pose))
            or np.any(np.abs(q) > 2*np.pi) or np.any(np.abs(pose) > 2*np.pi)
            or np.any(np.abs(pos) > 10000)):
        raise ValueError("G1 source shape/FPS/finite check failed")
    rotvec = pose[:limit_frames, 0]
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    quat = np.concatenate((rotvec * .5 * np.sinc(angle/(2*np.pi)), np.cos(angle/2)), axis=-1)
    trajectory = np.column_stack((pos[:limit_frames], quat, q[:limit_frames])).astype(np.float32)
    if not np.isfinite(trajectory).all() or not np.allclose(np.linalg.norm(trajectory[:, 3:7], axis=-1), 1, atol=1e-5):
        raise ValueError("G1 bridge output finite/quaternion check failed")
    return {"joint_q":trajectory,
            "dof_names":np.asarray(source_joint_names), "fps":fps, "root_quat_format":"xyzw"}


def verified_ufo_clip(clip, source_joint_names, source_axes, *, limit_frames=100):
    """The acquired UFO file uses `dof`; verify its order against per-hinge pose axes."""
    q = np.asarray(clip.get("dof_pos", clip.get("dof")), dtype=np.float64)
    pose = np.asarray(clip.get("pose_aa"), dtype=np.float64)
    axes = np.asarray(source_axes, dtype=np.float64)
    if (q.ndim != 2 or q.shape[1] != 29 or axes.shape != (29, 3)
            or pose.shape != (len(q), 30, 3) or not np.isfinite(q).all()
            or not np.isfinite(pose).all()
            or not np.allclose(np.linalg.norm(axes, axis=-1), 1, atol=1e-6)
            or not np.allclose(pose[:, 1:], q[..., None]*axes, atol=1e-5)):
        raise ValueError("G1 explicit DOFs and source hinge axis/order differ")
    return joint_trajectory({**clip, "dof_pos":q}, source_joint_names, limit_frames=limit_frames)
