"""Small Xsens → frozen RPO IK trial; this is not a qualified motion dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q, RPO_LAB_JOINT_ORDER
    from eb.runners.rpo_bfm_motion import digest
except ModuleNotFoundError:
    from rpo_bfm import ACTION_GAIN, DEFAULT_Q, RPO_LAB_JOINT_ORDER
    from rpo_bfm_motion import digest


# Xsens centimetres, +Y up/+X left/+Z forward → RPO metres, +Z up/+Y left/+X forward.
BASIS = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
BODY_MAP = {
    f"{side}{human}": f"{side.lower()}_{robot}_link"
    for side in ("Left", "Right")
    for human, robot in (
        ("Shoulder", "arm_pitch"),
        ("Elbow", "elbow_pitch"),
        ("Wrist", "elbow_yaw"),
        ("Knee", "knee"),
        ("Ankle", "ankle_roll"),
    )
}


def parse_bvh(text):
    hierarchy, motion = text.split("MOTION", 1)
    tokens = re.findall(r"[^\s{}]+|[{}]", hierarchy)
    if tokens.pop(0) != "HIERARCHY":
        raise ValueError("Xsens BVH lacks a hierarchy")
    names, parents, offsets, channels = [], [], [], []
    cursor = 0

    def node(parent):
        nonlocal cursor
        kind = tokens[cursor]
        cursor += 1
        name = tokens[cursor]
        cursor += 1
        if tokens[cursor] != "{":
            raise ValueError("Xsens BVH node syntax is invalid")
        cursor += 1
        if tokens[cursor] != "OFFSET":
            raise ValueError("Xsens BVH node has no offset")
        offset = [float(x) for x in tokens[cursor + 1 : cursor + 4]]
        cursor += 4
        if kind == "End":
            if tokens[cursor] != "}":
                raise ValueError("Xsens BVH end site is invalid")
            cursor += 1
            return
        if kind not in {"ROOT", "JOINT"} or len(names) >= 64:
            raise ValueError("Xsens BVH node type/count is invalid")
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(offset)
        if tokens[cursor] != "CHANNELS":
            raise ValueError("Xsens BVH channels are missing")
        count = int(tokens[cursor + 1])
        channel = tokens[cursor + 2 : cursor + 2 + count]
        expected = ["Yrotation", "Xrotation", "Zrotation"]
        if parent == -1:
            expected = ["Xposition", "Yposition", "Zposition", *expected]
        if channel != expected:
            raise ValueError("Unsupported Xsens BVH channel order")
        channels.append(count)
        cursor += 2 + count
        while tokens[cursor] != "}":
            node(index)
        cursor += 1

    node(-1)
    header = re.match(r"\s*Frames:\s*(\d+)\s+Frame Time:\s*([\d.]+)\s*", motion)
    if not header or names[0] != "Hips" or cursor != len(tokens):
        raise ValueError("Xsens BVH header/root is invalid")
    count, dt = int(header[1]), float(header[2])
    if not 2 <= count <= 100000 or abs(dt - 1 / 60) > 1e-5:
        raise ValueError("Xsens BVH length or frequency is invalid")
    values = np.fromstring(motion[header.end() :], sep=" ")
    if values.size != count * sum(channels) or not np.isfinite(values).all():
        raise ValueError("Xsens BVH channel values are invalid")
    values = values.reshape(count, -1)
    positions = np.zeros((count, len(names), 3))
    rotations = np.zeros((count, len(names), 3, 3))
    start = 0
    for index, (parent, offset, width) in enumerate(zip(parents, offsets, channels)):
        data = values[:, start : start + width]
        start += width
        local = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        for axis, degrees in zip((1, 0, 2), data[:, -3:].T):
            angle = np.deg2rad(degrees)
            c, s = np.cos(angle), np.sin(angle)
            turn = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
            a, b = (axis + 1) % 3, (axis + 2) % 3
            turn[:, a, a], turn[:, b, b], turn[:, a, b], turn[:, b, a] = c, c, -s, s
            local = local @ turn
        if parent == -1:
            positions[:, index] = data[:, :3] + offset
            rotations[:, index] = local
        else:
            positions[:, index] = (
                positions[:, parent] + (rotations[:, parent] @ np.asarray(offset)[..., None])[..., 0]
            )
            rotations[:, index] = rotations[:, parent] @ local
    return names, (positions / 100) @ BASIS.T, BASIS @ rotations @ BASIS.T, 1 / dt


def limb_targets(names, positions, root_rotation, chains, scale):
    targets = {}
    for side, chain in chains.items():
        for start, middle, end, anchor, lengths in (
            ("Hip", "Knee", "Ankle", "hip", "leg_lengths"),
            ("Shoulder", "Elbow", "Wrist", "shoulder", "arm_lengths"),
        ):
            first = (root_rotation @ chain[anchor][..., None])[..., 0]
            source = [positions[:, names.index(side + part)] for part in (start, middle, end)]
            directions = [source[1] - source[0], source[2] - source[1]]
            norms = [np.linalg.norm(value, axis=-1, keepdims=True) for value in directions]
            if any(np.any(norm < 1e-6) for norm in norms):
                raise ValueError("Xsens limb has degenerate bone length")
            second = first + chain[lengths][0] * directions[0] / norms[0]
            third = second + chain[lengths][1] * directions[1] / norms[1]
            targets.update({side + start: first, side + middle: second, side + end: third})
    source_feet = positions[:, [names.index("LeftAnkle"), names.index("RightAnkle")], 2]
    floor = np.percentile(source_feet, 5)
    relative_feet = np.stack([targets[side + "Ankle"][:, 2] for side in ("Left", "Right")], -1)
    support = np.argmin(source_feet, axis=1)
    root = positions[:, 0].copy() * scale
    root[:, :2] -= root[0, :2].copy()
    # Fit the stance ankle height once per frame; no q clipping or physics teleport.
    root[:, 2] = (0.04 + scale * (source_feet - floor) - relative_feet)[np.arange(len(root)), support]
    return np.stack([targets[name] for name in BODY_MAP], 1) + root[:, None], root


def retarget(source, assets, output):
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation, Slerp

    if output.exists():
        raise ValueError("100STYLE retarget refuses to overwrite a trial")
    manifest = json.loads((source / "source-manifest.json").read_text())
    expected = {item["file"]: item["sha256"] for item in manifest["files"]}
    if manifest["license"] != "CC-BY-4.0" or len(expected) != 5:
        raise ValueError("100STYLE trial source manifest is invalid")
    for name, sha in expected.items():
        path = source / name
        if path.is_symlink() or path.parent != source or digest(path) != sha:
            raise ValueError("100STYLE trial source changed after acquisition")
    with (source / "Frame_Cuts.csv").open() as stream:
        cuts = next(row for row in csv.DictReader(stream) if row["STYLE_NAME"] == "Neutral")
    model_path = assets / "rpo_description/mjcf/rpo.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RPO_LAB_JOINT_ORDER]
    if min(jids) < 0 or model.nq != 30 or model.nv != 29:
        raise ValueError("RPO retarget model is not the native 23 joint/free-root asset")
    addresses = model.jnt_qposadr[jids]
    default = np.asarray(DEFAULT_Q)
    bound = 0.25 * np.asarray(ACTION_GAIN)
    lower = np.maximum(model.jnt_range[jids, 0], default - bound)
    upper = np.minimum(model.jnt_range[jids, 1], default + bound)
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODY_MAP.values()]
    feet = [i for i, name in enumerate(BODY_MAP) if "Ankle" in name]
    data.qpos[:7] = [0, 0, 0.75, 1, 0, 0, 0]
    data.qpos[addresses] = 0
    mujoco.mj_forward(model, data)
    foot_rest_rotation = data.xmat[np.asarray(bodies)[feet]].reshape(-1, 3, 3).copy()
    hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_thigh_pitch_link")
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
    ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    robot_leg = np.linalg.norm(data.xpos[hip] - data.xpos[knee]) + np.linalg.norm(
        data.xpos[knee] - data.xpos[ankle]
    )
    chains = {}
    for side in ("Left", "Right"):

        def body(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side.lower() + "_" + name + "_link")

        points = [
            data.xpos[body(name)].copy()
            for name in ("thigh_pitch", "knee", "ankle_roll", "arm_pitch", "elbow_pitch", "elbow_yaw")
        ]
        chains[side] = {
            "hip": points[0] - data.qpos[:3],
            "shoulder": points[3] - data.qpos[:3],
            "leg_lengths": (np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[1])),
            "arm_lengths": (np.linalg.norm(points[4] - points[3]), np.linalg.norm(points[5] - points[4])),
        }
    output.mkdir(parents=True)
    rows = []
    for movement in ("ID", "FW", "BW", "SW"):
        name = f"Neutral_{movement}.bvh"
        names, positions, rotations, fps = parse_bvh((source / name).read_text())
        ids = [names.index(item) for item in BODY_MAP]
        begin, end = int(cuts[movement + "_START"]), int(cuts[movement + "_STOP"])
        # Cuts are supplied by the author; select 3 seconds near the middle, away from T poses.
        begin = (begin + end) // 2 - 90
        times = begin / fps + np.arange(150) / 50
        source_times = np.arange(len(positions)) / fps
        if times[0] < 0 or times[-1] >= source_times[-1]:
            raise ValueError("100STYLE author cut range is outside the file")
        p = np.stack(
            [np.interp(times, source_times, col) for col in positions.reshape(len(positions), -1).T], -1
        ).reshape(150, len(names), 3)
        root_rot = Slerp(source_times, Rotation.from_matrix(rotations[:, 0]))(times)
        foot_rot = np.stack(
            [
                Slerp(source_times, Rotation.from_matrix(rotations[:, ids[i]]))(times).as_matrix()
                for i in feet
            ],
            1,
        )
        human_leg = np.linalg.norm(
            p[0, names.index("LeftHip")] - p[0, names.index("LeftKnee")]
        ) + np.linalg.norm(p[0, names.index("LeftKnee")] - p[0, names.index("LeftAnkle")])
        scale = robot_leg / human_leg
        if not 0.4 < scale < 1.2:
            raise ValueError("RPO/Xsens leg scale is outside plausible calibration range")
        mapped_targets, roots = limb_targets(names, p, root_rot.as_matrix(), chains, scale)
        q, errors, boundary, fk_positions = [], [], [], []
        previous = np.clip(default, lower + 1e-6, upper - 1e-6)
        for frame in range(150):
            data.qpos[:3] = roots[frame]
            data.qpos[3:7] = root_rot[frame].as_quat()[[3, 0, 1, 2]]
            targets = mapped_targets[frame]

            def residual(joints):
                data.qpos[addresses] = joints
                mujoco.mj_forward(model, data)
                position = (data.xpos[bodies] - targets).flatten()
                actual = data.xmat[np.asarray(bodies)[feet]].reshape(-1, 3, 3)
                desired = foot_rot[frame] @ foot_rest_rotation
                orientation = Rotation.from_matrix(desired.transpose(0, 2, 1) @ actual).as_rotvec().flatten()
                return np.r_[position, 0.15 * orientation, 0.01 * (joints - previous)]

            solution = least_squares(
                residual, previous, bounds=(lower, upper), max_nfev=30, ftol=1e-5, xtol=1e-5, gtol=1e-5
            )
            residual(solution.x)
            q.append(solution.x.copy())
            errors.append(np.linalg.norm(data.xpos[bodies] - targets, axis=-1))
            fk_positions.append(data.xpos[bodies].copy())
            boundary.append(bool(np.any(np.minimum(solution.x - lower, upper - solution.x) < 1e-4)))
            previous = solution.x
        path = output / (name[:-4] + ".npz")
        np.savez(
            path,
            q=np.asarray(q, np.float32),
            root_pos=roots.astype(np.float32),
            root_quat=root_rot.as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
            fps=50,
            mujoco_body_pos=np.asarray(fk_positions, np.float32),
            source_frame=times * fps,
        )
        row = {
            "file": path.name,
            "sha256": digest(path),
            "source_file": name,
            "source_sha256": expected[name],
            "frames": 150,
            "duration_s": 3.0,
            "leg_scale": float(scale),
            "body_position_rmse_m": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
            "max_body_position_error_m": float(np.max(errors)),
            "boundary_frames": sum(boundary),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    receipt = {
        "schema_version": "eb/100style-rpo-retarget-trial/v1",
        "status": "succeeded",
        "source_manifest_sha256": digest(source / "source-manifest.json"),
        "license": "CC-BY-4.0",
        "asset_sha256": digest(assets / "asset-manifest.json"),
        "mjcf_sha256": digest(model_path),
        "recipe": "xsens-yxz-native23-limb-ground-bounded-ik-v2",
        "joint_order": list(RPO_LAB_JOINT_ORDER),
        "body_map": BODY_MAP,
        "clips": rows,
        "training_ready": False,
        "calibration": ("format T-pose axes; RPO native segment lengths, hip/shoulder anchors "
                        "and zero-joint foot rotations; 5th-percentile ankle plane"),
        "unchecked": ["Isaac physical tracking", "whole-clip coverage", "independent dataset splits"],
    }
    (output / "retarget-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retarget(args.source, args.assets, args.output)
