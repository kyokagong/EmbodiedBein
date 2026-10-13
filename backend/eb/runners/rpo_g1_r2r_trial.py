"""Pinned hhtools CPU R2R: one G1 walk → raw RPO reference, never a controller."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from eb.rpo_bfm import BODY_ORDER, RPO_LAB_JOINT_ORDER
    from eb.runners.rpo_bfm_motion import digest, geometry_screen, reference_mask
    from eb.runners.g1_lafan_export import source_model
except ModuleNotFoundError:
    from rpo_bfm import BODY_ORDER, RPO_LAB_JOINT_ORDER
    from rpo_bfm_motion import digest, geometry_screen, reference_mask
    from g1_lafan_export import source_model

HHTOOLS_REVISION = "1181cc98593dbef20b23f92ebbf9ad38ba6a1e7e"
SOURCE_MAP = {"hips":"pelvis", "chest":"torso_link", **{
    f"{side}_{key}":f"{side}_{body}_link" for side in ("left", "right")
    for key, body in (("hip", "hip_pitch"), ("knee", "knee"), ("ankle", "ankle_roll"),
                      ("shoulder", "shoulder_pitch"), ("elbow", "elbow"), ("wrist", "wrist_yaw"))}}
TARGET_MAP = {"hips":"base_link", "chest":"torso_link", **{
    f"{side}_{key}":f"{side}_{body}_link" for side in ("left", "right")
    for key, body in (("hip", "thigh_pitch"), ("knee", "knee"), ("ankle", "ankle_roll"),
                      ("shoulder", "arm_pitch"), ("elbow", "elbow_pitch"), ("wrist", "elbow_yaw"))}}


def fk_error(robot, model, frames, names):
    from hhtools.retarget.calibration.calibration import _collect_link_transforms_at_q
    data = mujoco.MjData(model)
    addresses = model.jnt_qposadr[[model.joint(n).id for n in names]]
    error = 0.
    for frame in frames:
        transforms = _collect_link_transforms_at_q(robot, dict(zip(names, frame[7:])))
        data.qpos[:3], data.qpos[3:7] = frame[:3], frame[[6, 3, 4, 5]]
        data.qpos[addresses] = frame[7:]
        mujoco.mj_forward(model, data)
        rot = Rotation.from_quat(frame[3:7]).as_matrix()
        for name, transform in transforms.items():
            body = model.body(name)
            error = max(error, float(np.linalg.norm(rot @ transform[:3, 3]+frame[:3]-data.xpos[body.id])))
    return error


def retarget(source, assets, output, report, *, asset_sha256, frames=100):
    from hhtools.robot.base import RobotPreset
    from hhtools.robot.loader import load_robot
    from hhtools.retarget import robot_to_robot as r2r

    if output.exists() or digest(assets / "asset-manifest.json") != asset_sha256:
        raise ValueError("G1 retarget refuses output overwrite or changed RPO asset")
    receipt = json.loads((source / "export-receipt.json").read_text())
    if (receipt.get("schema_version") != "eb/g1-trajectory-export/v1"
            or frames not in {100,400} or receipt.get("status") != "succeeded" or receipt.get("frames") != frames
            or receipt.get("source_use") != "local_noncommercial_no_redistribution"):
        raise ValueError("G1 retarget needs a verified bounded trajectory export")
    for name, key in (("source_joint_trajectory.npz", "trajectory_sha256"), ("g1_fk.urdf", "fk_urdf_sha256"),
                      ("g1_29dof.xml", "source_xml_sha256")):
        path = source / name
        if path.is_symlink() or path.stat().st_size > 1024*1024 or digest(path) != receipt.get(key):
            raise ValueError("G1 retarget source bytes differ from export receipt")
    output.mkdir(parents=True, exist_ok=False)
    urdf = ET.parse(assets / "rpo_description/urdf/rpo.urdf")
    for mesh in urdf.getroot().iter("mesh"):
        path = assets / "rpo_description/meshes" / Path(mesh.get("filename")).name
        if not path.is_file():
            raise ValueError("Missing frozen RPO mesh")
        mesh.set("filename", str(path))
    target_urdf = output / "rpo_fk.urdf"
    urdf.write(target_urdf, encoding="utf-8", xml_declaration=True)
    source_robot = load_robot(RobotPreset("g1_29_trial", "G1 FK", source, source / "g1_fk.urdf",
        ik_map=SOURCE_MAP, dof_order=tuple(receipt["source_joint_order"])), compile_mjcf=False)
    xml = ET.parse(assets / "rpo_description/mjcf/rpo.xml").getroot()
    # hhtools IM packs hinges in MJCF order, regardless of the preset's DOF names.
    target_dof_order = tuple(j.get("name") for j in xml.find("worldbody").iter("joint") if j.get("type") != "free")
    if len(target_dof_order) != 23 or set(target_dof_order) != set(RPO_LAB_JOINT_ORDER):
        raise ValueError("Frozen target MJCF is not RPO23")
    target_robot = load_robot(RobotPreset("rpo23_trial", "RPO V2", output, target_urdf,
        ik_map=TARGET_MAP, dof_order=target_dof_order), compile_mjcf=False)
    xml.find("compiler").set("meshdir", str(assets / "rpo_description/meshes"))
    target_robot.mjcf_xml = ET.tostring(xml, encoding="unicode")
    target_robot.mujoco_model = model = mujoco.MjModel.from_xml_string(target_robot.mjcf_xml)
    if tuple(model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE) != target_dof_order:
        raise ValueError("hhtools packed hinge names differ from MuJoCo declaration order")
    limits_by_name = {j.get("name"):j.find("limit") for j in urdf.getroot().findall("joint") if j.find("limit") is not None}
    limits = np.array([[float(limits_by_name[n].get(k)) for k in ("lower", "upper")] for n in RPO_LAB_JOINT_ORDER])
    jids = np.array([model.joint(n).id for n in RPO_LAB_JOINT_ORDER])
    model.jnt_range[jids] = limits
    trajectory = r2r.load_source_trajectory(source / "source_joint_trajectory.npz", source_model=source_robot)
    if (tuple(trajectory.dof_names) != tuple(receipt["source_joint_order"])
            or trajectory.joint_q.shape != (frames, 36) or trajectory.framerate != 50):
        raise ValueError("hhtools source order/frames/frequency mismatch")
    g1_model, _ = source_model(source / "g1_29dof.xml")
    # Preserve source collision primitives so R2R height does not fall back to 1.3m.
    g1_xml = ET.parse(source / "g1_29dof.xml").getroot()
    for parent in g1_xml.iter():
        for child in list(parent):
            if child.tag == "mesh" or (child.tag == "geom" and child.get("mesh")):
                parent.remove(child)
    source_robot.mjcf_xml = ET.tostring(g1_xml, encoding="unicode")
    source_robot.mujoco_model = mujoco.MjModel.from_xml_string(source_robot.mjcf_xml)
    source_error = fk_error(source_robot, g1_model, trajectory.joint_q[::10], trajectory.dof_names)
    target_zero = np.tile(np.r_[0., 0., 1., 0., 0., 0., 1., np.zeros(23)], (1, 1))
    target_error = fk_error(target_robot, model, target_zero, RPO_LAB_JOINT_ORDER)
    if source_error > 1e-5 or target_error > .002:
        raise ValueError(f"Source/target URDF FK differs from frozen models: {source_error}, {target_error}")
    from g1_rpo_calibration import calibrate,retarget_calibrated
    calibration, calibration_receipt = calibrate(source_robot,target_robot)
    calibration_receipt.update(source="g1_29_trial", target="rpo23_trial", asset_sha256=asset_sha256,
        source_export_sha256=digest(source / "export-receipt.json"),
        source_fk_error_m=source_error, target_fk_error_m=target_error)
    (output / "r2r-calibration.json").write_text(json.dumps(calibration_receipt, indent=2))
    motion = r2r.source_trajectory_to_motion(source_robot, trajectory.joint_q, trajectory.dof_names,
        framerate=50, name=receipt["clip_id"])
    def progress(cur, total):
        if cur % 20 == 0 or cur == total:
            print(json.dumps({"retarget_frame":cur, "total":total}), flush=True)
    result,solver_calibration = retarget_calibrated(source_robot,target_robot,motion,calibration,progress)
    dofs = tuple(result.dof_names)
    if len(dofs) != 23 or set(dofs) != set(RPO_LAB_JOINT_ORDER):
        raise ValueError("hhtools target is not the native RPO23 joint set")
    raw = np.asarray(result.joint_q, dtype=np.float64)
    if raw.shape != (frames, 30) or not np.isfinite(raw).all() or not np.allclose(np.linalg.norm(raw[:, 3:7], axis=-1), 1, atol=1e-4):
        raise ValueError("hhtools output shape/finite/quaternion mismatch")
    q = raw[:, [7+dofs.index(n) for n in RPO_LAB_JOINT_ORDER]].astype(np.float32)
    roots, rotations = raw[:, :3].astype(np.float32), raw[:, [6, 3, 4, 5]].astype(np.float32)
    data = mujoco.MjData(model)
    positions, quaternions = [], []
    bids = [model.body(n).id for n in BODY_ORDER]
    for joints, pos, quat in zip(q, roots, rotations):
        data.qpos[:3], data.qpos[3:7], data.qpos[model.jnt_qposadr[jids]] = pos, quat, joints
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[bids].copy())
        quaternions.append(data.xquat[bids].copy())
    positions = np.asarray(positions, np.float32)
    dq = np.gradient(q, .02, axis=0)
    up = Rotation.from_quat(rotations[:, [1, 2, 3, 0]]).as_matrix()[:, 2, 2]
    valid = reference_mask(q, dq, roots[:, 2], up, limits)
    geometry_valid, quality = geometry_screen(model, data, q, roots, rotations, positions, RPO_LAB_JOINT_ORDER, BODY_ORDER)
    valid &= geometry_valid
    windows = int((np.convolve(valid.astype(int), np.ones(9, int), mode="valid") == 9).sum())
    path = output / "rpo_g1_walk.npz"
    np.savez(path, q=q, dq=dq, root_pos=roots, root_quat=rotations, fps=50,
        joint_names=np.asarray(RPO_LAB_JOINT_ORDER), body_names=np.asarray(BODY_ORDER),
        body_pos=positions, body_quat=np.asarray(quaternions, np.float32), valid=valid,
        timestamps=np.arange(frames, dtype=np.float64)/50)
    fk_after = fk_error(target_robot, model, np.column_stack((raw[:, :7], q))[::10], RPO_LAB_JOINT_ORDER)
    if fk_after > .002:
        raise ValueError("Retargeted RPO FK differs from the native model")
    result_receipt = {"schema_version":"eb/rpo-g1-r2r-trial/v1", "status":"succeeded",
        "hhtools_revision":HHTOOLS_REVISION, "runtime_recipe":"pinned_cpu_subset_python311_v1",
        "backend":"hhtools_interaction_mesh", "asset_sha256":asset_sha256,
        "solver_iterations":{"sqp_inner_iters":3, "first_frame_sqp_iters":5},
        "source_export_sha256":digest(source / "export-receipt.json"), "source_refs":receipt["source_refs"],
        "source_clip_id":receipt["clip_id"], "source_group":receipt["source_group"],
        "source_use":"local_noncommercial_no_redistribution", "joint_order":list(RPO_LAB_JOINT_ORDER),
        "solver_joint_order":list(target_dof_order),
        "frames":frames, "fps":50, "trajectory_sha256":digest(path), "valid_frames":int(valid.sum()),
        "continuous_9frame_starts":windows, "source_fk_error_m":source_error,
        "target_fk_error_m":fk_after, "geometry":quality,
        "calibration_verified":True, "calibration_sha256":digest(output / "r2r-calibration.json"),
        **solver_calibration,
        "qualification":"diagnostic_reference", "training_ready":False,
        "unchecked":["BFM_Isaac_FK_codec_view", "source_visual_meshes", "physical_hand_tip_and_sole_alignment",
                     "independent_holdouts", "controller_evaluation", "full_hhtools_application"]}
    (output / "retarget-receipt.json").write_text(json.dumps(result_receipt, indent=2))
    files = list(output.iterdir())
    report.write_text(json.dumps({"format":"rpo_g1_joint_trajectory", "clips_count":1, "frames_count":frames,
        "total_bytes":sum(p.stat().st_size for p in files), "source_refs":receipt["source_refs"],
        "retarget":result_receipt, "validation":{"scope":"rpo_g1_r2r_trial_v1", "status":"validated",
        "storage_ref":str(output), "training_ready":False, "retarget_candidate_ready":windows > 0,
        "errors":[], "unchecked":result_receipt["unchecked"]}}, indent=2))
    print(json.dumps(result_receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "assets", "output", "report"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--asset-sha256", required=True)
    a = p.parse_args()
    retarget(a.source, a.assets, a.output, a.report, asset_sha256=a.asset_sha256)
