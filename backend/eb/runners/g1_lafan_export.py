"""One bounded G1 walk export, with a source-model-derived FK-only URDF."""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

try:
    from eb.runners.g1_lafan_source import load_verified, sha, SOURCES
    from eb.runners.g1_lafan_bridge import verified_ufo_clip
except ModuleNotFoundError:
    from g1_lafan_source import load_verified, sha, SOURCES
    from g1_lafan_bridge import verified_ufo_clip


def source_model(xml_path):
    import mujoco
    xml = ET.parse(xml_path)
    root = xml.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"geom", "asset"}:
                parent.remove(child)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    jids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if model.nq != 36 or len(jids) != 29 or np.any(np.abs(model.jnt_pos[jids]) > 1e-8):
        raise ValueError("Fixed G1 source is not the expected zero-pivot 29-hinge model")
    return model, jids


def fk_urdf(model, output):
    import mujoco
    root = ET.Element("robot", name="ufo_g1_29dof_fk")
    for body in range(1, model.nbody):
        name = model.body(body).name
        ET.SubElement(root, "link", name=name)
        parent = int(model.body_parentid[body])
        if parent == 0:
            continue
        jids = range(int(model.body_jntadr[body]), int(model.body_jntadr[body]+model.body_jntnum[body]))
        hinge = [j for j in jids if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if len(hinge) > 1:
            raise ValueError("G1 FK URDF export only supports one hinge per source link")
        joint = ET.SubElement(root, "joint", name=model.joint(hinge[0]).name if hinge else name+"_fixed",
                              type="revolute" if hinge else "fixed")
        ET.SubElement(joint, "parent", link=model.body(parent).name)
        ET.SubElement(joint, "child", link=name)
        rpy = Rotation.from_quat(model.body_quat[body, [1, 2, 3, 0]]).as_euler("xyz")
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, model.body_pos[body])), rpy=" ".join(map(str, rpy)))
        if hinge:
            j = hinge[0]
            ET.SubElement(joint, "axis", xyz=" ".join(map(str, model.jnt_axis[j])))
            ET.SubElement(joint, "limit", lower=str(model.jnt_range[j, 0]), upper=str(model.jnt_range[j, 1]), effort="100", velocity="50")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def export(source, output, *, clip_id=None, frames=100, source_data=None):
    if frames not in {100,400}:
        raise ValueError("G1 export supports only the fixed trial/pool frame budgets")
    data = load_verified(source) if source_data is None else source_data
    keys = sorted(k for k in data if k.lower().startswith("walk") and k.endswith("_clip3"))
    if not keys:
        raise ValueError("G1 bounded trial cannot find a walk clip away from the initial pose")
    name = clip_id or keys[0]
    clip = data[name]
    model, jids = source_model(source / "g1_29dof.xml")
    names = [model.joint(j).name for j in jids]
    original = verified_ufo_clip(clip, names, model.jnt_axis[jids], limit_frames=500)
    q = original["joint_q"]
    source_fps = original["fps"]
    old_t = np.arange(len(q))/source_fps
    times = (2 if frames==100 else 1) + np.arange(frames)/50
    if times[-1] >= old_t[-1]:
        raise ValueError("G1 walk clip is too short for the frozen 2s..4s trial")
    root_rotation = Rotation.from_quat(q[:, 3:7])
    raw_rotation = np.asarray(clip["root_rot"])
    candidates = {"xyzw":raw_rotation, "wxyz":raw_rotation[:, [1, 2, 3, 0]]}
    errors = {key:float(np.max((Rotation.from_quat(value)*root_rotation.inv()).magnitude())) for key, value in candidates.items()}
    layout = min(errors, key=errors.get)
    if errors[layout] > 1e-4:
        raise ValueError("G1 root_rot and root pose_aa differ")
    interp = lambda values:np.column_stack([np.interp(times, old_t, c) for c in values.T])
    trajectory = np.column_stack((interp(q[:, :3]), Slerp(old_t, root_rotation)(times).as_quat(), interp(q[:, 7:]))).astype(np.float32)
    if not np.isfinite(trajectory).all():
        raise ValueError("G1 exported trajectory is not finite")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "source_joint_trajectory.npz"
    np.savez(path, joint_q=trajectory, dof_names=np.asarray(names), fps=50., root_quat_format="xyzw")
    fk_urdf(model, output / "g1_fk.urdf")
    (output / "g1_29dof.xml").write_bytes((source / "g1_29dof.xml").read_bytes())
    receipt = {"schema_version":"eb/g1-trajectory-export/v1", "status":"succeeded",
        "clip_id":name, "source_group":re.sub(r"_clip\d+$", "", name),
        "source_fps":source_fps, "fps":50, "frames":frames, "source_time_range_s":[float(times[0]),float(times[-1])],
        "root_rot_layout_verified":layout, "root_rotation_error_rad":errors[layout],
        "source_joint_order":names, "source_xml_sha256":SOURCES["g1_29dof.xml"]["sha256"],
        "source_manifest_sha256":sha(source / "source-manifest.json"),
        "trajectory_sha256":sha(path), "fk_urdf_sha256":sha(output / "g1_fk.urdf"),
        "source_refs":list(SOURCES.values()), "source_use":"local_noncommercial_no_redistribution",
        "unchecked":["source_visual_meshes_and_sole_geometry", "physical_controller"]}
    (output / "export-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.source, a.output)
