"""RPO RobotState/UFO -> native PBFM motion, in an isolated CPU Job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from eb.pbfm_data import SCHEMA, validate_data_plan
from eb.pbfm_io import file_index, read_json, safe_file, sha256, write_json


def validate_records(records, robot, ancestry: set[str]) -> dict:
    import numpy as np
    from pbfm.motion.data.schema import validate_ufo_motion_dict

    validate_ufo_motion_dict(records, "RPO PBFM")
    seen = set()
    frames = 0
    duration = 0.0
    for name, row in records.items():
        metadata = row.get("metadata") or {}
        ancestor = metadata.get("ancestor_id") or metadata.get("source_group")
        if ancestor not in ancestry:
            raise ValueError(f"{name} lacks matching declared original ancestry")
        seen.add(ancestor)
        if list(row.get("joint_names", [])) != list(robot.control_joint_names):
            raise ValueError("motion joint order differs from frozen RPO")
        if list(row.get("body_names", [])) != list(robot.body_names):
            raise ValueError("motion body order differs from frozen RPO")
        n = len(row["root_trans_offset"])
        if n < 2:
            raise ValueError("motion requires two or more frames")
        for key, shape in (
            ("pose_aa", (n, len(robot.body_names), 3)),
            ("dof_pos", (n, 23)),
            ("root_trans_offset", (n, 3)),
            ("root_quat", (n, 4)),
        ):
            value = np.asarray(row.get(key))
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"nonfinite or incorrect RPO {key} shape")
        quats = np.asarray(row["root_quat"])
        if not np.allclose(np.linalg.norm(quats, axis=1), 1, atol=1e-3, rtol=0):
            raise ValueError("RPO root quaternions must be normalized xyzw")
        fps = float(row["fps"])
        if not 1 <= fps <= 1000:
            raise ValueError("invalid motion fps")
        frames += n
        duration += (n - 1) / fps
    if seen != ancestry:
        raise ValueError("actual source ancestry differs from frozen declaration")
    return {
        "episodes_count": len(records),
        "frames": frames,
        "duration_s": duration,
        "ancestor_ids": sorted(seen),
    }


def qualify_native(path: Path, records: dict, native) -> tuple[dict, dict]:
    import mujoco
    import numpy as np
    import torch
    from easydict import EasyDict
    from pbfm.motion.library.motion_lib_robot import MotionLibRobot

    robot = native.robot
    cfg = EasyDict(
        motion_file=str(path),
        asset={"assetRoot": str(Path(robot.xml_path).parent), "assetFileName": Path(robot.xml_path).name},
        extend_config=native.to_env_dict()["motion_extensions"],
    )
    library = MotionLibRobot(cfg, num_envs=1, device="cpu")
    library.load_all_motions()
    for key in ("gts", "grs", "gvs", "gavs", "dvs", "dof_pos", "_motion_lengths"):
        if not torch.isfinite(getattr(library, key)).all():
            raise ValueError(f"native MotionLib produced nonfinite {key}")
    model = mujoco.MjModel.from_xml_path(str(robot.xml_path))
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in robot.body_names]
    if min(body_ids) < 0:
        raise ValueError("RPO body missing from frozen XML")
    max_error, count = 0.0, 0
    for record in records.values():
        pose = torch.as_tensor(np.asarray(record["pose_aa"]), dtype=torch.float32)[None]
        root = torch.as_tensor(np.asarray(record["root_trans_offset"]), dtype=torch.float32)[None]
        fk = library.mesh_parsers.fk_batch(pose, root, return_full=False).global_translation[0].numpy()
        for i, position in enumerate(record["root_trans_offset"]):
            data.qpos[:3] = position
            data.qpos[3:7] = np.asarray(record["root_quat"])[i][[3, 0, 1, 2]]
            for index, joint in enumerate(robot.control_joint_names):
                data.qpos[robot.joint_qpos_addr[joint]] = record["dof_pos"][i, index]
            mujoco.mj_forward(model, data)
            error = float(np.abs(data.xpos[body_ids] - fk[i, : len(body_ids)]).max())
            if not np.isfinite(error) or error > 2e-4:
                raise ValueError(f"native MotionLib/MuJoCo FK disagrees: {error}")
            max_error = max(max_error, error)
            count += 1
    return (
        {
            "implementation": "pbfm.motion.library.motion_lib_robot.MotionLibRobot",
            "device": "cpu",
            "all_motions_loaded": True,
            "motions": int(library.num_motions()),
            "frames": int(library.gts.shape[0]),
        },
        {
            "verified": True,
            "implementation": "native Torch FK versus frozen MuJoCo FK",
            "all_frames": True,
            "frames_checked": count,
            "max_abs_error_m": max_error,
            "tolerance_m": 2e-4,
        },
    )


def prepare(
    plan: dict,
    dataset_id: str,
    storage_ref: str,
    output: Path = Path("/outputs"),
    source: Path = Path("/motion-source"),
    pbfm_source: Path = Path("/sources/pbfm"),
) -> dict:
    import joblib

    validate_data_plan(plan)
    from eb.runners.pbfm_source import prepare_pbfm_source, rpo_robot_config
    from eb.runners.rlinf_pbfm_runtime import verify_source_refs

    verify_source_refs(
        {
            "source_refs": {
                name: {
                    **ref,
                    "path": "/sources/pbfm" if name == "pbfm" else "/robot-assets",
                    "manifest_path": f"/sources/{name}-manifest.json",
                }
                for name, ref in plan["source_refs"].items()
            }
        }
    )

    private = output / "runtime/pbfm"
    source_receipt = prepare_pbfm_source(pbfm_source, private)
    sys.path.insert(0, str(private))
    os.chdir(private)
    from pbfm.motion.data.robot_state_convert import robot_state_dict_to_ufo_motion_dict
    from pbfm.motion.data.robot_state_readers import read_robot_state_npz
    from pbfm.robots import load_robot_training_spec

    native = load_robot_training_spec(rpo_robot_config(private))
    robot = native.robot
    if list(robot.control_joint_names) != plan["source_manifest"]["joint_order"] or robot.name not in {
        "rpo",
        "rpo_23dof",
    }:
        raise ValueError("source manifest differs from frozen native RPO")
    summaries, manifests, all_keys = {}, {}, set()
    bridge = plan["source_manifest"].get("bridge")
    selection = (
        {
            "mode": "registered_ufo_groups",
            "parent_dataset_id": plan["parent_dataset_id"],
            "parent_run_id": plan["parent_run_id"],
            "split_groups": bridge["split_groups"],
            "excluded_groups": bridge["excluded_groups"],
            "qualification": "diagnostic",
            "formal_acceptance": False,
        }
        if bridge
        else None
    )
    # Freeze held-out manifests first so the training manifest binds their bytes
    # without mounting their contents in training.
    for split in ("validation", "acceptance", "train"):
        records, ancestors = {}, set()
        for row in plan["source_manifest"]["splits"][split]:
            path = safe_file(source, row["path"])
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("frozen source bytes changed before conversion")
            ancestry = set(row["ancestor_ids"])
            if row["format"] in {"ufo_pickle", "ufo_grouped_pickle"}:
                loaded = joblib.load(path)
                if row["format"] == "ufo_grouped_pickle":
                    from eb.pbfm_ufo_bridge import select_grouped_records

                    loaded = select_grouped_records(loaded, row, split)
            else:
                raw = read_robot_state_npz(str(path), source_name=row["path"], robot_spec=robot)
                if len(ancestry) != 1:
                    raise ValueError("RobotState NPZ requires exactly one declared original ancestry")
                loaded = robot_state_dict_to_ufo_motion_dict(raw, robot, row["path"])
                for record in loaded.values():
                    record["metadata"]["ancestor_id"] = next(iter(ancestry))
            validate_records(loaded, robot, ancestry)
            if set(loaded) & (all_keys | records.keys()):
                raise ValueError("motion identities overlap across source files/splits")
            records.update(loaded)
            ancestors.update(ancestry)
        all_keys.update(records)
        summaries[split] = validate_records(records, robot, ancestors)
        root = output / ("data" if split == "train" else "holdout" if split == "validation" else "acceptance")
        root.mkdir(parents=True, exist_ok=False)
        filename = "train.pkl" if split == "train" else split + ".pkl"
        joblib.dump(records, root / filename)
        loader, fk = qualify_native(root / filename, records, native)
        manifest = {
            "schema_version": SCHEMA,
            "dataset_id": dataset_id,
            "split": split,
            "train_file" if split == "train" else "motion_file": filename,
            "file_sha256": sha256(root / filename),
            "joint_order": list(robot.control_joint_names),
            "body_order": list(robot.body_names),
            "robot_name": robot.name,
            "robot_config_sha256": sha256(Path(native.config_path)),
            "pbfm_patch_sha256": source_receipt["patch_sha256"],
            "pbfm_snapshot": plan["source_refs"]["pbfm"]["snapshot_hash"],
            "asset_snapshot": plan["source_refs"]["asset"]["snapshot_hash"],
            "robot_xml_sha256": sha256(Path(robot.xml_path)),
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "parent_version_id": plan["parent_version_id"],
            "source_selection": selection,
            "holdout_manifests": {key: item["sha256"] for key, item in manifests.items()}
            if split == "train"
            else {},
            "native_loader": loader,
            "fk": fk,
            **summaries[split],
        }
        write_json(root / "manifest.json", manifest)
        manifests[split] = {
            "sha256": sha256(root / "manifest.json"),
            "relative_path": str(root.relative_to(output) / "manifest.json"),
        }
    train = read_json(output / "data/manifest.json")
    report = {
        "dataset_id": dataset_id,
        "format": "pbfm_rpo_motion",
        "storage_ref": storage_ref,
        "source_selection": selection,
        **summaries["train"],
        "splits": manifests,
        "validation": {
            "scope": "pbfm_rpo_motion_v1",
            "status": "validated",
            "training_ready": True,
            "manifest_sha256": manifests["train"]["sha256"],
            "file_sha256": train["file_sha256"],
            "errors": [],
            "unchecked": [
                "Kinematics and native loading do not establish policy quality or source licensing."
            ],
        },
    }
    write_json(output / "dataset-index.json", report)
    write_json(
        output / "data-receipt.json",
        {
            "schema_version": "eb/pbfm-data-receipt/v1",
            "operation": plan["operation"],
            "plan_hash": plan["hash"],
            "run_id": os.environ["EB_RUN_ID"],
            "dataset_id": dataset_id,
            "source_receipt": source_receipt,
            "files": file_index(output),
        },
    )
    return report


def main():
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--storage-ref", required=True)
    args = parser.parse_args()
    prepare(json.loads(args.plan_json), args.dataset_id, args.storage_ref)


if __name__ == "__main__":
    main()
