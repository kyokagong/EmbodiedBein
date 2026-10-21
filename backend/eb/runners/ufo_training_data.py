"""CPU-only native motion qualification and split freezing (Python 3.10)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def regular(path: Path, maximum: int = 256 * 1024 * 1024) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"missing, linked or oversized motion input: {path}")
    return path


def read_json(path: Path) -> dict:
    value = json.loads(regular(path, 4 * 1024 * 1024).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected metadata object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path.chmod(0o444)


def validate_groups(qualification: dict, provenance: dict) -> dict[str, set[str]]:
    groups = {}
    for split in ("train", "validation", "test"):
        values = qualification.get(f"{split}_groups")
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"invalid {split} source groups")
        groups[split] = set(values)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"source groups overlap: {left}/{right}")
    if set(provenance.get("splits", {}).get("test", {}).get("groups", [])) != groups["test"]:
        raise ValueError("test group declaration differs from original provenance")
    return groups


def validate_records(data: dict, split: str, groups: set[str], robot, provenance: dict) -> dict:
    import numpy as np
    from ufo.motion.formats import validate_ufo_motion_dict

    validate_ufo_motion_dict(data, split)
    audit = {row["name"]: row for row in provenance.get("audit", [])}
    episodes, actual_groups, fps_values = [], set(), set()
    for name, record in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError("native motion keys must be nonempty strings")
        metadata = record.get("metadata") or {}
        group = metadata.get("source_group")
        if group not in groups or metadata.get("split") != split:
            raise ValueError(f"motion {name} has an inconsistent source group/split")
        # Source names can themselves contain '__'; retain the importer identity.
        source = metadata.get("motion_key")
        if (
            source not in audit
            or audit[source].get("group") != group
            or not name.startswith(f"{source}__")
            or metadata.get("source_sha256") != audit[source].get("sha256")
            or metadata.get("source_path") != audit[source].get("source")
        ):
            raise ValueError(f"motion {name} has no matching original import audit")
        actual_groups.add(group)
        fps = float(np.asarray(record["fps"]).reshape(-1)[0])
        fps_values.add(fps)
        frames = len(record["root_trans_offset"])
        if frames < 2:
            raise ValueError(f"motion {name} requires at least two frames")
        widths = {
            "root_trans_offset": (frames, 3),
            "pose_aa": (frames, len(robot.body_names), 3),
            "dof_pos": (frames, len(robot.control_joint_names)),
            "root_quat": (frames, 4),
            "dof_vel": (frames, len(robot.control_joint_names)),
            "body_vel": (frames, len(robot.body_names), 3),
            "body_ang_vel": (frames, len(robot.body_names), 3),
        }
        for field, shape in widths.items():
            if field not in record:
                if field in {"root_trans_offset", "pose_aa", "dof_pos"}:
                    raise ValueError(f"motion {name} missing {field}")
                continue
            array = np.asarray(record[field])
            if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"motion {name} has nonfinite or incompatible {field} shape")
        joints = record.get("joint_names")
        if joints is not None and list(joints) != list(robot.control_joint_names):
            raise ValueError(f"motion {name} joint order differs from native RPO")
        if record.get("body_names") is not None and list(record["body_names"]) != list(robot.body_names):
            raise ValueError(f"motion {name} body order differs from native RPO")
        episodes.append(
            {
                "id": name,
                "frames": frames,
                "fps": fps,
                "duration_s": (frames - 1) / fps,
                "source_group": group,
                "source_motion": source,
            }
        )
    if actual_groups != groups:
        raise ValueError(f"{split} actual source groups differ from qualification")
    return {
        "episodes": episodes,
        "episodes_count": len(episodes),
        "frames": sum(row["frames"] for row in episodes),
        "duration_s": sum(row["duration_s"] for row in episodes),
        "fps_values": sorted(fps_values),
        "groups": sorted(actual_groups),
    }


def native_loader(path: Path, robot, extensions: list) -> dict:
    import torch
    from easydict import EasyDict
    from ufo.motion.library import MotionLib

    cfg = EasyDict(
        motion_file=str(path),
        asset={"assetRoot": str(Path(robot.xml_path).parent), "assetFileName": Path(robot.xml_path).name},
        extend_config=extensions,
    )
    library = MotionLib(cfg, num_envs=1, device="cpu")
    library.load_all_motions()
    for field in ("gts", "grs", "gavs", "gvs", "dvs", "dof_pos", "_motion_lengths"):
        if not bool(torch.isfinite(getattr(library, field)).all()):
            raise ValueError(f"native MotionLib generated nonfinite {field}")
    return {
        "implementation": "ufo.motion.library.MotionLib",
        "device": "cpu",
        "all_motions_loaded": True,
        "motions": library.num_motions(),
        "frames": int(library.gts.shape[0]),
    }


def prepare(
    source: Path,
    provenance_path: Path,
    ufo_root: Path,
    output: Path,
    storage_ref: str,
    dataset_id: str,
    source_uri: str,
) -> dict:
    import joblib
    from ufo.tools.robot_config import load_robot_training_spec

    output.mkdir(parents=True, exist_ok=True)
    if (output / "data").exists() or (output / "holdout").exists():
        raise ValueError("refusing to overwrite frozen motion outputs")
    qualification = read_json(source / "qualification.json")
    provenance = read_json(provenance_path)
    physics_path = source / "physics_audit/mask_feasibility.json"
    physics = read_json(physics_path)
    groups = validate_groups(qualification, provenance)
    robot_config = ufo_root / "configs/robots/rpo_23dof.yaml"
    native = load_robot_training_spec(robot_config)
    robot = native.robot
    if len(robot.control_joint_names) != 23 or provenance.get("joint_order") != list(
        robot.control_joint_names
    ):
        raise ValueError("original import joint order differs from native RPO configuration")
    if provenance.get("robot_xml_sha256") != digest(Path(robot.xml_path)):
        raise ValueError("original import robot XML differs from frozen native RPO")
    hashes = qualification.get("sha256", {})
    for path in (source / "train_clipped.pkl", source / "validation.pkl", physics_path):
        relative = f"runs/{Path(source_uri).name}/{path.relative_to(source)}"
        if hashes.get(relative) != digest(regular(path)):
            raise ValueError(f"motion qualification checksum differs: {relative}")
    loaded = {
        split: joblib.load(source / filename)
        for split, filename in (("train", "train_clipped.pkl"), ("validation", "validation.pkl"))
    }
    if set(loaded["train"]) & set(loaded["validation"]):
        raise ValueError("train and validation motion identities overlap")
    summaries = {
        split: validate_records(data, split, groups[split], robot, provenance)
        for split, data in loaded.items()
    }
    for split, field in (("train", "train_clips"), ("validation", "validation_segments")):
        if summaries[split]["episodes_count"] != qualification.get(field):
            raise ValueError(f"{split} motion count differs from qualification")
    limitations = list(
        dict.fromkeys(
            [
                "Official RPO source ancestry and licensing are not independently verified.",
                "Geometry and kinematic screening do not prove dynamic feasibility or learned "
                "behavior quality.",
                "Test content was not loaded; test isolation is checked against declared "
                "original source groups.",
                *qualification.get("limitations", []),
                *physics.get("limitations", []),
            ]
        )
    )
    provenance_dir = output / "provenance"
    provenance_dir.mkdir()
    receipts = {}
    for name, path in (
        ("qualification.json", source / "qualification.json"),
        ("import_report.json", provenance_path),
        ("mask_feasibility.json", physics_path),
    ):
        shutil.copyfile(path, provenance_dir / name)
        (provenance_dir / name).chmod(0o444)
        receipts[name] = digest(provenance_dir / name)
    for split, dirname, filename in (
        ("train", "data", "train.pkl"),
        ("validation", "holdout", "validation.pkl"),
    ):
        destination = output / dirname
        destination.mkdir()
        original = source / ("train_clipped.pkl" if split == "train" else "validation.pkl")
        shutil.copyfile(original, destination / filename)
        qualified_hash = hashes[f"runs/{Path(source_uri).name}/{original.name}"]
        if qualified_hash != digest(original) or qualified_hash != digest(destination / filename):
            raise ValueError("source motion changed while freezing")
        (destination / filename).chmod(0o444)
        probe = native_loader(destination / filename, robot, native.motion_extensions)
        manifest = {
            "schema_version": "eb/ufo-native-motion/v1",
            "dataset_id": dataset_id,
            "split": split,
            "train_file" if split == "train" else "validation_file": filename,
            "file_sha256": digest(destination / filename),
            "source_uri": source_uri,
            "source_filename": original.name,
            "robot_config_sha256": digest(robot_config),
            "robot_xml_sha256": digest(Path(robot.xml_path)),
            "joint_order": list(robot.control_joint_names),
            **summaries[split],
            "native_loader": probe,
            "provenance": receipts,
            "declared_test_groups": sorted(groups["test"]),
            "unchecked": limitations,
        }
        write_json(destination / "manifest.json", manifest)
    manifest_hash = digest(output / "data/manifest.json")
    report = {
        "format": "ufo_native_motion",
        "storage_ref": storage_ref,
        "source_refs": [{"kind": "ufo_native_motion", "uri": source_uri, "provenance_sha256": receipts}],
        "total_bytes": (output / "data/train.pkl").stat().st_size,
        **summaries["train"],
        "holdout": {
            **summaries["validation"],
            "storage_ref": str(Path(storage_ref).parent / "holdout"),
            "manifest_sha256": digest(output / "holdout/manifest.json"),
        },
        "validation": {
            "scope": "ufo_native_motion_v1",
            "status": "validated",
            "training_ready": True,
            "storage_ref": storage_ref,
            "manifest_sha256": manifest_hash,
            "errors": [],
            "unchecked": limitations,
        },
    }
    write_json(output / "dataset-index.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "provenance", "ufo-root", "output", "storage-ref", "dataset-id", "source-uri"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.ufo_root)
    os.chdir(args.ufo_root)
    output = Path(args.output)
    try:
        report = prepare(
            Path(args.source),
            Path(args.provenance),
            Path(args.ufo_root),
            output,
            args.storage_ref,
            args.dataset_id,
            args.source_uri,
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "dataset-index.json",
            {
                "format": "ufo_native_motion",
                "validation": {
                    "scope": "ufo_native_motion_v1",
                    "status": "invalid",
                    "training_ready": False,
                    "errors": [str(exc)],
                    "unchecked": [],
                },
            },
        )
        raise
    print(
        json.dumps(
            {
                "format": report["format"],
                "episodes_count": report["episodes_count"],
                "duration_s": report["duration_s"],
                "validation": report["validation"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
