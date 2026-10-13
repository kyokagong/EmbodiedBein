"""Derive RobotState from preserved raw, frozen RPO/G1 source views, never inverse codec."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

try:
    from eb.rpo_bfm import (
        BODY_ORDER,
        CODEC_HASH,
        CODEC_V2_HASH,
        ENV_V2_ID,
        G1_CONVERTER_REVISION,
        MIXED_POOL_CONVERTER_REVISION,
        REFERENCE_POOL_CONVERTER_REVISION,
        RPO_LAB_JOINT_ORDER,
    )
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer
except ModuleNotFoundError:
    from bfm_ufo_replay import UFOExpertSlicer
    from rpo_bfm import (
        BODY_ORDER,
        CODEC_HASH,
        CODEC_V2_HASH,
        ENV_V2_ID,
        G1_CONVERTER_REVISION,
        MIXED_POOL_CONVERTER_REVISION,
        REFERENCE_POOL_CONVERTER_REVISION,
        RPO_LAB_JOINT_ORDER,
    )


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_robot_state_arrays(arrays, *, frames):
    widths = {
        "root_pos": 3,
        "root_quat": 4,
        "q": 23,
        "dq": 23,
        "root_lin_vel_w": 3,
        "root_ang_vel_w": 3,
        "states": 52,
        "privileged_state": 358,
    }
    if not set(widths) <= arrays.keys() or "valid" not in arrays:
        raise ValueError("Full raw RobotState is required; codec-only views cannot reconstruct world root")
    for key, width in widths.items():
        value = arrays[key]
        if value.shape != (frames, width) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid raw RobotState {key} shape/dtype/finite values")
    if arrays["valid"].shape != (frames,) or arrays["valid"].dtype != bool:
        raise ValueError("Invalid raw RobotState validity")
    if not np.allclose(np.linalg.norm(arrays["root_quat"], axis=-1), 1, atol=1e-4):
        raise ValueError("RobotState quaternion must be normalized wxyz")


def validate_robot_state_manifest(manifest, *, asset_sha256, expected_codec_sha256=CODEC_HASH):
    if (
        manifest.get("schema_version") != "eb/rpo-robot-state/v1"
        or manifest.get("asset_sha256") != asset_sha256
        or manifest.get("codec_sha256") != expected_codec_sha256
        or manifest.get("quaternion") != "wxyz"
        or manifest.get("velocity_frame") != "world"
        or manifest.get("velocity_point") != "root_link_origin"
        or manifest.get("joint_order") != list(RPO_LAB_JOINT_ORDER)
        or manifest.get("body_order") != list(BODY_ORDER)
        or not manifest.get("parents")
        or not manifest.get("source_refs")
        or not manifest.get("clips")
    ):
        raise ValueError("RobotState identity/asset/codec/order/source differs")
    if expected_codec_sha256 not in {CODEC_HASH, CODEC_V2_HASH}:
        raise ValueError("Unknown frozen RobotState codec")
    if expected_codec_sha256 == CODEC_V2_HASH and (
        manifest.get("interface_id") != ENV_V2_ID
        or manifest.get("control_profile") != "reachable_v2"
        or (manifest.get("control_view") or {}).get("source_codec_sha256") != CODEC_HASH
        or not re.fullmatch(r"[0-9a-f]{64}", str((manifest.get("control_view") or {}).get("source_manifest_sha256", "")))
        or not any(parent.get("manifest_sha256") == manifest["control_view"]["source_manifest_sha256"]
                   and parent.get("codec_sha256") == CODEC_HASH for parent in manifest["parents"])
        or (manifest.get("control_view") or {}).get("preservation") != "NPZ bytes, raw q/dq, observations and source splits unchanged"
    ):
        raise ValueError("RobotState v2 requires explicit preserved v1 control-view lineage")
    groups, ids, files = {}, set(), set()
    for clip in manifest["clips"]:
        if (
            not clip.get("source_group")
            or clip.get("split") not in {"train", "validation", "test", "acceptance"}
            or clip.get("source") not in {"official_rpo", "g1"}
            or clip.get("fps") != 50
            or type(clip.get("frames")) is not int
            or clip["frames"] < 2
            or clip.get("id") in ids
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(clip.get("id", "")))
            or clip.get("file") in files
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.npz", str(clip.get("file", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(clip.get("sha256", "")))
        ):
            raise ValueError("Invalid or duplicate RobotState clip")
        group, split = clip["source_group"], clip["split"]
        if group in groups and groups[group] != split:
            raise ValueError("RobotState ancestor leaks across split")
        groups[group] = split
        ids.add(clip["id"])
        files.add(clip["file"])
    for source in manifest["source_refs"]:
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) or not source.get("license"):
            raise ValueError("RobotState source hash/license missing")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(parent.get("manifest_sha256", "")))
        for parent in manifest["parents"]
    ):
        raise ValueError("RobotState parent manifest hash missing")
    return manifest


def read_arrays(root, item):
    path = root / item["file"]
    if (
        root.is_symlink()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
        or path.stat().st_size > 256 * 1024**2
        or digest(path) != item["sha256"]
    ):
        raise ValueError("RobotState source path/size/hash differs")
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 256 * 1024**2:
            raise ValueError("RobotState expanded clip exceeds budget")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_robot_state(root: Path, *, asset_sha256: str, seed=0, expected_codec_sha256=CODEC_HASH):
    manifest_path = root / "manifest.json"
    if root.is_symlink() or manifest_path.is_symlink() or manifest_path.stat().st_size > 1024**2:
        raise ValueError("Invalid RobotState manifest path/size")
    manifest = validate_robot_state_manifest(json.loads(manifest_path.read_text()), asset_sha256=asset_sha256,
                                             expected_codec_sha256=expected_codec_sha256)
    clips = []
    for item in manifest["clips"]:
        arrays = read_arrays(root, item)
        validate_robot_state_arrays(arrays, frames=item["frames"])
        clips.append({**item, "arrays": arrays})
    return manifest, UFOExpertSlicer(clips, manifest_sha256=digest(manifest_path), seed=seed)


def verify_parent_codec(arrays):
    import torch

    try:
        from eb.runners.rpo_bfm_codec import encode_privileged, encode_state, quaternion_matrix
    except ModuleNotFoundError:
        from rpo_bfm_codec import encode_privileged, encode_state, quaternion_matrix

    body_keys = ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel")
    if not set(body_keys) <= arrays.keys():
        raise ValueError("Parent raw FK fields are required for codec reencoding")
    values = [torch.from_numpy(arrays[key]) for key in body_keys]
    rotation = quaternion_matrix(torch.from_numpy(arrays["root_quat"]))
    body_rotation = quaternion_matrix(values[1][:, 0])
    if not np.allclose(arrays["body_pos"][:, 0], arrays["root_pos"], atol=1e-5) or not torch.allclose(
        rotation, body_rotation, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Parent raw root/link frames differ from codec source")
    inverse = rotation.transpose(-1, -2)
    gravity = (inverse @ torch.tensor([0.0, 0.0, -1.0]).expand(len(rotation), 3)[..., None]).squeeze(-1)
    angular = (inverse @ torch.from_numpy(arrays["root_ang_vel_w"])[..., None]).squeeze(-1)
    state = encode_state(torch.from_numpy(arrays["q"]), torch.from_numpy(arrays["dq"]), gravity, angular)
    privileged = encode_privileged(*values)
    if not np.allclose(state.numpy(), arrays["states"], atol=1e-5, rtol=1e-5) or not np.allclose(
        privileged.numpy(), arrays["privileged_state"], atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Parent raw RobotState disagrees with frozen codec reencoding")


def convert_robot_state(source_manifest: Path, output: Path, *, asset_manifest: Path, control_profile="v1") -> dict:
    if output.exists() or source_manifest.is_symlink() or asset_manifest.is_symlink():
        raise ValueError("RobotState conversion requires new output and regular frozen sources")
    parent = json.loads(source_manifest.read_text())
    asset_sha256 = digest(asset_manifest)
    if control_profile == "reachable_v2":
        validate_robot_state_manifest(parent, asset_sha256=asset_sha256)
        staging = output.with_name(output.name + ".staging")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for item in parent["clips"]:
                arrays = read_arrays(source_manifest.parent, item)
                validate_robot_state_arrays(arrays, frames=item["frames"])
                shutil.copyfile(source_manifest.parent / item["file"], staging / item["file"])
                if digest(staging / item["file"]) != item["sha256"]:
                    raise ValueError("RobotState control-view copy hash differs")
            manifest = {**parent, "codec_sha256": CODEC_V2_HASH, "interface_id": ENV_V2_ID,
                "control_profile": "reachable_v2", "converter_revision": "robot-state-reachable-control-v2",
                "parents": [{"manifest_sha256": digest(source_manifest), "codec_sha256": CODEC_HASH,
                             "schema_version": parent["schema_version"]}, *parent["parents"]],
                "control_view": {"source_codec_sha256": CODEC_HASH,
                    "source_manifest_sha256": digest(source_manifest),
                    "preservation": "NPZ bytes, raw q/dq, observations and source splits unchanged"}}
            validate_robot_state_manifest(manifest, asset_sha256=asset_sha256, expected_codec_sha256=CODEC_V2_HASH)
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
            load_robot_state(staging, asset_sha256=asset_sha256, expected_codec_sha256=CODEC_V2_HASH)
            staging.rename(output)
            return manifest
        except Exception:
            shutil.rmtree(staging)
            raise
    if control_profile != "v1":
        raise ValueError("Unknown RobotState control profile")
    if (
        parent.get("schema_version") != "eb/rpo-bfm-motion/v1"
        or parent.get("asset_sha256") != asset_sha256
        or parent.get("converter_revision")
        not in {REFERENCE_POOL_CONVERTER_REVISION, G1_CONVERTER_REVISION, MIXED_POOL_CONVERTER_REVISION}
        or parent.get("output_quaternion") != "wxyz"
        or not parent.get("source_refs")
    ):
        raise ValueError("RobotState requires frozen official/calibrated G1 raw lineage")
    # These versions preserve original world root/q/FK arrays beside their codec view.
    # No pickle is loaded here and no discarded translation/yaw is reconstructed.
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        clips = []
        for item in parent["clips"]:
            arrays = read_arrays(source_manifest.parent, item)
            for key in ("root_pos", "root_quat", "q", "dq", "body_lin_vel", "body_ang_vel"):
                if key not in arrays:
                    raise ValueError("Full raw RobotState required; cannot inverse codec-only parent")
            arrays["root_lin_vel_w"] = arrays["body_lin_vel"][:, 0].copy()
            arrays["root_ang_vel_w"] = arrays["body_ang_vel"][:, 0].copy()
            validate_robot_state_arrays(arrays, frames=item["frames"])
            verify_parent_codec(arrays)
            group = item["source_group"]
            official = group == "official-rpo-unverified-ancestry"
            if not official and parent["converter_revision"] == REFERENCE_POOL_CONVERTER_REVISION:
                raise ValueError("Unknown official source ancestry")
            file = item["file"]
            np.savez(staging / file, **arrays)
            clips.append(
                {
                    **item,
                    "source": "official_rpo" if official else "g1",
                    "sha256": digest(staging / file),
                    "parent_clip_sha256": item["sha256"],
                }
            )
        manifest = {
            "schema_version": "eb/rpo-robot-state/v1",
            "asset_sha256": asset_sha256,
            "codec_sha256": CODEC_HASH,
            "joint_order": list(RPO_LAB_JOINT_ORDER),
            "body_order": list(BODY_ORDER),
            "quaternion": "wxyz",
            "velocity_frame": "world",
            "velocity_point": "root_link_origin",
            "velocity_method": parent.get("velocity_method"),
            "source_refs": parent["source_refs"],
            "parents": [
                {
                    "manifest_sha256": digest(source_manifest),
                    "converter_revision": parent["converter_revision"],
                }
            ],
            "clips": clips,
            "source_use": "local noncommercial research; no derived motion redistribution",
            "codec_reencoding": "verified against preserved parent FK/state fields",
        }
        validate_robot_state_manifest(manifest, asset_sha256=asset_sha256)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _, sampler = load_robot_state(staging, asset_sha256=asset_sha256)
        manifest["window_counts"] = {
            str(length): {source: len(windows) for source, windows in sources.items()}
            for length, sources in sampler.windows.items()
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--control-profile", choices=("v1", "reachable_v2"), default="v1")
    args = parser.parse_args()
    manifest = convert_robot_state(args.source_manifest, args.output, asset_manifest=args.asset_manifest,
                                   control_profile=args.control_profile)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "format": "rpo_ufo_robot_state",
                "storage_ref": str(args.output),
                "validation": {
                    "status": "validated",
                    "scope": "rpo_ufo_robot_state_v2" if manifest["codec_sha256"] == CODEC_V2_HASH else "rpo_ufo_robot_state_v1",
                    "manifest_sha256": digest(args.output / "manifest.json"),
                    "asset_sha256": manifest["asset_sha256"],
                    "codec_sha256": manifest["codec_sha256"],
                    "window_counts": manifest["window_counts"],
                    "training_ready": False,
                    "unchecked": ["RSI physics probe", "native learning qualification"],
                },
                "source_refs": manifest["source_refs"],
            },
            indent=2,
        )
    )
