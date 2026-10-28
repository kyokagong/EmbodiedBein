"""Frozen RPO adaptation of P-BFM, applied only to a task-private source tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

CAMERA_CONTRACT = {
    "schema_version": "eb/pbfm-rpo-camera/v1",
    "simulation_only": True,
    "mount_body": "torso_link",
    "mount_pos": [0.08, 0.0, 0.10],
    "camera_frame_quat_wxyz": [0.9914448613738104, 0.0, 0.13052619222005157, 0.0],
    "frame_convention": "world",
    "depth_convention": "optical_top_down_v1",
    "processing_profile": "optical_z_maxcrop18_reset_v2",
    "source_shape": [36, 64],
    "crop": [18, 0, 16, 16],
    "output_shape": [8, 18, 32],
    "sampled_ages": [35, 30, 25, 20, 15, 10, 5, 0],
    "policy_hz": 50,
    "horizontal_fov_deg": 89.51,
    "vertical_fov_deg": 58.29,
    "near_m": 0.0,
    "far_m": 2.5,
    "invalid_depth_m": 2.5,
    "include_geom_groups": [0, 2, 5],
    "exclude_parent_body": False,
    "gaussian_sigma": 1.0,
    "image_randomization": False,
    "position_error_m": 0.01,
    "rpy_error_deg": [2.0, 10.0, 2.0],
    "installation_frame": "camera_link",
    "installation_resample": "reset",
    "delay_frames": [0, 1],
    "dropout": "none",
    "reset_history": "fill_current_frame",
    "storage_dtype": "uint8",
    "onnx_input": "float32_uint8_div_255_once",
}
TERRAIN_CONTRACT = {
    "schema_version": "eb/pbfm-rpo-terrain/v1",
    "mode": "training_mix",
    "curriculum_start_global_steps": 0,
    "curriculum_interval_global_steps": 19200000,
    "curriculum_clock": "global_main_transitions",
    "curriculum_apply": "reset_only",
    "families": [
        "flat",
        "perlin_rough",
        "low_stairs_up",
        "low_stairs_down",
        "low_platforms",
        "hf_pyramid_slope_inv",
        "boxes",
    ],
    "geometry_profile": "pbfm_training_mix_v1",
    "boundary_reset_is_success": False,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_root() -> Path:
    bundled = Path("/opt/eb-patches/pbfm")
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[3] / "deploy/patches/pbfm"


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("P-BFM source symlinks are forbidden")
        if path.is_file():
            result[relative.as_posix()] = file_sha256(path)
    return result


def rpo_robot_config(source: Path) -> Path:
    return Path(source) / "pbfm/configs/robots/rpo_23dof.yaml"


def prepare_pbfm_source(source: Path, destination: Path) -> dict:
    """Fail closed on upstream drift or reuse, without writing to the source."""
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if any(p.is_symlink() for p in (source, *source.parents, destination, *destination.parents)):
        raise ValueError("P-BFM source/destination symlink is forbidden")
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("P-BFM private source destination must be new and separate")
    patches = patch_root()
    frozen = json.loads((patches / "upstream-manifest.json").read_text())
    actual = _files(source)
    if actual != frozen["files"]:
        raise ValueError("P-BFM frozen upstream content differs; review a new adaptation")
    patch_files = _files(patches)
    patch_digest = hashlib.sha256(
        json.dumps(patch_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache")
    )
    try:
        operations = json.loads((patches / "replacements.json").read_text())
        for operation in operations:
            path = destination / operation["path"]
            content = path.read_text()
            if content.count(operation["before"]) != 1:
                raise ValueError("Frozen P-BFM patch hunk does not apply exactly once")
            path.write_text(content.replace(operation["before"], operation["after"], 1))
        for name, relative in (
            ("rpo_23dof.yaml", "pbfm/configs/robots/rpo_23dof.yaml"),
            ("rpo-hydra.yaml", "pbfm/configs/simulation/robot/rpo/rpo_23dof.yaml"),
            ("rpo_23dof.xml", "pbfm/data/robots/rpo/rpo_23dof.xml"),
            ("rpo_geometry.py", "pbfm/robots/rpo_geometry.py"),
        ):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(patches / name, target)
        robot = rpo_robot_config(destination)
        robot.write_text(
            robot.read_text().replace(
                "EB_PRIVATE_RPO_XML", str(destination / "pbfm/data/robots/rpo/rpo_23dof.xml")
            )
        )
        result = {
            "schema_version": "eb/pbfm-source-adaptation/v1",
            "upstream_revision": frozen["revision"],
            "upstream_files": actual,
            "patch_sha256": patch_digest,
            "patch_files": patch_files,
            "camera_contract": CAMERA_CONTRACT,
            "terrain_contract": TERRAIN_CONTRACT,
            "result_files": _files(destination),
        }
        (destination / ".eb-pbfm-source.json").write_text(json.dumps(result, sort_keys=True, indent=2))
        return result
    except BaseException:
        # Keep the failed private tree for diagnosis; never make it importable.
        (destination / ".eb-adaptation-failed").touch()
        raise
