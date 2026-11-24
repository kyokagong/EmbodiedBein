"""Pinned ChingMu raw motion acquisition; executed only in a data task container."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "CMRobot/MotionDecode"
REVISION = "6380f7420c83b20b6ad50124bc6cb829a7c0b178"
ENV_ID = "ee_g1_motiondecode_reference_29dof_v1"
EXPECTED_FILES = 92567
JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
HEADER = [
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
] + [f"dof_{j}(rad)" for j in JOINTS]


def safe_member(root: Path, name: str) -> Path:
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or relative.parts[0] in {".cache", ".embodiedbein"}
    ):
        raise ValueError(f"unsafe repository member: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in (path, *path.parents)
    ):
        raise ValueError(f"linked repository member: {name}")
    return path


def verify_file(root: Path, info: dict) -> str:
    path = safe_member(root, info["path"])
    if not path.is_file() or path.stat().st_size != info["size"]:
        raise ValueError(f"file size mismatch: {info['path']}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {info['size']}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
            git.update(chunk)
    expected, actual = (
        (info["lfs_sha256"], sha.hexdigest())
        if info.get("lfs_sha256")
        else (info["blob_id"], git.hexdigest())
    )
    if expected != actual:
        raise ValueError(f"file hash mismatch: {info['path']}")
    return sha.hexdigest()


def read_motion(path: Path):
    import numpy as np

    with path.open(newline="", encoding="utf-8-sig") as source:
        if next(csv.reader(source), []) != HEADER:
            raise ValueError("motion header does not match the 36-column source schema")
        data = np.loadtxt(source, delimiter=",", ndmin=2)
    if data.size == 0:
        raise ValueError("empty motion")
    if data.shape[1] != 36:
        raise ValueError("motion must have 36 columns")
    if not np.isfinite(data).all():
        raise ValueError("motion contains non-finite values")
    if not np.all(np.abs(np.linalg.norm(data[:, 3:7], axis=1) - 1) <= 0.01):
        raise ValueError("motion contains a non-unit quaternion (tolerance 0.01)")
    return data


def inspect_motion(path: Path) -> dict:
    return {"length": len(read_motion(path)), "columns": 36}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


def acquire(root: Path, output: Path, *, local_only: bool = False) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if root != root.resolve() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("download root must be canonical and unlinked")
    state = root / ".embodiedbein"
    if state.is_symlink():
        raise ValueError("linked acquisition state")
    state.mkdir(exist_ok=True)
    manifest_path = state / "source-manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("linked source manifest")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("repo_id") != REPO or manifest.get("revision") != REVISION:
            raise ValueError("destination belongs to a different source revision")
    else:
        if local_only:
            raise ValueError("local indexing requires the frozen source manifest")
        for attempt in range(5):
            try:
                info = HfApi(token=False).dataset_info(REPO, revision=REVISION, files_metadata=True)
                if info.sha != REVISION:
                    raise ValueError("HF returned a different source revision")
                manifest = {
                    "repo_id": REPO,
                    "revision": REVISION,
                    "files": [
                        {
                            "path": f.rfilename,
                            "size": f.size,
                            "blob_id": f.blob_id,
                            "lfs_sha256": f.lfs.sha256 if f.lfs else None,
                        }
                        for f in info.siblings
                    ],
                }
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        write_json(manifest_path, manifest)
    files = manifest["files"]
    if len(files) != EXPECTED_FILES or len({item["path"] for item in files}) != EXPECTED_FILES:
        raise ValueError("source manifest does not contain the complete pinned repository")
    for item in files:
        safe_member(root, item["path"])
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError("source file sizes must be known")
    needed = sum(item["size"] for item in files)
    remaining = sum(
        item["size"]
        for item in files
        if not (root / item["path"]).is_file() or (root / item["path"]).stat().st_size != item["size"]
    )
    if (
        not files
        or not any(f["path"].startswith("samples/") for f in files)
        or needed > 400 * 1024**3
        or (not local_only and shutil.disk_usage(root).free < remaining + 20 * 1024**3)
    ):
        raise ValueError("download exceeds 400 GiB limit or 20 GiB free-space reserve")
    print(json.dumps({"phase": "download_and_index", "files": len(files), "total_bytes": needed}), flush=True)

    def fetch(item):
        path = safe_member(root, item["path"])
        for attempt in range(5):
            try:
                if not local_only:
                    hf_hub_download(
                        REPO,
                        item["path"],
                        repo_type="dataset",
                        revision=REVISION,
                        local_dir=root,
                        token=False,
                        force_download=attempt > 0,
                    )
                sha = verify_file(root, item)
                break
            except Exception as exc:
                if local_only or attempt == 4:
                    cause = exc.__cause__ or exc.__context__
                    detail = f"{type(exc).__name__}: {exc}"
                    if cause:
                        detail += f"; {type(cause).__name__}: {cause}"
                    print(json.dumps({"path": item["path"], "error": detail}), flush=True)
                    return {**item, "error": detail}
                time.sleep(min(30, 2**attempt))
        try:
            motion = (
                inspect_motion(path) if item["path"].startswith("samples/") and path.suffix == ".csv" else {}
            )
            return {**item, "sha256": sha, **motion}
        except Exception as exc:
            return {**item, "sha256": sha, "error": f"{type(exc).__name__}: {exc}"}

    episodes, errors, tasks = [], [], {}
    completed = completed_bytes = frames = 0
    digest = hashlib.sha256()
    # shortcut: one bounded acquisition per dataset; add shared disk leases for concurrent bulk imports.
    with (output / "file-index.jsonl").open("w") as journal, ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch, item): item for item in files}
        # Stable final identity comes from source order, independent of completion order.
        identities = {}
        for future in as_completed(futures):
            result = future.result()
            journal.write(json.dumps(result) + "\n")
            journal.flush()
            completed += 1
            if "sha256" in result:
                completed_bytes += result["size"]
                identities[result["path"]] = result["sha256"]
            if "error" in result:
                errors.append({"path": result["path"], "error": result["error"]})
            else:
                if "length" in result:
                    task = Path(result["path"]).parent.name
                    tasks[task] = tasks.get(task, 0) + 1
                    frames += result["length"]
                    episodes.append(
                        {
                            "episode_id": Path(result["path"]).stem,
                            "path": result["path"],
                            "length": result["length"],
                            "tasks": [task],
                            "streams": [],
                        }
                    )
            if completed % 100 == 0 or completed == len(files):
                progress = {
                    "files_completed": completed,
                    "files_total": len(files),
                    "bytes_completed": completed_bytes,
                    "bytes_total": needed,
                    "invalid_files": len(errors),
                    "phase": "download_and_index",
                }
                write_json(output / "download-progress.json", progress)
                print(json.dumps(progress), flush=True)
    for item in files:
        digest.update(f"{item['path']}\0{identities.get(item['path'], 'invalid')}\n".encode())
    episodes.sort(key=lambda e: e["path"])
    write_json(output / "invalid-files.json", {"files": errors})
    return {
        "schema_version": "eb/motiondecode-raw/v1",
        "format": "motiondecode_csv",
        "storage_ref": str(root),
        "files": len(files),
        "total_bytes": completed_bytes,
        "bytes": completed_bytes,
        "episodes": episodes,
        "episodes_count": len(episodes),
        "frames_count": frames,
        "task_counts": tasks,
        "index_hash": digest.hexdigest(),
        "source_refs": [{"kind": "huggingface", "uri": REPO, "revision": REVISION}],
        "motion_schema": {
            "columns": HEADER,
            "joint_order": JOINTS,
            "position_unit": "m",
            "joint_unit": "rad",
            "quaternion_order": "wxyz",
            "declared_fps": 120,
            "timestamp_present": False,
            "applied_action_present": False,
        },
        "validation": {
            "scope": "motiondecode_raw_csv_v1",
            "status": "invalid" if errors else "validated",
            "training_ready": False,
            "evaluation_ready": False,
            "errors": [f"{e['path']}: {e['error']}" for e in errors[:100]],
            "invalid_files": len(errors),
            "unchecked": [
                "actual_sample_rate",
                "coordinate_frame",
                "requested_applied_actions",
                "target_loader",
                "physical_feasibility",
                "split_leakage",
                "immutable_training_publication",
            ],
            "warnings": [
                "CSV header uses metres; dataset card says millimetres",
                "120 Hz is an upstream declaration; no per-frame timestamps",
                "ChingMu attribution required; non-commercial use only",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = acquire(Path(args.dest), output, local_only=args.local_only)
    except Exception as exc:
        result = {
            "format": "motiondecode_csv",
            "validation": {
                "scope": "motiondecode_raw_csv_v1",
                "status": "invalid",
                "training_ready": False,
                "errors": [str(exc)],
            },
        }
    write_json(output / "dataset-index.json", result)
    if result["validation"]["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    main()
