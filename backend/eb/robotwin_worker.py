"""Pinned RoboTwin acquisition task; runs in the tools container, never in the API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path

from eb.robotwin import get_download_profile, raw_archive_profile


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            mode = member.external_attr >> 16
            if (target != root and not target.is_relative_to(root)) or stat.S_ISLNK(mode):
                raise ValueError(f"unsafe zip member: {member.filename}")
        bundle.extractall(destination)


def _shape(features: dict, key: str) -> list[int]:
    return list((features.get(key) or {}).get("shape") or [])


def _find_lerobot_root(root: Path) -> Path | None:
    direct = root / "meta" / "info.json"
    if direct.is_file():
        return root
    found = list(root.glob("**/meta/info.json"))
    return found[0].parent.parent if len(found) == 1 else None


def validate_robotwin_content(root: Path, profile_id: str, profile=None) -> dict:
    profile = profile or get_download_profile(profile_id)
    result = {
        "schema_version": "eb/robotwin-dataset/v1", "source_revision": profile.revision,
        "native_embodiment": profile.native_embodiment,
        "embodiment_env_id": profile.embodiment_env_id, "scenario": profile.scenario,
        "format": profile.format, "compatibility": "native_direct",
    }
    if not profile.format.startswith("lerobot"):
        result["validation"] = {"status":"validated", "scope":"robotwin_raw_archive",
                                "training_ready":False, "errors":[]}
        return result
    dataset = _find_lerobot_root(root)
    errors: list[str] = []
    info: dict = {}
    if dataset is None:
        errors.append("archive must contain exactly one LeRobot meta/info.json")
    else:
        info = json.loads((dataset / "meta" / "info.json").read_text())
        features = info.get("features") or {}
        if float(info.get("fps") or 0) != 15:
            errors.append("ALOHA LeRobot package must declare 15 fps")
        if _shape(features, "action")[-1:] != [14]:
            errors.append("ALOHA action must have 14 channels")
        if _shape(features, "observation.state")[-1:] != [14]:
            errors.append("ALOHA state must have 14 channels")
        cameras = {"observation.images.cam_high", "observation.images.cam_left_wrist",
                   "observation.images.cam_right_wrist"}
        if not cameras <= set(features):
            errors.append("ALOHA dataset requires head, left wrist and right wrist cameras")
        expected = "v2.1" if profile.format == "lerobot_v2.1" else "v3.0"
        if str(info.get("codebase_version")) != expected:
            errors.append(f"expected LeRobot {expected}")
    result.update({"data_root": str(dataset or root), "features": info.get("features", {}),
                   "fps":info.get("fps"),
                   "episodes_count": info.get("total_episodes"), "frames_count": info.get("total_frames")})
    result["validation"] = {"status":"invalid" if errors else "validated",
        "scope":"robotwin_aloha_lerobot", "training_ready":not errors and profile.format == "lerobot_v2.1",
        "errors":errors}
    return result


def _profile(args):
    if args.profile_id:
        return get_download_profile(args.profile_id)
    return raw_archive_profile(args.task, args.scenario, args.embodiment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-id")
    parser.add_argument("--task")
    parser.add_argument("--scenario")
    parser.add_argument("--embodiment")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = _profile(args)
    if args.revision != profile.revision:
        raise ValueError("RoboTwin source revision does not match the catalog")
    destination, output = Path(args.dest), Path(args.output)
    staging = destination.parent / f".{destination.name}.{os.environ.get('EB_RUN_ID', 'run')}.partial"
    try:
        if destination.exists() or staging.exists():
            raise ValueError("RoboTwin destination or staging path already exists")
        from huggingface_hub import HfApi, snapshot_download
        siblings = HfApi().dataset_info(profile.repo_id, revision=profile.revision,
                                        files_metadata=True).siblings
        by_name = {item.rfilename: item for item in siblings}
        selected = [by_name.get(name) for name in profile.allow_patterns]
        if any(item is None or item.size is None for item in selected):
            raise ValueError("pinned RoboTwin archive is missing or has unknown size")
        needed = sum(int(item.size) for item in selected if item is not None)
        if needed > 100 * 1024**3 or shutil.disk_usage(destination.parent).free < needed * 2 + 20 * 1024**3:
            raise ValueError("download/extraction exceeds 100 GiB task limit or 20 GiB disk reserve")
        download = staging / "download"
        unpacked = staging / "unpacked"
        staging.mkdir(parents=True)
        snapshot_download(profile.repo_id, repo_type="dataset", revision=profile.revision,
                          local_dir=str(download), token=False, allow_patterns=list(profile.allow_patterns))
        archives = []
        for relative in profile.allow_patterns:
            archive = download / relative
            if not archive.is_file():
                raise ValueError(f"downloaded archive missing: {relative}")
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            archives.append({"path":relative, "bytes":archive.stat().st_size, "sha256":digest})
            safe_extract_zip(archive, unpacked)
        publish = _find_lerobot_root(unpacked) if profile.format.startswith("lerobot") else unpacked
        if publish is None:
            raise ValueError("unable to locate the LeRobot dataset root")
        report = validate_robotwin_content(publish, profile.id, profile)
        if report["validation"]["status"] != "validated":
            raise ValueError("; ".join(report["validation"]["errors"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish.rename(destination)
        shutil.rmtree(staging, ignore_errors=True)
        report["data_root"] = str(destination)
        report["validation"]["storage_ref"] = str(destination)
        report.update({"root":str(destination), "archives":archives,
                       "license":"mit", "source_refs":[{"kind":"huggingface_dataset",
                           "repo_id":profile.repo_id, "revision":profile.revision,
                           "files":archives}],
                       "total_bytes":sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())})
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        report = {"schema_version":"eb/robotwin-dataset/v1", "native_embodiment":profile.native_embodiment,
                  "format":profile.format, "validation":{"status":"invalid", "training_ready":False,
                  "errors":[str(exc)]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False))
    tmp.replace(output)
    if report["validation"]["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
