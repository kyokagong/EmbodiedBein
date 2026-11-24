"""A hash-verified logical view over unmodified MotionDecode source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eb.motiondecode import EXPECTED_FILES, HEADER, JOINTS, REPO, REVISION, verify_file, write_json

OPERATION = "motiondecode_valid_view_v1"
EXCLUDED_PATH = "samples/5.Dance/5.3.Pop Dance/Summer_Wind/Summer_Wind.csv"
EXCLUDED_SHA256 = "5cea3e9e5684860ce1b45c799dac91d69b015b8e3df5c1575b2c47966b7d38c4"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_view(root: Path, manifest: dict, index: Path, output: Path) -> dict:
    evidence = {}
    with index.open() as source:
        for line in source:
            entry = json.loads(line)
            if entry["path"] in evidence:
                raise ValueError("duplicate source evidence")
            evidence[entry["path"]] = entry
    files = manifest["files"]
    if len({f["path"] for f in files}) != len(files) or set(evidence) != {f["path"] for f in files}:
        raise ValueError("source evidence does not cover the full manifest")
    output.mkdir(parents=True, exist_ok=True)
    episodes, exclusions, tasks = [], [], {}
    frames = total_bytes = verified = 0
    digest = hashlib.sha256()

    def check(item):
        entry = evidence[item["path"]]
        sha = verify_file(root, item)
        if entry.get("sha256") != sha or entry.get("size") != item["size"]:
            raise ValueError(f"prior evidence hash differs: {item['path']}")
        return entry

    with (output / "included-motions.jsonl").open("w") as members, ThreadPoolExecutor(max_workers=8) as pool:
        for entry in pool.map(check, files):
            verified += 1
            if entry["path"] == EXCLUDED_PATH:
                if (
                    entry["sha256"] != EXCLUDED_SHA256
                    or entry["size"] != 970
                    or entry.get("error") != "ValueError: empty motion"
                ):
                    raise ValueError("excluded source differs from the approved empty motion")
                exclusions.append(
                    {
                        "path": EXCLUDED_PATH,
                        "sha256": EXCLUDED_SHA256,
                        "bytes": 970,
                        "reason": "header_only_no_motion_frames",
                    }
                )
            elif entry.get("error"):
                raise ValueError(f"unapproved source error: {entry['path']}: {entry['error']}")
            else:
                total_bytes += entry["size"]
                if entry["path"].startswith("samples/") and entry["path"].endswith(".csv"):
                    if (
                        type(entry.get("length")) is not int
                        or entry["length"] <= 0
                        or entry.get("columns") != 36
                    ):
                        raise ValueError(f"missing full-frame validation evidence: {entry['path']}")
                    task = Path(entry["path"]).parent.name
                    episode = {
                        "episode_id": Path(entry["path"]).stem,
                        "path": entry["path"],
                        "length": entry["length"],
                        "sha256": entry["sha256"],
                        "bytes": entry["size"],
                        "tasks": [task],
                        "streams": [],
                    }
                    episodes.append(episode)
                    line = json.dumps(episode, sort_keys=True, separators=(",", ":")) + "\n"
                    members.write(line)
                    digest.update(line.encode())
                    frames += entry["length"]
                    tasks[task] = tasks.get(task, 0) + 1
            if verified % 1000 == 0 or verified == len(files):
                progress = {
                    "phase": "verify_existing",
                    "files_completed": verified,
                    "files_total": len(files),
                    "bytes_completed": total_bytes,
                    "bytes_total": sum(f["size"] for f in files),
                    "invalid_files": 0,
                }
                write_json(output / "download-progress.json", progress)
                print(json.dumps(progress), flush=True)
    if len(exclusions) != 1:
        raise ValueError("approved empty motion is missing from the source")
    receipt = {
        "schema_version": "eb/motiondecode-view/v1",
        "operation": OPERATION,
        "format": "motiondecode_csv",
        "storage_ref": str(root),
        "files": len(files) - 1,
        "raw_files": len(files),
        "total_bytes": total_bytes,
        "bytes": total_bytes,
        "episodes_count": len(episodes),
        "frames_count": frames,
        "task_counts": tasks,
        "index_hash": digest.hexdigest(),
        "exclusions": exclusions,
        "source_refs": [{"kind": "huggingface", "uri": REPO, "revision": REVISION}],
        "motion_schema": {
            "columns": HEADER,
            "joint_order": JOINTS,
            "position_unit": "m",
            "joint_unit": "rad",
            "quaternion_order": "wxyz",
            "declared_fps": 120,
        },
        "validation": {
            "scope": OPERATION,
            "status": "validated",
            "errors": [],
            "training_ready": False,
            "evaluation_ready": False,
            "preview_ready": True,
            "unchecked": ["actual_sample_rate", "target_training_loader", "physical_feasibility"],
            "warnings": [
                "Kinematic preview uses declared 120 Hz; CSV has no timestamps",
                "ChingMu attribution required; non-commercial use only",
            ],
        },
    }
    write_json(output / "validation-receipt.json", receipt)
    return {**receipt, "episodes": episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--index-sha", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-ref", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        manifest_path = Path(args.source) / ".embodiedbein/source-manifest.json"
        if file_hash(Path(args.index)) != args.index_sha or file_hash(manifest_path) != args.manifest_sha:
            raise ValueError("frozen source metadata changed")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("repo_id") != REPO
            or manifest.get("revision") != REVISION
            or len(manifest.get("files", [])) != EXPECTED_FILES
        ):
            raise ValueError("source manifest is not the complete pinned repository")
        result = derive_view(Path(args.source), manifest, Path(args.index), output)
        result["storage_ref"] = args.storage_ref
        result["source_evidence_sha256"] = args.index_sha
        # Compact receipt uses the same host storage identity as the dataset report.
        write_json(output / "validation-receipt.json", {k: v for k, v in result.items() if k != "episodes"})
        write_json(output / "dataset-index.json", result)
    except Exception as exc:
        write_json(
            output / "dataset-index.json",
            {
                "validation": {
                    "status": "invalid",
                    "scope": OPERATION,
                    "training_ready": False,
                    "errors": [str(exc)],
                }
            },
        )
        raise


if __name__ == "__main__":
    main()
