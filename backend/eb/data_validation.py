"""LeRobot v3 structural indexing for the container worker, never an API scan."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def require_amp_motion_dataset(row) -> dict:
    from eb.rpo_wbc import AMP_MOTION_CONVERTER_REVISION, JOINT_ORDER, RPO_LAB_JOINT_ORDER

    report = (row.summary_stats or {}).get("validation") or {}
    root = Path(row.dest_path)
    manifest = root / "manifest.json"
    if (row.status != "validated" or row.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1"
            or (row.meta or {}).get("usage") != "train"
            or (row.meta or {}).get("purpose") != "motion_prior"
            or report.get("status") != "validated" or report.get("scope") != "rpo_amp_motion_v1"
            or report.get("training_ready") is not True
            or report.get("storage_ref") != row.dest_path
            or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("manifest_sha256", "")))
            or root.is_symlink() or not manifest.is_file() or manifest.is_symlink()):
        raise ValueError("AMP requires a validated train motion_prior DatasetVersion")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != report["manifest_sha256"]:
        raise ValueError("AMP motion manifest changed after validation")
    identity = json.loads(manifest.read_text())
    if (identity.get("converter_revision") != AMP_MOTION_CONVERTER_REVISION
            or identity.get("joint_order") != list(JOINT_ORDER)
            or identity.get("source_joint_order") != list(RPO_LAB_JOINT_ORDER)):
        raise ValueError("AMP motion joint mapping is obsolete or invalid")
    return report


def validate_amp_motion(root: Path, asset_manifest: Path, source_refs: list[dict]) -> dict:
    from eb.runners.rpo_amp_motion import load_motion_prior

    root = root.resolve()
    asset_manifest = asset_manifest.resolve()
    receipt = json.loads(asset_manifest.read_text())
    if receipt.get("status") != "verified":
        raise ValueError("RPO asset receipt is not verified")
    asset_sha256 = hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
    sampler = load_motion_prior(root, asset_sha256=asset_sha256)
    manifest = root / "manifest.json"
    members = json.loads(manifest.read_text())["clips"]
    behavior_frames = {label: sum(len(data) for _, behavior, data in sampler.clips
                                  if behavior == label) for label in ("stand", "walk")}
    return {"schema_version": "eb/rpo-amp-motion-index/v1", "format": "rpo_amp_motion",
            "source_refs": source_refs, "storage_ref": str(root),
            "clips_count": len(sampler.clips), "frames_count": sum(len(data) for _, _, data in sampler.clips),
            "source_groups_count": len({item["source_group"] for item in members}),
            "behavior_seconds": {label: frames / 50 for label, frames in behavior_frames.items()},
            "asset_sha256": asset_sha256,
            "validation": {"status": "validated", "scope": "rpo_amp_motion_v1",
                           "training_ready": True, "qualification":"diagnostic",
                           "storage_ref": str(root),
                           "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                           "asset_sha256": asset_sha256, "errors": [],
                           "unchecked": ["source_license_authenticity", "time_scaled_dynamic_replay",
                                         "isaac_replay_quality"]}}


def require_validated_dataset(row, embodiment_env_id: str | None = None) -> dict:
    stats = row.summary_stats or {}
    report = stats.get("validation", {})
    valid_scope = report.get("scope") in {"lerobot_v3_structure", "robotwin_aloha_lerobot",
                                           "opendm_robotwin2_view"}
    if (row.status != "validated" or report.get("status") != "validated" or not valid_scope
            or report.get("storage_ref") != row.dest_path
            or (report.get("scope") == "robotwin_aloha_lerobot" and not report.get("training_ready"))):
        raise ValueError("dataset needs training-ready structural validation; registration/ready is insufficient")
    if embodiment_env_id == "ee_libero_franka_delta_v1":
        features = stats.get("features", {})
        if (row.embodiment_env_id != embodiment_env_id
                or features.get("action", {}).get("shape") != [7]
                or features.get("observation.state", {}).get("shape") != [8]):
            raise ValueError("LIBERO 数据接口或状态/动作维度不匹配")
    if embodiment_env_id == "ee_robotwin_aloha_joint14_v1":
        features = stats.get("features", {})
        if (row.embodiment_env_id != embodiment_env_id
                or stats.get("native_embodiment") != "aloha-agilex"
                or features.get("action", {}).get("shape") != [14]
                or features.get("observation.state", {}).get("shape") != [14]):
            raise ValueError("RoboTwin ALOHA 数据接口或 14 维状态/动作不匹配")
    return report


def validate_opendm_robotwin2(root: Path, source_refs: list[dict]) -> dict:
    root = root.resolve()
    index = json.loads((root / "index_cache.json").read_text()).get("data", {})
    norm = json.loads((root / "norm_stats.json").read_text())
    if not index or norm.get("default_robot_type") != "Aloha RoboTwin2":
        raise ValueError("OpenDM view requires an Aloha RoboTwin2 index and norm stats")

    def local(path: str, *, jsonl: bool = False) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.resolve()
        if jsonl:
            candidate = root / candidate.name
        elif "images" in candidate.parts:
            candidate = root / Path(*candidate.parts[candidate.parts.index("images"):])
        else:
            candidate = root / candidate.name
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"missing OpenDM view member: {path}")
        return candidate

    samples = 0
    for raw_path, count in index.items():
        path = local(raw_path, jsonl=True)
        lines = path.read_text().splitlines()
        if int(count) != len(lines) or len(lines) < 2:
            raise ValueError("OpenDM JSONL count mismatch or episode too short")
        samples += len(lines)
        for line in (lines[0], lines[-1]):
            row = json.loads(line)
            if len(row.get("state", [])) != 14 or len(row.get("action", [])) != 14 or not str(row.get("prompt", "")).strip():
                raise ValueError("OpenDM RoboTwin2 requires prompt and 14D state/action")
            for key in ("images_1", "images_2", "images_3"):
                local(str((row.get(key) or {}).get("url", "")))
    for field in ("state", "action"):
        stats = norm.get("norm_stats", {}).get(field, {})
        if any(len(stats.get(key, [])) != 14 for key in ("q01", "q99", "mean", "std")):
            raise ValueError(f"OpenDM {field} norm stats must be 14D")
    report = {"status": "validated", "scope": "opendm_robotwin2_view",
              "storage_ref": str(root), "training_ready": True, "errors": [],
              "unchecked": ["full_video_decode", "closed_loop_replay"]}
    return {"schema_version": "eb/opendm-robotwin2-view/v1", "format": "opendm_jsonl",
            "storage_ref": str(root), "native_embodiment": "aloha-agilex",
            "features": {"action": {"shape": [14]}, "observation.state": {"shape": [14]},
                         "images": {"count": 3}}, "frames_count": samples,
            "source_refs": source_refs, "validation": report}


def validate_lerobot_v3(root: Path, source_refs: list[dict]) -> dict:
    import pyarrow.parquet as pq

    root = root.resolve()
    def member(relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or escaping dataset member: {relative}")
        return path

    info = json.loads(member("meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("reader supports LeRobot v3.0 only")
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    features = info["features"]
    for key in ("action", "observation.state", "episode_index", "frame_index", "timestamp"):
        if key not in features:
            raise ValueError(f"missing feature: {key}")
    cameras = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    if not cameras:
        raise ValueError("no video features")
    shards = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not shards:
        raise ValueError("missing v3 episode metadata shards")
    # ponytail: metadata index held in memory; use PG episode rows when datasets exceed this bound.
    if int(info["total_episodes"]) > 100_000:
        raise ValueError("episode metadata exceeds 100000 entry worker limit")
    task_labels = {}
    if "task_index" in features:
        for task in pq.read_table(member("meta/tasks.parquet")).to_pylist():
            label = task.get("task") or task.get("__index_level_0__")
            index = int(task["task_index"])
            if not isinstance(label, str) or not label.strip() or index in task_labels:
                raise ValueError("invalid or duplicate task metadata")
            task_labels[index] = label
        if not task_labels or ("total_tasks" in info and len(task_labels) != int(info["total_tasks"])):
            raise ValueError("task metadata totals mismatch")
    episodes, seen, data_files, media_files = [], set(), {}, set()
    for shard in shards:
        metadata = pq.read_table(member(str(shard.relative_to(root)))).to_pylist()
        for row in metadata:
            index, length = int(row["episode_index"]), int(row["length"])
            start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            if index in seen or index < 0 or length <= 0 or end - start != length:
                raise ValueError("duplicate episode or invalid dataset interval")
            seen.add(index)
            data_path = info["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                                                  file_index=int(row["data/file_index"]))
            if data_path not in data_files:
                data_files[data_path] = pq.ParquetFile(member(data_path))
                if not set(features).difference(cameras).issubset(data_files[data_path].schema_arrow.names):
                    raise ValueError(f"data shard missing feature columns: {data_path}")
            streams = []
            for key in cameras:
                prefix = f"videos/{key}"
                begin, finish = float(row[f"{prefix}/from_timestamp"]), float(row[f"{prefix}/to_timestamp"])
                if not all(math.isfinite(t) for t in (begin, finish)) or begin < 0 or finish <= begin:
                    raise ValueError(f"invalid video interval: {index}/{key}")
                path = info["video_path"].format(video_key=key, chunk_index=int(row[f"{prefix}/chunk_index"]),
                                                file_index=int(row[f"{prefix}/file_index"]))
                if member(path).stat().st_size == 0:
                    raise ValueError(f"empty video: {path}")
                media_files.add(path)
                streams.append({"key": key, "path": path, "from_timestamp": begin, "to_timestamp": finish})
            episodes.append({"episode_id": str(index), "source_episode_id": str(index), "length": length,
                             "dataset_from_index": start, "dataset_to_index": end, "data_path": data_path,
                             "tasks": row.get("tasks", []), "streams": streams, "outcome": "unknown"})
    episodes.sort(key=lambda e: e["dataset_from_index"])
    cursor = 0
    for episode in episodes:
        if episode["dataset_from_index"] != cursor:
            raise ValueError("episode intervals overlap or have gaps")
        cursor = episode["dataset_to_index"]
    if len(episodes) != int(info["total_episodes"]) or cursor != int(info["total_frames"]):
        raise ValueError("metadata totals mismatch")
    if sum(p.metadata.num_rows for p in data_files.values()) != cursor:
        raise ValueError("data shard row counts mismatch")
    indexed = {int(e["episode_id"]): e for e in episodes}
    counts = {index: 0 for index in indexed}
    episode_tasks = {index: set() for index in indexed}
    for path, parquet in data_files.items():
        columns = ["episode_index", "frame_index"] + (["task_index"] if task_labels else [])
        for batch in parquet.iter_batches(columns=columns):
            for item in batch.to_pylist():
                index = int(item["episode_index"])
                episode = indexed.get(index)
                if (episode is None or episode["data_path"] != path
                        or int(item["frame_index"]) != counts[index]):
                    raise ValueError("data episode/frame indices disagree with metadata")
                counts[index] += 1
                if task_labels:
                    task_index = int(item["task_index"])
                    if task_index not in task_labels:
                        raise ValueError("data references unknown task_index")
                    episode_tasks[index].add(task_index)
    if any(counts[index] != episode["length"] for index, episode in indexed.items()):
        raise ValueError("episode lengths disagree with data rows")
    if not episodes:
        raise ValueError("empty dataset")
    task_counts = {}
    for index, episode in indexed.items():
        if task_labels:
            episode["task_indices"] = sorted(episode_tasks[index])
            episode["tasks"] = [task_labels[task] for task in episode["task_indices"]]
        for task in set(episode["tasks"]):
            task_counts[task] = task_counts.get(task, 0) + 1
    report = {"status": "validated", "scope": "lerobot_v3_structure", "format_version": "v3.0",
              "storage_ref": str(root), "training_ready": False, "errors": [],
              "unchecked": ["video_decode", "per_frame_alignment", "action_semantics", "target_loader_smoke",
                            "immutable_publication", "split_leakage"]}
    manifest = {"schema_version": "eb/lerobot-index/v1", "source_refs": source_refs,
                "format": "lerobot", "format_version": "v3.0", "storage_ref": str(root),
                "features": features, "fps": fps, "episodes": episodes}
    # This hashes the index, not the payload; never advertise it as an immutable dataset content hash.
    index_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {**manifest, "episodes_count": len(episodes), "frames_count": cursor,
            "task_counts": task_counts, "tasks_count": len(task_counts),
            "tasks_unknown_episodes": sum(not e["tasks"] for e in episodes),
            "index_hash": index_hash, "validation": report}
