import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eb.data_validation import require_validated_dataset, validate_lerobot_v3, validate_opendm_robotwin2
from eb.domain.data_view import episode_media


def test_shared_v3_shard_offsets_and_invalid_members(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    features = {key: {"dtype": "float32", "shape": [1]} for key in
                ("action", "observation.state", "episode_index", "frame_index", "timestamp")}
    features["task_index"] = {"dtype": "int64", "shape": [1]}
    features["observation.images.image"] = {"dtype": "video", "shape": [8, 8, 3]}
    info = {"codebase_version": "v3.0", "fps": 1, "features": features, "total_episodes": 2,
            "total_frames": 4, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    data = tmp_path / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    columns = {key: [0, 1, 2, 3] for key in features if key != "observation.images.image"}
    columns.update(episode_index=[0, 0, 1, 1], frame_index=[0, 1, 0, 1], task_index=[0, 0, 1, 1])
    pq.write_table(pa.table({"task_index": [0, 1], "__index_level_0__": ["pick cup", "place cup"]}),
                   tmp_path / "meta/tasks.parquet")
    pq.write_table(pa.table(columns), data)
    video = tmp_path / "videos/observation.images.image/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"container-only fixture; decode explicitly unchecked")
    rows = []
    for ep in range(2):
        prefix = "videos/observation.images.image"
        rows.append({"episode_index": ep, "length": 2, "dataset_from_index": ep * 2,
                     "dataset_to_index": ep * 2 + 2, "data/chunk_index": 0, "data/file_index": 0,
                     f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0,
                     f"{prefix}/from_timestamp": ep * 2., f"{prefix}/to_timestamp": ep * 2. + 2})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    result = validate_lerobot_v3(tmp_path, [{"kind": "path", "uri": str(tmp_path)}])
    assert result["validation"]["training_ready"] is False
    assert result["episodes"][0]["tasks"] == ["pick cup"]
    assert result["task_counts"] == {"pick cup": 1, "place cup": 1}
    first = episode_media(tmp_path, "0", result["episodes"])
    second = episode_media(tmp_path, "1", result["episodes"])
    assert first["path"] == second["path"]
    assert (first["from_timestamp"], first["to_timestamp"]) == (0, 2)
    assert (second["from_timestamp"], second["to_timestamp"]) == (2, 4)
    row = SimpleNamespace(status="validated", dest_path=str(tmp_path), summary_stats=result)
    assert require_validated_dataset(row)["scope"] == "lerobot_v3_structure"
    row.status = "ready"
    with pytest.raises(ValueError):
        require_validated_dataset(row)
    video.unlink()
    with pytest.raises(ValueError, match="missing or escaping"):
        validate_lerobot_v3(tmp_path, [])


def test_opendm_robotwin2_view_requires_real_joint14_samples(tmp_path: Path):
    images = []
    for camera in ("high", "left", "right"):
        path = tmp_path / f"images/{camera}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
        images.append(path)
    row = {f"images_{index}": {"type": "image", "url": str(path)}
           for index, path in enumerate(images, 1)}
    row.update(state=[0.0] * 14, action=[0.0] * 14, prompt="pick", is_robot=True)
    data = tmp_path / "episode.jsonl"
    data.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    (tmp_path / "index_cache.json").write_text(json.dumps({"data": {str(data): 2}}))
    profile = {key: [0.0] * 14 for key in ("q01", "q99", "mean", "std")}
    (tmp_path / "norm_stats.json").write_text(json.dumps({
        "default_robot_type": "Aloha RoboTwin2",
        "norm_stats": {"state": profile, "action": profile},
    }))

    result = validate_opendm_robotwin2(tmp_path, [{"kind": "derived_view"}])

    assert result["validation"]["scope"] == "opendm_robotwin2_view"
    assert result["validation"]["training_ready"] is True
    assert result["features"]["action"]["shape"] == [14]
