"""Create the small OpenDM JSONL view used by a RoboTwin smoke run."""
import argparse
import json
from pathlib import Path

CAMERAS = {
    "images_1": "cam_high",
    "images_2": "cam_left_wrist",
    "images_3": "cam_right_wrist",
}


def build(source: Path, output: Path, frames: int = 64) -> Path:
    import av
    import numpy as np
    import pyarrow.parquet as pq

    episode = json.loads((source / "meta/episodes.jsonl").open().readline())
    count = min(frames, int(episode["length"]))
    if count < 51:
        raise ValueError("DM0.5 chunk_size=50 requires at least 51 frames")
    table = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
    rows = table.slice(0, count).to_pylist()
    image_paths: dict[str, list[str]] = {}
    for key, camera in CAMERAS.items():
        target = output / "images/episode_000000" / camera
        target.mkdir(parents=True, exist_ok=True)
        video = source / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
        saved = []
        with av.open(str(video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= count:
                    break
                path = target / f"{index:06d}.jpg"
                frame.to_image().save(path, quality=95)
                saved.append(str(path))
        if len(saved) != count:
            raise ValueError(f"{camera} decoded {len(saved)} frames, expected {count}")
        image_paths[key] = saved
    jsonl = output / "episode_000000.jsonl"
    with jsonl.open("w") as stream:
        for index, row in enumerate(rows):
            value = {
                key: {"type": "image", "url": paths[index]}
                for key, paths in image_paths.items()
            }
            value.update({
                "state": row["observation.state"],
                "action": row["action"],
                "prompt": episode["tasks"][0],
                "is_robot": True,
            })
            stream.write(json.dumps(value) + "\n")
    (output / "index_cache.json").write_text(json.dumps({"data": {str(jsonl): count}}, indent=2))
    profile = {}
    for source_key, target_key in (("observation.state", "state"), ("action", "action")):
        values = np.asarray([row[source_key] for row in rows], dtype=np.float32)
        profile[target_key] = {
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }
    robot = "Aloha RoboTwin2"
    (output / "norm_stats.json").write_text(json.dumps({
        "default_robot_type": robot,
        "norm_stats": profile,
        "norm_stats_by_robot": {robot: profile},
    }, indent=2))
    return jsonl


def rebase(source: Path, output: Path) -> Path:
    """Copy JSONL metadata while rebasing local image paths to the mounted view."""
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((source / "index_cache.json").read_text())["data"]
    rebased = {}
    for raw_path, count in index.items():
        src = Path(raw_path)
        if not src.is_file():
            src = source / src.name
        if not src.is_file():
            raise ValueError(f"missing JSONL member: {raw_path}")
        dest = output / src.name
        with src.open() as reader, dest.open("w") as writer:
            for line in reader:
                row = json.loads(line)
                for key in CAMERAS:
                    url = Path(row[key]["url"])
                    if not url.is_file():
                        marker = f"/{key.replace('images_', 'cam_')}/"
                        if marker in url.as_posix():
                            url = source / "images" / "episode_000000" / marker.strip("/") / url.name
                        else:
                            parts = url.parts
                            url = source / Path(*parts[parts.index("images"):]) if "images" in parts else source / url.name
                    if not url.is_file():
                        raise ValueError(f"missing image member: {row[key]['url']}")
                    row[key]["url"] = str(url)
                writer.write(json.dumps(row) + "\n")
        rebased[str(dest)] = int(count)
    (output / "index_cache.json").write_text(json.dumps({"data": rebased}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    build(args.source, args.output, args.frames)
