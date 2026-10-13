"""Build bounded playback proxies inside a CPU Job; never modify the source dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from eb.motion_preview import PREVIEW_SCHEMA, read_motion_manifest


def build_previews(source: Path, output: Path, manifest_sha256: str) -> dict:
    manifest, digest = read_motion_manifest(source, manifest_sha256)
    output.mkdir(parents=True, exist_ok=False)
    items = []
    for clip in manifest["clips"]:
        path = source / clip["file"]
        if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve())
                or not path.is_file() or path.stat().st_size > 64 * 1024**2):
            raise ValueError(f"{clip['id']}: 动作文件缺失或超过大小限制")
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"{clip['id']}: 动作文件 hash 不一致")
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 128 * 1024**2:
                raise ValueError(f"{clip['id']}: 解压大小超过限制")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in ("q", "root_pos", "root_quat", "timestamps", "valid")}
        frames = clip["frames"]
        shapes = {"q": (frames, 23), "root_pos": (frames, 3), "root_quat": (frames, 4),
                  "timestamps": (frames,), "valid": (frames,)}
        if (any(a.shape != shapes[key] or not np.isfinite(a).all() for key, a in arrays.items())
                or arrays["valid"].dtype != np.bool_
                or np.any(np.diff(arrays["timestamps"]) <= 0)
                or not np.allclose(np.diff(arrays["timestamps"]), 1 / clip["fps"], rtol=.01, atol=1e-6)
                or np.any(np.abs(np.linalg.norm(arrays["root_quat"], axis=1) - 1) > .01)):
            raise ValueError(f"{clip['id']}: 轨迹维度、时间轴、数值或四元数非法")
        payload = {"schema_version": PREVIEW_SCHEMA, "manifest_sha256": digest,
                   "clip_sha256": clip["sha256"], "joint_order": manifest["joint_order"],
                   "quaternion": "wxyz", "clip": clip,
                   **{key: value.tolist() for key, value in arrays.items()}}
        target = output / f"{clip['id']}.json"
        raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 8 * 1024**2:
            raise ValueError(f"{clip['id']}: 回放数据超过 8 MiB")
        target.write_bytes(raw)
        items.append({"id": clip["id"], "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    receipt = {"schema_version": PREVIEW_SCHEMA, "manifest_sha256": digest, "items": items}
    (output / "index.json").write_text(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build_previews(args.source, args.output, args.manifest_sha256)))
