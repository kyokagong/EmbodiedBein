from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

FORMATS = {"rpo_bfm_motion", "rpo_ufo_robot_state"}
PREVIEW_SCHEMA = "eb/motion-preview/v1"
PREVIEW_IMAGE = "eb/tools:motion-preview-v1"


def read_motion_manifest(root: Path, expected_sha256: str | None = None) -> tuple[dict, str]:
    path = root / "manifest.json"
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("参考动作索引缺失或路径不安全")
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("参考动作索引超过 2 MiB")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("参考动作索引与冻结版本的 hash 不一致")
    manifest = json.loads(raw)
    if manifest.get("schema_version") not in {"eb/rpo-bfm-motion/v1", "eb/rpo-robot-state/v1"}:
        raise ValueError("此格式尚不支持参考动作回放")
    if (manifest.get("output_quaternion") or manifest.get("quaternion")) != "wxyz":
        raise ValueError("参考动作必须明确使用 wxyz 根四元数")
    order = manifest.get("joint_order") or []
    if len(order) != 23 or len(set(order)) != 23 or not all(isinstance(j, str) for j in order):
        raise ValueError("参考动作的 23 关节顺序缺失或重复")
    clips = manifest.get("clips") or []
    if not 1 <= len(clips) <= 1000:
        raise ValueError("参考动作片段数必须在 1–1000 之间")
    ids, files = set(), set()
    for clip in clips:
        cid, filename = clip.get("id", ""), clip.get("file", "")
        frames, fps = clip.get("frames"), clip.get("fps")
        if (not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}", cid)
                or cid in ids or filename in files
                or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}\.npz", filename)
                or not re.fullmatch(r"[a-f0-9]{64}", clip.get("sha256", ""))
                or type(frames) is not int or not 2 <= frames <= 10000
                or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 500):
            raise ValueError("参考动作片段 ID、文件、hash、帧数或频率非法")
        ids.add(cid)
        files.add(filename)
    return manifest, digest
