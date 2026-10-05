"""Verified RPO motion windows for AMP. Training never deserializes pickle."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

try:
    from eb.rpo_wbc import AMP_MOTION_CONVERTER_REVISION, JOINT_ORDER, RPO_LAB_JOINT_ORDER
except ModuleNotFoundError:  # The RPO image exposes runners as top-level modules.
    from rpo_wbc import AMP_MOTION_CONVERTER_REVISION, JOINT_ORDER, RPO_LAB_JOINT_ORDER


FEATURE_SCHEMA = "rpo_amp_features_v1"
FEATURE_DIM = 74
# Frozen minimum scales in rad/s, rad, rad/s, m, unit gravity, m, m/s.
STD_FLOOR = np.array([0.1] * 3 + [0.05] * 23 + [0.2] * 23
                     + [0.05] * 18 + [0.05] * 3 + [0.05] + [0.1] * 3,
                     dtype=np.float32)


def encode_amp_frame(*, root_ang_vel_b, q_abs, dq, key_body_pos_b,
                     projected_gravity_b, root_height, root_lin_vel_b):
    pieces = [np.asarray(x, dtype=np.float32) for x in
              (root_ang_vel_b, q_abs, dq, key_body_pos_b,
               projected_gravity_b, root_height, root_lin_vel_b)]
    widths = (3, 23, 23, 18, 3, 1, 3)
    lead = pieces[0].shape[:-1]
    for index, (piece, width) in enumerate(zip(pieces, widths)):
        if index == 3 and piece.shape == (*lead, 6, 3):
            pieces[index] = piece.reshape(*lead, 18)
            piece = pieces[index]
        if piece.shape != (*lead, width) or not np.isfinite(piece).all():
            raise ValueError(f"AMP feature {index} has invalid shape or values")
    return np.concatenate(pieces, axis=-1)


def valid_window_ends(clip_id, valid):
    if len(clip_id) != len(valid):
        raise ValueError("clip IDs and validity mask differ in length")
    return [i for i in range(2, len(valid))
            if all(valid[i - 2:i + 1])
            and clip_id[i - 2] == clip_id[i - 1] == clip_id[i]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class MotionWindowSampler:
    def __init__(self, clips: list[tuple[str, str, np.ndarray]], manifest_sha256: str,
                 seed: int = 0):
        self.clips = clips
        self.manifest_sha256 = manifest_sha256
        self.rng = np.random.default_rng(seed)
        self.available = {behavior: [(clip_id, data, end)
                                     for clip_id, label, data in clips if label == behavior
                                     for end in range(2, len(data))]
                          for behavior in ("stand", "walk")}
        if any(not entries for entries in self.available.values()):
            raise ValueError("AMP train Slice needs valid stand and walk windows")

        total = sum(len(data) for _, _, data in clips)
        summed = sum((np.sum(data, axis=0, dtype=np.float64)
                      for _, _, data in clips), np.zeros(FEATURE_DIM, dtype=np.float64))
        squared = sum((np.sum(np.square(data, dtype=np.float64), axis=0)
                       for _, _, data in clips), np.zeros(FEATURE_DIM, dtype=np.float64))
        self.mean = (summed / total).astype(np.float32)
        self.std = np.maximum(np.sqrt(np.maximum(squared / total - (summed / total) ** 2, 0)),
                              STD_FLOOR).astype(np.float32)

    def sample(self, count: int) -> tuple[np.ndarray, list[tuple[str, int]]]:
        if count <= 0:
            raise ValueError("AMP sample count must be positive")
        windows = np.empty((count, 3, FEATURE_DIM), dtype=np.float32)
        ids = []
        for index in range(count):
            pool = self.available[("stand", "walk")[index % 2]]
            clip_id, data, end = pool[int(self.rng.integers(len(pool)))]
            windows[index] = data[end - 2:end + 1]
            ids.append((clip_id, end))
        return windows, ids

    def state_dict(self) -> dict:
        return {"manifest_sha256": self.manifest_sha256,
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        if state.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("AMP sampler manifest changed")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


def load_motion_prior(root: Path, *, asset_sha256: str, seed: int = 0) -> MotionWindowSampler:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("AMP motion manifest is missing or too large")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != "eb/rpo-amp-motion/v1"
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("usage") != "train" or manifest.get("purpose") != "motion_prior"
            or manifest.get("asset_sha256") != asset_sha256
            or manifest.get("joint_order") != list(JOINT_ORDER)
            or manifest.get("source_joint_order") != list(RPO_LAB_JOINT_ORDER)
            or manifest.get("converter_revision") != AMP_MOTION_CONVERTER_REVISION):
        raise ValueError("AMP motion identity, purpose, asset or joint order differs")
    members = manifest.get("clips")
    if not isinstance(members, list) or not 2 <= len(members) <= 10000:
        raise ValueError("AMP motion clips must be a bounded non-empty list")
    clips = []
    ids = set()
    groups = set()
    for item in members:
        if not isinstance(item, dict):
            raise ValueError("AMP clip entry is invalid")
        clip_id = item.get("id")
        name = item.get("file")
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (not isinstance(clip_id, str) or not clip_id or clip_id in ids
                or not isinstance(item.get("source_group"), str) or not item["source_group"]
                or not isinstance(item.get("source_license"), str) or not item["source_license"]
                or item.get("split") != "train" or item.get("behavior") not in ("stand", "walk")
                or item.get("fps") != 50 or relative is None or relative.is_absolute()
                or ".." in relative.parts or "\\" in name or relative.suffix != ".npy"):
            raise ValueError("AMP clip identity, split, timing or path is invalid")
        ids.add(clip_id)
        groups.add(item["source_group"])
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or not path.is_file() or path.stat().st_size > 512 * 1024 * 1024
                or _sha256(path) != item.get("sha256")):
            raise ValueError("AMP clip file hash or path differs")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if (data.dtype != np.float32 or data.ndim != 2 or data.shape[1] != FEATURE_DIM
                or len(data) < 3 or not np.isfinite(data).all()):
            raise ValueError("AMP clip feature array is invalid")
        clips.append((clip_id, item["behavior"], data))
    if not groups:
        raise ValueError("AMP motion source groups are empty")
    return MotionWindowSampler(clips, hashlib.sha256(raw).hexdigest(), seed)
