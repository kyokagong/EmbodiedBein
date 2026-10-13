"""Frozen reference package for the RLinf tracking teacher, not a BFM prior."""

import hashlib
import json
from pathlib import Path

import numpy as np

ACTOR_DIM, CRITIC_DIM = 547, 905
TEMPLATE = "rpo_v2_bfm_zero_tracking_teacher"
GYM_ID = "RPO-Tracking-Teacher-v1"


def future_indices(frame, count):
    return np.minimum(np.asarray(frame)[..., None] + [1, 5, 10], count - 1)


def residual_action(q, residual):
    try:
        from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q
    except ModuleNotFoundError:
        from rpo_bfm import ACTION_GAIN, DEFAULT_Q
    return np.clip((q - np.asarray(DEFAULT_Q)) / (0.25 * np.asarray(ACTION_GAIN)) + residual, -1, 1)


def reference_receipt(root, receipt_sha, asset_sha):
    root = Path(root)
    path = root / "retarget-receipt.json"
    if root.is_symlink() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != receipt_sha:
        raise ValueError("Tracking reference receipt changed")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema_version") != "eb/100style-rpo-retarget-trial/v1"
        or receipt.get("asset_sha256") != asset_sha
        or receipt.get("recipe") != "xsens-yxz-native23-limb-ground-bounded-ik-v2"
        or receipt.get("training_ready") is not False
        or len(receipt.get("clips", [])) != 4
    ):
        raise ValueError("Tracking reference recipe/asset binding mismatch")
    for clip in receipt["clips"]:
        path = root / clip["file"]
        if (
            path.parent != root
            or path.is_symlink()
            or clip["frames"] != 150
            or hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]
        ):
            raise ValueError("Tracking reference clip changed")
    return receipt
