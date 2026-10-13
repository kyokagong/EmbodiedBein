"""Frozen held-out physical diagnostic, independent of train prompt calibration."""

import hashlib
import math

import numpy as np

SUITE = {
    "id": "suite_rpo_ufo_isaac_heldout_tracking_v2",
    "split": "validation",
    "frames": 250,
    "selection": "first_and_last_fully_valid_validation_clip_by_id",
    "case_ids": [63000, 63001],
    "scene_seed": 63000,
    "policy": "native_tracking_inference_mean_actor",
    "qualification": "diagnostic_only",
}


def tracking_references(clips):
    eligible = sorted(
        (
            clip
            for clip in clips
            if clip["split"] == "validation"
            and clip["frames"] >= 250
            and clip.get("valid_frames") == clip["frames"]
        ),
        key=lambda clip: clip["id"],
    )
    if len(eligible) < 2:
        raise ValueError("Tracking needs two frozen held-out reference clips")
    return [eligible[0], eligible[-1]]


def tracking_cases(clips):
    cases = []
    for clip in tracking_references(clips):
        valid = clip["arrays"]["valid"]
        starts = np.flatnonzero(
            np.convolve(valid.astype(np.int32), np.ones(250, np.int32), mode="valid") == 250
        )
        if not len(starts):
            raise ValueError("Held-out tracking clip lacks a valid contiguous 250-frame window")
        cases.append((clip, int(starts[0])))
    return cases


def tracking_latents(model, arrays, start):
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        latents = model.tracking_inference(
            {
                "state": torch.from_numpy(arrays["states"][start : start + 250]).to(device),
                "privileged_state": torch.from_numpy(arrays["privileged_state"][start : start + 250]).to(
                    device
                ),
            }
        )
    if latents.shape != (250, 256) or not torch.isfinite(latents).all():
        raise ValueError("Held-out native B prompt is invalid")
    return latents


def tracking_action(model, observation, z, active):
    import torch

    with torch.inference_mode():
        action = model.act(observation, z, mean=True)
    if not torch.isfinite(action).all() or (action.abs() > 1.000001).any():
        raise ValueError("Nonfinite/out-of-range native tracking action")
    return action.masked_fill(~torch.as_tensor(active, device=action.device)[:, None], 0)


def validate_tracking_receipt(receipt, plan, output):
    if (
        receipt.get("schema_version") != "eb/ufo-heldout-tracking/v1"
        or receipt.get("suite") != SUITE
        or receipt.get("plan") != plan
        or receipt.get("qualification") != "not_requested"
    ):
        raise ValueError("Held-out tracking receipt identity differs")
    cases = receipt.get("cases") or []
    if (
        len(cases) != 2
        or [row.get("clip_id") for row in cases] != plan["clip_ids"]
        or [row.get("case_id") for row in cases] != SUITE["case_ids"]
    ):
        raise ValueError("Held-out tracking case coverage differs")
    for row in cases:
        seed, steps = row["case_id"], row.get("steps")
        integrity = row.get("video_integrity") or {}
        if (
            row.get("split") != "validation"
            or row.get("scene_seed") != SUITE["scene_seed"]
            or type(steps) is not int
            or not 1 <= steps <= 249
            or not math.isfinite(row.get("q_rmse_rad", float("nan")))
            or integrity.get("decoded_frames") != 1 + math.ceil(steps / 5)
            or integrity.get("resolution") != [320, 240]
            or integrity.get("fps") != 10
        ):
            raise ValueError("Held-out tracking physical/media evidence incomplete")
        for key, name, expected in (
            ("video", f"tracking-{seed}.mp4", integrity.get("sha256")),
            ("trace", f"tracking-{seed}.jsonl", row.get("trace_sha256")),
        ):
            path = output / name
            if (
                row.get(key) != name
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > 64 * 1024**2
            ):
                raise ValueError("Held-out tracking media/trace missing")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Held-out tracking media/trace changed")
