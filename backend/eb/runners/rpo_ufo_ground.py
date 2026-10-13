"""Finite offline ground covering the real grid and train RSI collision envelope."""

import numpy as np

PROFILE = "wide_plane_v9"
SIZE_M = (512., 512.)


def coverage_receipt(origins, reference_foot_bounds, size):
    origins, feet = np.asarray(origins), np.asarray(reference_foot_bounds)
    size = np.asarray(size)
    if (origins.ndim != 2 or origins.shape[1] != 3 or not len(origins)
            or feet.shape != (2, 2) or size.shape != (2,) or (size <= 0).any()
            or not all(np.isfinite(x).all() for x in (origins, feet, size))
            or (feet[0] > feet[1]).any()):
        raise ValueError("Invalid ground coverage inputs")
    low, high = origins[:, :2].min(0) + feet[0], origins[:, :2].max(0) + feet[1]
    margin = float(np.minimum(low + size / 2, size / 2 - high).min())
    if margin <= 20.:
        raise ValueError("Insufficient ground coverage for actual grid + RSI feet + 20m reserve")
    return {"ground_profile": PROFILE, "plane_size_m": size.tolist(), "num_envs": len(origins),
        "origin_min_xy_m": origins[:, :2].min(0).tolist(), "origin_max_xy_m": origins[:, :2].max(0).tolist(),
        "reference_foot_bounds_xy_m": feet.tolist(), "combined_foot_bounds_xy_m": [low.tolist(), high.tolist()],
        "minimum_margin_m": margin, "required_motion_reserve_m": 20.}
