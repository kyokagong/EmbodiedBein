"""Small independent CPU geometry witness for the native depth probe."""

import math

SAMPLE_PIXELS = tuple(row * 64 + col for row in (24, 28, 32, 35) for col in (18, 27, 36, 45))
SCHEMA = "eb/pbfm-probe-depth/v1"
RECEIPT_SCHEMA = "eb/pbfm-native-runtime-probe/v2"
PHASES = {"random_reset": "reset-", "random_step": "", "positive_control": "positive-"}
TOLERANCE = {"atol_m": 0.01, "rtol": 0.001, "limit_margin_m": 0.02}
RAW_NAMES = (
    "network-depth.npy", "reference-network-depth.npy", "sensor-ranges.npy", "metric-depth.npy",
    "simulation-qpos.npy", "sensor-cached-world-origins.npy", "sensor-cached-world-rays.npy", "qpos.npy",
)


def required_probe_names():
    names = {"depth-validation.json", "probe-stats.json", "network-depth.png", "robot-terrain.png",
             "positive-robot-terrain.png", "positive-network-depth.png"}
    names.update(prefix + name for prefix in PHASES.values() for name in RAW_NAMES)
    return names


def validate_range_rows(rows, limit, *, positive=False):
    if (
        not isinstance(rows, list) or len(rows) != 16
        or type(limit) not in (float, int) or not math.isfinite(limit) or not 2.5 < limit <= 5
    ):
        raise ValueError("P-BFM depth geometry witness is incomplete")
    witnesses, errors, compared = [], [], 0
    for pixel, row in zip(SAMPLE_PIXELS, rows):
        if not isinstance(row, dict):
            raise ValueError("P-BFM depth ray evidence must be an object")
        cpu, gpu = row.get("cpu_range_m"), row.get("gpu_range_m")
        if (
            row.get("pixel") != pixel
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in (cpu, gpu))
            or any(type(row.get(key)) is not int for key in ("geom_id", "welded_body_id"))
            or type(row.get("collidable")) is not bool
            or cpu < 0 and cpu != -1 or gpu < 0 and gpu != -1 or gpu > limit
        ):
            raise ValueError("P-BFM depth ray evidence is malformed or nonfinite")
        if cpu >= 0 and abs(cpu - limit) <= TOLERANCE["limit_margin_m"]:
            continue
        compared += 1
        hit = 0 <= cpu < limit - TOLERANCE["limit_margin_m"]
        if hit:
            if gpu < 0 or abs(cpu - gpu) > TOLERANCE["atol_m"] + TOLERANCE["rtol"] * abs(cpu):
                raise ValueError(f"P-BFM CPU/GPU ray distance differs at pixel {pixel}: cpu={cpu}, gpu={gpu}")
            errors.append(abs(cpu - gpu))
            if row["geom_id"] >= 0 and row["welded_body_id"] == 0 and row["collidable"]:
                witnesses.append(cpu)
        elif gpu >= 0:
            raise ValueError(f"P-BFM CPU/GPU ray hit classification differs at pixel {pixel}")
    span = max(witnesses) - min(witnesses) if witnesses else 0.0
    if not compared or positive and (len(witnesses) < 4 or span <= 0.1):
        raise ValueError("P-BFM positive depth control lacks four static hits spanning more than 0.1m")
    return {
        "rows": rows, "compared_count": compared, "witness_count": len(witnesses),
        "witness_span_m": span, "max_abs_error_m": max(errors, default=0.0),
    }


def scene_ray_comparison(model, qpos, origins, directions, gpu_ranges, limit, *, positive=False):
    import mujoco
    import numpy as np

    if (
        qpos.shape != (2, model.nq) or origins.shape != (2, 36 * 64, 3)
        or directions.shape != origins.shape or gpu_ranges.shape != (2, 36 * 64)
        or not all(np.isfinite(value).all() for value in (qpos, origins, directions, gpu_ranges))
        or not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-5, rtol=1e-5)
    ):
        raise ValueError("P-BFM depth witness requires finite same-frame physical qpos and actual unit rays")
    groups = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    data, lanes = mujoco.MjData(model), []
    for lane in range(2):
        data.qpos[:] = qpos[lane]
        mujoco.mj_forward(model, data)
        rows = []
        for pixel in SAMPLE_PIXELS:
            geom = np.asarray([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                model, data, origins[lane, pixel].astype(np.float64),
                directions[lane, pixel].astype(np.float64),
                groups, 1, -1, geom,
            )
            index = int(geom[0])
            rows.append({
                "pixel": pixel, "cpu_range_m": float(distance), "gpu_range_m": float(gpu_ranges[lane, pixel]),
                "geom_id": index,
                "welded_body_id": int(model.body_weldid[model.geom_bodyid[index]]) if index >= 0 else -1,
                "collidable": (
                    bool(model.geom_contype[index] or model.geom_conaffinity[index]) if index >= 0 else False
                ),
            })
        lanes.append(validate_range_rows(rows, limit, positive=positive))
    return lanes


def validate_depth_report(report):
    if (
        report.get("schema_version") != SCHEMA or report.get("tolerance") != TOLERANCE
        or type(report.get("rank")) is not int or not 0 <= report["rank"] <= 3
        or report.get("sample_pixels") != list(SAMPLE_PIXELS)
        or report.get("include_geom_groups") != [0, 2, 5] or report.get("bodyexclude") != -1
        or set(report.get("phases", {})) != set(PHASES)
    ):
        raise ValueError("P-BFM depth comparison protocol differs")
    for name, phase in report["phases"].items():
        lanes = phase.get("lanes")
        if not isinstance(lanes, list) or len(lanes) != 2:
            raise ValueError("P-BFM depth comparison lacks both lanes")
        for lane in lanes:
            expected = validate_range_rows(
                lane.get("rows"), report.get("range_limit_m"), positive=name == "positive_control"
            )
            if lane != expected:
                raise ValueError("P-BFM depth geometry summary differs from actual comparisons")
        history = phase.get("history") or {}
        if (
            history.get("shape") != [2, 8, 18, 32] or history.get("dtype") != "uint8"
            or history.get("history_length") != 37
            or history.get("sampled_ages") != [35, 30, 25, 20, 15, 10, 5, 0]
            or type(history.get("max_abs_error")) is not int or history["max_abs_error"] != 0
            or not isinstance(history.get("delay_frames"), list) or len(history["delay_frames"]) != 2
            or any(type(value) is not int or value not in (0, 1) for value in history["delay_frames"])
        ):
            raise ValueError("P-BFM depth preprocessing/history comparison differs")
    return report
