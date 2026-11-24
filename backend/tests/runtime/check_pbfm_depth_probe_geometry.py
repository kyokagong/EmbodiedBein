"""CPU geometry witness checks; no GPU or training execution."""

import importlib

import mujoco
import numpy as np


def main():
    try:
        probe = importlib.import_module("eb.runners.pbfm_depth_probe")
    except ModuleNotFoundError:
        raise AssertionError("the probe needs an independent CPU geometry witness") from None
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 .1" group="5"/></worldbody></mujoco>'
    )
    origins = np.zeros((2, 36 * 64, 3), dtype=np.float32)
    origins[..., 2] = 0.9
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    all_miss = np.full((2, 36 * 64), -1.0, dtype=np.float32)
    sky = probe.scene_ray_comparison(model, np.empty((2, 0)), origins, directions, all_miss, 4.0)
    assert all(row["compared_count"] == 16 and row["witness_count"] == 0 for row in sky)
    for index, pixel in enumerate(probe.SAMPLE_PIXELS):
        directions[:, pixel] = [0.2 + index * 0.02, 0.0, -0.9 + index * 0.025]
    directions[:, probe.SAMPLE_PIXELS[-1]] = [0.96, 0.0, -0.25]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    actual = all_miss.copy()
    for pixel in probe.SAMPLE_PIXELS:
        actual[:, pixel] = -origins[:, pixel, 2] / directions[:, pixel, 2]
    assert 2.5 < actual.max() < 4.0  # Raw range is longer than optical-Z far clipping.
    control = probe.scene_ray_comparison(
        model, np.empty((2, 0)), origins, directions, actual, 4.0, positive=True
    )
    assert all(row["witness_count"] == 16 and row["witness_span_m"] > 0.1 for row in control)
    one_lane_bad = actual.copy()
    one_lane_bad[1] = -1
    nan = actual.copy()
    nan[0, probe.SAMPLE_PIXELS[0]] = float("nan")
    for bad in (
        all_miss, actual + np.where(actual >= 0, 0.2, 0.0), one_lane_bad, nan,
        np.where(actual >= 0, 1.0, actual),
    ):
        try:
            probe.scene_ray_comparison(model, np.empty((2, 0)), origins, directions, bad, 4.0, positive=True)
        except ValueError:
            pass
        else:
            raise AssertionError("a dead or wrong-distance downward sensor must fail")
    rows = control[0]["rows"]
    rows[0].update(cpu_range_m=3.99, gpu_range_m=-1.0)
    assert probe.validate_range_rows(rows, 4.0, positive=True)["compared_count"] == 15
    rows[0]["cpu_range_m"] = 3.97
    try:
        probe.validate_range_rows(rows, 4.0, positive=True)
    except ValueError:
        pass
    else:
        raise AssertionError("the 2cm limit margin must not hide a clear missed hit")
    print("LEGAL_ALL_FAR_WITH_POSITIVE_CONTROL_AND_BAD_SENSOR_REJECTION_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
