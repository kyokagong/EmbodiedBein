import hashlib

import numpy as np
import pytest

from eb.runners.rpo_ufo_initial_contact import PROTOCOL, contact_metrics, validate_receipt


def test_paired_contact_probe_checks_raw_shapes_hash_and_measured_summary(tmp_path):
    arrays = {
        "q0": np.zeros((16, 32, 23)),
        "dq0": np.zeros((16, 32, 23)),
        "root0": np.zeros((16, 32, 13)),
        "foot_link_height0_m": np.zeros((16, 32, 2)),
        "decoder_projection_residual_rad": np.zeros((16, 32, 23)),
        "foot_vertical_N": np.zeros((2, 16, 4, 32, 2)),
        "foot_vertical_history_peak_N": np.zeros((2, 16, 4, 32, 2)),
        "joint_vel_rad_s": np.zeros((2, 16, 4, 32, 23)),
        "joint_acc_rad_s2": np.zeros((2, 16, 4, 32, 23)),
        "hard_flags": np.zeros((2, 16, 4, 32, 3), dtype=bool),
    }
    arrays["foot_vertical_N"][0, 0, 0, 0, 0] = 700
    arrays["hard_flags"][1, 0, :, 0, 1] = True
    metrics = contact_metrics(arrays)
    assert metrics[0]["statistics"]["peak_abs_foot_vertical_N"]["max"] == 700
    assert metrics[1]["hard_first_control_count"] == 1
    raw = tmp_path / "initial-contact.npz"
    np.savez_compressed(raw, **arrays)
    validation = {"asset_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    receipt = {
        "protocol": PROTOCOL,
        "status": "measured",
        "samples_per_mode": 512,
        **validation,
        "metrics": metrics,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }
    validate_receipt(receipt, validation, tmp_path)
    receipt["metrics"][0]["statistics"]["peak_abs_foot_vertical_N"]["max"] += 1e-5
    validate_receipt(receipt, validation, tmp_path)
    receipt["metrics"][1]["hard_first_control_count"] = 0
    with pytest.raises(ValueError, match="summary differs"):
        validate_receipt(receipt, validation, tmp_path)
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw evidence changed"):
        validate_receipt(receipt, validation, tmp_path)
