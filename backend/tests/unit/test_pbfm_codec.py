import zipfile

import numpy as np
import pytest

from eb.runners.pbfm_codec import PolicyCodec, depth_input
from eb.runners.pbfm_eval import diagnostic_metrics, forbidden_contact_summary
from eb.runners.pbfm_export import archive_package, compare_paired_traces


def config():
    return {
        "history": {key: 4 for key in ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")},
        "obs_scales": {
            key: 0.25 if key == "base_ang_vel" else 1.0
            for key in ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity", "history_actor")
        },
        "default_dof_pos": [0.0] * 23,
        "action_target_scale": [0.6] * 23,
        "action_clip": 5.0,
        "normalize_action_from": 1.0,
        "normalize_action_to": 5.0,
    }


def test_depth_boundary_normalizes_uint8_once_and_rejects_float_bytes():
    value = depth_input(np.asarray([0, 128, 255], dtype=np.uint8))
    np.testing.assert_allclose(value, [0.0, 128 / 255.0, 1.0])
    np.testing.assert_array_equal(depth_input(value), value)
    with pytest.raises(ValueError):
        depth_input(np.asarray([0.0, 128.0, 255.0], dtype=np.float32))
    with pytest.raises(ValueError):
        depth_input(np.asarray([np.nan], dtype=np.float32))


def test_history_is_previous_newest_first_and_partial_reset_is_isolated():
    codec = PolicyCodec(config(), batch=2)
    values = (
        np.ones((2, 23)),
        np.ones((2, 23)) * 2,
        np.asarray([[0.0, 0.0, -1.0]] * 2),
        np.ones((2, 3)) * 4,
        np.ones((2, 23)) * 3,
    )
    first = codec.observations(*values)
    assert first["state"].shape == (2, 52)
    assert first["history_actor"].shape == (2, 300)
    assert not first["history_actor"].any()
    np.testing.assert_array_equal(first["state"][:, -3:], np.ones((2, 3)))
    codec.reset([0])
    second = codec.observations(*values)
    assert not second["history_actor"][0].any()
    np.testing.assert_array_equal(second["history_actor"][1, :23], np.full(23, 3))
    assert not second["history_actor"][1, 23:92].any()


def test_action_target_preserves_predelay_command_and_delayed_pd_semantics():
    codec = PolicyCodec(config())
    clipped, target = codec.action_target(np.full((1, 23), 2.0), default_offset=0.1)
    np.testing.assert_array_equal(clipped, np.full((1, 23), 5.0))
    np.testing.assert_allclose(target, np.full((1, 23), 3.1))
    clipped, target = codec.action_target(
        np.full((1, 23), 2.0), default_offset=0.1, delayed_command=np.full((1, 23), -1.0)
    )
    np.testing.assert_array_equal(clipped, np.full((1, 23), 5.0))
    np.testing.assert_allclose(target, np.full((1, 23), -0.5))


def test_portable_archive_contains_manifest_assets_and_codecs(tmp_path):
    package = tmp_path / "policy"
    (package / "robot").mkdir(parents=True)
    (package / "manifest.json").write_text("{}")
    (package / "robot/rpo.xml").write_text("<mujoco/>")
    (package / "codec.json").write_text("{}")
    archive_package(package, tmp_path)
    with zipfile.ZipFile(tmp_path / "policy.zip") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "policy/manifest.json",
            "policy/robot/rpo.xml",
            "policy/codec.json",
        }


def test_physical_diagnostics_use_measured_contact_for_stance_slip():
    joints = np.full((2, 23), 0.1)
    bodies = np.zeros((2, 3, 3))
    velocities = np.zeros_like(bodies)
    velocities[:, 1, 0] = 1.0
    velocities[:, 2, 0] = 5.0
    forces = np.zeros_like(bodies)
    forces[:, 1, 2] = 10.0
    result = diagnostic_metrics(
        joints,
        bodies,
        velocities,
        forces,
        [1, 2],
        np.tile([0.3, 0.0], (2, 1)),
        [0.3, 0.0],
        np.ones((2, 23)),
        np.full((3, 2, 2), 0.02),
        np.zeros_like(joints),
        np.zeros_like(bodies),
    )
    assert result["foot_slip_m_s"] == 1.0
    assert result["stance_samples"] == 2
    assert result["velocity_rmse_m_s"] == 0
    assert result["joint_tracking_rmse_rad"] == pytest.approx(0.1)
    assert result["action_saturation_fraction"] == 1.0
    missing = diagnostic_metrics(
        joints,
        bodies,
        velocities,
        np.zeros_like(forces),
        [1, 2],
        np.tile([0.3, 0.0], (2, 1)),
        [0.3, 0.0],
        np.ones((2, 23)),
        np.full((3, 2, 2), 0.02),
        np.zeros_like(joints),
        np.zeros_like(bodies),
    )
    assert missing["foot_slip_m_s"] is None and missing["foot_slip_status"] == "not_observed"


def test_forbidden_torso_contact_cannot_hide_behind_valid_standing_height():
    forces = np.zeros((2, 24, 3))
    forces[:, [7, 13], 2] = 100.0
    assert forbidden_contact_summary(forces, [7, 13])["forbidden_contact_count"] == 0
    forces[1, 0, 2] = 10.0
    result = forbidden_contact_summary(forces, [7, 13])
    assert result == {"forbidden_contact_count": 1, "forbidden_contact_peak_n": 10.0}


def test_independent_paired_traces_report_drift_and_termination_without_quality_gate():
    native = {"actions": np.zeros((32, 23)), "root_xyz": np.zeros((33, 3)), "terminated": False}
    onnx = {
        "actions": np.full((1, 23), 1e-4),
        "root_xyz": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        "terminated": True,
    }
    summary = compare_paired_traces(native, onnx)
    assert summary["shared_prefix_steps"] == 1
    assert summary["termination_matches"] is False
    assert summary["max_root_abs_difference_m"] == 0.2
    assert summary["quality_passed"] is None
    onnx["root_xyz"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        compare_paired_traces(native, onnx)
