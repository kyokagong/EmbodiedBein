import numpy as np
import pytest


def codec():
    return {
        "default_pos": [0.0] * 23,
        "history_length": 4,
        "history_order": ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
        "action_gain": [0.72] * 23,
        "feedback_action_scale": 5.0,
    }


def test_codec_history_is_term_major_previous_only_and_feedback_is_applied():
    from eb.runners.ufo_native_codec import ObservationCodec

    c = ObservationCodec(codec())
    obs = c.observation(np.ones(23), np.full(23, 2.0), [0, 0, -1], [3, 4, 5])
    assert obs.shape == (375,)
    np.testing.assert_array_equal(obs[:23], 1.0)
    np.testing.assert_array_equal(obs[75:], 0.0)
    target = c.apply(np.full(23, 0.5))
    np.testing.assert_allclose(target, 0.36)
    next_obs = c.observation(np.full(23, 7.0), np.zeros(23), [0, 0, -1], [0, 0, 0])
    np.testing.assert_array_equal(next_obs[52:75], 2.5)
    # The previous frame's action was zero, then its omega group is newest-first.
    np.testing.assert_array_equal(next_obs[75 : 75 + 92], 0.0)
    np.testing.assert_array_equal(next_obs[167:170], [3, 4, 5])
    c.reset()
    np.testing.assert_array_equal(c.observation(np.zeros(23), np.zeros(23), [0, 0, -1], [0, 0, 0])[52:], 0.0)


def test_codec_rejects_bad_shape_nonfinite_input_and_missing_task():
    from eb.runners.ufo_native_codec import ObservationCodec

    c = ObservationCodec(codec())
    with pytest.raises(ValueError):
        c.observation(np.zeros(22), np.zeros(23), [0, 0, -1], [0, 0, 0])
    with pytest.raises(ValueError):
        c.apply(np.full(23, np.nan))


def test_export_check_exception_cannot_count_as_verification(tmp_path):
    from eb.runners.ufo_native_package import verify_export

    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        verify_export(None, tmp_path / "missing.onnx", {"actor_dim": 631}, {})


def test_independent_fall_is_not_erased_by_recovery():
    from eb.runners.ufo_native_sim import case_metrics

    values = {
        "height": [0.75, 0.2, 0.75],
        "tilt": [0.0, 0.0, 0.0],
        "velocity": [[0.0, 0.0]] * 3,
        "xy": [[0.0, 0.0]] * 3,
        "joint_violation": [False] * 3,
        "velocity_violation": [False] * 3,
        "torque_violation": [False] * 3,
        "latency_ms": [1.0, 1.0, 1.0],
    }
    status, reason, metrics = case_metrics(values, speed=0.0, yaw=0.0)
    assert status == "task_failure" and reason == "fall"
    assert metrics["first_fall_step"] == 2


def test_dc_motor_curve_preserves_braking_above_no_load_velocity():
    from eb.runners.ufo_native_sim import motor_torque

    result = motor_torque(
        np.array([120.0, -120.0]),
        np.array([30.0, -30.0]),
        np.array([120.0, 120.0]),
        np.array([25.0, 25.0]),
        np.array([5.0, 5.0]),
    )
    assert result[0] < 0 < result[1]
    np.testing.assert_allclose(result[0], -result[1])


def test_mutated_native_parent_is_rejected_before_environment_build(tmp_path):
    import json

    from eb.runners.ufo_native_package import verified_manifest
    from eb.ufo_native import plan_hash

    (tmp_path / "config.json").write_text("{}")
    manifest = {"files": [{"path": "config.json", "bytes": 2, "sha256": "0" * 64}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        verified_manifest(tmp_path, plan_hash(manifest))


def test_command_delay_backfills_first_target_and_resets():
    from eb.runners.ufo_native_sim import CommandDelay

    delay = CommandDelay(2)
    assert [float(delay.push(np.array([n]))[0]) for n in range(4)] == [0, 0, 0, 1]
    fresh = CommandDelay(2)
    assert float(fresh.push(np.array([9]))[0]) == 9
