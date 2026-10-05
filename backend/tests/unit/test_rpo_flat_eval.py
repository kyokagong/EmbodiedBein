import math

import numpy as np
import pytest

from eb.runners.rpo_flat_artifact import (
    case_success,
    command_frame_velocity,
    summarize_cases,
    tracking_metrics,
)


def valid_case(seed=50000, **changes):
    return {
        "seed": seed,
        "completed": True,
        "episode_len": 1000,
        "actual_steps": 1000,
        "metric_steps": 1000,
        "fall": False,
        "nan": False,
        "hard_limit": False,
        "velocity_rmse": 0.20,
        "yaw_rmse": 0.30,
        "success": True,
        **changes,
    }


def test_golden_check_allows_float_rounding_but_rejects_changed_actions():
    from eb.runners.rpo_flat_artifact import golden_error_metrics

    expected = np.array([[320.923767, 0.0, 99.0]], dtype=np.float32)
    actual = np.array([[320.923920, 0.00002, 99.00009]], dtype=np.float32)
    measured = golden_error_metrics(expected, actual)
    assert measured["max_abs_error"] > 1e-4
    assert measured["max_error_ratio"] <= 1
    assert measured["max_clipped_action_error"] <= 1e-4
    single = golden_error_metrics(np.array([-25.35654068]), np.array([-25.35664749]))
    assert single["max_clipped_action_error"] > 1e-4
    assert single["max_joint_target_error_rad"] < 1e-4
    for changed in (actual + 0.01, actual * np.nan):
        with pytest.raises(ValueError, match="golden numerical"):
            golden_error_metrics(expected, changed)


def test_success_requires_full_finite_horizon_and_both_velocity_limits():
    assert case_success(valid_case())
    for change in (
        {"completed": False},
        {"actual_steps": 999},
        {"metric_steps": 999},
        {"fall": True},
        {"nan": True},
        {"hard_limit": True},
        {"velocity_rmse": 0.200001},
        {"yaw_rmse": 0.300001},
        {"velocity_rmse": float("nan")},
        {"yaw_rmse": None},
    ):
        assert not case_success(valid_case(**change))


def test_tracking_uses_vector_norm_and_only_actual_finite_samples():
    commands = np.zeros((3, 3))
    measured = np.array([[0.3, 0.4, 0.2], [0.0, 0.0, -0.2], [np.nan, 0.0, 0.0]])
    metrics = tracking_metrics(commands, measured)
    assert metrics["metric_steps"] == 2
    assert metrics["velocity_rmse"] == pytest.approx(math.sqrt(0.25 / 2))
    assert metrics["yaw_rmse"] == pytest.approx(0.2)
    assert tracking_metrics([], []) == {
        "metric_steps": 0,
        "metrics_available": False,
        "velocity_rmse": None,
        "yaw_rmse": None,
    }
    velocity = command_frame_velocity(
        {
            "root_quat_w": np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]),
            "root_lin_vel_w": np.array([0.0, 0.4, 1.0]),
            "root_ang_vel_w": np.array([0.0, 0.0, 0.2]),
        }
    )
    assert velocity == pytest.approx([0.4, 0.0, 0.2])


def test_physical_limits_copy_device_joint_indices_before_numpy_indexing():
    from types import SimpleNamespace
    from eb.runners import rpo_flat_artifact as evaluator

    class DeviceTensor:
        def __init__(self, value): self.value = np.array(value)
        def __getitem__(self, key): return DeviceTensor(self.value[key])
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
        def __array__(self, *args): raise TypeError("device tensor requires cpu()")

    data = SimpleNamespace(joint_pos_limits=DeviceTensor([[[-1, 1]] * 3]),
        joint_vel_limits=DeviceTensor([[100, 100, 100]]),
        joint_effort_limits=DeviceTensor([[100, 100, 100]]))
    actuator = SimpleNamespace(joint_indices=DeviceTensor([0, 2]),
        velocity_limit=DeviceTensor([[5, 6]]), effort_limit=DeviceTensor([[7, 8]]))
    limits = evaluator._hard_limits(SimpleNamespace(robot=SimpleNamespace(
        data=data, actuators={"motor":actuator})))
    np.testing.assert_array_equal(limits["joint_vel"], [5, 100, 6])
    np.testing.assert_array_equal(limits["applied_torque"], [7, 100, 8])


def test_loop_uses_pre_step_command_and_records_actual_extent(monkeypatch, tmp_path):
    import gzip
    import json
    import sys
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    import eb.rpo_wbc as codec
    import eb.runners.rpo_flat_artifact as evaluator

    monkeypatch.setitem(sys.modules, "rpo_wbc", codec)
    tensors = {
        "root_pos_w": [[0.0, 0.0, 0.8]],
        "root_quat_w": [[1.0, 0.0, 0.0, 0.0]],
        "root_lin_vel_b": [[0.0, 0.0, 0.0]],
        "root_ang_vel_b": [[0.0, 0.0, 0.0]],
        "root_lin_vel_w": [[0.0, 0.0, 0.0]],
        "root_ang_vel_w": [[0.0, 0.0, 0.0]],
        "projected_gravity_b": [[0.0, 0.0, -1.0]],
        "joint_pos": [[0.0] * 23],
        "joint_vel": [[0.0] * 23],
        "applied_torque": [[0.0] * 23],
    }
    data = SimpleNamespace(**{key: torch.tensor(value) for key, value in tensors.items()})
    commands = SimpleNamespace(
        command=torch.tensor([[0.8, 0.1, 0.5]]),
        heading_target=torch.zeros(1),
        is_standing_env=torch.zeros(1, dtype=torch.bool),
    )
    obs = {"policy": torch.zeros(1, 780), "critic": torch.zeros(1, 1390)}
    env = SimpleNamespace(
        robot=SimpleNamespace(data=data, joint_names=codec.RPO_LAB_JOINT_ORDER),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        command_generator=commands,
        device="cpu",
        step_dt=0.02,
        clip_actions=100.0,
    )
    env.reset = lambda seed: (obs, {})
    steps = 0

    def step(action):
        nonlocal steps
        steps += 1
        data.root_lin_vel_w[0, :2] = commands.command[0, :2]
        data.root_ang_vel_w[0, 2] = commands.command[0, 2]
        # Resample after physics, at the official ten-second boundary.
        if steps == 500:
            commands.command[:] = torch.tensor([[-0.6, -0.5, -1.57]])
        return obs, 0.0, torch.tensor([False]), torch.tensor([steps == 1000]), {}

    env.step = step
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="observation")],
        run=lambda outputs, inputs: [np.zeros((1, 23), dtype=np.float32)],
    )
    monkeypatch.setattr(evaluator, "_initial_state", lambda *args: {})
    monkeypatch.setattr(
        evaluator,
        "_hard_limits",
        lambda *args: {
            "joint_pos": np.array([[-1.0, 1.0]] * 23),
            "joint_vel": np.ones(23),
            "applied_torque": np.ones(23),
        },
    )
    row = evaluator._evaluate_case(session, env, 50000, tmp_path, False)
    assert row["success"] and row["actual_steps"] == row["metric_steps"] == 1000
    assert row["velocity_rmse"] == row["yaw_rmse"] == 0.0
    with gzip.open(tmp_path / row["trace"]["path"], "rt") as stream:
        boundary = [json.loads(line) for line in stream][499]
    assert boundary["command_pre"] == pytest.approx([0.8, 0.1, 0.5])
    assert boundary["command_post"] == pytest.approx([-0.6, -0.5, -1.57])
    assert boundary["metric_velocity"] == pytest.approx([0.8, 0.1, 0.5])
    # Exercise the actual recording loop: 25 fps at 50 Hz must capture 501 frames,
    # and disabling macroblock resizing must preserve 1080 rather than 1088 rows.
    from types import ModuleType
    frames, options = [], {}
    imageio = ModuleType("imageio")
    imageio.v2 = ModuleType("imageio.v2")
    def writer(path, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(append_data=frames.append, close=lambda: None)
    imageio.v2.get_writer = writer
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio.v2)
    monkeypatch.setattr(evaluator, "_video_frame", lambda env: steps)
    steps = 0
    row = evaluator._evaluate_case(session, env, 60000, tmp_path, True, "hd")
    assert row["success"] and len(frames) == 501 and frames[:3] == [0, 2, 4]
    assert options["fps"] == 25 and options["macro_block_size"] == 1
    assert options["ffmpeg_params"][:2] == ["-crf", "18"]
    obs["policy"][0, 0] = float("nan")
    row = evaluator._evaluate_case(session, env, 50001, tmp_path, False)
    assert row["completed"] and row["nan"] and not row["success"]
    assert row["actual_steps"] == row["metric_steps"] == 0
    assert row["velocity_rmse"] is None


def test_summary_keeps_missing_infrastructure_in_denominator_and_rejects_forgery():
    good = valid_case()
    bad = valid_case(
        50001,
        episode_len=1,
        actual_steps=1,
        metric_steps=1,
        velocity_rmse=0.4,
        yaw_rmse=0.6,
        fall=True,
        success=False,
    )
    result = summarize_cases([good, bad], 100)
    assert result["cases"] == {"planned": 100, "completed": 2, "valid": 2, "invalid": 0, "infra_failed": 98}
    assert result["metrics"]["n_trials"] == 100
    assert result["metrics"]["success_rate"] == 0.01
    assert result["metrics"]["fall_count"] == 1
    assert result["metrics"]["velocity_rmse"] == pytest.approx(math.sqrt((0.2**2 * 1000 + 0.4**2) / 1001))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cases([good, good], 100)
    with pytest.raises(ValueError, match="success"):
        summarize_cases([valid_case(fall=True)], 100)
