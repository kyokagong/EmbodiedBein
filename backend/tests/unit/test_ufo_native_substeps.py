import json
from types import SimpleNamespace

import numpy as np
import pytest

from eb.runners import ufo_native_sim as sim


def test_simulation_loads_delivered_runtime_instead_of_image_copy(tmp_path):
    (tmp_path / "runtime.py").write_text(
        "class Policy:\n    def __init__(self, path):\n        self.payload = path\n"
    )
    assert sim.package_policy(tmp_path).payload == tmp_path


def test_native_state_com_uses_entity_indexing_without_reward_specific_cache(monkeypatch):
    class Tensor(np.ndarray):
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self)

    def tensor(value):
        return np.asarray(value, dtype=float).view(Tensor)

    backend = sim.MjlabBackend.__new__(sim.MjlabBackend)
    backend.qpos = np.zeros(30)
    data = SimpleNamespace(
        body_com_pose_w=tensor([[[3, 0, 0, 1, 0, 0, 0], [9, 0, 0, 1, 0, 0, 0]]]),
        model=SimpleNamespace(body_mass=tensor([[0, 3, 1]])),
    )
    backend.core = SimpleNamespace(
        robot=SimpleNamespace(data=data, indexing=SimpleNamespace(body_ids=np.array([2, 1]))),
        _body_ids=np.array([1, 0]),
        feet_indices=np.array([0, 1]),
        robot_root_states=tensor(np.zeros((1, 13))),
        **{key: tensor(np.zeros((1, 2, 3))) for key in ("body_pos", "body_vel", "contact_forces")},
        **{
            key: tensor(np.zeros((1, 3)))
            for key in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel", "torques")
        },
    )
    monkeypatch.setattr(sim, "orientation", lambda quat: (None, 0.0, 0.0))
    np.testing.assert_allclose(backend.state()["com_position_w"], [7.5, 0, 0])


def test_physical_sample_records_limit_excess_and_absolute_peaks():
    sample = sim.physical_sample(
        np.array([1.2, -0.4]),
        np.array([3.0, -0.5]),
        np.array([-5.0, 0.0]),
        (np.array([-1.0, -1.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([4.0, 4.0])),
    )
    np.testing.assert_allclose(sample, [0.2, 1.0, 1.0, 1.2, 3.0, 5.0])


def test_com_weights_true_body_centers_and_ignores_zero_mass_world():
    centers = np.array([[-100.0, -100.0, -100.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    np.testing.assert_allclose(sim.mass_weighted_com(centers, np.array([0.0, 1.0, 3.0])), [4, 5, 6])
    np.testing.assert_allclose(
        sim.mass_weighted_com(np.stack([centers, centers]), np.array([[0, 1, 3], [0, 3, 1]])),
        [[4, 5, 6], [2, 3, 4]],
    )


def test_transient_substep_violations_survive_a_clean_final_sample():
    samples = np.zeros((8, 6))
    samples[0] = [0.2, 0.0, 0.0, 1.2, 0.5, 1.0]
    samples[1] = [0.0, 0.3, 0.0, 0.5, 2.3, 1.0]
    samples[2] = [0.0, 0.0, 0.5, 0.5, 0.5, 4.5]
    metrics = sim.physical_metrics(samples, 0.0025)
    assert metrics["physics_substeps"] == 8
    for kind in ("joint", "velocity", "torque"):
        assert metrics[f"{kind}_violation_count"] == 1
        assert metrics[f"{kind}_violation_fraction"] == 0.125
        assert metrics[f"{kind}_violation_seconds"] == 0.0025
    assert metrics["max_joint_excess_rad"] == 0.2
    assert metrics["max_velocity_excess_rad_s"] == 0.3
    assert metrics["max_torque_excess_nm"] == 0.5
    assert metrics["max_abs_joint_position_rad"] == 1.2
    assert metrics["max_abs_velocity_rad_s"] == 2.3
    assert metrics["max_abs_torque_nm"] == 4.5


def test_substep_tolerance_does_not_count_roundoff_as_a_violation():
    metrics = sim.physical_metrics(np.array([[0.001, 0.0001, 0.0, 0.0, 0.0, 0.0]]), 0.0025)
    assert metrics["joint_violation_count"] == 0
    assert metrics["velocity_violation_count"] == 0
    assert metrics["torque_violation_count"] == 0


def test_native_substep_hook_samples_after_update_only_during_tracking():
    backend = sim.MjlabBackend.__new__(sim.MjlabBackend)
    data = SimpleNamespace(
        joint_pos=np.zeros((1, 2)),
        joint_vel=np.zeros((1, 2)),
        qfrc_actuator=np.zeros((1, 2)),
    )
    backend.core = SimpleNamespace(robot=SimpleNamespace(data=data), _joint_ids=[1])
    backend.torch = np
    backend.limits = (np.array([-1.0]), np.array([1.0]), np.array([2.0]), np.array([4.0]))
    backend._substeps = None
    backend._scene_update = lambda dt: data.joint_pos.__setitem__((0, 1), 1.2)
    backend._track_substep(0.0025)
    assert backend._substeps is None
    data.joint_pos[:] = 0.0
    backend._substeps = []
    backend._track_substep(0.0025)
    np.testing.assert_allclose(backend._substeps, [[0.2, 0.0, 0.0, 1.2, 0.0, 0.0]])


def test_displacement_uses_reset_position_and_substep_latch_fails_case():
    trace = {
        "height": [0.75] * 1000,
        "tilt": [0.0] * 1000,
        "velocity": [[0.3, 0.0]] * 1000,
        "xy": [[0.01, 0.0]] * 999 + [[4.0, 0.0]],
        "joint_violation": [False] * 1000,
        "velocity_violation": [False] * 1000,
        "torque_violation": [False] * 1000,
        "latency_ms": [1.0] * 1000,
        "substep_limits": np.zeros((1000, 8, 6)),
    }
    status, reason, metrics = sim.case_metrics(trace, speed=0.3, yaw=0.0, initial_xy=[0.0, 0.0])
    assert (status, reason) == ("success", "completed")
    assert metrics["forward_displacement_m"] == 4.0
    trace["substep_limits"][0, 0, 0] = 0.1
    status, reason, metrics = sim.case_metrics(trace, speed=0.3, yaw=0.0, initial_xy=[0.0, 0.0])
    assert (status, reason) == ("task_failure", "physical_limit")
    assert metrics["joint_violation_count"] == 1
    assert metrics["joint_violation_fraction"] == 1 / 8000


def test_delay_history_starts_with_first_target_and_cannot_leak_between_cases():
    delay = sim.CommandDelay(2)
    assert [float(delay.push(np.array([n]))[0]) for n in (7, 8, 9, 10)] == [7, 7, 7, 8]
    fresh = sim.CommandDelay(2)
    assert float(fresh.push(np.array([-3]))[0]) == -3


def test_feet_contact_forces_have_world_frame_signs_and_include_self_collisions():
    backend = sim.MujocoBackend.__new__(sim.MujocoBackend)
    backend.feet_ids = [1, 2]
    backend.model = SimpleNamespace(geom_bodyid=np.array([0, 1, 2]))
    contacts = [
        SimpleNamespace(geom1=0, geom2=1, frame=np.eye(3)),
        SimpleNamespace(geom1=1, geom2=2, frame=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])),
    ]
    backend.data = SimpleNamespace(ncon=2, contact=contacts, xpos=np.arange(9).reshape(3, 3))
    forces = [[0, 0, 10, 0, 0, 0], [0, 3, 0, 0, 0, 0]]

    def contact_force(model, data, index, result):
        result[:] = forces[index]

    def object_velocity(model, data, kind, body, result, local):
        result[:] = [0, 0, 0, body, body + 1, body + 2]

    backend.mj = SimpleNamespace(
        mj_contactForce=contact_force,
        mj_objectVelocity=object_velocity,
        mjtObj=SimpleNamespace(mjOBJ_XBODY=2),
    )
    state = backend._feet_state()
    np.testing.assert_array_equal(state["foot_position_w"], [[3, 4, 5], [6, 7, 8]])
    np.testing.assert_array_equal(state["foot_velocity_w"], [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(state["foot_contact_force_w"], [[-3, 0, 10], [3, 0, 0]])


def test_rollout_equivalence_checks_raw_actions_and_scaled_targets():
    action = np.array([0.2, -0.3])
    assert sim.action_errors(action, action.copy(), np.ones(2)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="equivalent"):
        sim.action_errors(np.array([1.2]), np.array([1.3]), np.ones(1))
    with pytest.raises(ValueError, match="equivalent"):
        sim.action_errors(np.array([0.2]), np.array([0.200001]), np.array([1000.0]))
    with pytest.raises(ValueError, match="equivalent"):
        sim.action_errors(np.array([np.nan]), np.array([0.0]), np.ones(1))


def test_completed_case_is_flushed_with_hashes_and_failed_next_case_keeps_it(tmp_path):
    from eb.ufo_native import sha256_file

    (tmp_path / "videos").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "cases.jsonl").write_text("")
    (tmp_path / "videos/first.mp4").write_bytes(b"verified-video")
    row = {"case_id": "first", "video": "videos/first.mp4", "trace": "traces/first.npz"}
    sim.save_case(tmp_path, row, {"q": [[1.0, 2.0]]})
    saved = json.loads((tmp_path / "cases.jsonl").read_text())
    assert saved["case_id"] == "first"
    assert [f["path"] for f in saved["evidence_files"]] == ["videos/first.mp4", "traces/first.npz"]
    for item in saved["evidence_files"]:
        path = tmp_path / item["path"]
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == sha256_file(path)
    np.testing.assert_array_equal(np.load(tmp_path / "traces/first.npz")["q"], [[1.0, 2.0]])
    with pytest.raises(FileNotFoundError):
        sim.save_case(tmp_path, {"video": "videos/missing.mp4", "trace": "traces/missing.npz"}, {"q": [0]})
    assert (tmp_path / "cases.jsonl").read_text().count("\n") == 1
