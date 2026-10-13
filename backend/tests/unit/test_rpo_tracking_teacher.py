import numpy as np
import pytest

from eb.domain.rlinf_form import build_overrides, get_template, template_image
from eb.runners.rpo_tracking_reference import ACTOR_DIM, CRITIC_DIM, future_indices, residual_action


def test_reference_residual_matches_frozen_controller():
    from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q

    q = np.array(DEFAULT_Q) + 0.25 * np.array(ACTION_GAIN) * 0.4
    np.testing.assert_allclose(residual_action(q, np.zeros(23)), 0.4)
    np.testing.assert_allclose(residual_action(q, np.ones(23)), 1.0)


def test_future_reference_never_crosses_clip_boundary():
    np.testing.assert_array_equal(
        future_indices(np.array([0, 145, 149]), 150), [[1, 5, 10], [146, 149, 149], [149, 149, 149]]
    )
    assert ACTOR_DIM == 547 and CRITIC_DIM == 905


def test_teacher_template_is_separate_and_single_gpu():
    t = get_template("rpo_v2_bfm_zero_tracking_teacher")
    assert t.embodiment_env_id == "ee_rpo_v2_tracking_23dof_v1"
    assert template_image(t) == "eb/rlinf:rpo-v2-bfm-zero"
    assert build_overrides(t.id, 1, "test", [])
    with pytest.raises(ValueError, match="single GPU"):
        build_overrides(t.id, 4, "test", [])


def test_teacher_contract_and_reward_are_separate():
    from pathlib import Path

    from eb.domain.contract import load_env

    contract = load_env(Path("/workspace/contracts/ee_rpo_v2_tracking_23dof_v1.yaml"))
    assert contract.observation.dim == 547
    assert contract.observation.critic_dim == 905
    assert contract.action.offset == "next_reference_joint_pos"


def test_reference_binding_rejects_modified_clip(tmp_path):
    import hashlib
    import json

    from eb.runners.rpo_tracking_reference import reference_receipt

    clips = []
    for i in range(4):
        file = tmp_path / f"{i}.npz"
        file.write_bytes(b"reference")
        clips.append(
            {"file": file.name, "frames": 150, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        )
    receipt = {
        "schema_version": "eb/100style-rpo-retarget-trial/v1",
        "asset_sha256": "a" * 64,
        "recipe": "xsens-yxz-native23-limb-ground-bounded-ik-v2",
        "training_ready": False,
        "clips": clips,
    }
    path = tmp_path / "retarget-receipt.json"
    path.write_text(json.dumps(receipt))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference_receipt(tmp_path, sha, "a" * 64) == receipt
    (tmp_path / "1.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="clip changed"):
        reference_receipt(tmp_path, sha, "a" * 64)


def test_reference_run_requires_finished_run_and_succeeded_job():
    from types import SimpleNamespace

    from eb.api.routes_train import require_tracking_source

    run = SimpleNamespace(status="finished")
    job = SimpleNamespace(status="succeeded", spec={"params": {"operation": "retarget_100style_probe"}})
    require_tracking_source(run, job)
    job.status = "failed"
    with pytest.raises(ValueError, match="succeeded"):
        require_tracking_source(run, job)


def test_teacher_admission_uses_ppo_memory_not_vla_default():
    from eb.capacity.estimator import estimate_job
    from eb.domain.job import JobSpec

    spec = JobSpec(
        kind="train.rl",
        embodiment_env_id="ee_rpo_v2_tracking_23dof_v1",
        task_id="tsk_rpo_tracking_teacher",
        resources={"gpus": 1},
        params={"template_id": "rpo_v2_bfm_zero_tracking_teacher"},
        overrides=["env.train.total_num_envs=64"],
    )
    estimate = estimate_job(spec)
    assert estimate.verdict == "fits"
    assert estimate.per_gpu_peak_gb < 10
    assert estimate.basis["inputs"]["optimizer_bytes"] == 16


def test_teacher_checkpoint_interval_must_be_positive():
    for interval in [0, -1]:
        with pytest.raises(ValueError, match="positive checkpoint"):
            build_overrides(
                "rpo_v2_bfm_zero_tracking_teacher", 1, "test", [f"runner.save_interval={interval}"]
            )


def test_nonfinite_case_keeps_failure_with_nullable_metrics():
    import json

    from eb.runners.rpo_bfm_motion_probe import finite_case, tracking_pass

    case = {
        "finite": False,
        "passed": False,
        "root_position_rmse_m": float("nan"),
        "minimum_height_m": float("inf"),
        "steps": 3,
        "ended_early": True,
    }
    clean = finite_case(case)
    assert clean["root_position_rmse_m"] is None
    assert clean["minimum_height_m"] is None
    assert clean["finite"] is False and tracking_pass(clean) is False
    json.dumps(clean, allow_nan=False)


def test_tracking_validation_uses_separate_startup_seed_processes(tmp_path):
    from eb.runners.rpo_bfm_motion_probe import tracking_validation_commands

    commands = tracking_validation_commands(
        tmp_path / "ref", tmp_path / "assets.json", tmp_path / "probe.json", tmp_path / "teacher.pt", "a" * 64
    )
    assert len(commands) == 3
    assert [int(c[c.index("--validation-seed") + 1]) for c in commands] == [142, 143, 144]
    assert all("--video" in c for c in commands)
