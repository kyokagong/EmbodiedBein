from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from eb.data_validation import require_amp_motion_dataset
from eb.adapters.data_convert import DataConvertAdapter
from eb.adapters.mujoco import ToolsAdapter
from eb.domain.contract import load_env
from eb.domain.environment_profile import rpo_eval_arguments, validate_definition
from eb.domain.job import JobKind, JobSpec
from eb.domain.rlinf_form import build_overrides, get_template, template_image
from eb.gates.engine import GateContext, g1_submit
from eb.paths import validate_allowed_path
from eb.orchestrator.collect import (validate_rpo_amp_motion_probe_receipt,
                                     validate_rpo_amp_probe_receipt)
from eb.rpo_wbc import AMP_MOTION_CONVERTER_REVISION, JOINT_ORDER, RPO_LAB_JOINT_ORDER


def test_amp_template_is_rlinf_ppo_with_valid_sampling_budget():
    tpl = get_template("rpo_v2_amp_stand_walk_ppo")
    assert tpl.kind == "ppo" and tpl.requires_dataset
    assert tpl.embodiment_env_id == "ee_rpo_v2_amp_locomotion_23dof_v1"
    assert template_image(tpl) == "eb/rlinf:rpo-v2-wbc"
    overrides = build_overrides(tpl.id, 1, "amp-smoke", [])
    assert "env.train.total_num_envs=512" in overrides
    assert "actor.global_batch_size=4096" in overrides
    assert "actor.micro_batch_size=1024" in overrides
    with pytest.raises(ValueError, match="单 learner"):
        build_overrides(tpl.id, 2, "bad-multi-gpu", [])
    with pytest.raises(ValueError, match="rollout"):
        build_overrides(tpl.id, 1, "bad-batch", ["env.train.total_num_envs=128"])
    with pytest.raises(ValueError, match="rollout"):
        build_overrides(tpl.id, 1, "uneven-batch", ["env.train.total_num_envs=256"])
    probe = build_overrides(tpl.id, 1, "probe", ["env.train.total_num_envs=256",
                                                "actor.global_batch_size=2048"])
    assert "actor.global_batch_size=2048" in probe


def test_amp_task_only_control_is_explicit_and_isolated():
    key = "amp_discriminator.style_weight_max=0"
    assert key in build_overrides("rpo_v2_amp_stand_walk_ppo", 1, "task-only", [key])
    with pytest.raises(ValueError, match="AMP"):
        build_overrides("rpo_v2_wbc_stand_walk_ppo", 1, "wrong-task", [key])
    with pytest.raises(ValueError, match="AMP"):
        build_overrides("rpo_v2_amp_stand_walk_ppo", 1, "bad-weight",
                        ["amp_discriminator.style_weight_max=2"])


def test_official_stand_initial_pose_is_profiled_for_train_and_eval():
    definition = {"task":"stand walk", "reward":"rl", "failure":"fall", "success":"20s",
                  "settings":{"initial_pose":"official_a1_stand_v1"}}
    train = validate_definition("train", "rpo_v2_amp_stand_walk_ppo", definition)
    eval_ = validate_definition("eval", "suite_rpo_v2_amp_isaac_stand_walk_v1", definition)
    assert train["settings"] == eval_["settings"]
    assert rpo_eval_arguments({"criterion":"stand_walk", "suite_id":"suite_rpo_v2_amp_isaac_stand_walk_v1",
                               "environment_settings":eval_["settings"]}) == [
        "--initial-pose", "official_a1_stand_v1"]
    with pytest.raises(ValueError):
        validate_definition("train", "rpo_v2_wbc_stand_walk_ppo", definition)
    with pytest.raises(ValueError):
        validate_definition("eval", "suite_rpo_v2_amp_isaac_stand_walk_v1",
                            {**definition, "settings":{"initial_pose":"made_up"}})


def test_amp_initial_exploration_std_is_train_profiled_and_bounded():
    definition = {"task":"stand walk", "reward":"rl", "failure":"fall", "success":"20s",
                  "settings":{"initial_pose":"official_a1_stand_v1", "initial_logstd":-2.0}}
    assert validate_definition("train", "rpo_v2_amp_stand_walk_ppo", definition)["settings"] == definition["settings"]
    with pytest.raises(ValueError):
        validate_definition("train", "rpo_v2_amp_stand_walk_ppo",
                            {**definition, "settings": {**definition["settings"], "initial_logstd":3.0}})
    with pytest.raises(ValueError):
        validate_definition("eval", "suite_rpo_v2_amp_isaac_stand_walk_v1", definition)


def test_amp_walk_speed_reward_tolerance_is_a_versioned_training_setting(monkeypatch):
    from eb.runners.rpo_wbc_extension import training_settings_from_env

    definition = {"task":"stand walk", "reward":"track commanded speed", "failure":"fall",
                  "success":"20s", "settings":{"velocity_tracking_std":0.2}}
    assert validate_definition("train", "rpo_v2_amp_stand_walk_ppo", definition)["settings"] == definition["settings"]
    with pytest.raises(ValueError):
        validate_definition("eval", "suite_rpo_v2_amp_isaac_stand_walk_v1", definition)
    with pytest.raises(ValueError):
        validate_definition("train", "rpo_v2_wbc_stand_walk_ppo", definition)
    monkeypatch.setenv("EB_RPO_VELOCITY_TRACKING_STD", "0.2")
    assert training_settings_from_env()["velocity_tracking_std"] == 0.2


def test_amp_walk_speed_reward_weight_is_a_versioned_training_setting(monkeypatch):
    from eb.runners.rpo_wbc_extension import training_settings_from_env

    definition = {"task":"stand walk", "reward":"track commanded speed", "failure":"fall",
                  "success":"20s", "settings":{"velocity_reward_weight":8.0}}
    assert validate_definition("train", "rpo_v2_amp_stand_walk_ppo", definition)["settings"] == definition["settings"]
    with pytest.raises(ValueError):
        validate_definition("eval", "suite_rpo_v2_amp_isaac_stand_walk_v1", definition)
    monkeypatch.setenv("EB_RPO_VELOCITY_REWARD_WEIGHT", "8")
    assert training_settings_from_env()["velocity_reward_weight"] == 8.0


def test_amp_actuator_delay_can_be_frozen_in_train_and_isaac_eval_profiles():
    definition = {"task":"stand walk", "reward":"rl", "failure":"fall", "success":"20s",
                  "settings":{"actuator_delay_steps":0}}
    assert validate_definition("train", "rpo_v2_amp_stand_walk_ppo", definition)["settings"] == definition["settings"]
    assert validate_definition("eval", "suite_rpo_v2_amp_isaac_stand_walk_v1", definition)["settings"] == definition["settings"]
    assert rpo_eval_arguments({"criterion":"stand_walk", "backend":"isaac",
                               "suite_id":"suite_rpo_v2_amp_isaac_stand_walk_v1",
                               "environment_settings":definition["settings"]}) == ["--actuator-delay-steps", "0"]
    with pytest.raises(ValueError):
        validate_definition("eval", "suite_rpo_v2_amp_mujoco_stand_walk_v1", definition)
    with pytest.raises(ValueError):
        validate_definition("train", "rpo_v2_amp_stand_walk_ppo",
                            {**definition, "settings":{"actuator_delay_steps":1.5}})


def test_official_stand_action_targets_the_same_canonical_pose_as_isaac():
    from eb.rpo_wbc import AMP_STAND_ACTION, AMP_STAND_Q
    from eb.runners.rpo_wbc_artifact import DEFAULT_URDF

    assert len(AMP_STAND_ACTION) == len(AMP_STAND_Q) == 23
    assert max(abs(float(default + 0.25 * action - target))
               for default, action, target in zip(DEFAULT_URDF, AMP_STAND_ACTION, AMP_STAND_Q)) < 1e-5


def test_official_motion_read_only_source_is_inside_platform_mount_allowlist():
    source = "/data/kk/roboparty/roboparty_train/robolab/data/motions/rpo_lab"
    assert validate_allowed_path(source) == Path(source).resolve()


def test_official_motion_job_refuses_networked_pickle_conversion():
    spec = JobSpec(kind=JobKind.DATA_CONVERT,
                   embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                   params={"operation":"convert_rpo_amp_official", "dataset_id":"ds_a"})
    with pytest.raises(ValueError, match="without network"):
        DataConvertAdapter().render(spec, "run_test")


def test_official_motion_audit_is_an_isolated_platform_job():
    spec = JobSpec(kind=JobKind.DATA_CONVERT,
                   embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                   params={"operation":"audit_rpo_amp_official"},
                   runtime={"network":"none"}, resources={"gpus":0})
    rendered = DataConvertAdapter().render(spec, "run_test")
    assert "--audit" in rendered.command
    assert rendered.image == "eb/tools:latest"
    assert all(mode == "ro" for _, _, mode in rendered.mounts[:-1])
    assert rendered.mounts[-1][2] == "rw"


def test_amp_asset_probe_can_run_on_draft_env_but_training_stays_blocked():
    env = load_env(Path(__file__).resolve().parents[3] /
                   "contracts/ee_rpo_v2_amp_locomotion_23dof_v1.yaml")
    probe = JobSpec(kind=JobKind.ASSET_VALIDATE, embodiment_env_id=env.id,
                    params={"operation":"probe_rpo_amp_env"}, resources={"gpus":1})
    train = JobSpec(kind=JobKind.TRAIN_RL, embodiment_env_id=env.id,
                    task_id="tsk_rpo_amp_stand_walk", resources={"gpus":1})
    assert g1_submit(probe, env, GateContext(dataset_versions=[])).passed
    assert not g1_submit(train, env, GateContext(dataset_versions=[])).passed


def test_amp_asset_probe_uses_pinned_isaac_command_and_read_only_assets(tmp_path, monkeypatch):
    assets = tmp_path / "assets" / "roboparty" / "rpo-v2"
    assets.mkdir(parents=True)
    (assets / "asset-manifest.json").write_text('{"status":"verified"}')
    (assets / "rpo_description").mkdir()
    monkeypatch.setattr("eb.settings.get_settings", lambda: SimpleNamespace(
        assets_root=tmp_path / "assets", data_root=tmp_path / "state"))
    spec = JobSpec(kind=JobKind.ASSET_VALIDATE,
                   embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                   params={"operation":"probe_rpo_amp_env", "initial_pose":"official_a1_stand_v1"}, resources={"gpus":1},
                   runtime={"network":"none"})
    rendered = ToolsAdapter().render(spec, "run_test")
    assert rendered.image == "eb/rlinf:rpo-v2-wbc"
    assert rendered.command[:2] == ["/workspace/isaaclab_extension_template/.venv/bin/python",
                                     "/workspace/eb_runner/rpo_amp_probe.py"]
    assert (str(assets), "/workspace/rpo-assets", "ro") in rendered.mounts
    assert rendered.gpus == 1
    assert rendered.env["EB_RPO_AMP_INITIAL_POSE"] == "official_a1_stand_v1"


def test_amp_motion_probe_mounts_managed_dataset_read_only(tmp_path, monkeypatch):
    assets = tmp_path / "assets/roboparty/rpo-v2"
    (assets / "rpo_description").mkdir(parents=True)
    (assets / "asset-manifest.json").write_text('{"status":"verified"}')
    dataset = tmp_path / "datasets/ds_a/content"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}")
    monkeypatch.setattr("eb.settings.get_settings", lambda: SimpleNamespace(
        assets_root=tmp_path / "assets", datasets_root=tmp_path / "datasets",
        data_root=tmp_path / "state"))
    spec = JobSpec(kind=JobKind.ASSET_VALIDATE,
                   embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                   params={"operation":"probe_rpo_amp_motion", "dataset_id":"ds_a"},
                   resources={"gpus":1}, runtime={"network":"none"})
    rendered = ToolsAdapter().render(spec, "run_test")
    assert (str(dataset), "/workspace/motion", "ro") in rendered.mounts
    assert "--motion-dir" in rendered.command


def test_amp_asset_probe_needs_a_passed_numeric_receipt(tmp_path):
    receipt = tmp_path / "probe.json"
    with pytest.raises(ValueError, match="probe"):
        validate_rpo_amp_probe_receipt(receipt)
    receipt.write_text(json.dumps({"schema_version":"eb/rpo-amp-isaac-probe/v1",
        "status":"passed", "num_envs":2, "actor_dim":780, "critic_dim":1390,
        "action_dim":23, "amp_frame_dim":74, "amp_window_frames":3,
        "partial_reset":True}))
    assert validate_rpo_amp_probe_receipt(receipt)["status"] == "passed"
    receipt.write_text(json.dumps({"schema_version":"eb/rpo-amp-isaac-probe/v1",
        "status":"failed", "error":"scene failed"}))
    with pytest.raises(ValueError, match="probe"):
        validate_rpo_amp_probe_receipt(receipt)


def test_amp_motion_probe_requires_dataset_bound_kinematic_parity(tmp_path):
    receipt = tmp_path / "probe.json"
    data = {"schema_version":"eb/rpo-amp-isaac-probe/v1", "status":"passed",
            "num_envs":2, "actor_dim":780, "critic_dim":1390,
            "action_dim":23, "amp_frame_dim":74, "amp_window_frames":3,
            "partial_reset":True,
            "motion":{"status":"passed", "dataset_manifest_sha256":"a" * 64,
                      "clips_checked":12, "max_key_body_error_m":0.01,
                      "dynamic_replay_verified":False}}
    receipt.write_text(json.dumps(data))
    assert validate_rpo_amp_motion_probe_receipt(receipt, "a" * 64)["motion"]["clips_checked"] == 12
    with pytest.raises(ValueError, match="kinematic"):
        validate_rpo_amp_motion_probe_receipt(receipt, "b" * 64)
    data["motion"]["max_key_body_error_m"] = 0.04
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="kinematic"):
        validate_rpo_amp_motion_probe_receipt(receipt, "a" * 64)


def test_amp_motion_dataset_requires_validated_train_slice_and_manifest(tmp_path):
    manifest = {"converter_revision": AMP_MOTION_CONVERTER_REVISION,
                "joint_order": list(JOINT_ORDER),
                "source_joint_order": list(RPO_LAB_JOINT_ORDER)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    row = SimpleNamespace(
        status="validated", dest_path=str(tmp_path),
        embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
        meta={"usage": "train", "purpose": "motion_prior"},
        summary_stats={"validation": {"status": "validated", "scope": "rpo_amp_motion_v1",
                                      "storage_ref": str(tmp_path), "training_ready": True,
                                      "manifest_sha256": digest}},
    )
    assert require_amp_motion_dataset(row)["manifest_sha256"] == digest
    row.summary_stats["validation"]["manifest_sha256"] = hashlib.sha256(b"{}").hexdigest()
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="mapping"):
        require_amp_motion_dataset(row)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    row.summary_stats["validation"]["manifest_sha256"] = digest
    row.meta["usage"] = "eval"
    with pytest.raises(ValueError):
        require_amp_motion_dataset(row)
    row.meta["usage"] = "train"
    row.summary_stats["validation"]["training_ready"] = False
    with pytest.raises(ValueError):
        require_amp_motion_dataset(row)
    row.summary_stats["validation"]["training_ready"] = True
    (tmp_path / "manifest.json").write_text("{ } ")
    with pytest.raises(ValueError, match="changed"):
        require_amp_motion_dataset(row)
    row.meta["usage"] = "train"
    row.summary_stats["validation"]["training_ready"] = False
    with pytest.raises(ValueError):
        require_amp_motion_dataset(row)
