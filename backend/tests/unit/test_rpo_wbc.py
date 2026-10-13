from pathlib import Path

import yaml

from eb.domain.contract import load_env
from eb.domain.rlinf_form import build_overrides, get_template, template_image
import pytest
from eb.adapters import get_adapter
from eb.domain.job import JobKind, JobSpec
from eb.rpo_wbc import (
    ACTOR_DIM,
    CRITIC_DIM,
    JOINT_ORDER,
    UPPER_BODY_JOINTS,
    validate_eval_receipt,
    verify_asset_bundle,
)
from eb.runners.rpo_wbc_extension import DirectRLEnvAutoResetCompat


ROOT = Path(__file__).resolve().parents[3]


def test_rpo_direct_env_resets_only_finished_episodes():
    class BaseEnv:
        def reset(self, seed=None, options=None):
            self.full_resets += 1
            return seed, options

    class Env(DirectRLEnvAutoResetCompat, BaseEnv):
        def __init__(self):
            self.full_resets = 0
            self.episode_lengths = [5, 7, 9]
            self.obs_buf = {"policy": self.episode_lengths.copy()}
            self.extras = {"log": "kept"}

        def _reset_idx(self, env_ids):
            for env_id in env_ids:
                self.episode_lengths[env_id] = 0

        def _get_observations(self):
            return {"policy": self.episode_lengths.copy()}

    env = Env()
    assert env.reset(seed=7) == (7, None)
    assert env.reset(env_ids=[1]) == ({"policy": [5, 0, 9]}, env.extras)
    env.episode_lengths[1] = 4  # the next episode also ends
    assert env.reset(env_ids=[1]) == ({"policy": [5, 0, 9]}, env.extras)
    assert env.full_resets == 1
    assert env.episode_lengths == [5, 0, 9]


def test_rpo_image_bundles_the_isaaclab_compatible_urdf_importer():
    dockerfile = (ROOT / "deploy/images/rlinf-rpo.Dockerfile").read_text()
    assert "isaacsim.asset.importer.urdf-2.4.31" in dockerfile
    assert "665cc13d48842ac6d0c203359819c1d5c767a1d920fb8a00179e147ec2fb5201" in dockerfile
    assert "omni.kit.pip_archive-d38fa9ecd1fb6df4" in dockerfile
    assert "3a0d0623be2349a2861f566dc8d5bdb5040c11b0977bb255c6d522b4be7bd682" in dockerfile
    assert "2.4.30" not in dockerfile


def test_rpo_rlinf_config_is_valid_yaml_with_omegaconf_interpolation():
    config = yaml.safe_load((ROOT / "backend/eb/runners/config/rpo_wbc_ppo.yaml").read_text())
    assert "weight_syncer/patch_syncer@weight_syncer" in config["defaults"]
    assert config["cluster"]["component_placement"] == {"actor, env, rollout": 0}
    assert config["rollout"]["model"]["precision"] == "${actor.model.precision}"
    assert config["actor"]["fsdp_config"]["mixed_precision"]["param_dtype"] == "${actor.model.precision}"
    assert config["env"]["train"]["max_steps_per_rollout_epoch"] == 24
    assert config["env"]["eval"]["max_steps_per_rollout_epoch"] == 1000


def test_rpo_contract_freezes_wbc_codec():
    contract = load_env(ROOT / "contracts/ee_rpo_v2_wbc_23dof_v1.yaml")
    assert contract.validate_self().passed
    assert contract.status == "active"
    assert contract.joint_order == list(JOINT_ORDER)
    assert contract.observation.dim == ACTOR_DIM == 890
    assert contract.observation.critic_dim == CRITIC_DIM == 1500
    assert contract.action.dim == 23
    assert contract.control.policy_hz == 50
    assert len(UPPER_BODY_JOINTS) == 11


def test_rpo_rlinf_recipe_is_selectable_and_frozen():
    template = get_template("rpo_v2_wbc_ppo")
    assert template.embodiment_id == "emb_roboparty_rpo"
    assert template.embodiment_env_id == "ee_rpo_v2_wbc_23dof_v1"
    assert template.task_id == "tsk_rpo_walk_upperbody_tracking"
    assert template_image(template) == "eb/rlinf:rpo-v2-wbc"
    overrides = build_overrides(template.id, 1, "smoke", [])
    assert "env.train.total_num_envs=512" in overrides
    assert "runner.max_epochs=1" in overrides
    assert "runner.save_interval=25" in overrides
    assert "runner.val_check_interval=25" in overrides
    assert "actor.micro_batch_size=128" in overrides
    assert "actor.global_batch_size=1024" in overrides
    assert r"cluster.component_placement={actor\,\ env\,\ rollout:0-0}" in overrides


def test_rpo_rejects_batch_sizes_incompatible_with_actor_world_size():
    with pytest.raises(ValueError, match="global_batch_size.*micro_batch_size.*GPU"):
        build_overrides("rpo_v2_wbc_ppo", 4, "bad-batch", [
            "actor.micro_batch_size=4096", "actor.global_batch_size=4096",
        ])


def test_rpo_rejects_incompatible_save_and_validation_intervals():
    with pytest.raises(ValueError, match="save_interval.*val_check_interval"):
        build_overrides("rpo_v2_wbc_ppo", 4, "bad-interval", [
            "runner.save_interval=1", "runner.val_check_interval=25",
        ])


def test_rpo_eval_receipt_rejects_incomplete_or_wrong_onnx():
    good = {
        "schema_version": "eb/rpo-wbc-eval/v1",
        "onnx": {"sha256": "a" * 64, "obs": ACTOR_DIM, "act": 23, "max_abs_error": 1e-5},
        "cases": {"planned": 12, "completed": 12, "valid": 12, "invalid": 0, "infra_failed": 0},
        "metrics": {"fall_count": 0, "hard_limit_count": 0, "nan_count": 0,
                    "velocity_rmse": 0.1, "upper_rmse": 0.08},
    }
    assert validate_eval_receipt(good) == []
    assert "onnx.max_abs_error" in validate_eval_receipt({**good, "onnx": {**good["onnx"], "max_abs_error": float("nan")}})
    assert "metrics.velocity_rmse" in validate_eval_receipt({**good, "metrics": {**good["metrics"], "velocity_rmse": float("nan")}})
    assert "onnx.obs" in validate_eval_receipt({**good, "onnx": {**good["onnx"], "obs": 89}})
    assert "cases.completed" in validate_eval_receipt({**good, "cases": {**good["cases"], "completed": 11}})
    stability = {**good, "criterion":"stability", "required_steps":1000,
                 "metrics":{**good["metrics"], "n_trials":12, "successes":12,
                            "velocity_rmse":0.45, "upper_rmse":0.2}}
    assert validate_eval_receipt(stability, criterion="stability") == []
    assert "criterion" in validate_eval_receipt(stability)
    assert "metrics.successes" in validate_eval_receipt(
        {**stability, "metrics":{**stability["metrics"], "successes":11}}, criterion="stability")
    stand_walk = {**stability, "criterion":"stand_walk"}
    assert validate_eval_receipt(stand_walk, criterion="stand_walk") == []
    assert "metrics.successes" in validate_eval_receipt(
        {**stand_walk, "metrics":{**stand_walk["metrics"], "successes":11}}, criterion="stand_walk")


def test_rpo_stability_success_requires_surviving_the_horizon():
    import math
    from eb.runners.rpo_wbc_artifact import case_success, physical_fall
    from eb.runners.rpo_wbc_artifact import FALL_BASE_HEIGHT_M as EVAL_FALL_HEIGHT
    from eb.runners.rpo_wbc_artifact import FALL_UPRIGHT_COS_MIN
    from eb.runners.rpo_wbc_extension import FALL_BASE_HEIGHT_M, FALL_MAX_TILT_RAD

    assert EVAL_FALL_HEIGHT == FALL_BASE_HEIGHT_M
    assert FALL_UPRIGHT_COS_MIN == pytest.approx(math.cos(FALL_MAX_TILT_RAD))

    assert case_success("stability", 1000, False, False, False, 0.2, 0.45)
    assert not case_success("stability", 97, True, False, False, 0.08, 0.1)
    assert not case_success("stability", 999, False, False, False, 0.08, 0.1)
    assert not case_success("stability", 1000, False, True, False, 0.08, 0.1)
    assert not case_success("tracking", 1000, False, False, False, 0.2, 0.45)
    assert physical_fall(0.49, -1.0) == "low_base"
    assert physical_fall(0.75, -0.5) == "excessive_tilt"
    assert physical_fall(0.75, -1.0) is None
    assert case_success("stand_walk", 1000, False, False, False, 0, 0.1,
                        scenario="stand", distance_m=0.4)
    assert case_success("stand_walk", 1000, False, False, False, 0, 0.1,
                        scenario="walk", distance_m=4.5)
    assert not case_success("stand_walk", 1000, True, False, False, 0, 0.1,
                            scenario="walk", distance_m=4.5)
    assert not case_success("stand_walk", 1000, False, False, False, 0, 0.1,
                            scenario="walk", distance_m=3.9)


def test_stand_walk_recipe_keeps_rpo_codec_and_changes_task():
    template = get_template("rpo_v2_wbc_stand_walk_ppo")
    config = yaml.safe_load((ROOT / "backend/eb/runners/config/rpo_wbc_stand_walk_ppo.yaml").read_text())
    assert template.task_id == "tsk_rpo_stand_walk"
    assert template_image(template) == "eb/rlinf:rpo-v2-wbc"
    assert config["env"]["train"]["init_params"]["id"] == "RPO-WBC-StandWalk-v1"
    assert config["actor"]["model"]["obs_dim"] == 890
    assert config["actor"]["model"]["critic_obs_dim"] == 1500
    assert config["actor"]["model"]["action_dim"] == 23


def test_asset_verifier_blocks_an_incomplete_bundle(tmp_path):
    receipt = verify_asset_bundle(tmp_path, {"robolab": "pinned"})
    assert receipt["status"] == "blocked"
    assert "asset_files_missing" in receipt["blockers"]
    assert receipt["hardware_verified"] is False


def test_rpo_export_and_mujoco_use_model_and_asset_roots(tmp_path, monkeypatch):
    from types import SimpleNamespace

    models = tmp_path / "models"
    state = tmp_path / "state"
    assets = tmp_path / "assets/roboparty/rpo-v2/rpo_description/mjcf"
    checkpoint = models / "source/model_state_dict/full_weights.pt"
    policy = models / "source/policy.onnx"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    policy.write_bytes(b"onnx")
    assets.mkdir(parents=True)
    (assets / "rpo.xml").write_text("<mujoco/>")
    settings = SimpleNamespace(models_root=models, data_root=state, assets_root=tmp_path / "assets")
    monkeypatch.setattr("eb.adapters.isaaclab_rslrl.get_settings", lambda: settings)
    monkeypatch.setattr("eb.adapters.mujoco.get_settings", lambda: settings)

    export_spec = JobSpec(kind=JobKind.EXPORT_POLICY,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", inputs={"base_model":str(checkpoint)},
        resources={"gpus":1})
    exported = get_adapter(export_spec).render(export_spec, "run_export")
    assert "/workspace/output/model" in exported.command
    assert (str(models / "runs/run_export"), "/workspace/output/model", "rw") in exported.mounts

    eval_spec = JobSpec(kind=JobKind.EVAL_SIM2SIM,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", task_id="tsk_rpo_walk_upperbody_tracking",
        inputs={"base_model":str(policy)}, params={"seeds":[3, 7]})
    evaluated = get_adapter(eval_spec).render(eval_spec, "run_eval")
    assert evaluated.command[-1] == "3,7"
    assert (str(policy), "/workspace/model/policy.onnx", "ro") in evaluated.mounts

    trained_spec = JobSpec(kind=JobKind.EVAL_SIM2SIM,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", task_id="tsk_rpo_walk_upperbody_tracking",
        inputs={"base_model":str(checkpoint.parent.parent)},
        params={"rpo_checkpoint_eval":True, "seeds":[3, 7], "record_video":True},
        resources={"gpus":1})
    trained = get_adapter(trained_spec).render(trained_spec, "run_trained_eval")
    assert "eval-checkpoint-mujoco" in trained.command
    assert "--record-video" in trained.command
    assert (str(checkpoint), "/workspace/model/full_weights.pt", "ro") in trained.mounts

    (tmp_path / "assets/roboparty/rpo-v2/asset-manifest.json").write_text("{}")
    monkeypatch.setattr("eb.adapters.rlinf.get_settings", lambda: settings)
    isaac_spec = JobSpec(kind=JobKind.EVAL_SIM,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", task_id="tsk_rpo_walk_upperbody_tracking",
        inputs={"base_model":str(checkpoint.parent.parent)},
        params={"stack":"rlinf", "template_id":"rpo_v2_wbc_eval",
                "seeds":[3, 7], "record_video":True}, resources={"gpus":1})
    isaac = get_adapter(isaac_spec).render(isaac_spec, "run_isaac_eval")
    assert "eval-checkpoint-isaac" in isaac.command
    assert "--record-video" in isaac.command
    assert (str(checkpoint), "/workspace/model/full_weights.pt", "ro") in isaac.mounts
    isaac_spec.task_id = "tsk_rpo_stand_walk"
    isaac_spec.params["rpo_eval_plan"] = {"criterion":"stand_walk"}
    stand_walk_command = get_adapter(isaac_spec).render(isaac_spec, "run_basic_eval").command
    assert stand_walk_command[stand_walk_command.index("--criterion") + 1] == "stand_walk"
    isaac_spec.task_id = "tsk_rpo_walk_upperbody_tracking"

    for spec in (trained_spec, isaac_spec):
        spec.params.update(seeds=list(range(12)),
                           rpo_eval_plan={"profile":"standard", "seed_override":False,
                                          "implicit_seed_defaults":True})
        command = get_adapter(spec).render(spec, "run_default_seeds").command
        assert "--seeds" not in command
        assert command[command.index("--profile") + 1] == "standard"
        spec.params["rpo_eval_plan"]["criterion"] = "stability"
        command = get_adapter(spec).render(spec, "run_stability_eval").command
        assert command[command.index("--criterion") + 1] == "stability"

        spec.params.update(seeds=[42],
                           rpo_eval_plan={"profile":"standard", "seed_override":True,
                                          "implicit_seed_defaults":True})
        command = get_adapter(spec).render(spec, "run_custom_seed").command
        assert command[command.index("--seeds") + 1] == "42"

        spec.params.update(seeds=[0, 1, 2],
                           rpo_eval_plan={"profile":"smoke", "seed_override":False})
        command = get_adapter(spec).render(spec, "run_legacy_eval").command
        assert command[command.index("--seeds") + 1] == "0,1,2"


def test_rpo_checkpoint_runner_uses_profile_seeds_when_not_overridden(monkeypatch):
    import sys

    from eb.runners import rpo_wbc_artifact as runner

    actual = []
    monkeypatch.setattr(runner, "export", lambda *args: None)
    monkeypatch.setattr(runner, "evaluate_isaac", lambda *args, **kwargs: actual.append((args[2], kwargs["criterion"])))
    monkeypatch.setattr(runner, "evaluate_mujoco", lambda *args, **kwargs: actual.append((args[3], kwargs["criterion"])))
    for mode in ("eval-checkpoint-isaac", "eval-checkpoint-mujoco"):
        base = [mode, "--checkpoint", "weights.pt", "--output", "output"]
        if mode.endswith("mujoco"):
            base += ["--mjcf", "rpo.xml"]
        for extra, expected in (([], [0, 1, 2]),
                                (["--profile", "standard"], list(range(12))),
                                (["--seeds", "42"], [42]),
                                (["--criterion", "stability"], [0, 1, 2])):
            monkeypatch.setattr(sys, "argv", ["runner", *base, *extra])
            runner.main()
            assert actual.pop() == (expected, "stability" if "--criterion" in extra else "tracking")


@pytest.mark.asyncio
async def test_rpo_eval_plan_binds_registered_checkpoint_and_video_mode(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from eb.api.routes_eval import RpoEvalIn, _prepare_rpo
    from eb.store.models import Artifact, EvalSuite, ModelVersion, Run

    manifest = {"run_id":"run_source", "step":100}
    digest = "sha256:" + hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    model = SimpleNamespace(id="mv_rpo", artifact_id="art_rpo",
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", name="trained-rpo",
        card={"schema_version":"eb/rpo-wbc-model/v1", "run_id":"run_source", "step":100})
    artifact = SimpleNamespace(kind="rlinf_checkpoint", storage=str(tmp_path / "models/run_source"), sha256=digest)
    source = SimpleNamespace(id="run_source", status="finished")
    suite = SimpleNamespace(id="suite_rpo_v2_wbc_sim2sim_v1", frozen=True,
        embodiment_env_id=model.embodiment_env_id, spec={})
    isaac_suite = SimpleNamespace(id="suite_rpo_v2_wbc_isaac_v1", frozen=True,
        embodiment_env_id=model.embodiment_env_id, spec={})
    stability_suites = [SimpleNamespace(id=f"suite_rpo_v2_wbc_{backend}_stability_v1",
        frozen=True, embodiment_env_id=model.embodiment_env_id,
        spec={"criterion":"stability", "required_steps":1000}) for backend in ("isaac", "mujoco")]
    basic_suites = [SimpleNamespace(id=f"suite_rpo_v2_wbc_{backend}_stand_walk_v2",
        frozen=True, embodiment_env_id=model.embodiment_env_id,
        spec={"criterion":"stand_walk", "required_steps":1000,
              "cases":"even_seed_stand_odd_seed_walk", "initial_joint_offset_rad":[-0.05, 0.05],
              "initial_base_velocity_mps":[-0.05, 0.05],
              "initial_yaw_rad":[-3.141592653589793, 3.141592653589793],
              "stand_max_drift_m":0.5, "walk_min_forward_m":4.0,
              "fall_height_m":0.5, "fall_tilt_rad":0.8}) for backend in ("isaac", "mujoco")]
    rows = {(ModelVersion, model.id):model, (Artifact, model.artifact_id):artifact,
            (Run, source.id):source, (EvalSuite, suite.id):suite,
            (EvalSuite, isaac_suite.id):isaac_suite,
            **{(EvalSuite, item.id):item for item in stability_suites + basic_suites}}

    class Session:
        async def get(self, table, key):
            return rows.get((table, key))

    class Ebd:
        async def image_info(self, name):
            assert name == "eb/rlinf:rpo-v2-wbc"
            return {"ok":True, "digest":"sha256:" + "a" * 64}

    monkeypatch.setattr("eb.api.routes_eval.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("eb.checkpoints.checkpoint_model", lambda *a, **k: {
        "path":artifact.storage, "step":100, "manifest":manifest})
    monkeypatch.setattr("eb.store.ebd.EbdClient", Ebd)
    spec, plan = await _prepare_rpo(RpoEvalIn(model_version_id=model.id, record_video=True), Session())
    assert plan["seeds"] == [0, 1, 2] and plan["record_video"] is True
    assert plan["implicit_seed_defaults"] is True
    assert plan["backend"] == "mujoco" and spec.kind == JobKind.EVAL_SIM2SIM
    assert plan["criterion"] == "stability" and plan["suite_id"] == stability_suites[1].id
    assert spec.inputs.base_model == artifact.storage
    assert spec.params["rpo_eval_plan"] == plan
    isaac_spec, isaac_plan = await _prepare_rpo(
        RpoEvalIn(model_version_id=model.id, backend="isaac", record_video=True), Session())
    assert isaac_plan["backend"] == "isaac" and isaac_plan["suite_id"] == stability_suites[0].id
    assert isaac_spec.kind == JobKind.EVAL_SIM
    assert isaac_spec.params["template_id"] == "rpo_v2_wbc_eval"
    assert isaac_spec.params["record_video"] is True
    custom_spec, custom_plan = await _prepare_rpo(
        RpoEvalIn(model_version_id=model.id, backend="isaac", profile="standard", seed=42), Session())
    assert custom_plan["seeds"] == [42] and custom_spec.params["seeds"] == [42]
    assert custom_plan["seed_override"] is True and isaac_plan["seed_override"] is False
    _, tracking_plan = await _prepare_rpo(
        RpoEvalIn(model_version_id=model.id, backend="isaac", suite_id=isaac_suite.id), Session())
    assert tracking_plan["criterion"] == "tracking" and tracking_plan["suite_id"] == isaac_suite.id
    model.card["template_id"] = "rpo_v2_wbc_stand_walk_ppo"
    basic_spec, basic_plan = await _prepare_rpo(
        RpoEvalIn(model_version_id=model.id, backend="isaac"), Session())
    assert basic_plan["criterion"] == "stand_walk"
    assert basic_plan["suite_id"] == basic_suites[0].id
    assert basic_spec.task_id == "tsk_rpo_stand_walk"
    assert basic_plan["suite_spec"] == basic_suites[0].spec
    basic_suites[0].spec["initial_yaw_rad"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="未冻结"):
        await _prepare_rpo(RpoEvalIn(model_version_id=model.id, backend="isaac"), Session())
    with pytest.raises(ValueError, match="不匹配"):
        await _prepare_rpo(RpoEvalIn(model_version_id=model.id, backend="isaac", suite_id=suite.id), Session())
    with pytest.raises(ValueError):
        RpoEvalIn(model_version_id=model.id, seed=-1)
    artifact.sha256 = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="不一致"):
        await _prepare_rpo(RpoEvalIn(model_version_id=model.id), Session())


@pytest.mark.asyncio
async def test_rpo_candidates_only_lists_rpo_model_versions():
    from types import SimpleNamespace
    from eb.api.routes_eval import rpo_candidates

    rpo = SimpleNamespace(id="mv_rpo", name="我的 RL 模型", created_at=None,
        embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
        card={"schema_version":"eb/rpo-wbc-model/v1", "step":1000, "readiness":"simulation_unverified"})
    other = SimpleNamespace(id="mv_other", name="其他模型", created_at=None, card={})
    class Session:
        async def scalars(self, query):
            return SimpleNamespace(all=lambda: [rpo, other])

    result = await rpo_candidates(Session(), SimpleNamespace(id="user:test"))

    assert result == {"items":[{"id":"mv_rpo", "name":"我的 RL 模型", "step":1000,
                                "embodiment_env_id":"ee_rpo_v2_wbc_23dof_v1",
                                "readiness":"simulation_unverified", "execution":None,
                                "progress_unit":None, "created_at":None}]}
