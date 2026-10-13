from dataclasses import replace

import pytest

from eb.domain.job import JobSpec
from eb.domain.rlinf_form import build_overrides, component_placement, get_template
from eb.domain.training_recipe import (
    RECIPES,
    UFO_STABLE_RECIPE,
    recipe_eval_image_digest,
    recipe_image_digest,
    recipe_profile_id,
    recipe_resources,
    recipe_source_receipt,
    validate_training_execution,
)


def qualification_spec():
    recipe = RECIPES[UFO_STABLE_RECIPE]
    return JobSpec(kind="train.rl", embodiment_env_id=recipe.embodiment_env_id,
        task_id=recipe.task_id, image=recipe_image_digest(UFO_STABLE_RECIPE),
        params={"template_id": UFO_STABLE_RECIPE, "execution": recipe.binding.model_dump(),
                "purpose": "qualification", "target_learner_updates": 16,
                "source_receipt": dict(recipe_source_receipt(UFO_STABLE_RECIPE))},
        resources=recipe_resources(UFO_STABLE_RECIPE), policy={"timeout_h": 1, "max_retries": 0})


def test_native_profile_protects_full_architecture_batch_replay_and_sources(monkeypatch):
    recipe = RECIPES[UFO_STABLE_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_STABLE_RECIPE, replace(recipe, status="dependency_ready"))
    spec = qualification_spec()
    assert validate_training_execution(spec) == recipe.binding
    overrides = build_overrides(UFO_STABLE_RECIPE, 4, "native-qualification", [])
    assert component_placement(UFO_STABLE_RECIPE, 4) in overrides
    assert "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}" in overrides
    for value in ("actor.model.hidden_dim=2048", "actor.model.hidden_layers=6",
                  "actor.model.batch_size=1024", "actor.model.replay_capacity=1000000",
                  "+actor.model.architecture_profile=native_fb_v1",
                  "+actor.model.rollout_expert_trajectories=False"):
        assert value in overrides
    with pytest.raises(ValueError, match="protects"):
        build_overrides(UFO_STABLE_RECIPE, 4, "bad", ["actor.model.hidden_dim=1024"])
    spec.params["source_receipt"].pop("rollout_profile_patch_hash")
    with pytest.raises(ValueError, match="source receipt"):
        validate_training_execution(spec)


def test_native_profile_cannot_train_before_gpu_qualification(monkeypatch):
    recipe = RECIPES[UFO_STABLE_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_STABLE_RECIPE, replace(recipe, status="dependency_ready"))
    spec = qualification_spec()
    spec.params.update(purpose="training", target_learner_updates=1000)
    with pytest.raises(ValueError, match="smoke_passed"):
        validate_training_execution(spec)


def test_native_profile_and_evaluator_preserve_old_frozen_identities():
    from eb.domain.training_recipe import UFO_FOUR_GPU_EVAL_IMAGE_DIGEST, UFO_SAFE_RECIPE
    from eb.environment_catalog import training_definition
    definition = training_definition(get_template(UFO_STABLE_RECIPE))
    assert "2048" in definition["task"] and "1024" in definition["task"]
    assert recipe_profile_id(UFO_STABLE_RECIPE) != recipe_profile_id(UFO_SAFE_RECIPE)
    assert recipe_eval_image_digest(UFO_STABLE_RECIPE) != UFO_FOUR_GPU_EVAL_IMAGE_DIGEST
    assert recipe_eval_image_digest(UFO_SAFE_RECIPE) == UFO_FOUR_GPU_EVAL_IMAGE_DIGEST
    assert recipe_resources(UFO_STABLE_RECIPE)["memory_gb"] == 96


def test_native_capacity_uses_actual_six_optimizer_parameter_census():
    from eb.capacity.estimator import estimate_job
    estimate = estimate_job(qualification_spec())
    assert estimate.basis["inputs"]["native_trainable_parameters"] == 435116484
    assert estimate.basis["inputs"]["batch_size"] == 1024
    assert estimate.basis["inputs"]["cpu_replay_bytes"] == 7038000000
    assert estimate.basis["inputs"]["gpu_peak_measured"] is False
    assert 14 < estimate.per_gpu_peak_gb < 24


@pytest.mark.parametrize("recipe_id", ["rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2", UFO_STABLE_RECIPE,
    "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4",
    "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5"])
def test_heldout_tracking_selects_encoder_compatible_image(tmp_path, monkeypatch, recipe_id):
    import hashlib
    import json
    from types import SimpleNamespace

    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.training_recipe import UFO_TRACKING_IMAGE_DIGEST
    from eb.rpo_bfm import BODY_ORDER, CODEC_HASH, RPO_LAB_JOINT_ORDER
    from eb.runners.bfm_ufo_tracking import SUITE
    from eb.settings import get_settings

    assets = tmp_path / "assets/roboparty/rpo-v2"
    assets.mkdir(parents=True)
    (assets / "asset-manifest.json").write_text("{}")
    asset_hash = hashlib.sha256(b"{}").hexdigest()
    motion, checkpoint = tmp_path / "motion", tmp_path / "model"
    motion.mkdir()
    checkpoint.mkdir()
    clips = [{"id": name, "file": name + ".npz", "source": source, "source_group": name,
              "split": split, "frames": 260, "valid_frames": 260, "fps": 50, "sha256": "c" * 64}
             for name, source, split in (("official", "official_rpo", "train"), ("g1", "g1", "train"),
                                         ("heldout_a", "g1", "validation"), ("heldout_b", "g1", "validation"))]
    manifest = {"schema_version": "eb/rpo-robot-state/v1", "asset_sha256": asset_hash,
        "codec_sha256": CODEC_HASH, "joint_order": list(RPO_LAB_JOINT_ORDER), "body_order": list(BODY_ORDER),
        "quaternion": "wxyz", "velocity_frame": "world", "velocity_point": "root_link_origin",
        "parents": [{"manifest_sha256": "a" * 64}], "source_refs": [{"sha256": "b" * 64, "license": "fixture"}],
        "clips": clips}
    (motion / "manifest.json").write_text(json.dumps(manifest))
    motion_hash = hashlib.sha256((motion / "manifest.json").read_bytes()).hexdigest()
    metadata = {"path": str(checkpoint), "step": 24, "manifest": {"format": "ufo_agent_state_v1",
        "execution": RECIPES[recipe_id].binding.model_dump(), "weights_path": "model/native/model.safetensors",
        "files": [{"path": "model/native/model.safetensors", "sha256": "e" * 64}]}}
    monkeypatch.setattr("eb.checkpoints.checkpoint_model", lambda *args, **kwargs: metadata)
    monkeypatch.setenv("EB_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("EB_ALLOWED_PATH_PREFIXES", str(tmp_path))
    get_settings.cache_clear()
    try:
        image = recipe_eval_image_digest(recipe_id) if recipe_id != "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2" else UFO_TRACKING_IMAGE_DIGEST
        plan = {"suite": SUITE, "checkpoint_step": 24, "source_run_id": "run_abcdef", "image_digest": image,
            "motion_manifest_sha256": motion_hash, "asset_sha256": asset_hash, "weights_sha256": "e" * 64,
            "clip_ids": ["heldout_a", "heldout_b"]}
        from eb.rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
        control = action_control_for_execution(RECIPES[recipe_id].binding.model_dump())
        if control:
            plan["action_control"] = control
        auxiliary = auxiliary_profile_for_execution(RECIPES[recipe_id].binding.model_dump())
        if auxiliary:
            plan["auxiliary_profile"] = auxiliary
        plan["hash"] = "sha256:" + hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        spec = JobSpec(kind="asset.validate", embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
            params={"operation": "probe_rpo_ufo_tracking", "tracking_plan": plan},
            inputs={"base_model": str(checkpoint), "datasets": [{"path": str(motion), "validation": {
                "status": "validated", "scope": "rpo_ufo_robot_state_v1",
                "manifest_sha256": motion_hash, "asset_sha256": asset_hash}}]},
            runtime={"network": "none"}, resources={"gpus": 1}, policy={"timeout_h": .25, "max_retries": 0})
        rendered = ToolsAdapter().render(spec, "run_tracking")
        assert rendered.image == image
        assert "rpo_ufo_tracking_diagnostic.py" in rendered.command[1]
    finally:
        get_settings.cache_clear()
