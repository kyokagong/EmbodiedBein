from dataclasses import replace

import pytest

from eb.adapters import get_adapter
from eb.domain.job import JobSpec
from eb.domain.training_recipe import (
    RECIPES, UFO_ISAAC_RECIPE, execution_binding, validate_update_target,
)


def spec(**params):
    from eb.ufo_source import SOURCE_RECEIPT
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    return JobSpec(kind="train.rl", task_id="tsk_rpo_bfm_pretrain",
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", image=recipe.image,
        params={"template_id": UFO_ISAAC_RECIPE, "stack": "rlinf",
            "source_receipt": SOURCE_RECEIPT,
            "execution": recipe.binding.model_dump(), "purpose": "qualification",
            "target_learner_updates": 16, **params},
        resources={"gpus": 1, "cpus": 8, "memory_gb": 64, "shm_gb": 32},
        policy={"timeout_h": 1, "max_retries": 0})


@pytest.mark.parametrize("field,value", [("runtime_id", "unknown"),
    ("provider_id", "forged"), ("topology_id", "one_learner_three_samplers"),
    ("schema_version", "unknown"), ("extra", "unsafe")])
def test_ufo_binding_rejects_conflict(field, value):
    job = spec()
    job.params["execution"][field] = value
    with pytest.raises(ValueError):
        get_adapter(job)


def test_stack_conflict_and_missing_binding_do_not_fallback():
    for job in (spec(stack="isaaclab"), spec(execution=None)):
        with pytest.raises(ValueError):
            get_adapter(job)


def test_ufo_update_target_bounds():
    validate_update_target(1024, start=24)
    for target in (1025, 24, 0, True, 24.5):
        with pytest.raises(ValueError):
            validate_update_target(target, start=24)


@pytest.mark.parametrize("seed", ["-1", "4294967296", "1.5", "${oc.env:HOME}"])
def test_ufo_seed_cannot_be_a_resolver_or_invalid_rng_seed(seed, monkeypatch):
    from eb.domain.training_recipe import validate_training_execution
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="dependency_ready"))
    job = spec()
    job.overrides = ["actor.seed=" + seed]
    with pytest.raises(ValueError, match="seed"):
        validate_training_execution(job)


def test_qualification_is_bounded_and_not_a_bypass(monkeypatch):
    job = spec()
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="adapter_ready"))
    with pytest.raises(ValueError, match="dependency_ready"):
        get_adapter(job).render(job, "run_unready")
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="dependency_ready"))
    from eb.domain.training_recipe import validate_training_execution
    assert validate_training_execution(job) == recipe.binding
    for bad in (spec(target_learner_updates=25), spec(purpose="training"),
                spec(execution=None), spec(purpose="arbitrary")):
        with pytest.raises(ValueError):
            validate_training_execution(bad)
    for mutate in (lambda s: setattr(s, "image", "forged"),
                   lambda s: setattr(s.resources, "gpus", 4),
                   lambda s: setattr(s.policy, "timeout_h", 2),
                   lambda s: setattr(s.policy, "max_retries", 1),
                   lambda s: s.overrides.append("runner.max_epochs=16"),
                   lambda s: s.runtime.update(env={"PYTHONPATH": "/evil"})):
        bad = spec()
        mutate(bad)
        with pytest.raises(ValueError):
            validate_training_execution(bad)


def test_untrained_baseline_is_zero_update_qualification_only(monkeypatch):
    from eb.api.routes_train import TrainIn
    from eb.domain.training_recipe import validate_training_execution
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="dependency_ready"))
    body = TrainIn(name="baseline", model_name="baseline", template_id=UFO_ISAAC_RECIPE,
        embodiment_id="emb_roboparty_rpo", gpus=1, purpose="qualification", target_learner_updates=0)
    baseline = spec(target_learner_updates=body.target_learner_updates)
    assert validate_training_execution(baseline) == recipe.binding
    for invalid in (spec(target_learner_updates=0, purpose="training"),
                    spec(target_learner_updates=0, resume_step=0)):
        invalid.inputs.resume_from = "run_source" if invalid.params.get("resume_step") == 0 else None
        with pytest.raises(ValueError):
            validate_training_execution(invalid)


def test_legacy_binding_is_not_invented():
    job = JobSpec(kind="train.sft", task_id="tsk_libero_spatial",
                  embodiment_env_id="ee_libero_franka_delta_v1",
                  params={"stack": "rlinf", "template_id": "libero_pi05_sft"})
    assert execution_binding(job) is None
    assert get_adapter(job).name == "rlinf"


def test_ufo_resume_uses_approved_immutable_image(monkeypatch):
    from eb.domain.training_recipe import UFO_ISAAC_IMAGE_DIGEST, validate_training_execution
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="dependency_ready"))
    job = spec(target_learner_updates=24, resume_step=16)
    job.image = UFO_ISAAC_IMAGE_DIGEST
    job.inputs.resume_from = "run_source"
    assert validate_training_execution(job) == recipe.binding
    job.image = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="image"):
        validate_training_execution(job)


def test_resume_fingerprint_only_ignores_ufo_termination_and_purpose(monkeypatch):
    from eb.checkpoints import resume_fingerprint
    original = spec()
    resumed = original.model_copy(deep=True)
    resumed.params.update(target_learner_updates=24, purpose="training")
    assert resume_fingerprint(original) == resume_fingerprint(resumed)
    resumed.params["batch_size"] = 256
    assert resume_fingerprint(original) != resume_fingerprint(resumed)


def test_qualified_recipe_can_resume_beyond_qualification_budget(monkeypatch):
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint
    from eb.domain.training_recipe import validate_training_execution
    source = spec(target_learner_updates=24)
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="smoke_passed"))
    checkpoint = {"compatibility_fingerprint": resume_fingerprint(source), "step": 24,
        "run_id": "run_source", "mode": "trainer_state"}
    resumed = build_resume_spec(source, "trn_source", checkpoint,
        ResumeTrainIn(name="next", target_learner_updates=1024))
    assert resumed.params["purpose"] == "training"
    assert validate_training_execution(resumed) == recipe.binding
    monkeypatch.setitem(RECIPES, UFO_ISAAC_RECIPE, replace(recipe, status="dependency_ready"))
    with pytest.raises(ValueError):
        validate_training_execution(build_resume_spec(source, "trn_source", checkpoint,
            ResumeTrainIn(name="next", target_learner_updates=1024)))
