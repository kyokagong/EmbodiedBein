import pytest

RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6"


def test_scale_profile_and_budget_are_frozen():
    from eb.domain.rlinf_form import build_overrides
    from eb.domain.training_recipe import RECIPES, validate_update_target, recipe_timeout_h
    from eb.rpo_bfm import auxiliary_profile_for_execution

    assert RECIPE in RECIPES
    values = build_overrides(RECIPE, 4, "scale", [])
    assert "env.train.total_num_envs=6144" in values
    assert "actor.model.batch_size=2048" in values
    assert "actor.global_batch_size=2048" in values
    assert "actor.micro_batch_size=2048" in values
    assert "+actor.model.scaling_profile=env6144_batch2048_v6" in values
    assert auxiliary_profile_for_execution({"recipe_id": RECIPE}) == "soft_velocity_v5"
    validate_update_target(5024, start=24, recipe_id=RECIPE)
    with pytest.raises(ValueError):
        validate_update_target(5025, start=24, recipe_id=RECIPE)
    with pytest.raises(ValueError):
        validate_update_target(5024, start=24)
    assert recipe_timeout_h(RECIPE, "training") == 6
    assert recipe_timeout_h(RECIPE, "qualification") == 1


def test_scale_sampling_preserves_partial_update_and_resume_boundary():
    pytest.importorskip("torch")
    from eb.runners.bfm_ufo_worker import UFOLearner
    from test_bfm_ufo_worker import Agent, Replay

    agent = Agent()
    learner = UFOLearner(agent, Replay(), object(), target=16, sample_interval=24576)
    assert learner.train_round({"count":24576})["learner_joint_updates"] == 0
    assert learner.train_round({"count":24576})["learner_joint_updates"] == 16
    assert learner.next_trigger == 59392
    learner.target = 24
    assert learner.train_round({"count":24576})["learner_joint_updates"] == 24
    assert learner.next_trigger == 83968
    assert agent.calls == [49152] * 16 + [73728] * 8


def test_scale_budget_does_not_change_learning_identity():
    pytest.importorskip("pydantic_settings")
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import RECIPES
    from eb.checkpoints import resume_fingerprint

    source = JobSpec(kind="train.rl", task_id="tsk_rpo_bfm_pretrain", embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={"template_id":RECIPE, "execution":RECIPES[RECIPE].binding.model_dump()}, policy={"timeout_h":1})
    longer = source.model_copy(deep=True)
    longer.policy.timeout_h = 6
    assert resume_fingerprint(source) == resume_fingerprint(longer)
    longer.resources.gpus = 3
    assert resume_fingerprint(source) != resume_fingerprint(longer)


def test_scale_native_configuration_has_nine_rewards_and_only_larger_batch():
    pytest.importorskip("torch")
    from eb.runners.bfm_ufo_agent import agent_config

    common = {"architecture_profile":"native_fb_v1", "auxiliary_profile":"soft_velocity_v5",
        "rollout_expert_trajectories":False, "hidden_dim":2048, "hidden_layers":6}
    previous = agent_config({**common, "batch_size":1024}, device="cpu").model_dump()
    larger = agent_config({**common, "batch_size":2048}, device="cpu").model_dump()
    assert larger["train"]["batch_size"] == 2048
    larger["train"]["batch_size"] = 1024
    assert larger == previous
    assert len(larger["aux_rewards"]) == 9


def test_scale_resume_moves_from_qualification_to_finite_training_budget(monkeypatch):
    pytest.importorskip("pydantic_settings")
    from dataclasses import replace
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import RECIPES, recipe_resources, recipe_source_receipt, validate_training_execution

    recipe = RECIPES[RECIPE]
    source = JobSpec(kind="train.rl", task_id=recipe.task_id, embodiment_env_id=recipe.embodiment_env_id,
        image=recipe.image, resources=recipe_resources(RECIPE), policy={"timeout_h":1, "max_retries":0},
        params={"template_id":RECIPE, "execution":recipe.binding.model_dump(), "purpose":"qualification",
            "target_learner_updates":24, "source_receipt":recipe_source_receipt(RECIPE)})
    checkpoint = {"compatibility_fingerprint":resume_fingerprint(source), "step":24,
        "run_id":"run_source", "mode":"trainer_state"}
    body = ResumeTrainIn(name="experiment", target_learner_updates=5024)
    monkeypatch.setitem(RECIPES, RECIPE, replace(recipe, status="dependency_ready"))
    with pytest.raises(ValueError):
        validate_training_execution(build_resume_spec(source, "trn_source", checkpoint, body))
    monkeypatch.setitem(RECIPES, RECIPE, replace(recipe, status="smoke_passed"))
    result = build_resume_spec(source, "trn_source", checkpoint, body)
    assert result.policy.timeout_h == 6 and result.params["purpose"] == "training"
    assert validate_training_execution(result) == recipe.binding
    assert resume_fingerprint(result) == resume_fingerprint(source)
    result.policy.timeout_h = 7
    with pytest.raises(ValueError, match="budget"):
        validate_training_execution(result)
