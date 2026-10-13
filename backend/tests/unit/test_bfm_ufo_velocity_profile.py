import pytest

RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5"


def test_velocity_profile_has_native_capacity_and_protected_objective():
    from eb.capacity.estimator import estimate_job
    from eb.domain.job import JobSpec
    from eb.domain.rlinf_form import build_overrides, get_template
    from eb.domain.training_recipe import RECIPES, recipe_profile_id, recipe_resources
    from eb.environment_catalog import training_definition

    recipe = RECIPES[RECIPE]
    assert recipe.status in {"adapter_ready", "dependency_ready"}
    assert recipe_profile_id(RECIPE) == "rpo_bfm_ufo_isaac_4gpu_rsi_slew_velocity_v5"
    assert recipe_resources(RECIPE)["memory_gb"] == 96
    values = build_overrides(RECIPE, 4, "velocity", [])
    for value in (
        "+actor.model.auxiliary_profile=soft_velocity_v5",
        "actor.model.batch_size=1024",
        "actor.model.replay_capacity=1000000",
        "+actor.model.rollout_expert_trajectories=False",
        "env.train.init_params.id=RPO-UFO-Isaac-RSI-slew-velocity-v5",
    ):
        assert value in values
    with pytest.raises(ValueError, match="protects"):
        build_overrides(RECIPE, 4, "bad", ["actor.model.auxiliary_profile=custom"])
    definition = training_definition(get_template(RECIPE))
    assert definition["settings"] == {} and "limits_dof_vel" in definition["reward"]
    estimate = estimate_job(
        JobSpec(
            kind="train.rl",
            embodiment_env_id=recipe.embodiment_env_id,
            task_id=recipe.task_id,
            params={"template_id": RECIPE},
            resources={"gpus": 4},
        )
    )
    assert estimate.basis["inputs"]["native_trainable_parameters"] == 435116484
    assert estimate.basis["inputs"]["cpu_replay_bytes"] == 7042000000


def test_velocity_profile_uses_identical_controller_with_separate_objective_identity():
    from eb.rpo_bfm import SLEW_CONTROL, action_control_for_execution, auxiliary_profile_for_execution

    execution = {"recipe_id": RECIPE}
    assert action_control_for_execution(execution) == SLEW_CONTROL
    assert auxiliary_profile_for_execution(execution) == "soft_velocity_v5"
    assert (
        auxiliary_profile_for_execution({"recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4"}) is None
    )
