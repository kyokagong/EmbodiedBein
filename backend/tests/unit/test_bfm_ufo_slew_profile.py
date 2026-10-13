from dataclasses import replace

import pytest


def test_slew_profile_preserves_native_architecture_and_freezes_control_sources(monkeypatch):
    from eb.domain.job import JobSpec
    from eb.domain.rlinf_form import build_overrides
    from eb.domain.training_recipe import (
        RECIPES,
        UFO_SLEW_RECIPE,
        recipe_image_digest,
        recipe_resources,
        recipe_source_receipt,
        validate_training_execution,
    )

    recipe = RECIPES[UFO_SLEW_RECIPE]
    from eb.domain.rlinf_form import get_template
    from eb.environment_catalog import training_definition
    assert training_definition(get_template(UFO_SLEW_RECIPE))["settings"] == {}
    monkeypatch.setitem(RECIPES, UFO_SLEW_RECIPE, replace(recipe, status="dependency_ready"))
    spec = JobSpec(
        kind="train.rl",
        embodiment_env_id=recipe.embodiment_env_id,
        task_id=recipe.task_id,
        image=recipe_image_digest(UFO_SLEW_RECIPE),
        params={
            "template_id": UFO_SLEW_RECIPE,
            "execution": recipe.binding.model_dump(),
            "purpose": "qualification",
            "target_learner_updates": 16,
            "source_receipt": dict(recipe_source_receipt(UFO_SLEW_RECIPE)),
        },
        resources=recipe_resources(UFO_SLEW_RECIPE),
        policy={"timeout_h": 1, "max_retries": 0},
    )
    assert validate_training_execution(spec) == recipe.binding
    values = build_overrides(UFO_SLEW_RECIPE, 4, "slew", [])
    assert "actor.model.hidden_dim=2048" in values and "actor.model.batch_size=1024" in values
    assert "env.train.init_params.id=RPO-UFO-Isaac-RSI-slew-v4" in values
    assert "+actor.model.rollout_expert_trajectories=False" in values
    spec.params["source_receipt"].pop("action_control_source_sha256")
    with pytest.raises(ValueError, match="source receipt"):
        validate_training_execution(spec)
