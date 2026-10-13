import pytest

H="rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8"


def test_contact_profile_has_6144_envs_500_update_budget_and_v2_codec():
    from eb.domain.training_recipe import RECIPES,validate_update_target,recipe_eval_image_digest
    from eb.domain.rlinf_form import build_overrides
    from eb.rpo_bfm import ENV_V2_ID,CODEC_V2_HASH,codec_hash_for_execution,action_control_for_execution
    recipe=RECIPES[H]
    assert recipe.embodiment_env_id==ENV_V2_ID
    assert codec_hash_for_execution(recipe.binding.model_dump())==CODEC_V2_HASH
    assert action_control_for_execution(recipe.binding.model_dump())['physics']['physics_dt_s']==.0025
    values=build_overrides(H,4,'contact',[])
    assert 'env.train.total_num_envs=6144' in values
    assert 'actor.model.batch_size=2048' in values
    assert '+actor.model.contact_profile=contact_stable_v8' in values
    validate_update_target(524,start=24,recipe_id=H)
    with pytest.raises(ValueError):validate_update_target(525,start=24,recipe_id=H)
    with pytest.raises(ValueError):recipe_eval_image_digest(H)
