"""Native simulator identity stays separate from the published PhysX recipes."""
import pytest


def test_viability_physics_probe_uses_existing_four_card_admission():
    from eb.capacity import estimate_job
    from eb.domain.job import JobKind, JobSpec
    result = estimate_job(JobSpec(kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':'probe_rpo_ufo_mjlab_viability_v13'}, resources={'gpus':4}))
    assert result.basis['inputs']['num_envs'] == 6144
    assert result.basis['inputs']['sampler_count'] == 3


def test_native_recipe_topology_codec_and_budget():
    from eb.domain.training_recipe import (RECIPES, UFO_MJLAB_RECIPE, recipe_resources,
        recipe_profile_id, validate_update_target, recipe_eval_image_digest)
    from eb.domain.rlinf_form import get_template
    from eb.rpo_bfm import CODEC_V2_HASH, codec_hash_for_execution, action_control_for_execution
    recipe = RECIPES[UFO_MJLAB_RECIPE]
    assert recipe.binding.simulator_id == "mjlab"
    assert recipe.binding.topology_id == "one_learner_three_samplers"
    assert recipe_resources(UFO_MJLAB_RECIPE)["gpus"] == 4
    assert recipe_resources(UFO_MJLAB_RECIPE)["memory_gb"] == 96
    assert "mjlab" in recipe_profile_id(UFO_MJLAB_RECIPE)
    tpl = get_template(UFO_MJLAB_RECIPE)
    assert "env.train.env_type=mjlab" in tpl.default_overrides
    assert "env.train.total_num_envs=6144" in tpl.default_overrides
    assert "actor.model.batch_size=2048" in tpl.default_overrides
    assert "actor.model.replay_capacity=4000000" in tpl.default_overrides
    assert codec_hash_for_execution(recipe.binding.model_dump()) == CODEC_V2_HASH
    assert action_control_for_execution(recipe.binding.model_dump())["physics"]["backend"] == "mjlab"
    validate_update_target(524, start=24, recipe_id=UFO_MJLAB_RECIPE)
    with pytest.raises(ValueError):
        validate_update_target(525, start=24, recipe_id=UFO_MJLAB_RECIPE)
    with pytest.raises(ValueError, match="evaluation"):
        recipe_eval_image_digest(UFO_MJLAB_RECIPE)


@pytest.mark.parametrize("start,target", [(0, 500), (16, 516), (524, 1024)])
def test_native_experiment_cannot_bypass_frozen_segment(start, target):
    from eb.domain.training_recipe import UFO_MJLAB_RECIPE, validate_update_target
    with pytest.raises(ValueError, match="24.*524"):
        validate_update_target(target, start=start, recipe_id=UFO_MJLAB_RECIPE)


@pytest.mark.asyncio
async def test_completed_native_budget_is_not_a_resume_candidate(monkeypatch):
    from types import SimpleNamespace
    from eb.api import routes_train
    from eb.domain.training_recipe import UFO_MJLAB_RECIPE, RECIPES
    source = SimpleNamespace(params={"template_id": UFO_MJLAB_RECIPE, "purpose": "training",
        "execution": RECIPES[UFO_MJLAB_RECIPE].binding.model_dump()})
    checkpoint = {"step": 524, "path": "/tmp/model", "run_id": "run_native", "bytes": 1,
                  "mode": "trainer_state", "state_coverage": {}}
    async def context(*args, **kwargs):
        return SimpleNamespace(id="trn_native"), None, source, checkpoint
    monkeypatch.setattr(routes_train, "_resume_context", context)
    result = await routes_train.get_resume_candidate("trn_native", None, None)
    assert result["available"] is False
    assert "预算" in result["reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize('step,available', [(100,True),(600,False)])
async def test_viability_resume_candidate_closes_after_observation(monkeypatch,step,available):
    from types import SimpleNamespace
    from eb.api import routes_train
    from eb.domain.training_recipe import UFO_MJLAB_VIABILITY_RECIPE, RECIPES
    recipe = RECIPES[UFO_MJLAB_VIABILITY_RECIPE]
    from dataclasses import replace
    monkeypatch.setitem(RECIPES,UFO_MJLAB_VIABILITY_RECIPE,replace(recipe,status='smoke_passed'))
    source=SimpleNamespace(params={'template_id':UFO_MJLAB_VIABILITY_RECIPE,'purpose':'qualification',
        'execution':recipe.binding.model_dump()})
    checkpoint={'step':step,'path':'/tmp/model','run_id':'run_viability','bytes':1,
        'mode':'trainer_state','state_coverage':{}}
    async def context(*args,**kwargs):
        return SimpleNamespace(id='trn_viability'),None,source,checkpoint
    monkeypatch.setattr(routes_train,'_resume_context',context)
    result=await routes_train.get_resume_candidate('trn_viability',None,None)
    assert result['available'] is available
    if available:
        assert result['suggested_target'] == 600
