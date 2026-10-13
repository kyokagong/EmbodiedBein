import pytest


def test_tracking_contact_probe_requires_four_card_frozen_parameters():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec,JobKind
    spec=JobSpec(kind=JobKind.ASSET_VALIDATE,embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        resources={'gpus':1},params={'operation':'probe_rpo_ufo_tracking_contact'},runtime={'network':'none'})
    with pytest.raises(ValueError,match='four-card'):
        ToolsAdapter().render(spec,'run_probe')


def test_streamed_native_export_reaches_binding_validation(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from eb.orchestrator.collect import validate_rpo_bfm_export
    output=tmp_path/'output/exported';output.mkdir(parents=True)
    (output/'export-manifest.json').write_text('{}')
    (output/'prompt-manifest.json').write_text('{}')
    execution={'checkpoint_format':'ufo_agent_state_v2'}
    monkeypatch.setattr('eb.orchestrator.collect.get_settings',lambda:SimpleNamespace(
        models_root=tmp_path/'models',data_root=tmp_path/'state'))
    monkeypatch.setattr('eb.checkpoints.checkpoint_model',lambda *a,**k:{'manifest':{
        'format':'ufo_agent_state_v2','native_checkpoint_path':'model/ckpt'}})
    monkeypatch.setattr('eb.runners.bfm_ufo_checkpoint.validate_ufo_checkpoint',lambda *a,**k:{
        'execution':execution,'format':'ufo_agent_state_v2','step':16})
    job=SimpleNamespace(spec={'params':{'operation':'rpo_bfm_export','source_run_id':'run_abcdef',
        'asset_sha256':'a'*64,'model_execution':execution},'inputs':{'datasets':[{'path':str(tmp_path/'motion')}]}})
    # Recognition of v2 must not bypass the independent export/checkpoint binding guard.
    with pytest.raises(ValueError,match='differs from its checkpoint'):
        validate_rpo_bfm_export(tmp_path,job)


@pytest.mark.asyncio
async def test_compliance_resume_candidate_does_not_offer_failed_long_gate(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from eb.api import routes_train
    from eb.domain.training_recipe import UFO_COMPLIANCE_RECIPE,RECIPES
    source=SimpleNamespace(params={'template_id':UFO_COMPLIANCE_RECIPE,'purpose':'training',
        'execution':RECIPES[UFO_COMPLIANCE_RECIPE].binding.model_dump()})
    checkpoint={'step':524,'path':str(tmp_path),'run_id':'run_example','bytes':1,
        'mode':'trainer_state','state_coverage':{}}
    async def context(*args,**kwargs):return SimpleNamespace(id='trn_example'),None,source,checkpoint
    monkeypatch.setattr(routes_train,'_resume_context',context)
    result=await routes_train.get_resume_candidate('trn_example',None,None)
    assert not result['available']
