import hashlib
import json
from pathlib import Path

import pytest


def test_integration_adapter_freezes_50hz_source_and_rejects_second_factor():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_INTEGRATION_FACTOR_IMAGE_DIGEST
    raw = json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round3/B0-corrected-job-request.json').read_text())['spec']
    source = Path('/data/kk/embodiedbein/runs/run_6ed60271580c4d6c98f2/outputs/dynamics-diagnostics.json')
    raw['image'] = UFO_INTEGRATION_FACTOR_IMAGE_DIGEST
    raw['params'] = {'operation':'probe_rpo_ufo_integration_factor_v1','candidate':'I1',
        'trace_source_run_id':'run_6ed60271580c4d6c98f2',
        'trace_source_receipt_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    result = ToolsAdapter().render(JobSpec.model_validate(raw),'run_integration_test')
    assert result.gpus==4
    assert result.env['EB_UFO_INTEGRATION_FACTOR']=='I1'
    assert result.command[-1].endswith('/rpo_ufo_integration_factor.py')
    assert ('/data/kk/embodiedbein/runs/run_6ed60271580c4d6c98f2/outputs','/workspace/trace-source','ro') in result.mounts
    raw['params']['control_hz'] = 100
    with pytest.raises(ValueError,match='frozen'):
        ToolsAdapter().render(JobSpec.model_validate(raw),'run_integration_test')
    raw['params'].pop('control_hz'); raw['params']['trace_source_receipt_sha256']='0'*64
    with pytest.raises(ValueError,match='immutable'):
        ToolsAdapter().render(JobSpec.model_validate(raw),'run_integration_test')


def test_integration_capacity_has_three_samplers_without_learner():
    from eb.capacity import estimate_job
    from eb.domain.job import JobKind,JobSpec
    result = estimate_job(JobSpec(kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':'probe_rpo_ufo_integration_factor_v1','candidate':'I1'},resources={'gpus':4}))
    assert result.basis['inputs']['learner_count']==0
    assert result.basis['inputs']['num_envs']==6144
