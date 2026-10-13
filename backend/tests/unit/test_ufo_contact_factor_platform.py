import json
from pathlib import Path

import pytest


def test_contact_factor_uses_existing_four_gpu_diagnostic_admission():
    from eb.capacity import estimate_job
    from eb.domain.job import JobKind, JobSpec
    result = estimate_job(JobSpec(kind=JobKind.ASSET_VALIDATE,
        embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':'probe_rpo_ufo_contact_factor_v1','candidate':'C1'},resources={'gpus':4}))
    assert result.basis['inputs'].get('num_envs') == 6144
    assert result.basis['inputs'].get('learner_count') == 0


def test_adapter_binds_candidate_and_rejects_second_factor():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_CONTACT_FACTOR_IMAGE_DIGEST
    raw = json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round2/diagnostic-v1-job-request.json').read_text())['spec']
    raw['params'] = {'operation':'probe_rpo_ufo_contact_factor_v1','candidate':'C1'}
    raw['image'] = UFO_CONTACT_FACTOR_IMAGE_DIGEST
    rendered = ToolsAdapter().render(JobSpec.model_validate(raw),'run_contact_test')
    assert rendered.env['EB_UFO_CONTACT_FACTOR'] == 'C1'
    assert rendered.command[-1].endswith('/rpo_ufo_contact_factor.py')
    assert rendered.gpus == 4
    raw['params']['ankle_kd'] = 5.
    with pytest.raises(ValueError, match='frozen'):
        ToolsAdapter().render(JobSpec.model_validate(raw),'run_contact_test')
