import hashlib
import json
from pathlib import Path

import pytest


def test_capture_uses_frozen_source_and_rejects_second_factor_and_uncaptured_replay():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_ACTION_REPLAY_IMAGE_DIGEST
    raw=json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round4/I0-corrected-job-request.json').read_text())['spec']
    source=Path('/data/kk/embodiedbein/runs/run_743dba585660411ca866/outputs/dynamics-diagnostics.json')
    raw['image']=UFO_ACTION_REPLAY_IMAGE_DIGEST
    raw['params']={'operation':'probe_rpo_ufo_action_replay_v1','candidate':'I0','stage':'capture',
        'trace_source_run_id':'run_743dba585660411ca866',
        'trace_source_receipt_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    result=ToolsAdapter().render(JobSpec.model_validate(raw),'run_action_test')
    assert result.gpus==4 and result.env['EB_UFO_ACTION_STAGE']=='capture'
    assert result.command[-1].endswith('/rpo_ufo_action_replay.py')
    raw['params']['candidate']='I1'
    with pytest.raises(ValueError,match='stage'): ToolsAdapter().render(JobSpec.model_validate(raw),'run_action_test')
    raw['params'].update(candidate='I0',stage='replay')
    with pytest.raises(ValueError,match='captured'): ToolsAdapter().render(JobSpec.model_validate(raw),'run_action_test')
    raw['params'].update(stage='capture',kd=5.)
    with pytest.raises(ValueError,match='frozen'): ToolsAdapter().render(JobSpec.model_validate(raw),'run_action_test')


def test_action_capture_capacity_keeps_no_learner():
    from eb.capacity import estimate_job
    from eb.domain.job import JobKind,JobSpec
    value=estimate_job(JobSpec(kind=JobKind.ASSET_VALIDATE,embodiment_env_id='ee_rpo_v2_bfm_23dof_v2',
        params={'operation':'probe_rpo_ufo_action_replay_v1'},resources={'gpus':4}))
    assert value.basis['inputs']['learner_count']==0
    assert value.basis['inputs']['num_envs']==6144
