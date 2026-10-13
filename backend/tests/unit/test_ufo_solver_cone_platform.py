import json
from pathlib import Path

import pytest


def test_solver_adapter_requires_same_I1_replay_and_fixed_cone():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_SOLVER_CONE_IMAGE_DIGEST
    raw=json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round6/R1-job-request.json').read_text())['spec']
    raw['image']=UFO_SOLVER_CONE_IMAGE_DIGEST;raw['params']['operation']='probe_rpo_ufo_solver_cone_v1'
    result=ToolsAdapter().render(JobSpec.model_validate(raw),'run_solver_test')
    assert result.env['EB_UFO_SOLVER_FACTOR']=='S1'
    assert result.env['EB_UFO_INTEGRATION_FACTOR']=='I1' and result.env['EB_UFO_ACTION_STAGE']=='replay'
    raw['params']['candidate']='I0'
    with pytest.raises(ValueError,match='cone'):ToolsAdapter().render(JobSpec.model_validate(raw),'run_solver_test')
    raw['params']['candidate']='I1';raw['params']['friction']=.5
    with pytest.raises(ValueError,match='frozen'):ToolsAdapter().render(JobSpec.model_validate(raw),'run_solver_test')
