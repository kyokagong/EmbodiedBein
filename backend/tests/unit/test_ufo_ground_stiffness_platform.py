import json
from pathlib import Path
import pytest


def test_C2_adapter_is_fixed_S2_with_only_ground_stiffness_factor():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_GROUND_STIFFNESS_IMAGE_DIGEST
    raw=json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round6/R1-job-request.json').read_text())['spec']
    raw['image']=UFO_GROUND_STIFFNESS_IMAGE_DIGEST;raw['params']['operation']='probe_rpo_ufo_ground_stiffness_v1'
    result=ToolsAdapter().render(JobSpec.model_validate(raw),'run_ground_test')
    assert result.env['EB_UFO_GROUND_STIFFNESS_FACTOR']=='C2'
    assert result.env['EB_UFO_FRICTION_FACTOR']=='S2' and result.env['EB_UFO_SOLVER_FACTOR']=='S1'
    assert 'EB_UFO_CONTROL_FACTOR' not in result.env
    raw['params']['dampratio']=.2
    with pytest.raises(ValueError,match='frozen'):ToolsAdapter().render(JobSpec.model_validate(raw),'run_ground_test')
