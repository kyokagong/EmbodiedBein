import json
from pathlib import Path
import pytest


def test_S2_adapter_preserves_S1_and_rejects_extra_parameters():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_FRICTION_IMPEDANCE_IMAGE_DIGEST
    raw=json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round6/R1-job-request.json').read_text())['spec']
    raw['image']=UFO_FRICTION_IMPEDANCE_IMAGE_DIGEST;raw['params']['operation']='probe_rpo_ufo_friction_impedance_v1'
    result=ToolsAdapter().render(JobSpec.model_validate(raw),'run_friction_test')
    assert result.env['EB_UFO_SOLVER_FACTOR']=='S1' and result.env['EB_UFO_FRICTION_FACTOR']=='S2'
    assert 'EB_UFO_CONTROL_FACTOR' not in result.env
    raw['params']['impratio']=.2
    with pytest.raises(ValueError,match='frozen'):ToolsAdapter().render(JobSpec.model_validate(raw),'run_friction_test')
