import json
from pathlib import Path

import pytest


def test_A1_adapter_only_changes_contact_margin_factor():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_CONTACT_MARGIN_IMAGE_DIGEST
    raw = json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round6/R1-job-request.json').read_text())['spec']
    raw['image'] = UFO_CONTACT_MARGIN_IMAGE_DIGEST
    raw['params']['operation'] = 'probe_rpo_ufo_contact_margin_v1'
    result = ToolsAdapter().render(JobSpec.model_validate(raw), 'run_margin_test')
    assert result.env['EB_UFO_CONTACT_MARGIN_FACTOR'] == 'A1'
    assert result.env['EB_UFO_GROUND_STIFFNESS_FACTOR'] == 'C2'
    assert result.env['EB_UFO_FRICTION_FACTOR'] == 'S2' and result.env['EB_UFO_SOLVER_FACTOR'] == 'S1'
    assert 'EB_UFO_CONTROL_FACTOR' not in result.env
    raw['params']['margin'] = .003
    with pytest.raises(ValueError, match='frozen'):
        ToolsAdapter().render(JobSpec.model_validate(raw), 'run_margin_test')


def test_C2_default_validator_rejects_A1_header():
    from eb.runners.rpo_ufo_ground_stiffness import validate_receipt
    with pytest.raises(ValueError, match='margin'):
        validate_receipt({'ground_stiffness_factor':'C2','contact_margin_factor':'A1'}, {}, Path('/tmp'))
