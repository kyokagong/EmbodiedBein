import json
from pathlib import Path
import pytest


def test_D1_adapter_is_fixed_I1_replay_without_cone_factor():
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_CONTROL_DAMPING_IMAGE_DIGEST
    raw=json.loads((Path(__file__).parents[3]/'docs/progress/artifacts/2026-10-13-rpo-learning-round6/R1-job-request.json').read_text())['spec']
    raw['image']=UFO_CONTROL_DAMPING_IMAGE_DIGEST;raw['params']['operation']='probe_rpo_ufo_control_damping_v1'
    result=ToolsAdapter().render(JobSpec.model_validate(raw),'run_damping_test')
    assert result.env['EB_UFO_CONTROL_FACTOR']=='D1' and result.env['EB_UFO_ACTION_STAGE']=='replay'
    assert 'EB_UFO_SOLVER_FACTOR' not in result.env
    raw['params']['candidate']='I0'
    with pytest.raises(ValueError,match='damping'):ToolsAdapter().render(JobSpec.model_validate(raw),'run_damping_test')


def test_D1_validator_has_no_torch_dependency(monkeypatch,tmp_path):
    from eb.runners import rpo_ufo_control_damping as control,rpo_ufo_action_replay as action
    from eb.rpo_bfm import RPO_LAB_JOINT_ORDER
    monkeypatch.setattr(action,'validate_receipt',lambda *args:None)
    phases=[{'mode':m,'rank':r} for m in ('anchor','policy') for r in range(3)]
    baseline=Path('/data/kk/embodiedbein/runs/run_8a8b89522bdf4487a1af/outputs/startup-models.json')
    models=json.loads(baseline.read_text())
    for model in models.values():
        for n in RPO_LAB_JOINT_ORDER:
            if 'ankle' in n:model['actuator_parameters'][n][1]=10.
    p=tmp_path/'startup-models.json';p.write_text(json.dumps(models))
    receipt={'control_damping_factor':'D1','integration_factor':'I1','action_program_stage':'replay','phases':phases}
    control.validate_receipt(receipt,{},tmp_path)
    models['anchor-0']['actuator_parameters']['left_ankle_pitch_joint'][1]=2.5;p.write_text(json.dumps(models))
    with pytest.raises(ValueError,match='damping'):control.validate_receipt(receipt,{},tmp_path)
