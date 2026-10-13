import json
from dataclasses import asdict
import pytest


def test_D1_changes_only_four_ankle_damping_values(monkeypatch):
    import rpo_ufo_mjlab as physics
    from eb.runners.rpo_ufo_control_damping import install
    original=physics.actuator_parameters
    monkeypatch.setattr(physics,'actuator_parameters',original)
    before=physics.make_config('/data/kk/assets/roboparty/rpo-v2',num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    install()
    after=physics.make_config('/data/kk/assets/roboparty/rpo-v2',num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    changed=[]
    for a,b in zip(before.scene.entities['robot'].articulation.actuators,after.scene.entities['robot'].articulation.actuators):
        old,new=asdict(a),asdict(b)
        if old!=new:
            assert [k for k in old if old[k]!=new[k]]==['damping']
            assert old['damping']==2.5 and new['damping']==10.
            assert 'ankle' in old['target_names_expr'][0]
            changed.append(old['target_names_expr'][0])
    assert len(changed)==4 and asdict(before.sim)==asdict(after.sim)


def test_D1_receipt_checks_actual_PD_and_original_cone(monkeypatch,tmp_path):
    from eb.runners import rpo_ufo_control_damping as damping,rpo_ufo_action_replay as action
    import rpo_ufo_mjlab as physics
    monkeypatch.setattr(action,'validate_receipt',lambda *args:None)
    names=physics.RPO_LAB_JOINT_ORDER
    params={n:list(physics.actuator_parameters(n)) for n in names}
    for n in names:
        if 'ankle' in n:params[n][1]=10.
    receipt={'control_damping_factor':'D1','integration_factor':'I1','action_program_stage':'replay',
        'phases':[{'mode':m,'rank':r} for m in ('anchor','policy') for r in range(3)]}
    models={f'{m}-{r}':{'actuator_parameters':params.copy(),'compiled_solver':{'cone':0}} for m in ('anchor','policy') for r in range(3)}
    path=tmp_path/'startup-models.json';path.write_text(json.dumps(models))
    damping.validate_receipt(receipt,{},tmp_path)
    models['anchor-0']['compiled_solver']['cone']=1;path.write_text(json.dumps(models))
    with pytest.raises(ValueError,match='damping'):damping.validate_receipt(receipt,{},tmp_path)
