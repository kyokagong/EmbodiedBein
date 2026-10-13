import json

import pytest


def test_solver_factor_changes_only_cone_in_actual_native_config(monkeypatch):
    import rpo_ufo_mjlab as physics
    from eb.runners.rpo_ufo_solver_cone import install
    from dataclasses import asdict
    original=physics.make_config
    monkeypatch.setattr(physics,'make_config',original)
    before=original('/data/kk/assets/roboparty/rpo-v2',num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    install()
    after=physics.make_config('/data/kk/assets/roboparty/rpo-v2',num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    a,b=asdict(before.sim.mujoco),asdict(after.sim.mujoco)
    assert [k for k in a if a[k]!=b[k]]==['cone']
    assert a['cone']=='pyramidal' and b['cone']=='elliptic'
    assert before.scene.entities['robot'].articulation==after.scene.entities['robot'].articulation
    assert before.decimation==after.decimation==16


def test_solver_receipt_rejects_unapplied_cone_or_changed_stage(monkeypatch,tmp_path):
    from eb.runners import rpo_ufo_solver_cone as cone,rpo_ufo_action_replay as action
    monkeypatch.setattr(action,'validate_receipt',lambda *args,**kwargs:None)
    receipt={'solver_cone_factor':'S1','integration_factor':'I1','action_program_stage':'replay',
        'phases':[{'mode':m,'rank':r} for m in ('anchor','policy') for r in range(3)]}
    models={f'{m}-{r}':{'compiled_solver':{'cone':1}} for m in ('anchor','policy') for r in range(3)}
    path=tmp_path/'startup-models.json';path.write_text(json.dumps(models))
    cone.validate_receipt(receipt,{},tmp_path)
    models['anchor-0']['compiled_solver']['cone']=0;path.write_text(json.dumps(models))
    with pytest.raises(ValueError,match='cone'):cone.validate_receipt(receipt,{},tmp_path)
    receipt['action_program_stage']='capture'
    with pytest.raises(ValueError,match='cone'):cone.validate_receipt(receipt,{},tmp_path)
