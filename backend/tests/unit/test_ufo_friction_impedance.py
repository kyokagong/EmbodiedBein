import json
from dataclasses import asdict
import pytest


def test_S2_changes_only_impratio_of_actual_S1_config(monkeypatch):
    import rpo_ufo_mjlab as physics
    from eb.runners.rpo_ufo_solver_cone import install as cone
    from eb.runners.rpo_ufo_friction_impedance import install
    monkeypatch.setattr(physics,'make_config',physics.make_config)
    cone()
    before=physics.make_config('/data/kk/assets/roboparty/rpo-v2',num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    install()
    after=physics.make_config('/data/kk/assets/roboparty/rpo-v2',num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    a,b=asdict(before.sim.mujoco),asdict(after.sim.mujoco)
    assert [k for k in a if a[k]!=b[k]]==['impratio']
    assert a['impratio']==1. and b['impratio']==.1 and b['cone']=='elliptic'
    assert before.scene.entities['robot'].articulation==after.scene.entities['robot'].articulation
    import mujoco,torch,warp as wp
    from types import SimpleNamespace as NS
    from eb.runners.rpo_ufo_action_replay import ControlRecorder
    model=mujoco.MjModel.from_xml_string('<mujoco/>');after.sim.mujoco.apply(model)
    monkeypatch.setattr(wp,'to_torch',lambda value:value)
    env=NS(lab=NS(sim=NS(mj_model=model,wp_data=NS(qpos=torch.zeros(2048,30),qvel=torch.zeros(2048,29)))))
    recorder=ControlRecorder(None,'anchor-0');recorder.begin(env)
    assert recorder.solver_options=={'cone':1,'impratio':.1,'tolerance':1e-8,'ls_iterations':50,'ls_tolerance':.01}


def test_S2_receipt_requires_actual_impratio_and_original_solver_options(monkeypatch,tmp_path):
    from eb.runners import rpo_ufo_friction_impedance as friction,rpo_ufo_solver_cone as cone
    monkeypatch.setattr(cone,'validate_receipt',lambda *args,**kwargs:None)
    receipt={'solver_friction_factor':'S2','phases':[{'mode':m,'rank':r} for m in ('anchor','policy') for r in range(3)]}
    options={'cone':1,'impratio':.1,'tolerance':1e-8,'ls_iterations':50,'ls_tolerance':.01}
    for p in receipt['phases']:(tmp_path/f"actions-{p['mode']}-{p['rank']}.json").write_text(json.dumps({'actual_solver_options':options}))
    friction.validate_receipt(receipt,{},tmp_path)
    options['impratio']=1.;(tmp_path/'actions-anchor-0.json').write_text(json.dumps({'actual_solver_options':options}))
    with pytest.raises(ValueError,match='impratio'):friction.validate_receipt(receipt,{},tmp_path)
