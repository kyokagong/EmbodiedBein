import pytest
import torch


def limits():
    return {'joint_pos':torch.tensor([[-10.,10.]]*23),
        'joint_vel':torch.full((23,),8.),'applied_torque':torch.full((23,),27.)}


@pytest.mark.parametrize('substeps,dt',[(8,.0025),(16,.00125)])
def test_interval_keeps_20ms_mean_exposure_impulse_and_strict_latch(substeps,dt):
    from eb.runners.rpo_ufo_isaac import PhysicsInterval
    value = PhysicsInterval(1,limits(),device='cpu',substeps=substeps,physics_dt=dt)
    q = torch.zeros(1,23); dq = torch.full_like(q,9.)
    force = torch.full((1,2),100.)
    value.begin()
    for index in range(substeps):
        value.sample(index,q,dq,q,q,force,torch.zeros(1,5,3))
    torch.testing.assert_close(value.velocity_penalty(),torch.tensor([23*(9/8-.7)**2]))
    torch.testing.assert_close(value.exposure,torch.ones(1,23))
    torch.testing.assert_close(value.force_impulse,torch.full((1,2),2.))
    assert value.strict_flags[0,1] and not value.training_flags[0,1]
    assert value.trace(torch.ones(1,1,dtype=torch.long)).shape==(1,223)
    assert value.trace(torch.ones(1,1,dtype=torch.long))[0,164]==substeps
    with pytest.raises(ValueError):
        value.sample(substeps,q,dq,q,q,force,torch.zeros(1,5,3))


def test_unapproved_step_or_frequency_is_rejected():
    from eb.runners.rpo_ufo_isaac import PhysicsInterval
    for count,dt in ((8,.00125),(16,.0025),(4,.005)):
        with pytest.raises(ValueError,match='integration'):
            PhysicsInterval(1,limits(),device='cpu',substeps=count,physics_dt=dt)


def test_native_config_changes_only_timestep_and_preserves_control_and_actuators():
    import numpy as np
    from eb.runners.rpo_ufo_mjlab import make_config
    asset = '/data/kk/assets/roboparty/rpo-v2'
    before = make_config(asset,num_envs=1,seed=42)
    after = make_config(asset,num_envs=1,seed=42,substeps=16,physics_dt=.00125)
    assert before.decimation*before.sim.mujoco.timestep == after.decimation*after.sim.mujoco.timestep == .02
    assert before.scene.entities['robot'].articulation == after.scene.entities['robot'].articulation
    a = before.scene.entities['robot'].spec_fn().compile()
    b = after.scene.entities['robot'].spec_fn().compile()
    for key in ('body_mass','body_inertia','jnt_range','dof_armature','geom_solref','geom_solimp',
                'geom_friction','geom_contype','geom_conaffinity'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert a.opt.timestep==.0025 and b.opt.timestep==.00125
    assert a.opt.integrator==b.opt.integrator and a.opt.solver==b.opt.solver


def test_source_case_selection_is_hash_bound_and_bounded():
    from pathlib import Path
    import hashlib
    from eb.runners.rpo_ufo_integration_factor import source_cases
    path = Path('/data/kk/embodiedbein/runs/run_6ed60271580c4d6c98f2/outputs')
    sha = hashlib.sha256((path/'dynamics-diagnostics.json').read_bytes()).hexdigest()
    selected = source_cases(path,sha)
    assert len(selected)==6 and all(1<=len(ids)<=5 for ids in selected.values())
    assert 798 in selected['policy-2']
    with pytest.raises(ValueError,match='changed'):
        source_cases(path,'0'*64)


def test_packaged_trace_accepts_native_int32_contact_world_ids(monkeypatch,tmp_path):
    from types import SimpleNamespace as NS
    import numpy as np
    import warp as wp
    import rpo_ufo_mjlab as physics
    import rpo_ufo_contact_factor as contact
    import rpo_ufo_integration_factor as integration
    data = NS(qpos=torch.zeros(2,30),qvel=torch.zeros(2,29),nefc=torch.zeros(2,dtype=torch.int32),
        qfrc_actuator=torch.zeros(2,29),qfrc_constraint=torch.zeros(2,29),nacon=torch.tensor([2]),
        efc=NS(J=torch.zeros(2,0,29),force=torch.zeros(2,0),type=torch.zeros(2,0,dtype=torch.int32)),
        contact=NS(worldid=torch.tensor([0,1],dtype=torch.int32),dist=torch.tensor([-.01,-.005])))
    class FakePhysics:
        def __init__(self,**kwargs):
            self.lab=NS(sim=NS(wp_data=data,mj_model=NS(nv=29)))
            self.num_envs=2; self.population=[None]; self.dof_ids=list(range(6,29)); self.feet=[0,1]
            self.target=self.requested=self.last=torch.zeros(2,23)
            self.alive=torch.ones(2,dtype=torch.bool); self.reference=torch.zeros(2,3,dtype=torch.long)
            self.steps=torch.ones(2,1,dtype=torch.long)
        def device(self): return 'cpu'
        def step(self,actions): pass
        def _refresh(self,dt): self._measured_samples=len(self.population)
        def _forces(self): return torch.zeros(2,2,3)
    monkeypatch.setattr(physics,'RPOUFOMJLabPhysics',FakePhysics)
    monkeypatch.setattr(contact,'install_measurement',lambda *args,**kwargs:None)
    monkeypatch.setattr(wp,'to_torch',lambda value:value)
    integration.install_trace('I1',0,'anchor',tmp_path,[1])
    env=physics.RPOUFOMJLabPhysics(); env.step(torch.zeros(2,23)); env._refresh(0.)
    assert len(env._braking_rows)==1
    np.testing.assert_allclose(env._braking_rows[0]['contact_depth_after'],[.005])


@pytest.mark.parametrize('damage',['time','resurrection','continuity','constraint'])
def test_braking_receipt_rejects_internally_inconsistent_arrays(tmp_path,monkeypatch,damage):
    import json
    import numpy as np
    from eb.runners import rpo_ufo_contact_factor as contact, rpo_ufo_integration_factor as integration
    # Parent hashes/population/RSI checks are tested separately; exercise the new trace checks.
    monkeypatch.setattr(contact,'validate_receipt',lambda *args,**kwargs:None)
    phases = [{'rank':r,'mode':m} for m in ('anchor','policy') for r in range(3)]
    cases,models,members = {},{},[{'file':'trace-cases.json'},{'file':'startup-models.json'}]
    n=768
    for phase in phases:
        suffix=f"{phase['mode']}-{phase['rank']}"; cases[suffix]=[0]
        models[suffix]={'compiled_solver':{'timestep_s':.00125},
            'integration_actual':{'physics_dt_s':.00125,'substeps':16,'control_dt_s':.02},
            'geom_solref':[[.02,1.]],'physics_dt_s':.00125,'decimation':16}
        arrays={**{k:np.zeros((n,1,23)) for k in ('target','requested','applied','motor_torque','contact_torque','limit_torque','other_constraint','constraint_torque')},
            **{k:np.zeros((n,1,30)) for k in ('qpos_before','qpos_after')},
            **{k:np.zeros((n,1,29)) for k in ('qvel_before','qvel_after')},
            'foot_force':np.zeros((n,1,2,3)),'original_alive':np.ones((n,1),dtype=bool),
            'reference':np.zeros((n,1,3),dtype=int),'episode_step':np.ones((n,1)),
            'contact_depth_after':np.zeros((n,1)),'case_ids':np.array([0]),'dof_ids':np.arange(6,29),
            'pos_ids':np.arange(7,30),'joint_vel_limits':np.full(23,8.),'torque_limits':np.full(23,27.),
            'time_s':np.arange(1,n+1)*.00125}
        np.savez_compressed(tmp_path/f'braking-{suffix}.npz',**arrays)
        np.savez_compressed(tmp_path/f'rsi-{suffix}.npz',reference=np.zeros((2048,3),dtype=int))
        (tmp_path/f'braking-{suffix}.json').write_text(json.dumps({'factor':'I1','physics_dt_s':.00125,
            'substeps':16,'control_dt_s':.02,'samples':n,'case_ids':[0]}))
        members += [{'file':f'braking-{suffix}.{ext}'} for ext in ('json','npz')]
    (tmp_path/'trace-cases.json').write_text(json.dumps(cases))
    (tmp_path/'startup-models.json').write_text(json.dumps(models))
    receipt={'integration_factor':'I1','contact_factor':'B0','files':members,'phases':phases}
    integration.validate_receipt(receipt,{},tmp_path)
    path=tmp_path/'braking-anchor-0.npz'
    with np.load(path) as data: arrays={k:data[k].copy() for k in data.files}
    if damage=='time': arrays['time_s'][10]+=.00125
    if damage=='resurrection': arrays['original_alive'][10]=False
    if damage=='continuity': arrays['qvel_before'][10,0,0]=10.
    if damage=='constraint': arrays['contact_torque'][10,0,0]=10.
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError): integration.validate_receipt(receipt,{},tmp_path)
