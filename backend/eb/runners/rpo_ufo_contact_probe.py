"""Paired train-RSI physics diagnostic; no learner or independent evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

PHASES=('baseline','grounded','substeps')
PROTOCOL={'schema_version':'eb/ufo-contact-probe/v3','phases':list(PHASES),
    'envs':32,'rounds':16,'steps':16,'seed':64042,'split':'train',
    'pairing':'same reference state, requested action sequence and startup seed; solver warm start unverified',
    'scope':'contact-eligible reference starts; diagnostic only, no learner update or acceptance',
    'physics_samples':'new integration frames only; exclude prior interval/startup contact'}


def measured_summary(raw):
    a,t,valid=raw['trace'],raw['terminated'],raw['active']
    return {'first_terminated':int(t[:,0].sum()),'initial_states':512,
        'max_dq_active':float(raw['peak_dq'][valid].max()),
        'max_boundary_dq_active':float(raw['boundary_dq'][valid].max()),
        'max_foot_force_active_N':float(raw['foot_force_N'][valid].max()),
        'fk_before_max_m':float(np.max(raw['fk_before'])),'fk_after_max_m':float(np.max(raw['fk_after'])),
        'first_hard_flags':a[:,0,:,115:118].sum(axis=(0,1)).astype(int).tolist()}


def phase(name,output):
    sys.path.insert(0,str(Path(os.environ['RPO_ASSET_ROOT'])/'robolab'))
    if name!='baseline':os.environ['EB_UFO_CONTACT_PROFILE']='contact_stable_v8'
    os.environ.update(EB_UFO_LEARNING_PROFILE='physical_limits_stable_slew_v4',EB_UFO_CONTROL_PROFILE='reachable_v2',EB_UFO_AUXILIARY_PROFILE='soft_velocity_v7')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=False).app
    env=None
    try:
        import gymnasium as gym
        import torch
        import mujoco
        from rpo_bfm import RPO_LAB_JOINT_ORDER,BODY_ORDER
        from rpo_bfm_extension import RPOBFMEnvCfg
        from robolab.tasks.direct.base.scene_cfg import SceneCfg
        from rpo_ufo_isaac import register_task,write_reference_state
        from rpo_ufo_contact import ContactResetPool
        register_task();cfg=RPOBFMEnvCfg();cfg.seed=64042;cfg.scene.num_envs=32
        if name=='grounded':
            cfg.sim.dt=.005;cfg.decimation=4;cfg.sim.render_interval=4
            cfg.scene=SceneCfg(config=cfg.scene_context,physics_dt=.005,step_dt=.02)
            cfg.scene.terrain.use_terrain_origins=False
            cfg.scene.num_envs=32
        env=gym.make('RPO-UFO-Isaac-RSI-reachable-v7',cfg=cfg).unwrapped
        paired=output/'paired-inputs.npz'
        if name=='baseline':
            pool=ContactResetPool(env.ufo_reference,os.environ['RPO_ASSET_ROOT'])
            records=[pool.sample(32,'cpu') for _ in range(16)]
            arrays={key:np.stack([r[i].numpy() for r in records]) for i,key in enumerate(('root','q','dq','lift'))}
            arrays['root'][:,:,2]-=arrays['lift'];np.savez_compressed(paired,**arrays)
        with np.load(paired) as archive:inputs={k:archive[k] for k in archive.files}
        model=mujoco.MjModel.from_xml_path(str(Path(os.environ['RPO_ASSET_ROOT'])/'rpo_description/mjcf/rpo.xml'));data=mujoco.MjData(model)
        adr=model.jnt_qposadr[[model.joint(n).id for n in RPO_LAB_JOINT_ORDER]]
        free=int(model.jnt_qposadr[model.joint('floating_base_joint').id]);bids=[model.body(n).id for n in BODY_ORDER]
        traces=[];terms=[];active=[];fk_before=[];fk_after=[];contact=[];peakdq=[];boundarydq=[]
        original_apply=env._apply_action
        peaks={}
        def observe_apply():
            if peaks['count']:
                peaks['dq']=torch.maximum(peaks['dq'],env.robot.data.joint_vel.abs().amax(-1))
                force=env.contact_sensor.data.net_forces_w[:,env.ufo_mapping['contact_feet'],2].abs().amax(-1)
                peaks['force']=torch.maximum(peaks['force'],force)
            peaks['count']+=1
            return original_apply()
        env._apply_action=observe_apply
        ids=torch.arange(32,device=env.device);rng=np.random.default_rng(64043)
        for r in range(16):
            env.reset();root=inputs['root'][r].copy()
            if name!='baseline':root[:,2]+=inputs['lift'][r]
            expected=[]
            for i in range(32):
                data.qpos[:]=model.qpos0;data.qpos[free:free+3]=root[i,:3];data.qpos[free+3:free+7]=root[i,3:7];data.qpos[adr]=inputs['q'][r,i]
                mujoco.mj_forward(model,data);expected.append(data.xpos[bids].copy())
            expected=torch.tensor(np.stack(expected),device=env.device)+env.scene.env_origins[:,None,:]
            write_reference_state(env.robot,torch.tensor(root,device=env.device),torch.tensor(inputs['q'][r],device=env.device),torch.tensor(inputs['dq'][r],device=env.device),ids,env.scene.env_origins)
            if name!='baseline':env.ufo_reset_lift[:]=torch.tensor(inputs['lift'][r],device=env.device)
            before=float((env.robot.data.body_pos_w-expected).norm(dim=-1).max())
            fk_before.append(before)
            env._reset_slew_anchor(ids);env.ufo_cache.invalidate()
            if name!='baseline':
                env.contact_sensor.reset(ids);env.scene.write_data_to_sim();env.sim.forward();env.robot.update(0.)
            fk_after.append(float((env.robot.data.body_pos_w-expected).norm(dim=-1).max()))
            alive=np.ones(32,bool);tv=[];dv=[];av=[];cv=[];pv=[];bv=[]
            for step in range(16):
                peaks['count']=0
                peaks['dq']=torch.zeros(32,device=env.device);peaks['force']=torch.zeros(32,device=env.device)
                requested=(env.bfm_last_action.clone() if step<4 else torch.tensor(rng.uniform(-1,1,(32,23)),device=env.device,dtype=torch.float32))
                _,_,done,timeout,_=env.step(requested)
                values=env._get_observations()['ufo_action_trace'].cpu().numpy()
                tv.append(values);dv.append(done.cpu().numpy());av.append(alive.copy())
                force=env.contact_sensor.data.net_forces_w[:,env.ufo_mapping['contact_feet'],2].abs().amax(-1)
                velocity=env.robot.data.joint_vel.abs().amax(-1)
                cv.append(torch.maximum(peaks['force'],force).cpu().numpy())
                pv.append(torch.maximum(peaks['dq'],velocity).cpu().numpy());bv.append(velocity.cpu().numpy())
                alive &= ~done.cpu().numpy()
            traces.append(tv);terms.append(dv);active.append(av);contact.append(cv);peakdq.append(pv);boundarydq.append(bv)
        a=np.array(traces);t=np.array(terms);valid=np.array(active);forces=np.array(contact);dq=np.array(peakdq)
        raw=dict(trace=a,terminated=t,active=valid,foot_force_N=forces,peak_dq=dq,boundary_dq=np.array(boundarydq),fk_before=fk_before,fk_after=fk_after)
        np.savez_compressed(output/f'{name}.npz',**raw)
        receipt={'phase':name,'physics_dt_s':env.physics_dt,'decimation':env.cfg.decimation,
            **measured_summary(raw),
            'feet_controller_stiffness_range':[float(env.robot.actuators['feet'].stiffness.min()),float(env.robot.actuators['feet'].stiffness.max())],
            'feet_controller_damping_range':[float(env.robot.actuators['feet'].damping.min()),float(env.robot.actuators['feet'].damping.max())]}
        (output/f'{name}.json').write_text(json.dumps(receipt,indent=2))
    finally:
        if env is not None:env.close()
        app.close()


def probe(output):
    output.mkdir(parents=True,exist_ok=True)
    for name in PHASES:
        subprocess.run([sys.executable,__file__,'--phase',name,str(output)],check=True)
    files=[output/'paired-inputs.npz',*[output/f'{n}.npz' for n in PHASES]]
    receipt={'protocol':PROTOCOL,'status':'measured','asset_sha256':os.environ['EB_UFO_ASSET_SHA256'],
        'manifest_sha256':hashlib.sha256((Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json').read_bytes()).hexdigest(),
        'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        'phases':[json.loads((output/f'{n}.json').read_text()) for n in PHASES]}
    (output/'contact-probe.json').write_text(json.dumps(receipt,indent=2))


def validate_probe(receipt,validation,output):
    if receipt.get('protocol')!=PROTOCOL or receipt.get('status')!='measured' or any(receipt.get(k)!=validation.get(k) for k in ('asset_sha256','manifest_sha256')):
        raise ValueError('Contact probe frozen input/protocol differs')
    if [x['name'] for x in receipt['files']]!=['paired-inputs.npz',*[f'{n}.npz' for n in PHASES]]:
        raise ValueError('Contact probe file coverage differs')
    for member in receipt['files']:
        p=output/member['name']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=member['bytes'] or p.stat().st_size>32*1024**2 or hashlib.sha256(p.read_bytes()).hexdigest()!=member['sha256']:
            raise ValueError('Contact probe raw evidence invalid')
    if [x['phase'] for x in receipt['phases']]!=list(PHASES):raise ValueError('Contact phase coverage differs')
    for row in receipt['phases']:
        with np.load(output/f"{row['phase']}.npz") as raw:
            width=118 if row['phase']=='baseline' else 128
            if raw['trace'].shape!=(16,16,32,width) or not np.isfinite(raw['trace']).all():raise ValueError('Contact probe trace invalid')
            for key in ('active','terminated'):
                if raw[key].shape!=(16,16,32) or raw[key].dtype!=bool:raise ValueError('Contact probe mask invalid')
            if not raw['active'][:,0].all() or not np.array_equal(raw['active'][:,1:],raw['active'][:,:-1]&~raw['terminated'][:,:-1]):
                raise ValueError('Contact probe active mask invalid')
            for key in ('foot_force_N','peak_dq','boundary_dq'):
                if raw[key].shape!=(16,16,32) or not np.isfinite(raw[key]).all():raise ValueError('Contact physics samples invalid')
            for key in ('fk_before','fk_after'):
                if raw[key].shape!=(16,) or not np.isfinite(raw[key]).all():raise ValueError('Contact FK samples invalid')
            for key,value in measured_summary(raw).items():
                if isinstance(value,list):
                    if row[key]!=value:raise ValueError('Contact probe summary differs')
                elif not np.isclose(row[key],value,rtol=1e-6,atol=1e-6):raise ValueError('Contact probe summary differs')


if __name__=='__main__':
    if len(sys.argv)>1:phase(sys.argv[2],Path(sys.argv[3]))
    else:probe(Path('/outputs'))
