"""Paired contact/PD diagnostic on three 2048-env cards; no policy evaluation."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PHASES=('baseline','compliant','compliant_kd30')
PROTOCOL={'schema_version':'eb/ufo-contact-compliance-probe/v1','phases':list(PHASES),
    'samplers':3,'envs_per_sampler':2048,'steps':32,'detail_envs':256,
    'seeds':[42,43,44],'split':'train','qualification':'not_requested',
    'requested_action':'anchor + ankle .25sin(2Hz), hip/knee .12sin(2Hz), clipped to [-1,1]',
    'pairing':'same RNG seeds, flexion-filtered RSI references and requested actions; fresh simulator each phase/rank'}


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def phase(name,rank,output):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    devices=visible.split(',') if visible else ['0','1','2','3']
    os.environ['CUDA_VISIBLE_DEVICES']=devices[rank+1]
    os.environ.update(EB_UFO_LEARNING_PROFILE='physical_limits_stable_slew_v4',
        EB_UFO_CONTROL_PROFILE='reachable_v2',EB_UFO_AUXILIARY_PROFILE='soft_velocity_v7',
        EB_UFO_CONTACT_PROFILE='contact_stable_v8',EB_UFO_GROUND_PROFILE='wide_plane_v9',
        EB_UFO_CONTEXT_PROFILE='environment_tracking_v10',
        EB_UFO_DYNAMICS_PROFILE='contact_compliance_v11',
        EB_UFO_CONTACT_MATERIAL='rigid' if name=='baseline' else 'compliant',
        EB_UFO_ANKLE_CONTROL='kd30' if name=='compliant_kd30' else 'kd25',
        EB_UFO_DIAGNOSTICS_DIR=str(output/f'{name}-rank{rank}-pools'))
    sys.path.insert(0,str(Path(os.environ['RPO_ASSET_ROOT'])/'robolab'))
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=False).app
    env=None
    try:
        import random
        import numpy as np
        import torch
        import gymnasium as gym
        from rpo_bfm import RPO_LAB_JOINT_ORDER,SLEW_CONTROL
        from rpo_ufo_compliance import material_receipt
        from rpo_bfm_extension import RPOBFMEnvCfg
        from rpo_ufo_isaac import register_task,TRACKING_GYM_ID,INTERVAL_TRACE_WIDTH
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
        register_task();cfg=RPOBFMEnvCfg();cfg.seed=42+rank;cfg.scene.num_envs=2048
        env=gym.make(TRACKING_GYM_ID,cfg=cfg).unwrapped
        obs,_=env.reset()
        initial_reference=obs['ufo_reset_reference'].cpu().numpy()
        gains={'stiffness':env.robot.data.joint_stiffness[0].cpu().tolist(),
               'damping':env.robot.data.joint_damping[0].cpu().tolist(),
               'joint_names':list(RPO_LAB_JOINT_ORDER),
               'actuators':{key:{'stiffness':actuator.stiffness[0].cpu().tolist(),
                                'damping':actuator.damping[0].cpu().tolist(),
                                'effort_limit':actuator.effort_limit[0].cpu().tolist()}
                            for key,actuator in env.robot.actuators.items()},
               'mass_kg':env.robot.data.default_mass[0].sum().item(),
               'actual_mass_range_kg':[float(v) for v in (env.robot.root_physx_view.get_masses().sum(-1).min(),env.robot.root_physx_view.get_masses().sum(-1).max())]}
        anchor=env.bfm_last_action.clone()
        ankles=[i for i,n in enumerate(RPO_LAB_JOINT_ORDER) if 'ankle' in n]
        detail=np.random.default_rng(1000).choice(2048,256,replace=False)
        alive=np.ones(2048,bool);initial_terminated=0;exposure=0.;rows=0
        velocity=[];force=[];traces=[];intervals=[];active=[]
        for step in range(32):
            requested=anchor.clone()
            requested[:,ankles]+=.25*math.sin(2*math.pi*2*step*.02)
            requested[:,[7,8,11,12]]+=.12*math.sin(2*math.pi*2*step*.02)
            requested.clamp_(-1,1)
            _,_,terminated,truncated,_=env.step(requested)
            current=env._get_observations()
            interval=current['ufo_interval_trace'].cpu().numpy()
            trace=current['ufo_action_trace'].cpu().numpy()
            if interval.shape!=(2048,INTERVAL_TRACE_WIDTH) or not np.isfinite(interval).all() or (interval[:,164]!=8).any():
                raise ValueError('Incomplete paired physical interval')
            done=(terminated|truncated).cpu().numpy().reshape(-1)
            if step==0:initial_terminated=int(done.sum())
            velocity.append((interval[alive,:23]/np.array(SLEW_CONTROL['joint_velocity_limits_rad_s'])).max(-1))
            force.append(trace[alive,118:120].max(-1))
            exposure+=float(interval[alive,138:161].sum());rows+=int(alive.sum())
            traces.append(trace[detail]);intervals.append(interval[detail]);active.append(alive[detail].copy())
            alive &= ~done
            if done.any():
                env.reset(env_ids=torch.as_tensor(np.flatnonzero(done),device=env.device))
        file=output/f'{name}-rank{rank}.npz'
        np.savez_compressed(file,trace=np.stack(traces),interval=np.stack(intervals),
            active=np.stack(active),detail_env_ids=detail,initial_reference=initial_reference)
        speeds=np.concatenate(velocity);forces=np.concatenate(force)
        result={'phase':name,'rank':rank,'seed':42+rank,'envs':2048,'rows':rows,
            'first_terminated':initial_terminated,'first_samples':2048,
            'velocity_exposure_fraction':exposure/(rows*23),
            'normalized_velocity_p99':float(np.quantile(speeds,.99)),
            'normalized_velocity_max':float(speeds.max()),
            'foot_vertical_net_force_p99_N':float(np.quantile(forces,.99)),
            'foot_vertical_net_force_max_N':float(forces.max()),
            'initial_reference_sha256':hashlib.sha256(initial_reference.tobytes()).hexdigest(),
            'file':file.name,'bytes':file.stat().st_size,'sha256':digest(file),
            'ground_coverage':env.ufo_ground_coverage,
            'gains':gains,'contact_material':material_receipt(env.sim.stage,compliant=name!='baseline'),
            'pool':env.ufo_reset_pool.rows}
        (output/f'{name}-rank{rank}.json').write_text(json.dumps(result,indent=2))
    finally:
        if env is not None:env.close()
        app.close()


def validate_receipt(receipt,validation,root):
    if receipt.get('protocol')!=PROTOCOL or receipt.get('status')!='passed':
        raise ValueError('Tracking/contact diagnostic protocol differs')
    if receipt.get('manifest_sha256')!=validation['manifest_sha256'] or receipt.get('asset_sha256')!=validation['asset_sha256']:
        raise ValueError('Tracking/contact diagnostic inputs differ')
    cases=receipt.get('cases',[])
    if len(cases)!=9 or {(c['phase'],c['rank']) for c in cases}!={(p,r) for p in PHASES for r in range(3)}:
        raise ValueError('Tracking/contact diagnostic lacks paired ranks')
    for row in cases:
        path=root/row['file']
        if (row['file']!=f"{row['phase']}-rank{row['rank']}.npz" or path.is_symlink()
                or not path.is_file() or path.stat().st_size!=row['bytes'] or not 0<row['bytes']<128*1024**2):
            raise ValueError('Tracking/contact diagnostic evidence unavailable')
        if row['envs']!=2048 or row['seed']!=42+row['rank'] or row['first_samples']!=2048:
            raise ValueError('Tracking/contact diagnostic topology differs')
        if digest(path)!=row.get('sha256'):
            raise ValueError('Compliant contact diagnostic trace hash differs')
    for rank in range(3):
        if len({row['initial_reference_sha256'] for row in cases if row['rank']==rank})!=1:
            raise ValueError('Paired diagnostic RSI references changed')


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    for name in PHASES:
        processes=[subprocess.Popen([sys.executable,__file__,'--phase',name,'--rank',str(rank),
                                    '--output',str(output)]) for rank in range(3)]
        codes=[process.wait() for process in processes]
        if any(codes):raise RuntimeError(f'Physical diagnostic phase {name} failed: {codes}')
    cases=[json.loads((output/f'{name}-rank{rank}.json').read_text()) for name in PHASES for rank in range(3)]
    result={'protocol':PROTOCOL,'status':'passed','cases':cases,
        'manifest_sha256':digest(Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json'),
        'asset_sha256':os.environ['EB_UFO_ASSET_SHA256']}
    for row in cases:
        if digest(output/row['file'])!=row['sha256']:raise ValueError('Diagnostic trace changed')
    (output/'contact-compliance-probe.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=PHASES)
    parser.add_argument('--rank',type=int,choices=range(3));parser.add_argument('--output',type=Path,default=Path('/outputs'))
    args=parser.parse_args()
    if args.phase:phase(args.phase,args.rank,args.output)
    else:main(args.output)
