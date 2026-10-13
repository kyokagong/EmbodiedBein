"""Bounded same-run action and substep evidence; replay never invokes the actor."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

OPERATION = 'probe_rpo_ufo_action_replay_v1'


def validate_program(arrays, *, policy):
    shapes = {'requested':(48,2048,23),'applied':(48,2048,23),'target':(48,2048,23),
        'states':(48,2048,52),'history_actor':(48,2048,300),'reference':(48,2048,3),
        'episode':(48,2048),'step':(48,2048),'done':(48,2048),'original_alive':(48,2048),
        'interval':(48,2048,223),'initial_qpos':(2048,30),'initial_qvel':(2048,29)}
    if policy:
        shapes.update(z=(48,2048,256),context=(48,2048,7))
    if (not shapes.keys() <= set(arrays) or any(arrays[k].shape!=s for k,s in shapes.items())
            or any(not np.isfinite(arrays[k]).all() for k in arrays)
            or arrays['done'].dtype!=bool or arrays['original_alive'].dtype!=bool
            or not arrays['original_alive'][0].all()
            or not np.array_equal(arrays['original_alive'][1:],
                arrays['original_alive'][:-1]&~arrays['done'][:-1])):
        raise ValueError('Action program incomplete, nonfinite or episode identity differs')


def check_initial(program, qpos, qvel):
    if (not np.allclose(program['initial_qpos'],qpos,atol=1e-6,rtol=0)
            or not np.allclose(program['initial_qvel'],qvel,atol=1e-6,rtol=0)):
        raise ValueError('Action replay initial full physical state differs')


def validate_population(arrays, samples):
    shapes = {'qpos_after':(30,),'qvel_after':(29,),'motor_torque':(23,),
        'contact_torque':(23,),'limit_torque':(23,),'foot_force':(2,3),
        'original_alive':(),'contact_depth_after':()}
    ids=arrays['case_ids']
    for key,tail in shapes.items():
        name='population_'+key
        if (name not in arrays or arrays[name].shape!=(samples,2048,*tail)
                or not np.isfinite(arrays[name]).all()
                or not np.array_equal(arrays[key],arrays[name][:,ids])):
            raise ValueError('Same-run population history differs from selected braking trace')
    alive=arrays['population_original_alive']
    if (alive.dtype!=bool or not alive[0].all() or (np.diff(alive.astype(int),axis=0)>0).any()
            or (arrays['population_contact_depth_after']<0).any()):
        raise ValueError('Same-run population episode or depth identity differs')


def rng_hashes(policy):
    import torch
    def sha(value):
        if torch.is_tensor(value):
            value=value.cpu().numpy().tobytes()
        else:
            value=repr(value).encode()
        return hashlib.sha256(value).hexdigest()
    result={'torch':sha(torch.get_rng_state()),'cuda':sha(torch.cuda.get_rng_state()),
        'numpy':sha(np.random.get_state()),'python':sha(random.getstate())}
    if policy is not None:
        result.update({k:sha(getattr(policy,k)) for k in ('rng_torch','rng_cuda','rng_numpy','rng_python')})
    return result


def context_snapshot(policy, steps):
    import torch
    if policy.context_profile:
        ctx=policy.rollout_context
        return torch.cat((ctx.mask[:,None].float(),(ctx.last_steps-ctx.starts)[:,None].float(),
            ctx.references.float(),ctx.encoding_versions[:,None].float(),ctx.starts[:,None].float()),-1).cpu().numpy(),None
    ids=policy.native.env_idx_with_expert_rollout
    ids=torch.empty(0,dtype=torch.long,device=steps.device) if ids is None else ids
    counts=torch.bincount(ids,minlength=len(steps))
    # Legacy tracking samples discard source IDs; preserve unknowns, not RSI substitutes.
    result=torch.full((len(steps),7),-1.,device=steps.device)
    result[:,0]=counts;result[:,1]=steps%250;result[:,5]=policy.version
    return result.cpu().numpy(),ids.cpu().numpy()


class ControlRecorder:
    def __init__(self, output, suffix, program=None):
        self.output,self.suffix,self.program=output,suffix,program
        self.rows,self.rng=[],[]

    def begin(self, env):
        import warp as wp
        d=env.lab.sim.wp_data
        self.initial={'initial_qpos':wp.to_torch(d.qpos).cpu().numpy().copy(),
            'initial_qvel':wp.to_torch(d.qvel).cpu().numpy().copy()}
        opt=env.lab.sim.mj_model.opt
        self.solver_options={'cone':int(opt.cone),'impratio':float(opt.impratio),
            'tolerance':float(opt.tolerance),'ls_iterations':int(opt.ls_iterations),'ls_tolerance':float(opt.ls_tolerance)}
        self.contact_options={'geom_margin':env.lab.sim.mj_model.geom_margin.tolist(),
                              'geom_gap':env.lab.sim.mj_model.geom_gap.tolist()}
        if self.program is not None:
            check_initial(self.program,self.initial['initial_qpos'],self.initial['initial_qvel'])

    def before(self, env, policy):
        obs=env.get_observations()
        self.row={k:obs[k].cpu().numpy().copy() for k in ('states','history_actor')}
        self.row.update(reference=env.reference.cpu().numpy().copy(),
            episode=env.episode[:,0].cpu().numpy().copy(),step=env.steps[:,0].cpu().numpy().copy(),
            original_alive=env.alive.cpu().numpy().copy())
        self.rng.append({'before':rng_hashes(policy)})

    def after(self, env, policy, done):
        self.row.update({k:v.cpu().numpy().copy() for k,v in
            {'requested':env.requested,'applied':env.last,'target':env.target,'done':done,
             'interval':env.get_observations()['ufo_interval_trace']}.items()})
        if policy is not None:
            self.row['z']=policy.z.cpu().numpy().copy()
            self.row['context'],ids=context_snapshot(policy,env.steps[:,0]-1)
            if ids is not None:
                self.row['legacy_tracking_ids']=ids.copy()
            self.context_profile=policy.context_profile or 'legacy_global'
        self.rng[-1]['after']=rng_hashes(policy)
        self.rows.append(self.row)

    def close(self):
        if len(self.rows)!=48:
            return
        from rpo_ufo_dynamics_diagnostics import write_json,digest
        arrays={**self.initial,**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]}}
        validate_program(arrays,policy=self.suffix.startswith('policy') and self.program is None)
        file=self.output/f'actions-{self.suffix}.npz'
        np.savez_compressed(file,**arrays)
        source=json.loads(Path('/opt/action-replay-source.json').read_text())
        write_json(self.output/f'actions-{self.suffix}.json',{'schema_version':'eb/rpo-action-program/v1',
            'stage':'capture' if self.program is None else 'replay','samples':48,'num_envs':2048,
            'control_dt_s':.02,'sha256':digest(file),'rng':self.rng,'source':source,
            'actual_solver_options':self.solver_options,
            'actual_contact_options':self.contact_options,
            'context_profile':getattr(self,'context_profile',None),
            'context_columns':['membership_count_or_mask','phase','source_or_unknown','clip_or_unknown',
                'frame_or_unknown','encoding_version','starts_or_unknown'],
            'action_mode':'train_stochastic' if self.suffix.startswith('policy') and self.program is None else 'fixed_actions',
            'actor_invocations':48 if self.suffix.startswith('policy') and self.program is None else 0})


def validate_receipt(receipt, validation, output, *, ground_damping_ratio=1.):
    try:
        from eb.runners.rpo_ufo_integration_factor import validate_receipt as integration_validate,SETTINGS
    except ImportError:
        from rpo_ufo_integration_factor import validate_receipt as integration_validate,SETTINGS
    integration_validate(receipt,validation,output,ground_damping_ratio=ground_damping_ratio)
    stage=receipt.get('action_program_stage')
    if stage not in ('capture','replay') or (stage=='capture' and receipt['integration_factor']!='I0'):
        raise ValueError('Frozen action stage differs')
    count,_=SETTINGS[receipt['integration_factor']]
    members={r['file'] for r in receipt['files']}
    for phase in receipt['phases']:
        suffix=f"{phase['mode']}-{phase['rank']}"
        if not {f'actions-{suffix}.npz',f'actions-{suffix}.json'}<=members:
            raise ValueError('Action program evidence missing')
        metadata=json.loads((output/f'actions-{suffix}.json').read_text())
        expected_calls=48 if phase['mode']=='policy' and stage=='capture' else 0
        row=next(r for r in receipt['files'] if r['file']==f'actions-{suffix}.npz')
        if (metadata['stage']!=stage or metadata['samples']!=48 or metadata['num_envs']!=2048
                or metadata['control_dt_s']!=.02 or metadata['sha256']!=row['sha256']
                or len(metadata['rng'])!=48 or metadata['actor_invocations']!=expected_calls
                or metadata['source']['hashes']['bfm_ufo_policy.py']!=
                    'f884aeae9768d40c447e6b2aae16b2a8aa62cf727b70b285c931f48971ed09e2'):
            raise ValueError('Action program temporal or frozen policy identity differs')
        with np.load(output/f'actions-{suffix}.npz',allow_pickle=False) as a:
            validate_program(a,policy=phase['mode']=='policy' and stage=='capture')
            if phase['mode']=='policy' and stage=='capture':
                if (metadata.get('context_profile')!='legacy_global' or metadata.get('action_mode')!='train_stochastic'
                        or 'legacy_tracking_ids' not in a or a['legacy_tracking_ids'].shape!=(48,1024)
                        or a['legacy_tracking_ids'].dtype.kind not in 'iu'
                        or (a['legacy_tracking_ids']<0).any() or (a['legacy_tracking_ids']>=2048).any()
                        or not np.array_equal(a['context'][:,:,1],a['step']%250)
                        or (a['context'][:,:,2:5]!=-1).any() or (a['context'][:,:,6]!=-1).any()):
                    raise ValueError('Actual legacy context identity differs')
                for step,ids in enumerate(a['legacy_tracking_ids']):
                    if not np.array_equal(a['context'][step,:,0],np.bincount(ids,minlength=2048)):
                        raise ValueError('Legacy tracking duplicate membership differs')
            with np.load(output/f'rsi-{suffix}.npz',allow_pickle=False) as rsi:
                if not np.array_equal(a['reference'][0],rsi['reference']) or not np.array_equal(a['states'][0],rsi['online_states']):
                    raise ValueError('Action program initial RSI differs')
            with np.load(output/f'braking-{suffix}.npz',allow_pickle=False) as b:
                validate_population(b,48*count)
                if not np.array_equal(b['population_original_alive'],np.repeat(a['original_alive'],count,axis=0)):
                    raise ValueError('Control and population episode identity differs')


def main():
    from rpo_ufo_dynamics_diagnostics import phase,digest,write_json,SCHEMA
    from rpo_ufo_integration_factor import SETTINGS,source_cases,install_trace
    factor,stage=os.environ['EB_UFO_INTEGRATION_FACTOR'],os.environ['EB_UFO_ACTION_STAGE']
    if factor not in SETTINGS or stage not in ('capture','replay') or (stage=='capture' and factor!='I0'):
        raise ValueError('Unapproved action replay stage/factor')
    solver_factor=os.environ.get('EB_UFO_SOLVER_FACTOR')
    if solver_factor is not None and (solver_factor!='S1' or factor!='I1' or stage!='replay'):
        raise ValueError('Unapproved solver cone stage/factor')
    control_factor=os.environ.get('EB_UFO_CONTROL_FACTOR')
    if control_factor is not None and (control_factor!='D1' or solver_factor or factor!='I1' or stage!='replay'):
        raise ValueError('Unapproved control damping stage/factor')
    friction_factor=os.environ.get('EB_UFO_FRICTION_FACTOR')
    if friction_factor is not None and (friction_factor!='S2' or solver_factor!='S1' or control_factor or factor!='I1' or stage!='replay'):
        raise ValueError('Unapproved friction impedance stage/factor')
    ground_factor=os.environ.get('EB_UFO_GROUND_STIFFNESS_FACTOR')
    if ground_factor is not None and (ground_factor!='C2' or friction_factor!='S2'):
        raise ValueError('Unapproved ground stiffness stage/factor')
    margin_factor=os.environ.get('EB_UFO_CONTACT_MARGIN_FACTOR')
    if margin_factor is not None and (margin_factor!='A1' or ground_factor!='C2'):
        raise ValueError('Unapproved contact margin stage/factor')
    output=Path('/outputs'); output.mkdir(parents=True,exist_ok=True)
    source=Path('/workspace/trace-source')
    if len(sys.argv)==3:
        rank,mode=int(sys.argv[1]),sys.argv[2]
        if rank not in range(3) or mode not in ('anchor','policy'):
            raise ValueError('Unbounded action replay phase')
        import rpo_ufo_learning_diagnostics as learning
        admitted=learning.visible_devices(); os.environ['CUDA_VISIBLE_DEVICES']=admitted[rank+1]
        learning.visible_devices=lambda:admitted
        suffix=f'{mode}-{rank}'
        cases=json.loads((output/'trace-cases.json').read_text())[suffix]
        program=None
        if stage=='replay':
            with np.load(source/f'actions-{suffix}.npz',allow_pickle=False) as a:
                program={k:a[k] for k in ('requested','initial_qpos','initial_qvel')}
        if solver_factor:
            from rpo_ufo_solver_cone import install
            install()
        if control_factor:
            from rpo_ufo_control_damping import install
            install()
        if friction_factor:
            from rpo_ufo_friction_impedance import install
            install()
        install_trace(factor,rank,mode,output,cases,record_population=True)
        if ground_factor:
            from rpo_ufo_ground_stiffness import install
            install()
        if margin_factor:
            from rpo_ufo_contact_margin import install
            install()
        observer=ControlRecorder(output,suffix,program)
        write_json(output/f'phase-{suffix}.json',phase(rank,mode,output,observer=observer,
            action_program=None if program is None else program['requested'])); return
    if sys.argv[1:]:
        raise ValueError('Unbounded action replay arguments')
    cases=source_cases(source,os.environ['EB_UFO_TRACE_SOURCE_SHA256'])
    if stage=='replay':
        receipt=json.loads((source/'dynamics-diagnostics.json').read_text())
        if receipt.get('action_program_stage')!='capture' or receipt['integration_factor']!='I0':
            raise ValueError('Replay requires an I0 captured action program')
        validate_receipt(receipt,{'manifest_sha256':receipt['dataset_manifest_sha256']},source)
    write_json(output/'trace-cases.json',cases)
    for mode in ('anchor','policy'):
        children=[subprocess.Popen([sys.executable,__file__,str(r),mode]) for r in range(3)]
        results=[c.wait() for c in children]
        if any(results):
            raise RuntimeError(f'Action {stage}/{factor}/{mode} failed: {results}')
    models={}
    for mode in ('anchor','policy'):
        for rank in range(3):
            suffix=f'{mode}-{rank}'; files=list((output/f'{suffix}-startup').glob('physics-startup-*.json'))
            if len(files)!=1: raise ValueError('Compiled startup evidence missing')
            models[suffix]=json.loads(files[0].read_text())
    write_json(output/'startup-models.json',models)
    receipt={'schema_version':SCHEMA,'qualification':'diagnostics_only','baseline_step':600,'optimizer_updates':0,
        **({'solver_cone_factor':solver_factor} if solver_factor else {}),
        **({'control_damping_factor':control_factor} if control_factor else {}),
        **({'solver_friction_factor':friction_factor} if friction_factor else {}),
        **({'ground_stiffness_factor':ground_factor} if ground_factor else {}),
        **({'contact_margin_factor':margin_factor} if margin_factor else {}),
        'contact_factor':'B0','integration_factor':factor,'action_program_stage':stage,
        'trace_source_run_id':os.environ['EB_UFO_TRACE_SOURCE_RUN_ID'],
        'trace_source_receipt_sha256':os.environ['EB_UFO_TRACE_SOURCE_SHA256'],
        'phases':[json.loads((output/f'phase-{m}-{r}.json').read_text()) for m in ('anchor','policy') for r in range(3)],
        'dataset_manifest_sha256':digest(Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json'),
        'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    check=validate_receipt
    if solver_factor:
        from rpo_ufo_solver_cone import validate_receipt as check
    if control_factor:
        from rpo_ufo_control_damping import validate_receipt as check
    if friction_factor:
        from rpo_ufo_friction_impedance import validate_receipt as check
    if ground_factor:
        from rpo_ufo_ground_stiffness import validate_receipt as check
    if margin_factor:
        from rpo_ufo_contact_margin import validate_receipt as check
    check(receipt,{'manifest_sha256':receipt['dataset_manifest_sha256']},output)
    write_json(output/'dynamics-diagnostics.json',receipt)


if __name__=='__main__':
    main()
