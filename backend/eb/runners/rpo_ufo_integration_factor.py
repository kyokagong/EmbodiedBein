"""Frozen integration sensitivity and continuous braking evidence; no learner."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

OPERATION = 'probe_rpo_ufo_integration_factor_v1'
SETTINGS = {'I0':(8,.0025), 'I1':(16,.00125)}


def source_cases(source, expected_sha256):
    try:
        from eb.runners.rpo_ufo_contact_factor import validate_receipt as contact_validate
        from eb.runners.rpo_ufo_dynamics_diagnostics import digest
    except ImportError:
        from rpo_ufo_contact_factor import validate_receipt as contact_validate
        from rpo_ufo_dynamics_diagnostics import digest
    pointer = source/'dynamics-diagnostics.json'
    if digest(pointer) != expected_sha256:
        raise ValueError('Braking trace source receipt changed')
    receipt = json.loads(pointer.read_text())
    factor = receipt.get('integration_factor','I0')
    if factor not in SETTINGS or receipt.get('contact_factor') != 'B0':
        raise ValueError('Braking source must retain original contact')
    count, dt = SETTINGS[factor]
    contact_validate(receipt,{'manifest_sha256':receipt['dataset_manifest_sha256']},source,
        substeps=count,physics_dt=dt)
    cases = {}
    for phase in receipt['phases']:
        suffix = f"{phase['mode']}-{phase['rank']}"
        events = json.loads((source/f'events-{suffix}.json').read_text())
        ids = sorted({event['env'] for event in events})
        if not 1 <= len(ids) <= 5 or any(type(i) is not int or not 0 <= i < 2048 for i in ids):
            raise ValueError('Braking case selection is missing or unbounded')
        cases[suffix] = ids
    return cases


def install_trace(factor, rank, mode, output, cases, *, record_population=False):
    import torch
    import warp as wp
    import rpo_ufo_mjlab as physics
    from rpo_ufo_contact_factor import install_measurement
    from rpo_ufo_dynamics_diagnostics import write_json

    count, dt = SETTINGS[factor]
    install_measurement('B0',rank,mode,output,substeps=count)
    original = physics.RPOUFOMJLabPhysics

    class TracePhysics(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, substeps=count, physics_dt=dt)
            self._trace_ids = torch.tensor(cases,device=self.device(),dtype=torch.long)
            self._braking_rows = []

        def step(self, actions):
            d = self.lab.sim.wp_data
            self._trace_before_q = wp.to_torch(d.qpos)[self._trace_ids].clone()
            self._trace_before_v = wp.to_torch(d.qvel)[self._trace_ids].clone()
            return super().step(actions)

        def _refresh(self, dt=.0025):
            sample = (hasattr(self,'population') and hasattr(self,'_trace_ids')
                and len(self.population)>getattr(self,'_measured_samples',0))
            if sample:
                data, ids = self.lab.sim.wp_data, self._trace_ids
                model = self.lab.sim.mj_model
                force_ids = slice(None) if record_population else ids
                jac = wp.to_torch(data.efc.J)[force_ids,:,:model.nv]
                force = wp.to_torch(data.efc.force)[force_ids]
                types = wp.to_torch(data.efc.type)[force_ids]
                valid = torch.arange(jac.shape[1],device=self.device())[None,:] < wp.to_torch(data.nefc)[force_ids,None]
                groups = {'contact':(types>=5)&(types<=7), 'limit':(types==3)|(types==4)}
                groups['other'] = ~(groups['contact']|groups['limit'])
                generalized = {k:torch.einsum('civ,ci->cv',jac,force*(valid&mask))[:,self.dof_ids]
                    for k,mask in groups.items()}
                population_generalized = generalized
                if record_population:
                    generalized = {k:v[ids] for k,v in generalized.items()}
                row = {'qpos_before':self._trace_before_q.clone(),'qvel_before':self._trace_before_v.clone(),
                    'qpos_after':wp.to_torch(data.qpos)[ids].clone(),'qvel_after':wp.to_torch(data.qvel)[ids].clone(),
                    'target':self.target[ids].clone(),'requested':self.requested[ids].clone(),'applied':self.last[ids].clone(),
                    'motor_torque':wp.to_torch(data.qfrc_actuator)[ids][:,self.dof_ids].clone(),
                    'contact_torque':generalized['contact'],'limit_torque':generalized['limit'],'other_constraint':generalized['other'],
                    'constraint_torque':wp.to_torch(data.qfrc_constraint)[ids][:,self.dof_ids].clone(),
                    'foot_force':self._forces()[ids][:,self.feet].clone(),
                    'original_alive':self.alive[ids].clone(),'reference':self.reference[ids].clone(),
                    'episode_step':self.steps[ids,0].clone()}
                if record_population:
                    row.update(population_qpos_after=wp.to_torch(data.qpos).clone(),
                        population_qvel_after=wp.to_torch(data.qvel).clone(),
                        population_motor_torque=wp.to_torch(data.qfrc_actuator)[:,self.dof_ids].clone(),
                        population_contact_torque=population_generalized['contact'],
                        population_limit_torque=population_generalized['limit'],
                        population_foot_force=self._forces()[:,self.feet].clone(),
                        population_original_alive=self.alive.clone())
                self._trace_before_q, self._trace_before_v = row['qpos_after'].clone(), row['qvel_after'].clone()
            result = super()._refresh(dt)
            if sample:
                total = int(data.nacon.numpy()[0])
                worlds = wp.to_torch(data.contact.worldid)[:total].long()
                depths = (-wp.to_torch(data.contact.dist)[:total]).clamp_min(0.)
                maximum = torch.zeros(self.num_envs,device=self.device())
                maximum.scatter_reduce_(0,worlds,depths,reduce='amax',include_self=True)
                row['contact_depth_after'] = maximum[ids]
                if record_population:
                    row['population_contact_depth_after'] = maximum
                self._braking_rows.append({k:v.cpu().numpy().copy() for k,v in row.items()})
            return result

        def close(self):
            if hasattr(self,'_braking_rows') and self._braking_rows:
                suffix = f'{mode}-{rank}'
                rows = {k:np.stack([r[k] for r in self._braking_rows]) for k in self._braking_rows[0]}
                rows.update(case_ids=np.asarray(cases),dof_ids=np.asarray(self.dof_ids),pos_ids=np.asarray(self.pos_ids),
                    joint_vel_limits=self.limits['joint_vel'].cpu().numpy(),
                    torque_limits=self.limits['applied_torque'].cpu().numpy(),
                    time_s=np.arange(1,len(self._braking_rows)+1)*SETTINGS[factor][1])
                np.savez_compressed(output/f'braking-{suffix}.npz',**rows)
                write_json(output/f'braking-{suffix}.json',{'factor':factor,'rank':rank,'mode':mode,
                    'substeps':count,'physics_dt_s':SETTINGS[factor][1],'control_dt_s':.02,
                    'samples':len(self._braking_rows),'case_ids':cases,
                    'force_time':'solver forces before forward; contact depth after integrated geometry refresh',
                    'reset_rows':'original_alive=False; exclude from original episode decisions'})
            super().close()

    physics.RPOUFOMJLabPhysics = TracePhysics


def validate_receipt(receipt, validation, output, *, ground_damping_ratio=1.):
    try:
        from eb.runners.rpo_ufo_contact_factor import validate_receipt as contact_validate
    except ImportError:
        from rpo_ufo_contact_factor import validate_receipt as contact_validate
    factor = receipt.get('integration_factor')
    if factor not in SETTINGS or receipt.get('contact_factor') != 'B0':
        raise ValueError('Integration factor identity differs')
    count, dt = SETTINGS[factor]
    contact_validate(receipt,validation,output,substeps=count,physics_dt=dt,ground_damping_ratio=ground_damping_ratio)
    members = {r['file'] for r in receipt['files']}
    if not {'trace-cases.json','startup-models.json'} <= members:
        raise ValueError('Frozen braking case selection missing')
    cases = json.loads((output/'trace-cases.json').read_text())
    models = json.loads((output/'startup-models.json').read_text())
    for phase in receipt['phases']:
        suffix = f"{phase['mode']}-{phase['rank']}"
        model = models[suffix]
        if (model['compiled_solver']['timestep_s']!=dt
                or model['integration_actual']!={'physics_dt_s':dt,'substeps':count,'control_dt_s':.02}
                or model['geom_solref'][0]!=[.02,ground_damping_ratio] or model['physics_dt_s']!=dt or model['decimation']!=count):
            raise ValueError('Compiled integration or original ground differs')
        if not {f'braking-{suffix}.json',f'braking-{suffix}.npz'} <= members:
            raise ValueError('Continuous braking evidence missing')
        metadata = json.loads((output/f'braking-{suffix}.json').read_text())
        ids = cases[suffix]
        if (metadata['factor']!=factor or metadata['physics_dt_s']!=dt or metadata['substeps']!=count
                or metadata['control_dt_s']!=.02 or metadata['samples']!=48*count or metadata['case_ids']!=ids
                or not 1<=len(ids)<=5 or len(set(ids))!=len(ids) or any(type(i)is not int or not 0<=i<2048 for i in ids)):
            raise ValueError('Braking case or temporal identity differs')
        with np.load(output/f'braking-{suffix}.npz',allow_pickle=False) as arrays:
            n, c = 48*count, len(ids)
            shapes = {'qpos_before':(n,c,30),'qpos_after':(n,c,30),'qvel_before':(n,c,29),'qvel_after':(n,c,29),
                **{k:(n,c,23) for k in ('target','requested','applied','motor_torque','contact_torque','limit_torque','other_constraint','constraint_torque')},
                'foot_force':(n,c,2,3),'original_alive':(n,c),'reference':(n,c,3),'episode_step':(n,c),'contact_depth_after':(n,c),
                'case_ids':(c,),'dof_ids':(23,),'pos_ids':(23,),'joint_vel_limits':(23,),'torque_limits':(23,),'time_s':(n,)}
            if (not shapes.keys()<=set(arrays.files) or any(arrays[k].shape!=shape for k,shape in shapes.items())
                    or any(not np.isfinite(arrays[k]).all() for k in arrays.files)
                    or arrays['original_alive'].dtype!=bool or not arrays['original_alive'][0].all()
                    or (np.diff(arrays['original_alive'].astype(int),axis=0)>0).any()
                    or not np.array_equal(arrays['case_ids'],ids)
                    or not np.allclose(arrays['time_s'],np.arange(1,n+1)*dt,atol=1e-10,rtol=0)
                    or not np.allclose(arrays['contact_torque']+arrays['limit_torque']+arrays['other_constraint'],
                        arrays['constraint_torque'],atol=1e-3,rtol=1e-5)):
                raise ValueError('Continuous braking arrays differ')
            alive = arrays['original_alive']
            continuous = alive[1:] & alive[:-1]
            if (not np.array_equal(arrays['reference'][alive],np.broadcast_to(arrays['reference'][0],(n,c,3))[alive])
                    or not np.allclose(arrays['qvel_before'][1:][continuous],arrays['qvel_after'][:-1][continuous],atol=1e-6,rtol=1e-6)
                    or not np.allclose(arrays['qpos_before'][1:][continuous],arrays['qpos_after'][:-1][continuous],atol=1e-6,rtol=1e-6)):
                raise ValueError('Original episode reference or integrated continuity differs')
            with np.load(output/f'rsi-{suffix}.npz',allow_pickle=False) as initial:
                if not np.array_equal(arrays['reference'][0],initial['reference'][ids]):
                    raise ValueError('Braking reference differs from original RSI')


def main():
    from rpo_ufo_dynamics_diagnostics import phase,digest,write_json,SCHEMA
    factor = os.environ['EB_UFO_INTEGRATION_FACTOR']
    if factor not in SETTINGS:
        raise ValueError('Unknown integration factor')
    output = Path('/outputs'); output.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)==3:
        rank,mode = int(sys.argv[1]),sys.argv[2]
        if rank not in range(3) or mode not in ('anchor','policy'):
            raise ValueError('Unknown integration phase')
        import rpo_ufo_learning_diagnostics as learning
        admitted = learning.visible_devices()
        os.environ['CUDA_VISIBLE_DEVICES'] = admitted[rank+1]
        learning.visible_devices = lambda:admitted
        cases = json.loads((output/'trace-cases.json').read_text())[f'{mode}-{rank}']
        install_trace(factor,rank,mode,output,cases)
        write_json(output/f'phase-{mode}-{rank}.json',phase(rank,mode,output)); return
    if sys.argv[1:]:
        raise ValueError('Unbounded integration arguments')
    write_json(output/'trace-cases.json',source_cases(Path('/workspace/trace-source'),os.environ['EB_UFO_TRACE_SOURCE_SHA256']))
    for mode in ('anchor','policy'):
        children = [subprocess.Popen([sys.executable,__file__,str(r),mode]) for r in range(3)]
        results = [c.wait() for c in children]
        if any(results):
            raise RuntimeError(f'Integration {factor}/{mode} failed: {results}')
    models = {}
    for mode in ('anchor','policy'):
        for rank in range(3):
            suffix = f'{mode}-{rank}'
            files = list((output/f'{suffix}-startup').glob('physics-startup-*.json'))
            if len(files)!=1:
                raise ValueError('Compiled model startup evidence missing')
            models[suffix] = json.loads(files[0].read_text())
    write_json(output/'startup-models.json',models)
    receipt = {'schema_version':SCHEMA,'qualification':'diagnostics_only','baseline_step':600,'optimizer_updates':0,
        'contact_factor':'B0','integration_factor':factor,'trace_source_run_id':os.environ['EB_UFO_TRACE_SOURCE_RUN_ID'],
        'trace_source_receipt_sha256':os.environ['EB_UFO_TRACE_SOURCE_SHA256'],
        'phases':[json.loads((output/f'phase-{m}-{r}.json').read_text()) for m in ('anchor','policy') for r in range(3)],
        'dataset_manifest_sha256':digest(Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json'),
        'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    validate_receipt(receipt,{'manifest_sha256':receipt['dataset_manifest_sha256']},output)
    write_json(output/'dynamics-diagnostics.json',receipt)


if __name__=='__main__':
    main()
