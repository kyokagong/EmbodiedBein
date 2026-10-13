"""Bounded v13 paired physics and reward audit, without optimizer updates."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

OPERATION = 'probe_rpo_ufo_learning_diagnostics_v4'
SCHEMA = 'eb/rpo-learning-diagnostics/v1'


def visible_devices():
    # ebd freezes all four cards via Docker devices, without setting this variable.
    result=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if len(result)!=4 or any(not value for value in result):
        raise ValueError('Learning diagnostic requires four admitted GPU devices')
    return result


class CompletedFrames:
    def __init__(self):
        self.frames = []

    def append(self, index, fields):
        if index != len(self.frames) or not 0 <= index < 8:
            raise ValueError('Completed substeps must be recorded exactly once in order')
        if not fields or any(not np.isfinite(v).all() for v in fields.values()):
            raise ValueError('Invalid completed frame')
        self.frames.append({k: np.asarray(v).copy() for k, v in fields.items()})

    def finish(self):
        if len(self.frames) != 8:
            raise ValueError('Incomplete completed interval')
        return {k: np.stack([f[k] for f in self.frames]) for k in self.frames[0]}


def position_audit(frames, limits):
    frames, limits = np.asarray(frames), np.asarray(limits)
    if (frames.ndim != 3 or frames.shape[0] != 8 or limits.shape != (frames.shape[-1], 2)
            or not np.isfinite(frames).all() or not np.isfinite(limits).all()
            or (limits[:, 0] >= limits[:, 1]).any()):
        raise ValueError('Invalid position audit shape or limits')
    cost = (np.maximum(limits[:, 0] - frames, 0) + np.maximum(frames - limits[:, 1], 0)).sum(-1)
    return {'endpoint': cost[-1], 'mean': cost.mean(0), 'peak': cost.max(0),
            'transient_missed': (cost.max(0) > 0) & (cost[-1] == 0)}


def reward_audit(raw, weights):
    raw, weights = np.asarray(raw), np.asarray(weights)
    if (raw.ndim != 2 or weights.shape != (raw.shape[-1],)
            or not np.isfinite(raw).all() or not np.isfinite(weights).all() or (raw < 0).any()):
        raise ValueError('Invalid raw physical costs or weights')
    return {'raw_mean': raw.mean(0), 'weighted_mean': (raw * weights).mean(0),
            'weighted_sum': raw @ weights,
            'activation_rate': (raw > 0).mean(0),
            'zero_weight_columns': np.flatnonzero(weights == 0).tolist()}


def frozen_aux_reward(normalizer, value):
    import torch
    if (normalizer.counter<=0).any():
        raise ValueError('Frozen reward EMA has no observations')
    correction=1-normalizer.tau**normalizer.counter
    mean=normalizer.mean/correction
    variance=(normalizer.mean_square/correction-mean.square()).clamp_min(normalizer.epsilon)
    return (value-(mean if normalizer.translate else 0))/(torch.sqrt(variance) if normalizer.scale else 1)


def actor_objective_audit(native, obs, z):
    import torch
    model=native._model
    params=tuple(model._actor.parameters())
    previous=[p.requires_grad for p in params]
    device=params[0].device
    try:
        for p in params:p.requires_grad_(True)
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(20261013)
            distribution=model._actor(obs,z,model.cfg.actor_std)
            action=distribution.sample(clip=native.cfg.train.stddev_clip)
            penalty=native.cfg.train.actor_pessimism_penalty
            qd=native.get_targets_uncertainty(model._critic(obs,z,action),penalty)[2]
            qa=native.get_targets_uncertainty(model._aux_critic(obs,z,action),penalty)[2]
            qfb=native.get_targets_uncertainty((model._forward_map(obs,z,action)*z).sum(-1),penalty)[2]
            w=(qfb.abs().mean().detach() if native.cfg.train.scale_reg else qfb.new_tensor(1.)).clamp(max=50.)
            terms={'FB':-qfb.mean(),'D':-qd.mean()*native.cfg.train.reg_coeff*w,
                   'aux':-qa.mean()*native.cfg.train.reg_coeff_aux*w}
            gradients={name:torch.autograd.grad(value,params,retain_graph=True) for name,value in terms.items()}
            norms={name:torch.sqrt(sum(g.square().sum() for g in values)) for name,values in gradients.items()}
            cosine={f'{a}/{b}':float(sum((x*y).sum() for x,y in zip(gradients[a],gradients[b]))/
                (norms[a]*norms[b]).clamp_min(1e-20)) for a,b in (('FB','D'),('FB','aux'),('D','aux'))}
            return {'Q_mean':{'FB':float(qfb.mean().detach()),'D':float(qd.mean().detach()),'aux':float(qa.mean().detach())},
                'actor_loss_contributions':{n:float(v.detach()) for n,v in terms.items()},
                'actor_gradient_norms':{n:float(v) for n,v in norms.items()},
                'actor_gradient_cosines':cosine,'actor_reg_weight':float(w)}
    finally:
        for p,flag in zip(params,previous):p.requires_grad_(flag)


class EventWindows:
    def __init__(self, capacity=32, radius=8):
        if not 1 <= capacity <= 32 or not 0 <= radius <= 8:
            raise ValueError('Unbounded diagnostic events')
        self.capacity, self.radius = capacity, radius
        self.history = deque(maxlen=radius)
        self.events, self.counts = [], {}

    @staticmethod
    def row(step, env, fields):
        return {'control_step': step, 'fields': {k: v[:, env].copy() for k, v in fields.items()}}

    def append(self, step, episodes, fields, selected):
        for event in self.events:
            if step <= event['event_step'] or step > event['event_step'] + self.radius:
                continue
            env = event['env']
            if int(episodes[env]) != event['episode']:
                event['post_reset_truncated'] = True
            elif not event['post_reset_truncated']:
                event['frames'].append(self.row(step, env, fields))
        for category, ids in selected.items():
            for env in ids[:max(0, self.capacity-self.counts.get(category, 0))]:
                episode = int(episodes[env])
                before = [self.row(s, env, f) for s, e, f in self.history if int(e[env]) == episode]
                self.events.append({'category': category, 'env': int(env), 'episode': episode,
                    'event_step': step, 'post_reset_truncated': False,
                    'frames': [*before, self.row(step, env, fields)]})
                self.counts[category] = self.counts.get(category, 0) + 1
        self.history.append((step, episodes.copy(), fields))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def json_write(path, value):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False))


def physics_phase(rank, mode, output):
    visible = visible_devices()
    os.environ['CUDA_VISIBLE_DEVICES'] = visible[rank+1]
    os.environ['WARP_CACHE_PATH'] = str(output/f'warp-{rank}')
    os.environ['EB_UFO_DIAGNOSTICS_DIR'] = str(output/f'{mode}-{rank}-startup')
    import torch
    from rpo_ufo_mjlab import RPOUFOMJLabPhysics
    from bfm_ufo_policy import UFOBFMPolicy, load_pool
    from bfm_ufo_agent import AGENT_CONFIG_KEYS, VELOCITY_AUX_NAMES
    import safetensors.torch
    import yaml

    class CapturePhysics(RPOUFOMJLabPhysics):
        def _completed_substep(self):
            force, torque = super()._completed_substep()
            s = self._state()
            fields = {k: v.detach().cpu().numpy() for k, v in {
                'q': s['q'], 'dq': s['dq'], 'integrated_torque': torque,
                'integrated_force': force, 'target': self.target,
                'requested': self.requested, 'applied': self.last, 'previous': self.previous,
                'reference': self.reference, 'episode_control_step': self.steps,
                'body_pos': s['body_pos'], 'body_quat': s['body_quat'],
                'body_linear_velocity': s['body_lin_vel'], 'body_angular_velocity': s['body_ang_vel'],
            }.items()}
            self.capture.append(len(self.capture.frames), fields)
            return force, torque

    env = CapturePhysics(num_envs=2048, seed=42+rank)
    policy = None
    initial = env.get_observations()
    repeated = env.get_observations()
    assert all(torch.equal(initial[k], repeated[k]) for k in initial)
    anchor = env.last.clone()
    reference = env.reference.cpu().numpy().copy()
    a = env.pool.sampler
    reference_rows = []
    for source in a.source_weights:
        expert_ids = set(a.windows[8][source])
        rsi_ids = {row[:2] for row in env.pool.candidates[source]}
        tracking_ids = {row[:2] for row in env.pool.tracking_candidates[source]}
        assert tracking_ids <= rsi_ids <= expert_ids
        assert all(a.clips[i]['split'] == 'train' for i, _ in rsi_ids)
        reference_rows.append({'source': source, 'expert8_windows':len(expert_ids),
            'rsi_starts':len(rsi_ids), 'tracking250_starts':len(tracking_ids),
            'rsi_independent_ancestors':len({a.clips[i]['source_group'] for i,_ in rsi_ids}),
            'rsi_independent_clips':len({i for i,_ in rsi_ids}), 'behavior_label':'unknown'})
    expected = torch.as_tensor(np.stack([a.clips[int(i)]['arrays']['states'][int(f)]
                              for _,i,f in reference]), device=env.device())
    state_difference = (initial['states']-expected).abs().amax(0).cpu().numpy()
    assert float(state_difference.max()) < 1e-4
    for key in ('q','dq'):
        expected_raw = np.stack([a.clips[int(i)]['arrays'][key][int(f)] for _,i,f in reference])
        assert np.max(abs(env._state()[key].cpu().numpy()-expected_raw)) < 1e-5
    json_write(output/f'reference-{mode}-{rank}.json', {'sets':reference_rows,
        'state_abs_difference_by_column':state_difference, 'reference':reference,
        'expected_differences':['root Z lifted to sole clearance', 'history cleared'],
        'privileged_world_local_codec': 'not_yet_compared', 'rsi_rejections':env.pool.rows,
        'tracking_fraction':float(env.tracking.float().mean()), 'B_encoding':'original checkpoint model',
        'qualification':'data_consistency_only'})
    if mode == 'policy':
        package = Path('/workspace/baseline')
        cfg = yaml.safe_load((package/'resolved-config.yaml').read_text())['actor']['model']
        policy = UFOBFMPolicy({**{k:v for k,v in cfg.items() if k in AGENT_CONFIG_KEYS},
            'motion_root': os.environ['EB_UFO_MOTION_ROOT'], 'asset_sha256':os.environ['EB_UFO_ASSET_SHA256'],
            'codec_sha256':cfg['codec_sha256'], 'seed':42+rank, 'sampler_rank':rank})
        pointer = json.loads((package/'native-checkpoint.json').read_text())
        state = safetensors.torch.load_file(package/pointer['path']/'model/model.safetensors')
        keys = set(policy.model.state_dict())
        policy.model.load_state_dict({k:state[k] for k in keys}, strict=True)
        del state
        payload=torch.load(package/pointer['path']/'trainer-state.pt',map_location='cpu',weights_only=True)
        load_pool(policy.native,payload['pool']);del payload
        assert not policy.native.z_buffer.empty()
        policy.to('cuda:0'); policy.version=600; policy.online_transitions=983040
        policy.rng_cuda=torch.cuda.get_rng_state()
    weights = np.asarray([json.loads(os.environ['EB_UFO_REWARD_WEIGHTS'])[n] for n in VELOCITY_AUX_NAMES])
    # Six rank/mode phases × five events keep each category below the global 32 cap.
    events, stats, traces, raws = EventWindows(capacity=5), [], [], []
    alive = np.ones(2048, bool)
    try:
        for step in range(48):
            obs = env.get_observations()
            actions = anchor if policy is None else policy.predict_action_batch(env_obs=obs, mode='train')[1]['forward_inputs']['action']
            env.capture = CompletedFrames()
            obs, _, terminated, truncated, _ = env.step(actions)
            frames = env.capture.finish()
            repeated = env.get_observations()
            assert all(torch.equal(obs[k], repeated[k]) for k in obs)
            soft = env.robot.data.soft_joint_pos_limits[0, env.joint_ids].cpu().numpy()
            positions = position_audit(frames['q'], soft)
            raw = obs['ufo_aux_rewards'].cpu().numpy()
            np.testing.assert_allclose(raw[:,2], positions['endpoint'], atol=1e-5)
            done = (terminated|truncated).cpu().numpy()
            flags = env.interval.training_flags.cpu().numpy()
            ratio = abs(frames['dq'])/env.limits['joint_vel'].cpu().numpy()
            first = obs['step_id'].cpu().numpy().reshape(-1)==1
            snapshot = {**frames, 'raw_aux':np.repeat(raw[None],8,axis=0)}
            selected = {'first_position':np.flatnonzero(first & flags[:,0]).tolist(),
                'later_position':np.flatnonzero(~first & flags[:,0]).tolist(),
                'velocity_peak':np.argsort(-ratio.max((0,2)))[ratio.max((0,2))[np.argsort(-ratio.max((0,2)))]>2.5].tolist(),
                'normal_landing':np.flatnonzero(~done & ~env.interval.strict_flags.any(-1).cpu().numpy()
                    & (env.peak_force.max(-1).values.cpu().numpy()>20)).tolist()}
            events.append(step, env.episode[:,0].cpu().numpy(), snapshot, selected)
            alive &= ~done
            integrated_contact = np.linalg.norm(frames['integrated_force'][:,:,env.protected],axis=-1).max((0,2))>1
            endpoint_contact = raw[:,4]>0
            stats.append({'step':step, 'done':int(done.sum()), 'position_done':int(flags[:,0].sum()),
                'velocity_done':int(flags[:,1].sum()), 'first_steps':int(first.sum()),
                'first_done':int((done&first).sum()), 'initial_survivors':int(alive.sum()),
                'peak_velocity_ratio':float(ratio.max()),
                'position_transient_missed':int(positions['transient_missed'].sum()),
                'position_endpoint_mean':float(positions['endpoint'].mean()),
                'position_substep_mean':float(positions['mean'].mean()),
                'integrated_protected_contact':int(integrated_contact.sum()),
                'endpoint_protected_contact':int(endpoint_contact.sum()),
                'contact_event_missing_from_aux':int((integrated_contact & ~endpoint_contact).sum()),
                'slew_changed_elements_rate':float((abs(frames['requested'][-1]-frames['applied'][-1])>1e-6).mean()),
                'requested_applied_RMSE':float(np.sqrt(np.mean((frames['requested'][-1]-frames['applied'][-1])**2))),
                'dq_abs_quantiles':np.quantile(abs(frames['dq']),[.95,.99,1],axis=(0,1)),
                'reward':reward_audit(raw,weights)})
            traces.append(obs['ufo_interval_trace'][:256].cpu().numpy())
            raws.append(raw)
            if done.any():
                ids=torch.as_tensor(np.flatnonzero(done),device=env.device())
                before=obs['history_actor'][~torch.as_tensor(done,device=env.device())].clone()
                env.reset(env_ids=ids)
                after=env.get_observations()
                assert (after['history_actor'][ids]==0).all()
                assert (after['ufo_aux_rewards'][ids]==0).all()
                assert torch.equal(before,after['history_actor'][~torch.as_tensor(done,device=env.device())])
                anchor[ids]=env.last[ids]
        file=output/f'{mode}-{rank}-arrays.npz'
        np.savez_compressed(file, raw_aux=np.stack(raws), interval=np.stack(traces))
        event_rows=[]
        for i,event in enumerate(events.events):
            path=output/f'{mode}-{rank}-event-{i:03d}.npz'
            frames=event.pop('frames')
            np.savez_compressed(path, control_step=np.array([r['control_step'] for r in frames]),
                **{k:np.stack([r['fields'][k] for r in frames]) for k in frames[0]['fields']})
            event_rows.append({**event,'rank':rank,'mode':mode,'file':path.name,'sha256':digest(path),
                'frame_count':len(frames),'missing':['solver warmstart','mass matrix','contact Jacobian',
                    'runtime contact points/normals','requested motor torque','qpos/qvel full free joint'],
                'replay_class':'paired_diagnostic_not_exact'})
        json_write(output/f'physics-events-{mode}-{rank}.json',event_rows)
        json_write(output/f'physics-summary-{mode}-{rank}.json',stats)
        return {'rank':rank,'mode':mode,'seed':42+rank,'steps':48,'num_envs':2048,
                'initial_reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),
                'saved_z_pool_restored':policy is not None,
                'initial_prompt_context':'fresh episodes; saved step600 pool for mixed half, B600 for tracking half',
                'event_count':len(event_rows),'file':file.name,'sha256':digest(file),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    finally:
        env.close()


def validate_receipt(receipt, validation, output):
    if (receipt.get('schema_version') != SCHEMA or receipt.get('qualification') != 'diagnostics_only'
            or receipt.get('baseline_step') != 600 or receipt.get('optimizer_updates') != 0
            or receipt.get('dataset_manifest_sha256') != validation.get('manifest_sha256')
            or len(receipt.get('phases',[])) != 6
            or {(r.get('rank'),r.get('mode')) for r in receipt.get('phases',[])} !=
                {(r,m) for r in range(3) for m in ('anchor','policy')}):
        raise ValueError('Learning diagnostic identity differs')
    phases=receipt['phases']
    for row in phases:
        pair=[p for p in phases if p['rank']==row['rank']]
        if (row.get('seed') != 42+row['rank'] or row.get('steps') != 48 or row.get('num_envs') != 2048
                or row.get('initial_reference_sha256') != pair[1].get('initial_reference_sha256')):
            raise ValueError('Learning diagnostic paired identity differs')
    members=receipt.get('files',[])
    if not members or len(members)>1000:
        raise ValueError('Diagnostic artifacts missing or unbounded')
    for row in members:
        path=output/row['file']
        if (Path(row['file']).name != row['file'] or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row['bytes'] or digest(path)!=row['sha256']):
            raise ValueError('Diagnostic evidence file differs')


def replay_value_audit(output):
    """Differentiate the frozen actor objective; never step or update normalizers."""
    visible=visible_devices()
    os.environ['CUDA_VISIBLE_DEVICES']=visible[0]
    import torch
    import safetensors.torch
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent, FBcprAuxAgentConfig
    from bfm_ufo_policy import observation_space
    from bfm_ufo_checkpoint import numpyize
    package=Path('/workspace/baseline')
    pointer=json.loads((package/'native-checkpoint.json').read_text())
    root=package/pointer['path']
    cfg=FBcprAuxAgentConfig(**json.loads((root/'config.json').read_text()))
    model=cfg.model.build(observation_space(),23)
    saved=safetensors.torch.load_file(root/'model/model.safetensors',device='cuda:0')
    keys=set(model.state_dict())
    model.load_state_dict({k:saved[k] for k in keys},strict=True); del saved
    model.eval()
    native=FBcprAuxAgent.__new__(FBcprAuxAgent); native.cfg=cfg; native._model=model
    payload=torch.load(root/'trainer-state.pt',map_location='cpu',weights_only=True)
    replay=numpyize(payload['replay']['train']); del payload
    arrays={tuple(row['key']):np.load(root/'replay'/row['path'],mmap_mode='r') for row in replay['arrays']}
    ids=np.random.default_rng(20261013).choice(replay['count'],size=64,replace=False)
    obs={k[-1]:torch.as_tensor(np.asarray(v[ids]).copy(),device='cuda:0')
         for k,v in arrays.items() if k[0]=='observation'}
    z=torch.as_tensor(np.asarray(arrays[('z',)][ids]).copy(),device='cuda:0')
    raw=np.concatenate([arrays[('aux_rewards',n)][ids] for n in cfg.aux_rewards],axis=1)
    weights=np.asarray([cfg.aux_rewards_scaling[n] for n in cfg.aux_rewards])
    audit=reward_audit(raw,weights)
    def state_hash():
        h=hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=state_hash()
    with torch.no_grad():
        normalized_obs=model._obs_normalizer(obs)
    actor=actor_objective_audit(native,normalized_obs,z)
    weighted=torch.as_tensor(audit['weighted_sum'][:,None],dtype=torch.float32,device='cuda:0')
    normalized=frozen_aux_reward(model._aux_reward_normalizer,weighted)
    after=state_hash()
    assert before==after and all(p.grad is None and not p.requires_grad for p in model.parameters())
    json_write(output/'reward-value-audit.json',{
        'sampling_seed':20261013,'sample_size':len(ids),'replay_count':replay['count'],
        'replay_indices':ids,'aux_names':cfg.aux_rewards,'weights':weights,'reward':audit,
        'normalized_aux_reward':normalized.detach().cpu().numpy(),
        **actor,'state_hash_before':before,'state_hash_after':after,
        'normalization':'saved EMA statistics, without advancing its counter',
        'optimizer_updates':0,'normalizers_updated':False,'training_rng_consumed':False,
        'latent':'stored behavior latent; does not reproduce training 0.8 random relabel distribution',
        'multi_step_Q_aux_calibration':'pending same-policy/same-latent continuous rollout; historical replay is off-policy',
        'qualification':'frozen_replay_gradient_diagnosis_only'})


def validate_value_receipt(receipt, validation, output):
    if (receipt.get('schema_version')!='eb/rpo-learning-value/v1'
            or receipt.get('qualification')!='diagnostics_only' or receipt.get('optimizer_updates')!=0
            or receipt.get('baseline_step')!=600
            or receipt.get('dataset_manifest_sha256')!=validation.get('manifest_sha256')
            or {r.get('file') for r in receipt.get('files',[])}!={'reward-value-audit.json'}):
        raise ValueError('Frozen value diagnostic identity differs')
    for row in receipt['files']:
        path=output/row['file']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes']
                or digest(path)!=row['sha256']):
            raise ValueError('Frozen value diagnostic evidence differs')
    state=json.loads((output/'reward-value-audit.json').read_text())
    if (state.get('state_hash_before')!=state.get('state_hash_after') or not state.get('state_hash_before')
            or state.get('optimizer_updates')!=0 or state.get('normalizers_updated') is not False
            or state.get('training_rng_consumed') is not False):
        raise ValueError('Frozen value diagnostic state changed')


def main():
    output=Path('/outputs'); output.mkdir(parents=True,exist_ok=True)
    if sys.argv[1:]==['--value-only']:
        replay_value_audit(output)
        file=output/'reward-value-audit.json'
        json_write(output/'learning-value.json',{'schema_version':'eb/rpo-learning-value/v1',
            'qualification':'diagnostics_only','baseline_step':600,'optimizer_updates':0,
            'dataset_manifest_sha256':digest(Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json'),
            'files':[{'file':file.name,'bytes':file.stat().st_size,'sha256':digest(file)}]})
        return
    if len(sys.argv)==3:
        rank,mode=int(sys.argv[1]),sys.argv[2]
        json_write(output/f'phase-{mode}-{rank}.json',physics_phase(rank,mode,output))
        return
    for mode in ('anchor','policy'):
        children=[subprocess.Popen([sys.executable,__file__,str(r),mode]) for r in range(3)]
        results=[c.wait() for c in children]
        if any(results): raise RuntimeError(f'{mode} diagnostics failed: {results}')
    phases=[json.loads((output/f'phase-{m}-{r}.json').read_text()) for m in ('anchor','policy') for r in range(3)]
    assert all(phases[r]['initial_reference_sha256']==phases[r+3]['initial_reference_sha256'] for r in range(3))
    replay_value_audit(output)
    files=[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}
           for p in sorted(output.iterdir()) if p.is_file()]
    receipt={'schema_version':SCHEMA,'qualification':'diagnostics_only',
        'dataset_manifest_sha256':digest(Path(os.environ['EB_UFO_MOTION_ROOT'])/'manifest.json'),
        'baseline_step':600,'optimizer_updates':0,'phases':phases,'files':files}
    json_write(output/'learning-diagnostics.json',receipt)


if __name__=='__main__':
    main()
