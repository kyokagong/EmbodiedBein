"""Bounded integrated-force and RSI diagnostics of the frozen v13 environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SCHEMA = 'eb/rpo-dynamics-diagnostics/v1'
OPERATION = 'probe_rpo_ufo_dynamics_diagnostics_v1'


def dense_mass_matrix(raw, parent):
    raw, parent = np.asarray(raw, dtype=float), np.asarray(parent)
    nv = len(parent)
    if raw.shape == (nv, nv):
        matrix = np.tril(raw) + np.tril(raw, -1).T
    elif raw.ndim == 2 and raw.shape[0] == 1:
        matrix = np.zeros((nv, nv)); address = 0
        for i in range(nv):
            j = i
            while j >= 0:
                if address >= raw.size or j > i:
                    raise ValueError('Invalid sparse mass layout')
                matrix[i, j] = matrix[j, i] = raw[0, address]
                address += 1
                following = int(parent[j])
                if following >= j:
                    raise ValueError('Invalid sparse mass parent')
                j = following
        if address != raw.size:
            raise ValueError('Invalid sparse mass length')
    else:
        raise ValueError('Invalid mass layout')
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite mass matrix')
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError('Mass matrix is not positive definite') from error
    return matrix


def constraint_forces(jacobian, force, types):
    jacobian, force, types = np.asarray(jacobian), np.asarray(force), np.asarray(types)
    if (jacobian.ndim != 2 or force.shape != (len(jacobian),) or types.shape != force.shape
            or not np.isfinite(jacobian).all() or not np.isfinite(force).all()):
        raise ValueError('Invalid constraint force layout')
    contact, limit = np.isin(types, [5, 6, 7]), np.isin(types, [3, 4])
    return {name: jacobian[mask].T @ force[mask] for name, mask in
        [('contact', contact), ('limit', limit), ('other_constraint', ~(contact | limit))]}


def force_balance(matrix, terms, before, after, dt, solver_acceleration):
    matrix = np.asarray(matrix, dtype=float)
    values = [np.asarray(v, dtype=float) for v in (*terms.values(), before, after, solver_acceleration)]
    if (not terms or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or any(v.shape != (len(matrix),) or not np.isfinite(v).all() for v in values)
            or not np.isfinite(matrix).all() or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid force balance')
    accelerations = {name: np.linalg.solve(matrix, value) for name, value in terms.items()}
    predicted = sum(accelerations.values()) * dt
    measured = np.asarray(after) - before
    return {'delta_v_terms': {name: value * dt for name, value in accelerations.items()},
        'delta_v_predicted': predicted, 'delta_v_measured': measured,
        'delta_v_residual_max_abs': float(np.max(abs(measured - predicted))),
        'force_residual_max_abs': float(np.max(abs(matrix @ solver_acceleration - sum(terms.values())))),
        'interpretation': 'instantaneous M-inverse force contributions; implicit damping can alter integrated delta-v'}


def privileged_audit(reference, online, lift, atol=1e-4, rtol=1e-4):
    reference, online, lift = np.asarray(reference), np.asarray(online), np.asarray(lift)
    if (reference.ndim != 2 or reference.shape[1] != 358 or online.shape != reference.shape
            or lift.shape != (len(reference),) or not all(np.isfinite(v).all() for v in (reference, online, lift))):
        raise ValueError('Invalid privileged audit')
    adjusted = online.copy(); adjusted[:, 0] -= lift
    difference = abs(adjusted - reference)
    mismatch = difference > atol + rtol * abs(reference)
    groups = {'height': (0, 1), 'local_position': (1, 70), 'orientation': (70, 214),
        'linear_velocity': (214, 286), 'angular_velocity': (286, 358)}
    return {'sample_count': len(reference), 'atol': atol, 'rtol': rtol,
        'unexpected_mismatch_rows': int(mismatch.any(-1).sum()),
        'groups': {name: {'columns': [a, b], 'max_abs_after_lift': float(difference[:, a:b].max()),
            'mean_abs_after_lift': float(difference[:, a:b].mean()),
            'mismatch_rows': int(mismatch[:, a:b].any(-1).sum())} for name, (a, b) in groups.items()},
        'expected_difference': 'online root height includes RSI root lift; history is independently reset'}


def peak_joint_is_braking(torque, velocity, limits):
    import torch
    torque, velocity, limits = [torch.as_tensor(v) for v in (torque, velocity, limits)]
    if (torque.ndim != 2 or velocity.shape != torque.shape or limits.shape != (torque.shape[1],)
            or not all(torch.isfinite(v).all() for v in (torque, velocity, limits)) or (limits <= 0).any()):
        raise ValueError('Invalid peak braking inputs')
    joint = (velocity.abs() / limits).argmax(-1, keepdim=True)
    return (torque * velocity).gather(1, joint).squeeze(1) < 0


def same_audit_groups(first, second):
    return first.keys() == second.keys() and all(
        first[name]['columns'] == second[name]['columns']
        and first[name]['mismatch_rows'] == second[name]['mismatch_rows']
        and all(np.isclose(first[name][key], second[name][key], rtol=1e-6, atol=1e-10)
            for key in ('max_abs_after_lift', 'mean_abs_after_lift')) for name in first)


def kinematic_body_state(model, data, pos, quat, linear, angular, q, dq, joints, bodies):
    import mujoco
    data.qpos[:] = model.qpos0; data.qvel[:] = 0
    free = next(i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    qa, va = model.jnt_qposadr[free], model.jnt_dofadr[free]
    rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    data.qpos[qa:qa+3] = pos; data.qpos[qa+3:qa+7] = quat
    data.qvel[va:va+3] = linear; data.qvel[va+3:va+6] = rotation.reshape(3,3).T @ angular
    data.qpos[model.jnt_qposadr[joints]] = q; data.qvel[model.jnt_dofadr[joints]] = dq
    mujoco.mj_forward(model, data)
    velocities, angulars = [], []
    for body in bodies:
        jp, jr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
        mujoco.mj_jac(model, data, jp, jr, data.xpos[body], body)
        velocities.append(jp @ data.qvel); angulars.append(jr @ data.qvel)
    return {'body_pos':data.xpos[bodies].copy(), 'body_quat':data.xquat[bodies].copy(),
        'body_lin_vel':np.asarray(velocities), 'body_ang_vel':np.asarray(angulars)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
        else v.item() if isinstance(v, np.generic) else str(v), allow_nan=False))


def validate_receipt(receipt, validation, output, *, substeps=8, physics_dt=.0025):
    if (substeps, physics_dt) not in ((8,.0025),(16,.00125)):
        raise ValueError('Unapproved integration evidence')
    if (receipt.get('schema_version') != SCHEMA or receipt.get('qualification') != 'diagnostics_only'
            or receipt.get('baseline_step') != 600 or receipt.get('optimizer_updates') != 0
            or receipt.get('dataset_manifest_sha256') != validation.get('manifest_sha256')
            or len(receipt.get('phases', [])) != 6
            or {(p.get('rank'), p.get('mode')) for p in receipt.get('phases', [])} !=
                {(r, m) for r in range(3) for m in ('anchor', 'policy')}):
        raise ValueError('Dynamics diagnostic identity differs')
    for phase in receipt['phases']:
        paired = [p for p in receipt['phases'] if p['rank'] == phase['rank']]
        if (phase.get('seed') != 42 + phase['rank'] or phase.get('steps') != 48
                or phase.get('num_envs') != 2048 or not 0 <= phase.get('event_count', -1) <= 5
                or phase.get('optimizer_updates') != 0
                or not isinstance(phase.get('initial_reference_sha256'), str)
                or len(phase['initial_reference_sha256']) != 64
                or phase.get('initial_reference_sha256') != paired[1].get('initial_reference_sha256')):
            raise ValueError('Dynamics paired identity differs')
    files = receipt.get('files', [])
    if not files or len(files) > 100 or sum(row['bytes'] for row in files) > 20 * 1024**3:
        raise ValueError('Dynamics evidence missing or unbounded')
    for row in files:
        path = output / row['file']
        if (Path(row['file']).name != row['file'] or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']):
            raise ValueError('Dynamics evidence file differs')
    members = {row['file'] for row in files}
    if len(members) != len(files):
        raise ValueError('Duplicate dynamics evidence')
    for phase in receipt['phases']:
        suffix = f"{phase['mode']}-{phase['rank']}"
        required = {f'phase-{suffix}.json', f'rsi-{suffix}.json', f'rsi-{suffix}.npz',
            f'events-{suffix}.json', f'population-{suffix}.json'}
        if not required <= members:
            raise ValueError('Required dynamics and RSI evidence missing')
        if json.loads((output / f'phase-{suffix}.json').read_text()) != phase:
            raise ValueError('Phase evidence differs from receipt')
        population = np.asarray(json.loads((output / f'population-{suffix}.json').read_text()))
        if (population.shape != (48*substeps, 2) or not np.isfinite(population).all() or (population < 0).any()
                or (population[:, 1] > 2048).any() or (np.diff(population[:, 1]) > 0).any()
                or not np.isclose(population[:, 0].max(), phase['original_episode_peak_ratio'])):
            raise ValueError('Population evidence differs')
        rsi = json.loads((output / f'rsi-{suffix}.json').read_text())
        with np.load(output / f'rsi-{suffix}.npz', allow_pickle=False) as arrays:
            if (arrays['reference'].shape != (2048, 3)
                    or hashlib.sha256(arrays['reference'].tobytes()).hexdigest() != phase['initial_reference_sha256']
                    or arrays['reference_states'].shape != (2048, 52) or arrays['online_states'].shape != (2048, 52)):
                raise ValueError('RSI reference evidence differs')
            audit = privileged_audit(arrays['reference_privileged'], arrays['online_privileged'], arrays['lift'])
            if (audit['sample_count'] != 2048 or not same_audit_groups(audit['groups'], rsi['groups'])
                    or audit['unexpected_mismatch_rows'] != rsi['unexpected_mismatch_rows']
                    or audit['unexpected_mismatch_rows'] != phase['rsi_unexpected_mismatch_rows']
                    or not np.isfinite(arrays['online_states']).all()
                    or not np.isclose(abs(arrays['online_states'] - arrays['reference_states']).max(), rsi['state52_max_abs'])):
                raise ValueError('RSI state evidence differs')
        events = json.loads((output / f'events-{suffix}.json').read_text())
        if len(events) != phase['event_count']:
            raise ValueError('Event count evidence differs')
        for event in events:
            name = event['file']
            if name not in members or digest(output / name) != event['sha256']:
                raise ValueError('Event evidence missing or changed')
            with np.load(output / name, allow_pickle=False) as arrays:
                matrix = arrays['mass_matrix']; nv = len(matrix)
                if (matrix.shape != (nv, nv) or arrays['qpos'].shape != (nv+1,)
                        or arrays['qpos_before'].shape != (nv+1,) or arrays['qvel'].shape != (nv,)
                        or arrays['qvel_before'].shape != (nv,) or arrays['constraint_J'].shape[1:] != (nv,)
                        or any(not np.isfinite(arrays[key]).all() for key in arrays.files)):
                    raise ValueError('Integrated dynamics evidence incomplete')
                terms = constraint_forces(arrays['constraint_J'], arrays['constraint_force'], arrays['constraint_type'])
                terms.update(actuator=arrays['qfrc_actuator'], passive=arrays['qfrc_passive'],
                    bias=-arrays['qfrc_bias'], applied=arrays['qfrc_applied'])
                balance = force_balance(matrix, terms, arrays['qvel_before'], arrays['qvel'], physics_dt, arrays['qacc'])
                if not np.isclose(balance['force_residual_max_abs'], event['balance']['force_residual_max_abs']):
                    raise ValueError('Integrated dynamics balance evidence differs')


def phase(rank, mode, output, *, observer=None, action_program=None):
    from rpo_ufo_learning_diagnostics import visible_devices
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices()[rank + 1]
    os.environ['WARP_CACHE_PATH'] = str(output / f'warp-{rank}')
    os.environ['EB_UFO_DIAGNOSTICS_DIR'] = str(output / f'{mode}-{rank}-startup')
    import torch
    import warp as wp
    import mujoco
    import safetensors.torch
    import yaml
    from rpo_ufo_mjlab import RPOUFOMJLabPhysics, RPO_LAB_JOINT_ORDER
    from bfm_ufo_policy import UFOBFMPolicy, load_pool
    from bfm_ufo_agent import AGENT_CONFIG_KEYS

    class CapturePhysics(RPOUFOMJLabPhysics):
        def _completed_substep(self):
            sim, model = self.lab.sim, self.lab.sim.mj_model
            d = sim.wp_data
            before_q = wp.to_torch(d.qpos).clone()
            before_v = wp.to_torch(d.qvel).clone()
            warmstart = wp.to_torch(d.qacc_warmstart).clone()
            sim.step()
            if d.nacon.numpy()[0] >= d.naconmax or (wp.to_torch(d.nefc) >= d.njmax).any():
                raise ValueError('Native contact/constraint capacity exhausted')
            dt = float(model.opt.timestep)
            self.lab.scene.update(dt=dt); sim.sense()
            forces, torque = self._forces().clone(), self.robot.data.qfrc_actuator[:, self.joint_ids].clone()
            after_v = wp.to_torch(d.qvel)
            ratio = (after_v[:, self.dof_ids].abs() / self.limits['joint_vel']).amax(-1)
            q = wp.to_torch(d.qpos)[:, self.pos_ids]
            margin = torch.minimum(q - self.limits['joint_pos'][:, 0], self.limits['joint_pos'][:, 1] - q).amin(-1)
            contact_load = forces[:, self.feet, 2].abs().sum(-1)
            candidates = {'velocity_peak': (ratio, self.alive),
                'position_excursion': (-margin, self.alive & (margin < -.001)),
                'contact_peak': (contact_load, self.alive & (contact_load > 20)),
                'braking_peak': (ratio, self.alive & peak_joint_is_braking(torque, after_v[:, self.dof_ids], self.limits['joint_vel'])),
                'normal_landing': (contact_load, self.alive & (ratio <= 1) & (margin > .001) & (contact_load > 20))}
            for category, (scores, mask) in candidates.items():
                if not mask.any():
                    continue
                env_id = int(scores.masked_fill(~mask, -float('inf')).argmax())
                score = float(scores[env_id])
                old = self.events.get(category)
                if old is not None and (category == 'normal_landing' or score <= old['score']):
                    continue
                nefc = int(wp.to_torch(d.nefc)[env_id])
                raw_mass = wp.to_torch(d.qM)[env_id].cpu().numpy().copy()
                if raw_mass.shape[0] != 1:
                    raw_mass = raw_mass[:model.nv, :model.nv]
                matrix = dense_mass_matrix(raw_mass, model.dof_parentid)
                if wp.to_torch(d.efc.J).shape[1] == 1:
                    raise ValueError('Sparse runtime constraint Jacobian requires an explicit decoder')
                jacobian = wp.to_torch(d.efc.J)[env_id, :nefc, :model.nv].cpu().numpy().copy()
                constraint_force = wp.to_torch(d.efc.force)[env_id, :nefc].cpu().numpy().copy()
                constraint_type = wp.to_torch(d.efc.type)[env_id, :nefc].cpu().numpy().copy()
                fields = {name: wp.to_torch(getattr(d, name))[env_id].cpu().numpy().copy() for name in
                    ('qpos', 'qvel', 'qacc', 'qacc_smooth', 'qfrc_actuator', 'qfrc_passive',
                     'qfrc_bias', 'qfrc_constraint', 'qfrc_applied', 'qfrc_smooth')}
                fields.update(qpos_before=before_q[env_id].cpu().numpy(), qvel_before=before_v[env_id].cpu().numpy(),
                    qacc_warmstart_before=warmstart[env_id].cpu().numpy(), mass_matrix=matrix,
                    constraint_J=jacobian, constraint_force=constraint_force, constraint_type=constraint_type,
                    body_force=forces[env_id].cpu().numpy(), target=self.target[env_id].cpu().numpy(),
                    reference=self.reference[env_id].cpu().numpy(), dof_ids=np.asarray(self.dof_ids),
                    pos_ids=np.asarray(self.pos_ids))
                count = int(d.nacon.numpy()[0])
                contact_ids = torch.nonzero(wp.to_torch(d.contact.worldid)[:count] == env_id).flatten()
                for key in ('pos', 'frame', 'dist', 'geom', 'dim', 'efc_address', 'solref', 'solimp', 'friction', 'includemargin'):
                    fields['contact_' + key] = wp.to_torch(getattr(d.contact, key))[contact_ids].cpu().numpy().copy()
                terms = constraint_forces(jacobian, constraint_force, constraint_type)
                terms.update(actuator=fields['qfrc_actuator'], passive=fields['qfrc_passive'],
                    bias=-fields['qfrc_bias'], applied=fields['qfrc_applied'])
                balance = force_balance(matrix, terms, fields['qvel_before'], fields['qvel'], dt, fields['qacc'])
                peak_joint = int((after_v[env_id, self.dof_ids].abs() / self.limits['joint_vel']).argmax())
                self.events[category] = {'score': score, 'env': env_id, 'control_step': int(self.steps[env_id]),
                    'substep': self.interval.samples, 'joint': RPO_LAB_JOINT_ORDER[peak_joint],
                    'peak_joint_dof': self.dof_ids[peak_joint], 'peak_joint_velocity': float(after_v[env_id, self.dof_ids[peak_joint]]),
                    'constraint_reconstruction_error': float(np.max(abs(sum(constraint_forces(jacobian, constraint_force,
                        constraint_type).values()) - fields['qfrc_constraint']))), 'balance': balance, 'fields': fields}
            self.population.append([float(ratio[self.alive].max()) if self.alive.any() else 0., int(self.alive.sum())])
            self._refresh(dt=0.)
            return forces, torque

    env = CapturePhysics(num_envs=2048, seed=42 + rank)
    model = env.lab.sim.mj_model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    joint_ids = [next(i for i, name in enumerate(names) if name and name.split('/')[-1] == n) for n in RPO_LAB_JOINT_ORDER]
    env.dof_ids = model.jnt_dofadr[joint_ids].tolist(); env.pos_ids = model.jnt_qposadr[joint_ids].tolist()
    env.events, env.population = {}, []
    env.alive = torch.ones(2048, dtype=torch.bool, device=env.device())
    reference = env.reference.cpu().numpy().copy()
    initial = env.get_observations()
    source_rows = [env.pool.sampler.clips[int(clip)]['arrays'] for _, clip, _ in reference]
    ref_privileged = np.stack([a['privileged_state'][int(row[2])] for a, row in zip(source_rows, reference)])
    ref_states = np.stack([a['states'][int(row[2])] for a, row in zip(source_rows, reference)])
    rsi = privileged_audit(ref_privileged, initial['privileged_state'].cpu().numpy(), env.lift.cpu().numpy())
    rsi.update(state52_max_abs=float(abs(initial['states'].cpu().numpy() - ref_states).max()),
        history_is_zero=bool((initial['history_actor'] == 0).all()), sources=reference[:, 0])
    np.savez_compressed(output / f'rsi-{mode}-{rank}.npz', reference=reference, reference_states=ref_states,
        online_states=initial['states'].cpu().numpy(), reference_privileged=ref_privileged,
        online_privileged=initial['privileged_state'].cpu().numpy(), lift=env.lift.cpu().numpy())
    write_json(output / f'rsi-{mode}-{rank}.json', rsi)
    anchor, policy = env.last.clone(), None
    if mode == 'policy' and action_program is None:
        package = Path('/workspace/baseline')
        cfg = yaml.safe_load((package / 'resolved-config.yaml').read_text())['actor']['model']
        policy = UFOBFMPolicy({**{k: v for k, v in cfg.items() if k in AGENT_CONFIG_KEYS},
            'motion_root': os.environ['EB_UFO_MOTION_ROOT'], 'asset_sha256': os.environ['EB_UFO_ASSET_SHA256'],
            'codec_sha256': cfg['codec_sha256'], 'seed': 42 + rank, 'sampler_rank': rank})
        pointer = json.loads((package / 'native-checkpoint.json').read_text())
        state = safetensors.torch.load_file(package / pointer['path'] / 'model/model.safetensors')
        policy.model.load_state_dict({k: state[k] for k in policy.model.state_dict()}, strict=True); del state
        payload = torch.load(package / pointer['path'] / 'trainer-state.pt', map_location='cpu', weights_only=True)
        load_pool(policy.native, payload['pool']); del payload
        policy.to('cuda:0'); policy.version = 600; policy.online_transitions = 983040
        policy.rng_cuda = torch.cuda.get_rng_state()
    try:
        if observer is not None:
            observer.begin(env)
        for step in range(48):
            if observer is not None:
                observer.before(env, policy)
            actions = (torch.as_tensor(action_program[step],device=env.device()) if action_program is not None
                else anchor if policy is None else policy.predict_action_batch(env_obs=env.get_observations(),
                    mode='train')[1]['forward_inputs']['action'])
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if observer is not None:
                observer.after(env, policy, done)
            env.alive &= ~done
            ids = torch.nonzero(done).flatten()
            if len(ids):
                env.reset(env_ids=ids); anchor[ids] = env.last[ids]
        event_rows = []
        for category, event in env.events.items():
            fields = event.pop('fields')
            file = output / f'event-{mode}-{rank}-{category}.npz'
            np.savez_compressed(file, **fields)
            event_rows.append({**event, 'category': category, 'file': file.name, 'sha256': digest(file)})
        write_json(output / f'events-{mode}-{rank}.json', event_rows)
        write_json(output / f'population-{mode}-{rank}.json', env.population)
        return {'rank': rank, 'mode': mode, 'seed': 42 + rank, 'num_envs': 2048, 'steps': 48,
            'initial_reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
            'initial_survivors_48': int(env.alive.sum()), 'event_count': len(event_rows),
            'original_episode_peak_ratio': max(row[0] for row in env.population),
            'rsi_unexpected_mismatch_rows': rsi['unexpected_mismatch_rows'],
            'snapshot_time': 'forces and M/J after step before forward; qpos/qvel before and after integration',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0}
    finally:
        if observer is not None:
            observer.close()
        env.close()


def main():
    output = Path('/outputs'); output.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ['--finalize-rsi']:
        finalize_rsi(output); return
    if len(sys.argv) == 3:
        rank, mode = int(sys.argv[1]), sys.argv[2]
        if rank not in range(3) or mode not in ('anchor', 'policy'):
            raise ValueError('Unknown dynamics phase')
        write_json(output / f'phase-{mode}-{rank}.json', phase(rank, mode, output)); return
    for mode in ('anchor', 'policy'):
        children = [subprocess.Popen([sys.executable, __file__, str(rank), mode]) for rank in range(3)]
        results = [child.wait() for child in children]
        if any(results):
            raise RuntimeError(f'{mode} dynamics diagnostics failed: {results}')
    phases = [json.loads((output / f'phase-{mode}-{rank}.json').read_text())
        for mode in ('anchor', 'policy') for rank in range(3)]
    receipt = {'schema_version': SCHEMA, 'qualification': 'diagnostics_only', 'baseline_step': 600,
        'optimizer_updates': 0, 'phases': phases,
        'dataset_manifest_sha256': digest(Path(os.environ['EB_UFO_MOTION_ROOT']) / 'manifest.json'),
        'files': [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(output.iterdir()) if p.is_file()]}
    validate_receipt(receipt, {'manifest_sha256': receipt['dataset_manifest_sha256']}, output)
    write_json(output / 'dynamics-diagnostics.json', receipt)


def finalize_rsi(output):
    """Validate retained physics and trace RSI differences; never integrate or train."""
    import shutil
    import torch
    import mujoco
    from scipy.spatial.transform import Rotation
    from rpo_ufo_mjlab import prepare_spec
    from rpo_ufo_motion import load_robot_state
    from rpo_bfm import BODY_ORDER, RPO_LAB_JOINT_ORDER, CODEC_V2_HASH
    from rpo_bfm_codec import encode_privileged
    source = Path('/workspace/prior-evidence')
    original = json.loads((source / 'dynamics-diagnostics.json').read_text())
    validate_receipt(original, {'manifest_sha256': original['dataset_manifest_sha256']}, source)
    for row in original['files']:
        shutil.copy2(source / row['file'], output / row['file'])
    for root in source.glob('*-startup'):
        shutil.copytree(root, output / root.name)
    manifest, sampler = load_robot_state(Path(os.environ['EB_UFO_MOTION_ROOT']),
        asset_sha256=os.environ['EB_UFO_ASSET_SHA256'], expected_codec_sha256=CODEC_V2_HASH)
    model = prepare_spec(os.environ['RPO_ASSET_ROOT']).compile(); data = mujoco.MjData(model)
    joints = [model.joint(n).id for n in RPO_LAB_JOINT_ORDER]; bodies = [model.body(n).id for n in BODY_ORDER]
    cache, records, files = {}, [], []
    for rank in range(3):
        suffix = f'anchor-{rank}'
        with np.load(source / f'rsi-{suffix}.npz', allow_pickle=False) as arrays:
            native = []
            finite_linear, finite_angular, pose_difference = [], [], []
            for row in arrays['reference']:
                _, clip, frame = map(int,row)
                key = (clip,frame)
                raw = sampler.clips[clip]['arrays']
                if clip not in cache:
                    linear = np.gradient(raw['body_pos'], .02, axis=0).astype(np.float32)
                    angular = np.empty_like(linear)
                    for body in range(24):
                        rotations = Rotation.from_quat(raw['body_quat'][:,body,[1,2,3,0]])
                        delta = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / .02
                        angular[0,body], angular[-1,body] = delta[0], delta[-1]
                        angular[1:-1,body] = (delta[:-1]+delta[1:])/2
                    cache[clip] = (linear,angular)
                values = kinematic_body_state(model,data,raw['root_pos'][frame],raw['root_quat'][frame],
                    raw['root_lin_vel_w'][frame],raw['root_ang_vel_w'][frame],raw['q'][frame],raw['dq'][frame],joints,bodies)
                native.append(encode_privileged(*(torch.as_tensor(values[name][None],dtype=torch.float32)
                    for name in ('body_pos','body_quat','body_lin_vel','body_ang_vel'))).numpy()[0])
                finite_linear.append(float(abs(cache[clip][0][frame]-raw['body_lin_vel'][frame]).max()))
                finite_angular.append(float(abs(cache[clip][1][frame]-raw['body_ang_vel'][frame]).max()))
                pose_difference.append(float(np.linalg.norm(values['body_pos']-raw['body_pos'][frame],axis=-1).max()))
            native = np.asarray(native)
            alignment = privileged_audit(native,arrays['online_privileged'],arrays['lift'])
            records.append({'rank':rank,'sample_count':2048,'online_vs_native_cpu':alignment,
                'reference_linear_velocity_vs_recomputed_finite_difference_max':max(finite_linear),
                'reference_angular_velocity_vs_recomputed_finite_difference_max':max(finite_angular),
                'reference_vs_native_body_position_max_m':max(pose_difference),
                'reference_pose_screen_existing_tolerance_m':.002})
            file = output / f'rsi-native-{rank}.npz'
            np.savez_compressed(file,native_cpu_privileged=native,reference=arrays['reference'],
                online_privileged=arrays['online_privileged'],lift=arrays['lift'])
            files.append(file.name)
    audit = {'schema_version':'eb/rpo-rsi-source-audit/v1','records':records,
        'cpu_fk_steps':6144,'environment_integration_steps':0,'optimizer_updates':0,
        'source_velocity_definition':'reference converter uses 20ms centered finite differences of pose',
        'native_velocity_definition':'MJLab link-origin instantaneous velocity; CPU verification uses J(q)*qvel',
        'dataset_changed':False,'source_run_id':os.environ['EB_UFO_PRIOR_RUN_ID'],
        'source_receipt_sha256':digest(source / 'dynamics-diagnostics.json')}
    write_json(output / 'rsi-source-audit.json',audit)
    receipt = {**original,'source_run_id':os.environ['EB_UFO_PRIOR_RUN_ID'],
        'source_receipt_sha256':digest(source / 'dynamics-diagnostics.json'),
        'execution':'retained physics validation plus CPU-only RSI source audit',
        'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    validate_receipt(receipt,{'manifest_sha256':receipt['dataset_manifest_sha256']},output)
    write_json(output / 'dynamics-diagnostics.json',receipt)


if __name__ == '__main__':
    main()
