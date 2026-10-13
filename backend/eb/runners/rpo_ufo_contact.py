"""Train-only RSI contact candidates and explicit-PD physics resolution."""

from pathlib import Path
import numpy as np

try:
    from eb.rpo_bfm import RPO_LAB_JOINT_ORDER, JOINT_POSITION_MIN, JOINT_POSITION_MAX, SLEW_CONTROL
except ModuleNotFoundError:
    from rpo_bfm import RPO_LAB_JOINT_ORDER, JOINT_POSITION_MIN, JOINT_POSITION_MAX, SLEW_CONTROL

PROFILE = "contact_stable_v8"
SETTINGS = {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02,
    "sole_clearance_m": .002, "max_initial_sole_height_m": .02,
    "max_root_lift_m": .03, "max_self_penetration_m": .002,
    "min_root_height_m": .53, "max_root_tilt_rad": .70,
    "reset_velocity_limit_fraction": .70,
    "pd": "unchanged explicit gains/effort/velocity limits; twice the integration resolution",
    "hard_limits": "latch every physics substep; retain original 0.001 tolerance",
    "protected_contact": "latch protected-body net force >1N every physics substep"}


def configure_substeps(cfg):
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .0025, 8, 8


def contact_point_normal_velocity(link_velocity, link_angular_velocity, point_offset_world, ground_normal_world):
    values = (link_velocity, link_angular_velocity, point_offset_world, ground_normal_world)
    if any(np.asarray(v).shape[-1:] != (3,) or not np.isfinite(v).all() for v in values):
        raise ValueError("Invalid collision-point velocity/frame")
    return np.sum((link_velocity + np.cross(link_angular_velocity, point_offset_world)) * ground_normal_world, axis=-1)


def support_velocity_mask(point_heights, point_normal_velocities):
    if (point_heights.shape != point_normal_velocities.shape or point_heights.ndim != 2
            or not np.isfinite(point_heights).all() or not np.isfinite(point_normal_velocities).all()):
        raise ValueError("Invalid support-point heights/velocities")
    near = point_heights <= .0200001
    return near.any(-1) & ((abs(point_normal_velocities) <= .15) | ~near).all(-1)


def joint_lookahead_mask(q, dq, *, control_dt, margin):
    if (q.ndim != 2 or q.shape[1] != 23 or dq.shape != q.shape
            or not np.isfinite(q).all() or not np.isfinite(dq).all()
            or not np.isfinite(control_dt) or control_dt <= 0
            or not np.isfinite(margin) or margin < 0):
        raise ValueError("Invalid RSI joint lookahead state")
    low = np.asarray(JOINT_POSITION_MIN) + margin
    high = np.asarray(JOINT_POSITION_MAX) - margin
    future = q + dq * control_dt
    return ((q >= low) & (q <= high) & (future >= low) & (future <= high)).all(-1)


def physics_startup_receipt(masses, mode, *, seed):
    import hashlib
    masses=np.asarray(masses);mode=np.asarray(mode,dtype=bool)
    if (masses.ndim!=2 or masses.shape[1]!=24 or mode.shape!=(len(masses),)
            or not np.isfinite(masses).all() or (masses<0).any() or (masses.sum(-1)<=0).any()):
        raise ValueError('Invalid actual physics masses or tracking partition')
    totals=masses.sum(-1).astype(float)
    return {'schema_version':'eb/ufo-physics-startup/v1','seed':int(seed),'num_envs':len(masses),
        'actual_total_mass_kg':totals.tolist(),'body_weight_N':(totals*9.81).tolist(),
        'tracking_count':int(mode.sum()),'tracking_partition_sha256':hashlib.sha256(mode.tobytes()).hexdigest(),
        'physics_dt_s':.0025,'control_dt_s':.02}


def reset_mask_and_lift(root, q, dq, sole_height, self_depth, protected_gap=None):
    if (root.shape != (len(q),13) or q.shape != (len(q),23) or dq.shape != q.shape
            or sole_height.shape != (len(q),) or self_depth.shape != sole_height.shape
            or not all(np.isfinite(a).all() for a in (root,q,dq,sole_height,self_depth))):
        raise ValueError("Invalid RSI contact geometry/state")
    lift=np.maximum(.002-sole_height,0).astype(np.float32)
    quat=root[:,3:7]
    upright=1-2*(quat[:,1]**2+quat[:,2]**2)
    mask=(((q>=np.array(JOINT_POSITION_MIN)) & (q<=np.array(JOINT_POSITION_MAX))).all(-1)
        & (abs(dq)<=.7*np.array(SLEW_CONTROL['joint_velocity_limits_rad_s'])).all(-1)
        & (sole_height<=.02) & (lift<=.03) & (self_depth<=.002)
        & (root[:,2]+lift>=.53) & (upright>=np.cos(.7)))
    if protected_gap is not None:
        if protected_gap.shape != sole_height.shape or not np.isfinite(protected_gap).all():
            raise ValueError("Invalid RSI protected collision gap")
        mask &= protected_gap+lift>.025
    return mask,lift


class ContactResetPool:
    def __init__(self, sampler, asset_root, *, tracking_contact=False, flexion=False, native_mjcf=False,
                 joint_lookahead_s=None):
        import mujoco
        from scipy.spatial import ConvexHull

        path=Path(asset_root)/'rpo_description/mjcf/rpo.xml'
        if native_mjcf:
            try:
                from eb.runners.rpo_ufo_mjlab import prepare_spec
            except ModuleNotFoundError:
                from rpo_ufo_mjlab import prepare_spec
            model=prepare_spec(asset_root).compile()
        else:
            model=mujoco.MjModel.from_xml_path(str(path))
        data=mujoco.MjData(model)
        addresses=model.jnt_qposadr[[model.joint(n).id for n in RPO_LAB_JOINT_ORDER]]
        velocity_addresses=model.jnt_dofadr[[model.joint(n).id for n in RPO_LAB_JOINT_ORDER]]
        free=int(model.jnt_qposadr[model.joint('floating_base_joint').id])
        free_velocity=int(model.jnt_dofadr[model.joint('floating_base_joint').id])
        feet={model.body(s+'_ankle_roll_link').id for s in ('left','right')}
        protected={model.body(n).id for n in ('torso_link','left_thigh_yaw_link','right_thigh_yaw_link','left_thigh_roll_link','right_thigh_roll_link')}
        if not native_mjcf:
            model.geom_conaffinity[model.geom_contype!=0]=1
        masks=np.zeros(model.nbody,dtype=model.body_conaffinity.dtype)
        np.bitwise_or.at(masks,model.geom_bodyid,model.geom_conaffinity);model.body_conaffinity[:]=masks
        parents=model.body_parentid
        nearby={(a,int(parents[a])) for a in range(1,model.nbody)}
        nearby|={(a,int(parents[parents[a]])) for a in range(1,model.nbody)}
        geometry=[]
        for gid in range(model.ngeom):
            body=int(model.geom_bodyid[gid]);mesh=int(model.geom_dataid[gid])
            if body in feet|protected and model.geom_contype[gid] and mesh>=0:
                start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
                vertices=model.mesh_vert[start:start+count]
                # Linear height extrema are attained on the hull; avoid scanning STL triangles per reset frame.
                geometry.append((gid,body,vertices[ConvexHull(vertices).vertices]))
        if not geometry:
            raise ValueError("RSI collision meshes unavailable")
        self.sampler=sampler;self.candidates={source:[] for source in sampler.source_weights};self.rows=[]
        self.tracking_contact = tracking_contact
        self.tracking_candidates = {source: [] for source in sampler.source_weights}
        self.normal_velocity_rows = []
        self.foot_bounds=np.array([[np.inf,np.inf],[-np.inf,-np.inf]])
        self.foot_radius=max(float(np.linalg.norm(vertices,axis=-1).max()+np.linalg.norm(model.geom_pos[gid]))
            for gid,body,vertices in geometry if body in feet)
        for index,clip in enumerate(sampler.clips):
            frames=np.array([f for i,f in sampler.windows[8][clip['source']] if i==index],dtype=int)
            if not len(frames):continue
            a=clip['arrays']; sole=[];depth=[];gap=[];foot_bounds=[]; speed_ok=[]; normal_peak=[];support_legs=[]
            for frame in frames:
                data.qpos[:]=model.qpos0;data.qpos[free:free+3]=a['root_pos'][frame]
                data.qpos[free+3:free+7]=a['root_quat'][frame];data.qpos[addresses]=a['q'][frame]
                if tracking_contact:
                    data.qvel[:] = 0
                    data.qvel[velocity_addresses] = a['dq'][frame]
                    data.qvel[free_velocity:free_velocity+3] = a['root_lin_vel_w'][frame]
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, a['root_quat'][frame].astype(np.float64))
                    data.qvel[free_velocity+3:free_velocity+6] = rotation.reshape(3,3).T @ a['root_ang_vel_w'][frame]
                mujoco.mj_forward(model,data)
                low=protected_low=np.inf
                xy_low=np.full(2,np.inf);xy_high=np.full(2,-np.inf)
                point_heights=[];point_speeds=[];point_bodies=[]
                for gid,body,vertices in geometry:
                    points = vertices @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]
                    z=points[:,2].min()
                    if body in feet:
                        low=min(low,z)
                        xy=vertices@data.geom_xmat[gid].reshape(3,3)[:2].T+data.geom_xpos[gid,:2]
                        xy_low=np.minimum(xy_low,xy.min(0));xy_high=np.maximum(xy_high,xy.max(0))
                        if tracking_contact:
                            for point in points:
                                jacobian = np.zeros((3,model.nv)); angular = np.zeros_like(jacobian)
                                mujoco.mj_jac(model,data,jacobian,angular,point,body)
                                point_heights.append(float(point[2]))
                                point_speeds.append(float((jacobian @ data.qvel)[2]))
                                point_bodies.append(body)
                    else:protected_low=min(protected_low,z)
                penetration=0.
                for c in data.contact:
                    x,y=map(int,model.geom_bodyid[[c.geom1,c.geom2]])
                    if x and y and x!=y and (x,y) not in nearby and (y,x) not in nearby:
                        penetration=max(penetration,-float(c.dist))
                sole.append(low);gap.append(protected_low);depth.append(penetration)
                foot_bounds.append([xy_low,xy_high])
                if tracking_contact:
                    heights = np.array(point_heights) + max(.002-low, 0)
                    speeds = np.array(point_speeds)
                    speed_ok.append(bool(support_velocity_mask(heights[None],speeds[None])[0]))
                    normal_peak.append(float(abs(speeds[heights<=.0200001]).max()) if (heights<=.0200001).any() else float('inf'))
                    support_legs.append([bool((heights[np.asarray(point_bodies)==model.body(side+'_ankle_roll_link').id]<=.0200001).any())
                                         for side in ('left','right')])
            root=np.concatenate([a[k][frames] for k in ('root_pos','root_quat','root_lin_vel_w','root_ang_vel_w')],-1)
            mask,lift=reset_mask_and_lift(root,a['q'][frames],a['dq'][frames],np.array(sole),np.array(depth),np.array(gap))
            geometric_count=int(mask.sum())
            if tracking_contact:
                mask &= np.array(speed_ok)
            velocity_count=int(mask.sum())
            if joint_lookahead_s is not None:
                mask &= joint_lookahead_mask(a['q'][frames], a['dq'][frames],
                    control_dt=joint_lookahead_s, margin=.001)
            dynamic_count=int(mask.sum())
            if flexion:
                from rpo_ufo_compliance import flexion_mask
                supports=np.asarray(support_legs,dtype=bool)
                # Geometry-invalid frames can have no near foot; reject rather than fabricate support.
                valid_support=supports.any(-1)
                selected=np.zeros(len(frames),bool)
                selected[valid_support]=flexion_mask(a['q'][frames][valid_support],supports[valid_support])
                mask &= selected
            entries = [(index,int(f),float(h)) for f,h in zip(frames[mask],lift[mask])]
            self.candidates[clip['source']].extend(entries)
            windows = set(sampler.windows[250][clip['source']])
            self.tracking_candidates[clip['source']].extend(row for row in entries if row[:2] in windows)
            if mask.any():
                bounds=np.asarray(foot_bounds)[mask]
                self.foot_bounds[0]=np.minimum(self.foot_bounds[0],bounds[:,0].min(0))
                self.foot_bounds[1]=np.maximum(self.foot_bounds[1],bounds[:,1].max(0))
            self.rows.append({'clip':clip['id'],'source':clip['source'],'eligible':int(mask.sum()),'original_starts':len(frames),
                'max_root_lift_m':float(lift[mask].max()) if mask.any() else None})
            if joint_lookahead_s is not None:
                self.rows[-1].update(joint_lookahead_s=joint_lookahead_s, joint_margin_rad=.001,
                    joint_lookahead_rejected=velocity_count-dynamic_count)
            if tracking_contact:
                peaks=np.asarray(normal_peak)[mask]
                self.rows[-1].update(geometric_eligible=geometric_count,
                    velocity_rejected=geometric_count-velocity_count,
                    tracking_eligible=sum(row[:2] in windows for row in entries),
                    normal_speed_max_mps=float(peaks.max()) if len(peaks) else None,
                    eligible_frames=frames[mask].tolist())
                if flexion:
                    self.rows[-1].update(flexion_rejected=velocity_count-int(mask.sum()),
                        min_knee_flexion_rad=.15,max_hip_pitch_rad=.10)
        if any(not pool for pool in self.candidates.values()):
            raise ValueError("RSI contact pool has no candidates for a positive-weight train source")
        if tracking_contact and any(not pool for pool in self.tracking_candidates.values()):
            raise ValueError("RSI tracking contact pool has no 251-state windows for a positive-weight train source")

    def sample(self,count,device, *, tracking=None):
        import torch

        sources=self.sampler.rng.choice(tuple(self.candidates),count,p=list(self.sampler.source_weights.values()))
        if self.tracking_contact and (tracking is None or np.asarray(tracking).shape != (count,)):
            raise ValueError("Tracking RSI requires frozen per-environment mode")
        selected=[]
        for i, source in enumerate(sources):
            pool=(self.tracking_candidates if self.tracking_contact and tracking[i] else self.candidates)[source]
            selected.append(pool[int(self.sampler.rng.integers(len(pool)))])
        fields={k:[] for k in ('root_pos','root_quat','root_lin_vel_w','root_ang_vel_w','q','dq')}
        for i,f,_ in selected:
            for k in fields:fields[k].append(self.sampler.clips[i]['arrays'][k][f])
        values={k:np.stack(v) for k,v in fields.items()}
        root=np.concatenate([values[k] for k in ('root_pos','root_quat','root_lin_vel_w','root_ang_vel_w')],-1)
        lift=np.array([h for _,_,h in selected],np.float32);root[:,2]+=lift
        result=tuple(torch.as_tensor(v,device=device) for v in (root,values['q'],values['dq'],lift))
        if self.tracking_contact:
            reference=np.array([[("official_rpo","g1").index(source),i,f]
                                for source,(i,f,_) in zip(sources,selected)],np.int64)
            result=(*result,torch.as_tensor(reference,device=device))
        return result
