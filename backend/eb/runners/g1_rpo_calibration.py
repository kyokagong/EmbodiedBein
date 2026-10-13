"""Constrained native-link calibration using the pinned hhtools solver."""
from dataclasses import asdict

import numpy as np


def check_offsets(source, target, scales, offsets):
    position, angle = [], []
    for name in source:
        scale = float(scales[name])
        translation, quaternion = map(np.asarray, offsets[name])
        if (not np.isfinite(scale) or scale <= 0 or translation.shape != (3,)
                or quaternion.shape != (4,) or not np.isfinite(translation).all()
                or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion),1,atol=1e-5)):
            raise ValueError("Invalid calibration scale/offset")
        s, t = source[name], target[name]
        s_pos = s[:3,3] - source["hips"][:3,3]
        t_pos = t[:3,3] - target["hips"][:3,3]
        position.append(float(np.linalg.norm(s_pos*scale+t[:3,:3]@translation-t_pos)))
        x,y,z,w = quaternion
        cross = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        offset_rotation = np.eye(3)+2*w*cross+2*cross@cross
        error_rotation = (s[:3,:3]@offset_rotation).T@t[:3,:3]
        angle.append(float(np.arccos(np.clip((np.trace(error_rotation)-1)/2,-1,1))))
    result = {"max_position_error_m":max(position), "max_orientation_error_rad":max(angle)}
    if result["max_position_error_m"] > 1e-5 or result["max_orientation_error_rad"] > .001:
        raise ValueError("Calibration offsets do not close the native-link rest alignment")
    return result


def calibrate(source_robot, target_robot):
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.calibration.assistant import assess_calibration_pose, propose_calibration_pose
    from hhtools.retarget.calibration.calibration import _collect_link_transforms_at_q

    ref = r2r.build_source_reference_pose(source_robot)
    before = assess_calibration_pose(target_robot,ref.name,target_robot.zero_configuration(),reference_pose=ref)
    q, assessment = propose_calibration_pose(target_robot,ref.name,reference_pose=ref)
    if (not assessment.valid or max(assessment.edge_errors_deg.values()) > 20
            or max(assessment.symmetry_errors_deg.values()) > 5
            or assessment.foot_height_delta_m is None or assessment.foot_height_delta_m > .01):
        raise ValueError(f"G1/RPO constrained calibration failed: {asdict(assessment)}")
    cfg, _ = r2r._build_scaler_config(source_robot,target_robot,q)
    source_fk = _collect_link_transforms_at_q(source_robot,source_robot.zero_configuration())
    target_fk = _collect_link_transforms_at_q(target_robot,q)
    source = {key:source_fk[name] for key,name in source_robot.preset.ik_map.items()}
    target = {key:target_fk[name] for key,name in target_robot.preset.ik_map.items()}
    closure = check_offsets(source,target,cfg.joint_scales,cfg.joint_offsets)
    target_robot.apply_configuration(target_robot.zero_configuration())
    return q, {"method":"hhtools_constrained_chain_alignment_soma_offsets_v1", "verified":True,
        "keypoint_semantics":"native_link_origins_not_hand_tips_or_contact_soles",
        "source_map":dict(source_robot.preset.ik_map), "target_map":dict(target_robot.preset.ik_map),
        "calibrated_joint_q":q, "assessment_before":asdict(before), "assessment":asdict(assessment),
        "joint_scales":cfg.joint_scales, "joint_offsets":cfg.joint_offsets, "closure":closure,
        "unchecked":["physical_hand_tip_and_sole_alignment", "independent_motion_generalization"]}


def retarget_calibrated(source_robot,target_robot,motion,q,progress):
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
    from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline
    from hhtools.retarget.interaction_mesh.config import InteractionMeshPipelineConfig
    from scipy.spatial.transform import Rotation

    cfg,ref=r2r._build_scaler_config(source_robot,target_robot,q)
    class CalibratedPipeline(InteractionMeshPipeline):
        def _build_scaled_source_pose(self,source):
            if source.terrain is not None or source.objects:
                raise ValueError("Calibrated G1 trial supports only this flat-ground source")
            pos,quat,z_min,ratio=super()._build_scaled_source_pose(source)
            names=tuple(source.hierarchy.bone_names)
            root=names.index("hips")
            root_out=pos[:,root].copy()
            source_pos=np.asarray(source.positions)
            for name in target_robot.preset.ik_map:
                i=names.index(name)
                translation,offset_q=cfg.joint_offsets[name]
                rotation=Rotation.from_quat(source.quaternions[:,i])*Rotation.from_quat(offset_q)
                pos[:,i]=root_out+(source_pos[:,i]-source_pos[:,root])*cfg.joint_scales[name]+rotation.apply(translation)
                quat[:,i]=rotation.as_quat()
            return pos,quat,z_min,ratio
    scaler=HumanToRobotScaler(motion.hierarchy,cfg,human_height=ref.height_m)
    # Solver output retains joint bounds; no UI ankle-overlay shift or extra hip clamp.
    pipe=CalibratedPipeline(robot=target_robot,scaler=scaler,
        cfg=InteractionMeshPipelineConfig(post_mpc_foot_clamps=False))
    baseline=InteractionMeshPipeline(robot=target_robot,scaler=scaler,cfg=InteractionMeshPipelineConfig())
    before=baseline._build_scaled_source_pose(motion)[0]
    after=pipe._build_scaled_source_pose(motion)[0]
    change=float(np.linalg.norm(after-before,axis=-1).max())
    if not np.isfinite(change) or change<1e-4:
        raise ValueError("Calibration did not change the CPU solver targets")
    result=pipe.run(motion,progress_callback=lambda stage,cur,total:progress(cur,total))
    return result,{"calibration_applied_to_solver":True,"max_target_change_m":change,
        "post_mpc_foot_clamps":False,"ui_ankle_overlay_alignment":False}
