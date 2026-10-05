"""Two-environment Isaac probe run by ebd before publishing the RPO AMP interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _probe_motion(env, motion_dir: Path, asset_manifest: Path) -> dict:
    import torch
    from isaaclab.utils.math import quat_apply
    from rpo_amp_motion import load_motion_prior

    asset_sha = hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
    sampler = load_motion_prior(motion_dir, asset_sha256=asset_sha)
    robot = env.robot
    errors = []
    for clip_id, _, data in sampler.clips:
        frame = torch.from_numpy(data[len(data) // 2].copy()).to(env.device)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[0, 2] += frame[70]
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        root[:, 7:13] = 0
        joints = robot.data.default_joint_pos.clone()
        joints[0, env.codec_cfg.joint_ids] = frame[3:26]
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.sim.render()
        env.scene.update(env.step_dt)
        actual = env._amp_frame()[0]
        errors.append({"clip":clip_id,
                       "key_body_max_error_m":float((actual[49:67] - frame[49:67]).abs().max())})
    maximum = max(item["key_body_max_error_m"] for item in errors)
    if maximum > 0.03:
        raise ValueError(f"RPO motion key-body FK differs from Isaac asset: {maximum:.4f} m")
    pd_tracking = []
    for clip_id, _, data in sampler.clips:
        env.reset()
        first = torch.from_numpy(data[0].copy()).to(env.device)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[0, 2] += first[70]
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        root[:, 7:13] = 0
        gravity = first[67:70]
        pitch = torch.asin(gravity[0].clamp(-1.0, 1.0))
        roll = torch.atan2(-gravity[1], -gravity[2])
        cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
        cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
        root[0, 3:7] = torch.stack((cp * cr, cp * sr, sp * cr, -sp * sr))
        root[0, 7:10] = quat_apply(root[0:1, 3:7], first[71:74].unsqueeze(0))[0]
        root[0, 10:13] = quat_apply(root[0:1, 3:7], first[:3].unsqueeze(0))[0]
        joints = robot.data.default_joint_pos.clone()
        velocity = torch.zeros_like(joints)
        joints[0, env.codec_cfg.joint_ids] = first[3:26]
        velocity[0, env.codec_cfg.joint_ids] = first[26:49]
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joints, velocity)
        env.sim.render()
        env.scene.update(env.step_dt)
        tracking_errors = []
        reward_totals = torch.zeros(len(env.reward_manager._term_names), device=env.device)
        steps = 0
        for item in data[1:]:
            target = torch.from_numpy(item[3:26].copy()).to(env.device)
            actions = torch.zeros((env.num_envs, 23), device=env.device)
            actions[0] = (target - robot.data.default_joint_pos[0, env.codec_cfg.joint_ids]) / env.action_scale
            _, _, terminated, truncated, _ = env.step(actions)
            steps += 1
            tracking_errors.append(float((robot.data.joint_pos[0, env.codec_cfg.joint_ids] - target).square().mean()))
            reward_totals += env.reward_manager._step_reward[0] * env.step_dt
            if bool(terminated[0] or truncated[0]):
                break
        height = float(robot.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
        upright = float(-robot.data.projected_gravity_b[0, 2])
        reason = ("complete" if steps == len(data) - 1 else "low_base" if height < 0.50
                  else "excessive_tilt" if upright < 0.6967 else "contact_or_other")
        terms = sorted(zip(env.reward_manager._term_names, reward_totals.cpu().tolist()),
                       key=lambda pair: pair[1])
        pd_tracking.append({"clip": clip_id, "steps": steps, "available_steps": len(data) - 1,
                            "joint_rmse_rad": (sum(tracking_errors) / len(tracking_errors)) ** 0.5 if tracking_errors else None,
                            "ended_early": steps < len(data) - 1,
                            "end_reason": reason, "base_height_m": height,
                            "reward_terms": {name: round(value, 4) for name, value in (*terms[:5], *terms[-5:])}})
    stand_clip = next(data for clip_id, _, data in sampler.clips if "Stand" in clip_id)
    env.reset()
    env.command_generator.is_standing_env[:] = True
    env.command_generator.vel_command_b[:] = 0
    stand_target = torch.from_numpy(stand_clip[len(stand_clip) // 2, 3:26].copy()).to(env.device)
    root = robot.data.default_root_state.clone()
    root[1, 2] = env.scene.env_origins[1, 2] + float(stand_clip[len(stand_clip) // 2, 70])
    joints = robot.data.default_joint_pos.clone()
    joints[1, env.codec_cfg.joint_ids] = stand_target
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
    env.sim.render()
    env.scene.update(env.step_dt)
    stand_action = (stand_target - robot.data.default_joint_pos[0, env.codec_cfg.joint_ids]) / env.action_scale
    hold = torch.zeros((env.num_envs, 23), device=env.device)
    hold[:] = stand_action
    stand_pose_hold = [{"case":case, "steps":1000, "terminated":False}
                       for case in ("default_reset", "source_pose_reset")]
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for step in range(1, 1001):
        _, _, terminated, truncated, _ = env.step(hold)
        ended = active & (terminated | truncated)
        for index in ended.nonzero(as_tuple=False).flatten().tolist():
            stand_pose_hold[index].update(steps=step, terminated=bool(terminated[index]),
                                          base_height_m=float(robot.data.root_pos_w[index, 2]
                                                              - env.scene.env_origins[index, 2]))
        active &= ~ended
        if not bool(active.any()):
            break
    return {"status":"passed", "dataset_manifest_sha256":sampler.manifest_sha256,
            "clips_checked":len(errors), "max_key_body_error_m":maximum, "clips":errors,
            "dynamic_replay_verified":False, "pd_tracking_diagnostic":pd_tracking,
            "stand_pose_hold":stand_pose_hold}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import load_cfg_from_registry
        from rpo_wbc_extension import AMP_TASK_ID, register_task

        register_task()
        cfg = load_cfg_from_registry(AMP_TASK_ID, "env_cfg_entry_point")
        cfg.seed = 1234
        cfg.scene.num_envs = 2
        env = gym.make(AMP_TASK_ID, cfg=cfg, render_mode=None).unwrapped
        obs, _ = env.reset()
        source_init = None
        if os.getenv("EB_RPO_AMP_INITIAL_POSE"):
            from rpo_wbc import AMP_STAND_HEIGHT_M, AMP_STAND_Q
            target = torch.tensor(AMP_STAND_Q, device=env.device)
            q_error = float((env.robot.data.joint_pos[0, env.codec_cfg.joint_ids] - target).abs().max())
            height = float(env.robot.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
            frame_error = float((env.amp_history.frames[0, 2, 3:26] - target).abs().max())
            source_init = {"joint_max_error_rad":q_error, "amp_frame_max_error_rad":frame_error,
                           "base_height_m":height}
            if max(q_error, frame_error, abs(height - AMP_STAND_HEIGHT_M)) > .005:
                raise ValueError("official A1 reset did not reach the declared initial pose")
        if obs["policy"].shape != (2, 780) or obs["critic"].shape != (2, 1390):
            raise ValueError("RPO AMP actor/critic observations differ from the draft contract")

        for step in range(3):
            obs, reward, terminated, truncated, info = env.step(
                torch.zeros((2, 23), device=env.device))
            window, valid = info["amp_window"], info["amp_valid"]
            if (obs["policy"].shape != (2, 780) or obs["critic"].shape != (2, 1390)
                    or window.shape != (2, 3, 74) or valid.shape != (2,)
                    or not torch.isfinite(obs["policy"]).all()
                    or not torch.isfinite(obs["critic"]).all()
                    or not torch.isfinite(window).all()
                    or not torch.isfinite(reward).all()
                    or (step < 1 and bool(valid.any()))
                    or (step == 2 and not bool(valid.all()))):
                raise ValueError(f"RPO AMP numerical probe failed at step {step}")
        initial_terminated, initial_truncated = int(terminated.sum()), int(truncated.sum())

        untouched = env.amp_history.frames[1].clone()
        env.reset(env_ids=torch.tensor([0], device=env.device))
        if (int(env.amp_history.count[0]) != 1
                or not torch.equal(env.amp_history.frames[1], untouched)):
            raise ValueError("RPO AMP partial reset changed the untouched environment")

        env.reset()
        env.command_generator.is_standing_env[:] = torch.tensor([True, False], device=env.device)
        env.command_generator.vel_command_b[:] = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], device=env.device)
        origin_xy = env.robot.data.root_pos_w[:, :2].clone()
        neutral = []
        active = torch.ones(2, dtype=torch.bool, device=env.device)
        for policy_step in range(1, 1001):
            _, _, terminated, truncated, _ = env.step(torch.zeros((2, 23), device=env.device))
            ended = active & (terminated | truncated)
            for index in ended.nonzero(as_tuple=False).flatten().tolist():
                neutral.append({"env":index, "command_mps":float(env.command_generator.command[index, 0]),
                                "steps":policy_step,
                                "base_height_m":float(env.robot.data.root_pos_w[index, 2]),
                                "drift_m":float(torch.norm(env.robot.data.root_pos_w[index, :2] - origin_xy[index]))})
            active &= ~(terminated | truncated)
            if not bool(active.any()):
                break
        for index in active.nonzero(as_tuple=False).flatten().tolist():
            neutral.append({"env":index, "command_mps":float(env.command_generator.command[index, 0]),
                            "steps":1000,
                            "base_height_m":float(env.robot.data.root_pos_w[index, 2]),
                            "drift_m":float(torch.norm(env.robot.data.root_pos_w[index, :2] - origin_xy[index]))})

        report = {"schema_version":"eb/rpo-amp-isaac-probe/v1", "status":"passed",
                  "task":AMP_TASK_ID, "num_envs":2, "actor_dim":780, "critic_dim":1390,
                  "action_dim":23, "amp_frame_dim":74, "amp_window_frames":3,
                  "partial_reset":True,
                  **({"source_initial_pose":source_init} if source_init else {}),
                  "neutral_pd":sorted(neutral, key=lambda item:item["env"]),
                  "terminated_on_zero_action":initial_terminated,
                  "truncated_on_zero_action":initial_truncated}
        if args.motion_dir:
            if not args.asset_manifest:
                raise ValueError("motion probe requires an asset manifest")
            report["motion"] = _probe_motion(env, args.motion_dir, args.asset_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True))
        print(json.dumps(report, sort_keys=True), flush=True)
    except BaseException as exc:
        import sys
        import traceback

        failure = {"schema_version":"eb/rpo-amp-isaac-probe/v1", "status":"failed",
                   "error_type":type(exc).__name__, "error":str(exc),
                   "traceback":traceback.format_exc()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, sort_keys=True))
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)  # Isaac shutdown can otherwise replace the original failure with exit 0.
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
