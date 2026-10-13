"""Container-only RPO BFM probes; JSON receipts are evidence, not qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def probe_source(output: Path) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from robolab.tasks.direct.base.rpo_env_cfg import RPOFlatEnvCfg
        from rpo_flat_extension import TASK_ID, register_task
        from rpo_wbc import RPO_LAB_JOINT_ORDER

        register_task()
        cfg = RPOFlatEnvCfg()
        cfg.scene.num_envs = 2
        cfg.seed = 42
        cfg.commands.debug_vis = False
        cfg.scene.left_feet_scanner.debug_vis = False
        cfg.scene.right_feet_scanner.debug_vis = False
        env = gym.make(TASK_ID, cfg=cfg, render_mode=None).unwrapped
        obs, _ = env.reset()
        assert tuple(env.robot.joint_names) == RPO_LAB_JOINT_ORDER
        assert obs["policy"].shape == (2, 780)
        assert torch.isfinite(obs["policy"]).all()
        robot = env.robot
        receipt = {
            "schema_version": "eb/rpo-bfm-source-probe/v1",
            "status": "passed",
            "qualification": "not_requested",
            "joint_names": robot.joint_names,
            "body_names": robot.body_names,
            "privileged_dim": 15 * len(robot.body_names) - 2,
            "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
            "joint_limits": robot.data.joint_pos_limits[0].cpu().tolist(),
            "policy_hz": 1 / env.step_dt,
            "physics_hz": 1 / env.physics_dt,
            "root_body": robot.body_names[0],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "source-probe.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def probe_interface(output: Path, motion_root: Path | None = None) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from rpo_bfm_extension import GYM_ID, RPOBFMEnvCfg, register_task, rlinf_observation
        from rpo_bfm import BODY_ORDER, OBS_DIMS

        register_task()
        cfg = RPOBFMEnvCfg()
        cfg.scene.num_envs = 2
        cfg.seed = 42
        env = gym.make(GYM_ID, cfg=cfg, render_mode=None).unwrapped
        obs, _ = env.reset()
        assert tuple(env.robot.body_names) == BODY_ORDER
        assert obs["commands"].shape == (2, 3) and torch.isfinite(obs["commands"]).all()
        torch.testing.assert_close(obs["commands"], env.command_generator.command)
        wrapped = rlinf_observation(obs)
        assert wrapped["commands"].shape == (2, 3)
        assert wrapped["bfm_aux_rewards"].shape == (2, 3)
        for key, width in OBS_DIMS.items():
            assert obs[key].shape == (2, width) and torch.isfinite(obs[key]).all()
        golden = []
        if motion_root is not None:
            import numpy as np
            from rpo_bfm_codec import encode_privileged, encode_state, quaternion_matrix

            manifest = json.loads((motion_root / "manifest.json").read_text())
            for clip in manifest["clips"]:
                with np.load(motion_root / clip["file"], allow_pickle=False) as arrays:
                    valid = np.flatnonzero(arrays["valid"])
                    if not len(valid):
                        continue
                    index = int(valid[len(valid)//2])
                    q = torch.from_numpy(arrays["q"][index].copy()).to(env.device)
                    root = env.robot.data.default_root_state.clone()
                    root[:, :3] += env.scene.env_origins
                    root[0, :3] = env.scene.env_origins[0] + torch.from_numpy(arrays["root_pos"][index].copy()).to(env.device)
                    root[0, 3:7] = torch.from_numpy(arrays["root_quat"][index].copy()).to(env.device)
                    root[:, 7:13] = 0
                    joints = env.robot.data.default_joint_pos.clone()
                    joints[0] = q
                    env.robot.write_root_state_to_sim(root)
                    env.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.scene.update(env.step_dt)
                    reference_pos = torch.from_numpy(arrays["body_pos"][index:index+1].copy()).to(env.device)
                    reference_rot = torch.from_numpy(arrays["body_quat"][index:index+1].copy()).to(env.device)
                    zeros = torch.zeros_like(reference_pos)
                    expected_priv = encode_privileged(reference_pos, reference_rot, zeros, zeros)
                    actual = env._get_observations()
                    gravity = (quaternion_matrix(reference_rot[:, 0]).transpose(-1, -2)
                               @ q.new_tensor([0., 0., -1.])[None, :, None]).squeeze(-1)
                    expected_state = encode_state(q[None], torch.zeros_like(q[None]), gravity,
                                                  torch.zeros(1, 3, device=env.device))
                    # Static pose compares the shared geometry/state codec; derivative quality is separate.
                    state_error = float((actual["states"][:1]-expected_state).abs().max())
                    pose_error = float((actual["privileged_state"][:1, :214]-expected_priv[:, :214]).abs().max())
                    assert max(state_error, pose_error) < 2e-4, (clip["id"], state_error, pose_error)
                    golden.append({"clip":clip["id"], "state_max_error":state_error,
                                   "privileged_pose_max_error":pose_error})
            assert golden
            obs, _ = env.reset()
        for _ in range(6):
            obs, _, _, _, info = env.step(torch.zeros(2, 23, device=env.device))
            assert info["bfm_aux_rewards"].shape == (2, 3)
            torch.testing.assert_close(obs["bfm_aux_rewards"], info["bfm_aux_rewards"])
        before = {key: value[1].clone() for key, value in obs.items()}
        saved_history = env.bfm_history.snapshot()[1].clone()
        obs, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
        for key, value in before.items():
            torch.testing.assert_close(value, obs[key][1])
        assert not env.bfm_history.snapshot()[0].any()
        torch.testing.assert_close(saved_history, env.bfm_history.snapshot()[1])
        # Reading observations/bootstrap must not push history again.
        history = env.bfm_history.snapshot().clone()
        env._get_observations()
        assert torch.equal(env.bfm_history.snapshot(), history)
        env.reset()
        env.episode_length = 1
        root = env.robot.data.root_state_w.clone()
        root[0, 2] = env.scene.env_origins[0, 2] + .3
        env.robot.write_root_state_to_sim(root)
        env.sim.forward()
        env.scene.update(env.step_dt)
        terminal, _, terminated, truncated, _ = env.step(torch.zeros(2, 23, device=env.device))
        assert bool(terminated[0]) and bool(truncated[0]) and bool(truncated[1])
        assert float(terminal["privileged_state"][0, 0]) < .5
        assert int(terminal["step_id"][0]) == 1
        next_obs, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
        assert int(next_obs["step_id"][0]) == 0 and int(next_obs["step_id"][1]) == 1
        assert float(next_obs["privileged_state"][0, 0]) > .5
        receipt = {"schema_version": "eb/rpo-bfm-interface-probe/v1", "status": "passed",
                   "qualification": "not_requested", "dimensions": OBS_DIMS,
                   "body_order": list(BODY_ORDER), "partial_reset_verified": True,
                   "observation_read_is_pure": True, "real_terminal_before_reset": True,
                   "expert_pose_golden": golden}
        output.mkdir(parents=True, exist_ok=True)
        (output / "interface-probe.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["source", "interface"], default="source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path)
    args = parser.parse_args()
    if args.case == "source":
        probe_source(args.output)
    else:
        probe_interface(args.output, args.motion_root)
