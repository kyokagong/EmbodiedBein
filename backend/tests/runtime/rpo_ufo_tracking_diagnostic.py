"""Fixed held-out policy diagnostic; launched exclusively by a registered Job."""

import json
import math
import os
from contextlib import ExitStack
from pathlib import Path


def main():
    import traceback

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    env = None
    try:
        print("tracking: app ready", flush=True)
        import gymnasium as gym
        import imageio.v2 as imageio
        import numpy as np
        import rpo_flat_artifact as flat
        import torch
        from bfm_ufo_agent import OBS_DIMS, to_ufo_observation
        from bfm_ufo_artifact import load_ufo_export_policy, native_checkpoint_path
        from bfm_ufo_tracking import SUITE, tracking_action, tracking_cases, tracking_latents
        from rpo_bfm_extension import RPOBFMEnvCfg
        from rpo_bfm_parallel_eval import usd_camera_type, video_integrity
        from rpo_ufo_isaac import GYM_ID, register_task, write_reference_state
        from rpo_ufo_motion import digest, load_robot_state, read_arrays

        root, checkpoint, output = Path("/workspace/motion"), Path("/workspace/checkpoint"), Path("/outputs")
        plan = json.loads(os.environ["EB_UFO_TRACKING_PLAN"])
        manifest, _ = load_robot_state(root, asset_sha256=plan["asset_sha256"])
        clips = [{**clip, "arrays": read_arrays(root, clip)} for clip in manifest["clips"]]
        cases = tracking_cases(clips)
        print("tracking: eligible references loaded", flush=True)
        if [clip["id"] for clip, _ in cases] != plan["clip_ids"] or digest(root / "manifest.json") != plan[
            "motion_manifest_sha256"
        ]:
            raise ValueError("Frozen held-out references changed")
        policy, weights, _ = load_ufo_export_policy(
            checkpoint, checkpoint / "resolved-config.yaml", root, plan["asset_sha256"]
        )
        if digest(weights) != plan["weights_sha256"]:
            raise ValueError("Frozen diagnostic weights changed")
        from rpo_bfm_eval import configure_action_control
        execution = json.loads((native_checkpoint_path(checkpoint) / "resume-manifest.json").read_text())["execution"]
        configure_action_control({"execution": execution, "action_control": plan.get("action_control"),
                                  "auxiliary_profile": plan.get("auxiliary_profile")})
        model = policy.model.to("cuda")
        print("tracking: complete checkpoint loaded", flush=True)
        prompts = [tracking_latents(model, clip["arrays"], start) for clip, start in cases]
        register_task()
        cfg = RPOBFMEnvCfg()
        cfg.seed, cfg.scene.num_envs = 63000, 2
        flat._add_camera(cfg, "standard")
        cfg.scene.eval_camera.prim_path = "{ENV_REGEX_NS}/EvalCamera"
        cfg.scene.eval_camera.class_type = usd_camera_type()
        env = gym.make(GYM_ID, cfg=cfg).unwrapped
        print("tracking: physical scene created", flush=True)
        env.reset()
        arrays = [clip["arrays"] for clip, _ in cases]
        starts = [start for _, start in cases]
        fields = {
            key: torch.as_tensor(np.stack([a[key][s] for a, s in zip(arrays, starts)]), device=env.device)
            for key in ("root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_w", "q", "dq")
        }
        root_state = torch.cat(
            [fields[key] for key in ("root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_w")], -1
        )
        ids = torch.arange(2, device=env.device)
        write_reference_state(env.robot, root_state, fields["q"], fields["dq"], ids, env.scene.env_origins)
        env.bfm_history.reset(ids)
        env.bfm_last_action.zero_()
        env.bfm_previous_action.zero_()
        env.bfm_step_id.zero_()
        env.ufo_cache.invalidate()
        env.sim.forward()
        env.scene.update(0)
        if env.bfm_slew:
            env._reset_slew_anchor(ids)
        obs = env._get_observations()
        torch.testing.assert_close(env.robot.data.joint_pos, fields["q"], atol=1e-5, rtol=1e-5)
        limits = flat._hard_limits(env)
        rows, active, errors = [], [True, True], [[], []]
        output.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            writers, traces = [], []
            frames = flat._video_frames(env)
            for i, ((clip, start), seed) in enumerate(zip(cases, SUITE["case_ids"])):
                video = output / f"tracking-{seed}.mp4"
                writer = imageio.get_writer(video, fps=10, codec="libx264", macro_block_size=1)
                stack.callback(writer.close)
                writer.append_data(frames[i])
                writers.append(writer)
                trace = stack.enter_context((output / f"tracking-{seed}.jsonl").open("w"))
                traces.append(trace)
                rows.append(
                    {
                        "clip_id": clip["id"],
                        "source": clip["source"],
                        "split": clip["split"],
                        "start": start,
                        "case_id": seed,
                        "scene_seed": SUITE["scene_seed"],
                        "steps": 0,
                        "reason": "reference_end",
                        "video": video.name,
                        "trace": f"tracking-{seed}.jsonl",
                        "initial": flat._initial_state(env, SUITE["scene_seed"], limits, i),
                    }
                )
            for step in range(249):
                if not any(active):
                    break
                native = {key: value for key, value in to_ufo_observation(obs).items() if key in OBS_DIMS}
                z = torch.stack([prompt[step] for prompt in prompts])
                action = tracking_action(model, native, z, active)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = None
                for i in range(2):
                    if not active[i]:
                        continue
                    state = flat._state(env, i)
                    if not all(np.isfinite(value).all() for value in state.values()):
                        raise ValueError("Nonfinite tracking physics")
                    error = float(
                        np.sqrt(
                            np.mean(
                                (
                                    flat._numpy(env.robot.data.joint_pos[i])
                                    - arrays[i]["q"][starts[i] + step + 1]
                                )
                                ** 2
                            )
                        )
                    )
                    errors[i].append(error)
                    fall = bool(
                        terminated[i]
                        or state["root_pos_w"][2] - float(env.scene.env_origins[i, 2]) < 0.5
                        or -state["projected_gravity_b"][2] < math.cos(0.8)
                    )
                    hard_limit = flat._violates_hard_limit(state, limits)
                    stop = fall or hard_limit or bool(truncated[i]) or step == 248
                    rows[i]["steps"] += 1
                    traces[i].write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "q_rmse_rad": error,
                                "action": flat._trace_array(action[i].cpu()),
                                **({"applied_action": flat._trace_array(env.bfm_last_action[i].cpu())}
                                   if env.bfm_slew else {}),
                                "z": flat._trace_array(z[i].cpu()),
                                "fall": fall,
                                "hard_limit": hard_limit,
                                **{key: flat._trace_array(value) for key, value in state.items()},
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    if stop or (step + 1) % 5 == 0:
                        if frame is None:
                            frame = flat._video_frames(env)
                        writers[i].append_data(frame[i])
                    if stop:
                        rows[i]["reason"] = (
                            "hard_limit"
                            if hard_limit
                            else "fall"
                            if fall
                            else "early_timeout"
                            if bool(truncated[i])
                            else "reference_end"
                        )
                        active[i] = False
        for i, row in enumerate(rows):
            row.update(
                q_rmse_rad=float(np.sqrt(np.mean(np.square(errors[i])))),
                video_integrity=video_integrity(output / row["video"], "standard"),
                trace_sha256=digest(output / row["trace"]),
            )
            if row["video_integrity"]["decoded_frames"] != 1 + math.ceil(row["steps"] / 5):
                raise ValueError("Tracking video does not cover physical steps")
        receipt = {
            "schema_version": "eb/ufo-heldout-tracking/v1",
            "suite": SUITE,
            "plan": plan,
            "cases": rows,
            "official_rpo_heldout": "unavailable_in_frozen_parent_splits",
            "qualification": "not_requested",
        }
        (output / "tracking.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
        print(
            json.dumps(
                {
                    "cases": [
                        {key: row[key] for key in ("clip_id", "steps", "reason", "q_rmse_rad")}
                        for row in rows
                    ]
                }
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
