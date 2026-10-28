"""Capture the first invalid terrain/physics frame without changing termination or data."""

from eb.runners.rlinf_pbfm_runtime import OUTPUT_ROOT, write_json


def install_numerics_capture(rank):
    import torch
    from pbfm.envs.mjlab import observations

    original = observations._terrain_observations

    def observed(core):
        try:
            return original(core)
        except RuntimeError as error:
            if "terrain sensor observations contain non-finite" not in str(error):
                raise
            import numpy as np

            sensor = core.mjlab_env.scene.sensors["terrain_height"]
            heights = sensor.data.heights
            bad = (~torch.isfinite(heights)).reshape(core.num_envs, -1).any(-1)
            ids = bad.nonzero().flatten()[:16]
            tensors = {"heights": heights, "root_states": core.robot_root_states,
                       "body_pos": core.body_pos, "dof_pos": core.dof_pos, "dof_vel": core.dof_vel}
            for name in ("_cached_world_origins", "_cached_world_rays", "_cached_frame_pos",
                         "_cached_frame_mat", "_distances", "_hit_pos_w"):
                value = getattr(sensor, name, None)
                if value is not None:
                    tensors[name] = value
            for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl"):
                value = getattr(core.mjlab_env.sim.data, name, None)
                if value is not None:
                    tensors["physics_"+name] = value
            arrays = {name: value[ids].detach().cpu().numpy() for name,value in tensors.items()}
            directory = OUTPUT_ROOT / "outputs/numerics" / f"rank-{rank}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / "invalid-frame.npz", env_ids=ids.cpu().numpy(), **arrays)
            import mujoco

            mujoco.mj_saveModel(core.mjlab_env.sim.mj_model, str(directory / "scene.mjb"), None)
            write_json(directory / "invalid-frame.json", {
                "rank": rank, "error": str(error), "env_ids": ids.cpu().tolist(),
                "global_env_steps": int(getattr(core, "_training_terrain_global_step", -1)),
                "nonfinite_entries": {name:int((~np.isfinite(value)).sum()) for name,value in arrays.items()},
                "termination_changed": False, "observation_sanitized": False,
            })
            raise

    observations._terrain_observations = observed
