"""CPU contract probe; GPU reset/step/depth must use asset.validate through ebd.

Run inside the exact target image with /sources/pbfm and /robot-assets read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from eb.runners.pbfm_source import CAMERA_CONTRACT, TERRAIN_CONTRACT, prepare_pbfm_source, rpo_robot_config


def main():
    destination = Path("/tmp/pbfm-cpu-probe")
    adaptation = prepare_pbfm_source(Path("/sources/pbfm"), destination)
    sys.path.insert(0, str(destination))
    import mujoco
    import torch
    from pbfm.envs.mjlab.config import _action_target_scale, make_mjlab_ufo_env_cfg
    from pbfm.envs.mjlab.task_config import _compose_humanoidverse_config
    from pbfm.robots.training_spec import load_robot_training_spec, validate_g1_robot
    from pbfm.terrains.presets import TRAINING_TERRAIN_COMPONENT_NAMES

    from eb.runners.rlinf_pbfm_runtime import build_native_config

    robot = load_robot_training_spec(rpo_robot_config(destination))
    assert robot.robot.name == "rpo_23dof" and len(robot.robot.control_joint_names) == 23
    assert len(robot.robot.body_names) == 24 and len(robot.robot.feet) == 2
    geometry = robot.to_env_dict()["foot_geometry"]
    assert geometry["collision_geom_names"] == ["left_foot_rpo_collision", "right_foot_rpo_collision"]
    assert len(geometry["edge_points"]) == 100
    assert geometry["sole_grid_size"] != [0.23, 0.04]
    assert geometry["sole_center"] != [0.025, 0.0, -0.045]
    assert geometry["bound_strategy"] == "union_of_collision_mesh_aabbs_in_ankle_body_frames"
    validate_g1_robot("g1_29dof")
    try:
        validate_g1_robot("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown robot accepted")
    model = mujoco.MjModel.from_xml_path(robot.robot.xml_path)
    data = mujoco.MjData(model)
    for joint, value in robot.default_joint_angles.items():
        data.qpos[robot.robot.joint_qpos_addr[joint]] = value
    mujoco.mj_forward(model, data)
    assert torch.isfinite(torch.tensor(data.xpos)).all()
    args = {
        "work_dir": "/tmp/pbfm-no-training",
        "num_envs": 1024,
        "num_env_steps": 18432,
        "seed": 42,
        "checkpoint_every_steps": 18432,
        "data_path": "/training-data/train.pkl",
        "buffer_size": 32768,
        "robot_config": str(rpo_robot_config(destination)),
    }
    cfg = build_native_config(args, 0, 1)
    rank0, rank1 = build_native_config(args, 0, 2), build_native_config(args, 1, 2)
    assert rank0.seed == 42 and rank0.env.seed == 42
    assert rank1.seed == 43 and rank1.env.seed == 43
    assert rank1.distributed_gradient_sync == "ddp"
    assert cfg.prior_plane_envs == 0 and cfg.checkpoint_buffer and cfg.checkpoint_rank_buffers
    assert (
        cfg.env.terrain_curriculum_start_global_step == 0 and cfg.env.terrain_curriculum_interval == 19200000
    )
    assert list(TRAINING_TERRAIN_COMPONENT_NAMES) == TERRAIN_CONTRACT["families"]
    hv, _ = _compose_humanoidverse_config(
        num_envs=2,
        relative_config_path=cfg.env.relative_config_path,
        hydra_overrides=cfg.env.hydra_overrides,
        lafan_tail_path="/training-data/train.pkl",
        data_mix_weights=None,
        disable_obs_noise=False,
        disable_domain_randomization=False,
        max_episode_length_s=None,
        root_height_obs=cfg.env.root_height_obs,
        robot_training=cfg.env.robot_training,
        direct_depth_profile=cfg.env.direct_depth_profile,
    )
    assert hv.simulator.config.sim.fps == 400 and hv.simulator.config.sim.control_decimation == 8
    assert list(hv.robot.dof_names) == robot.robot.control_joint_names
    assert hv.robot.motion.nums_extend_bodies == 0 and len(hv.robot.motion.body_names) == 24
    assert hv.obs.obs_scales.base_ang_vel == 0.25
    assert cfg.env.root_height_obs is True and hv.obs.root_height_obs is True
    from pbfm.envs.mjlab.history import HistoryHandler
    from pbfm.envs.mjlab.observations import get_observation
    from pbfm.motion.observations import compute_humanoid_observations_max

    body_pos = torch.tensor(data.xpos[1:], dtype=torch.float32)[None]
    body_rot = torch.tensor(data.xquat[1:, [1, 2, 3, 0]], dtype=torch.float32)[None]
    privileged = compute_humanoid_observations_max(
        body_pos,
        body_rot,
        torch.zeros_like(body_pos),
        torch.zeros_like(body_pos),
        True,
        True,
    )
    assert torch.cat(list(privileged.values()), dim=-1).shape == (1, 358)
    defaults = torch.tensor(list(robot.default_joint_angles.values()))[None]
    core = SimpleNamespace(
        config=hv,
        device="cpu",
        is_evaluating=True,
        add_noise_currculum=False,
        _creation_config=cfg.env.model_copy(update={"evaluation_fast_path": True}),
        dof_pos=defaults,
        default_dof_pos=defaults.clone(),
        default_dof_pos_offset=torch.zeros_like(defaults),
        dof_vel=torch.zeros_like(defaults),
        actions=torch.zeros_like(defaults),
        base_ang_vel=torch.zeros(1, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        history_handler=HistoryHandler(1, hv.obs.obs_auxiliary, hv.obs.obs_dims, "cpu"),
        episode_length_buf=torch.zeros(1),
        robot_root_states=torch.zeros(1, 13),
        base_quat=body_rot[:, 0],
        terrain_enabled=False,
    )
    actor_obs = get_observation(core, to_numpy=False)
    assert actor_obs["state"].shape == (1, 52) and actor_obs["history_actor"].shape == (1, 300)
    actor_scalars = torch.cat(
        [actor_obs[key] for key in ("state", "last_action", "history_actor", "heading", "velocity_command")],
        dim=-1,
    )
    assert actor_scalars.shape == (1, 380)
    assert list(hv.terrain.direct_depth.mount_pos) == CAMERA_CONTRACT["mount_pos"]
    expected = torch.tensor(
        [0.6 * e / robot.stiffness[j] for e, j in zip(robot.effort_limits, robot.robot.control_joint_names)]
    )
    torch.testing.assert_close(_action_target_scale(hv), expected)
    simulation = make_mjlab_ufo_env_cfg(
        hv,
        num_envs=2,
        seed=42,
        mjcf_path=robot.robot.xml_path,
        auto_reset=False,
        robot_training=cfg.env.robot_training,
    )
    sensors = {sensor.name: sensor for sensor in simulation.scene.sensors}
    assert list(sensors["foot_collision_contacts"].primary.pattern) == geometry["collision_geom_names"]
    assert list(sensors["foot_height_scanners"].local_center) == geometry["sole_center"]
    print(
        json.dumps(
            {
                "status": "complete",
                "scope": "cpu_contract_only",
                "training_completed": False,
                "joint_count": 23,
                "body_count": 24,
                "actor_scalar_dim": 380,
                "privileged_fk_dim": 358,
                "fk_finite": True,
                "g1_identity_accepted": True,
                "camera_contract": CAMERA_CONTRACT,
                "terrain_contract": TERRAIN_CONTRACT,
                "patch_sha256": adaptation["patch_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
