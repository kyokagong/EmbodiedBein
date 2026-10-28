"""Every learner rank is an actual RLinf Worker hosting the native Workspace."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import os
from datetime import timedelta
from pathlib import Path

from rlinf.scheduler import Worker

from eb.runners.rlinf_pbfm_runtime import (
    OUTPUT_ROOT,
    PRIVATE_SOURCE,
    STAGING_ROOT,
    MemorySnapshots,
    build_native_config,
    configure_simulation_execution,
    install_simulation_memory_handoff,
    parameter_hash,
    record_cuda_memory,
    resolved_environment_config,
    rng_state_hash,
    simulation_execution_evidence,
    verify_native_state,
    write_json,
)


def verify_environment(env, cfg) -> dict:
    shapes = {
        name: list(space.shape)
        for name, space in env.single_observation_space.spaces.items()
        if name != "time"
    }
    required = {
        "state": [52],
        "last_action": [23],
        "history_actor": [300],
        "privileged_state": [358],
        "heading": [2],
        "velocity_command": [3],
        "depth_image": [8, 18, 32],
        "velocity_target_yaw": [3],
    }
    core = env.base_env
    actor = cfg.agent.model.archi.actor
    if (
        set(shapes) != set(required) | {"terrain_priv"}
        or any(shapes.get(key) != value for key, value in required.items())
        or env.single_action_space.shape != (23,)
        or abs(core.dt - 0.02) > 1e-9
        or float(core.config.simulator.config.sim.fps) != 400
        or int(core.config.simulator.config.sim.control_decimation) != 8
        or list(core.config.robot.dof_names) != list(cfg.env.robot_training["robot"]["control_joint_names"])
        or float(core.config.obs.obs_scales.base_ang_vel) != 0.25
        or float(core.config.robot.control.action_scale) != 0.6
        or not bool(core.config.robot.control.action_rescale)
        or float(core.config.robot.control.normalize_action_from) != 1.0
        or float(core.config.robot.control.normalize_action_to) != 5.0
        or float(core.config.robot.control.action_clip_value) != 5.0
        or not core._ctrl_delay_enabled
        or core._ctrl_delay_min_step != 0
        or core._ctrl_delay_max_step != 2
        or cfg.agent.model.archi.z_dim != 256
        or not cfg.agent.model.velocity_critic_enabled
        or list(actor.input_filter.key)
        != ["state", "last_action", "history_actor", "heading", "velocity_command"]
    ):
        raise ValueError("P-BFM runtime RPO observation/action/physics contract differs")
    from eb.runners.pbfm_source import CAMERA_CONTRACT

    camera = core.direct_depth_config
    if (
        camera.mount_body != CAMERA_CONTRACT["mount_body"]
        or list(camera.mount_pos) != CAMERA_CONTRACT["mount_pos"]
        or list(camera.sampled_ages) != CAMERA_CONTRACT["sampled_ages"]
        or camera.processing_profile != CAMERA_CONTRACT["processing_profile"]
    ):
        raise ValueError("P-BFM effective camera contract differs")
    return {
        "observation_shapes": shapes,
        "actor_scalar_dim": 380,
        "action_dim": 23,
        "z_dim": 256,
        "physics_hz": 400,
        "policy_hz": 50,
        "control_decimation": 8,
        "camera_contract": CAMERA_CONTRACT,
        "last_action_semantics": "normalized_clipped_pre_motor_delay",
        "simulation_execution": simulation_execution_evidence(env, eager=cfg.online_parallel_envs == 2048),
    }


class PbfmTrainWorker(Worker):
    def __init__(self, plan: dict):
        super().__init__()
        self.plan, self.used = plan, False

    def describe_runtime(self) -> dict:
        import torch
        from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        expected = NvidiaGPUManager.get_egl_device_id(visible)
        mapping = {
            "rank": rank, "world_size": world,
            "local_rank": os.environ.get("LOCAL_RANK"),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_visible_devices": visible, "expected_egl_device_id": expected,
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        }
        if expected is None:
            from rlinf.scheduler.hardware.accelerators.nvidia_gpu import _query_egl_index_by_cuda_ordinal

            try:
                mapping["egl_query"] = _query_egl_index_by_cuda_ordinal()
            except Exception as exc:
                mapping["egl_query_error"] = str(exc)[:500]
        if (
            world != self.plan["world_size"]
            or not 0 <= rank < world
            or int(os.environ["LOCAL_RANK"]) != 0
            or torch.cuda.device_count() != 1
            or "," in visible
            or expected is None
            or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(expected)
        ):
            raise RuntimeError("P-BFM RLinf rank/GPU isolation/EGL mapping differs: " + json.dumps(mapping))
        imports = {}
        for name, root in (
            ("pbfm", PRIVATE_SOURCE),
            ("rlinf", Path(self.plan["source_refs"]["rlinf"]["path"])),
        ):
            module = importlib.import_module(name)
            source = Path(module.__file__).resolve()
            if not source.is_relative_to(root.resolve()):
                raise RuntimeError(name + " imported outside the frozen source")
            imports[name] = str(source)
        dependencies = {
            name: importlib.metadata.version(name)
            for name in ("torch", "mjlab", "mujoco", "ray", "wandb", "numpy")
        }
        if (
            os.sys.version_info[:2] != (3, 10)
            or dependencies["torch"] != "2.7.1+cu128"
            or dependencies["mjlab"] != "1.4.0"
            or dependencies["mujoco"] != "3.8.0"
            or dependencies["ray"] != "2.47.0"
        ):
            raise RuntimeError("P-BFM numerical runtime versions differ")
        identity = {
            "rank": rank,
            "world_size": world,
            "local_rank": 0,
            "pid": os.getpid(),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "cuda_visible_devices": visible,
            "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
            "imports": imports,
            "dependencies": dependencies,
            "python": os.sys.version,
            "source_refs": self.plan["source_refs"],
            "worker_class": type(self).__name__,
            "worker_group": self._worker_address.root_group_name,
        }
        write_json(OUTPUT_ROOT / f"outputs/runtime/rank-{rank}.json", identity)
        return identity

    def run_segment(self, args: dict, probe: bool = False, probe_fail_rank: int | None = None) -> dict:
        import torch
        import torch.distributed as dist

        if self.used or dist.is_initialized():
            raise RuntimeError("A P-BFM Worker must own one segment and a new process group")
        self.used = True
        identity = self.describe_runtime()
        rank, world = identity["rank"], identity["world_size"]
        torch.cuda.set_device(0)
        if probe_fail_rank is not None and (not probe or not 0 <= probe_fail_rank < world):
            raise ValueError("Failure injection requires a diagnostic probe rank")
        if probe_fail_rank == rank:
            raise RuntimeError("Injected P-BFM diagnostic rank failure")
        trainer = None
        memory_snapshots = MemorySnapshots() if self.plan.get("long_exploration") else []
        try:
            if world > 1:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=rank,
                    world_size=world,
                    timeout=timedelta(seconds=self.plan.get("communication_timeout_seconds", 300)),
                    device_id=torch.device("cuda:0"),
                )
                value = torch.tensor([rank + 1.0], device="cuda:0")
                dist.all_reduce(value)
                if value.item() != world * (world + 1) / 2:
                    raise RuntimeError("P-BFM NCCL collective failed")
            eager = configure_simulation_execution(self.plan)
            if self.plan.get("recovery"):
                from eb.runners.pbfm_numerics import install_numerics_capture

                install_numerics_capture(rank)
            cfg = build_native_config(args, rank, world)
            write_json(OUTPUT_ROOT / f"outputs/runtime/config-rank-{rank}.json", cfg.model_dump(mode="json"))
            if probe:
                return self._probe(cfg, identity)
            record_cuda_memory(rank, "before_build", memory_snapshots)
            trainer = cfg.build()
            record_cuda_memory(rank, "after_build", memory_snapshots)
            contract = verify_environment(trainer.train_env, cfg)
            execution = {
                "main": contract["simulation_execution"], "sampling_cache_handoff": eager and world > 1,
            }
            write_json(OUTPUT_ROOT / f"outputs/runtime/execution-rank-{rank}.json", execution)
            if trainer.agent.cfg.model_dump() != cfg.agent.model_dump():
                raise ValueError("P-BFM loaded agent effective configuration differs from the frozen plan")
            if rank == 0:
                write_json(STAGING_ROOT / "effective_config.json", cfg.model_dump(mode="json"))
                write_json(
                    STAGING_ROOT / "effective-environment.json",
                    resolved_environment_config(trainer.train_env.base_env.config),
                )
            before = parameter_hash(trainer.agent._model)
            if self.plan.get("long_exploration"):
                from eb.runners.pbfm_longrun import verify_resume_start

                start_evidence = verify_resume_start(trainer.agent, self.plan, before)
                write_json(OUTPUT_ROOT / f"outputs/runtime/start-rank-{rank}.json",
                           {**identity, **start_evidence})
            rng_before = rng_state_hash()
            torch.cuda.synchronize()
            record_cuda_memory(rank, "before_cache_cleanup", memory_snapshots,
                               parameter_sha256=before, rng_sha256=rng_before)
            # QR/model/DDP initialization can leave unused Torch cache that Warp's
            # first graph executable cannot reclaim through its own allocator.
            gc.collect()
            torch.cuda.empty_cache()
            cleaned_hash, cleaned_rng = parameter_hash(trainer.agent._model), rng_state_hash()
            record_cuda_memory(rank, "after_cache_cleanup", memory_snapshots,
                               parameter_sha256=cleaned_hash, rng_sha256=cleaned_rng)
            if cleaned_hash != before or cleaned_rng != rng_before:
                raise ValueError("P-BFM initialization cache cleanup changed parameters or RNG state")
            if execution["sampling_cache_handoff"]:
                install_simulation_memory_handoff(trainer, rank, memory_snapshots, execution)
            if self.plan.get("long_exploration"):
                from eb.runners.pbfm_longrun import install_periodic_checkpoints

                install_periodic_checkpoints(trainer, self.plan, identity)
            trainer.train()
            record_cuda_memory(rank, "after_train", memory_snapshots)
            checkpoint = STAGING_ROOT / "checkpoint"
            status = json.loads((checkpoint / "train_status.json").read_text())
            counts = {key: status[key] for key in ("local_time", "global_time", "optimizer_steps")}
            if (
                counts["optimizer_steps"] != self.plan["target_updates"]
                or counts["global_time"] != args["num_env_steps"]
            ):
                raise ValueError("P-BFM native segment endpoint differs")
            verify_native_state(trainer.agent, expected_updates=counts["optimizer_steps"])
            after = parameter_hash(trainer.agent._model)
            if before == after:
                raise ValueError("P-BFM trainable parameters did not change")
            trainer.train_env.close()
            del trainer
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
            loaded = cfg.agent.object_class.load(checkpoint, device="cpu")
            state_evidence = verify_native_state(
                loaded, checkpoint, expected_updates=counts["optimizer_steps"]
            )
            del loaded
            from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim

            replay_path = checkpoint / "buffers" / ("train" if world == 1 else f"train_rank_{rank}")
            replay = TrajectoryDictBufferMultiDim.load_for_training(replay_path, device="cpu")
            if replay.empty():
                raise ValueError("P-BFM native Replay loader returned an empty buffer")
            replay._validate_sparse_storage()
            replay_rows = len(replay)
            del replay
            result = {
                **identity,
                **counts,
                **state_evidence,
                "status": "succeeded",
                "loader_verified": True,
                "replay_loader_verified": True,
                "full_state_finite_verified": True,
                "parameter_change": {"before_sha256": before, "after_sha256": after},
                "native_contract": contract,
                "seed": cfg.seed,
                "replay_rows": replay_rows,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            write_json(OUTPUT_ROOT / f"outputs/runtime/result-rank-{rank}.json", result)
            return result
        except BaseException as exc:
            record_cuda_memory(rank, "failed", memory_snapshots,
                               error=type(exc).__name__ + ": " + str(exc)[:500])
            raise
        finally:
            if trainer is not None:
                trainer.train_env.close()
            if world > 1 and dist.is_initialized():
                dist.destroy_process_group()

    def _probe(self, cfg, identity: dict) -> dict:
        import numpy as np
        import torch
        from pbfm.inference.runtime import MujocoQposRenderer, _assign_training_tile, policy_qpos_from_env
        from pbfm.perception.instinct_direct_depth import preprocess_depth, raycast_ranges_to_image_plane
        from PIL import Image

        from eb.runners.pbfm_depth_probe import (
            SAMPLE_PIXELS,
            SCHEMA,
            TOLERANCE,
            scene_ray_comparison,
            validate_depth_report,
        )

        env, _ = cfg.env.build(num_envs=2)
        renderer = None
        try:
            contract = verify_environment(env, cfg)
            rank = identity["rank"]
            folder = OUTPUT_ROOT / f"outputs/runtime/probe-rank-{rank}"
            folder.mkdir(parents=True)
            core = env.base_env
            sensor = core.mjlab_env.scene.sensors["g1_direct_depth"]
            snapshots = {}
            if tuple(sensor.cfg.include_geom_groups) != (0, 2, 5) or sensor.cfg.exclude_parent_body:
                raise ValueError("P-BFM depth geometry groups or body exclusion differs")

            def array(value):
                return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value)

            def statistics(value):
                finite = value[np.isfinite(value)]
                return {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "finite_count": int(finite.size), "negative_count": int((finite < 0).sum()),
                    "min": float(finite.min()) if finite.size else None,
                    "max": float(finite.max()) if finite.size else None,
                    "unique_count": int(np.unique(value).size),
                }

            def capture(observation, prefix):
                qpos, qvel = env._get_qpos_qvel(to_numpy=True)
                values = {
                    "network-depth": observation["depth_image"], "sensor-ranges": sensor.data.distances,
                    "metric-depth": raycast_ranges_to_image_plane(
                        sensor.data.distances, core.direct_depth_config.camera_config(), invalid_to_max=True
                    ),
                    "qpos": qpos, "qvel": qvel,
                    "depth-delay-frames": core._direct_depth_runtime.delay_frames,
                }
                for name in (
                    "robot_root_states", "base_quat", "projected_gravity", "motion_ids", "motion_start_times"
                ):
                    value = getattr(core, name, None)
                    if value is not None:
                        values[name.replace("_", "-")] = value
                for name in ("pos_w", "quat_w", "ray_starts_w", "ray_directions_w"):
                    value = getattr(sensor.data, name, None)
                    if value is not None:
                        values["camera-" + name.replace("_", "-")] = value
                for name in ("_cached_world_origins", "_cached_world_rays"):
                    value = getattr(sensor, name, None)
                    if value is not None:
                        values["sensor" + name.replace("_", "-")] = value
                simulation = getattr(core.mjlab_env, "sim", None)
                if simulation is not None:
                    values["simulation-qpos"] = simulation.data.qpos
                values = {name: array(value) for name, value in values.items()}
                snapshots[prefix] = values
                for name, value in values.items():
                    np.save(folder / f"{prefix}{name}.npy", value)
                return {
                    name.replace("-", "_"): statistics(values[name])
                    for name in ("network-depth", "sensor-ranges", "metric-depth")
                }

            def depth_comparison(prefix, *, previous_frame=None, done=None, positive=False):
                values = snapshots[prefix]
                lanes = scene_ray_comparison(
                    core.mjlab_env.sim.mj_model, values["simulation-qpos"],
                    values["sensor-cached-world-origins"], values["sensor-cached-world-rays"],
                    values["sensor-ranges"], sensor.cfg.max_distance, positive=positive,
                )
                metric = raycast_ranges_to_image_plane(
                    torch.from_numpy(values["sensor-ranges"]).to(core.device),
                    core.direct_depth_config.camera_config(), invalid_to_max=True,
                )
                np.testing.assert_allclose(array(metric), values["metric-depth"], atol=2e-5, rtol=2e-5)
                frame = array(preprocess_depth(metric, core.direct_depth_config, enable_noise=False))
                delays = values["depth-delay-frames"].tolist()
                expected = np.repeat(frame[:, None], 8, axis=1)
                if previous_frame is not None:
                    for lane in range(2):
                        if not done[lane]:
                            for index, age in enumerate(core.direct_depth_config.sampled_ages):
                                if age + delays[lane] > 0:
                                    expected[lane, index] = previous_frame[lane]
                np.save(folder / f"{prefix}reference-network-depth.npy", expected)
                network = values["network-depth"]
                if network.dtype != np.uint8 or network.shape != (2, 8, 18, 32):
                    raise ValueError("P-BFM real depth input is missing or malformed")
                error = int(np.abs(network.astype(np.int16) - expected.astype(np.int16)).max())
                return {
                    "lanes": lanes,
                    "history": {
                        "shape": list(network.shape), "dtype": str(network.dtype),
                        "history_length": core.direct_depth_config.history_length,
                        "sampled_ages": list(core.direct_depth_config.sampled_ages),
                        "delay_frames": delays, "max_abs_error": error,
                    },
                }, frame

            stats = {
                "rank": rank,
                "seed": {
                    "cfg_seed": cfg.seed, "environment_seed": cfg.env.seed,
                    "mjlab_seed": core.mjlab_env.cfg.seed, "torch_initial_seed": torch.initial_seed(),
                    "worker_explicit_seeded": False,
                },
            }
            obs, _ = env.reset(to_numpy=False)
            stats["reset"] = capture(obs, "reset-")
            write_json(folder / "probe-stats.json", stats)
            obs, _, terminated, truncated, _ = env.step(torch.zeros((2, 23), device="cuda:0"), to_numpy=False)
            depth = obs["depth_image"]
            stats["post_step"] = capture(obs, "")
            stats.update(terminated=array(terminated).tolist(), truncated=array(truncated).tolist())
            write_json(folder / "probe-stats.json", stats)
            if depth.dtype == torch.uint8 and depth.ndim == 4 and depth.shape[0] and depth.shape[1]:
                Image.fromarray(depth[0, -1].cpu().numpy()).save(folder / "network-depth.png")
            # The training MJLab wrapper has no RGB render mode. Use the same
            # scene/qpos renderer as independent evaluation, including terrain.
            try:
                renderer = MujocoQposRenderer(
                    None, render_size=360, scene_spec=core.mjlab_env.scene.spec,
                    source_xml_path=Path(cfg.env.mjcf_path), add_floor=False, expected_qpos_size=30,
                )
                image = renderer.render_qpos(policy_qpos_from_env(env, expected_qpos_size=30))
                if image is None or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("P-BFM EGL robot/terrain render is missing")
                Image.fromarray(image).save(folder / "robot-terrain.png")
                stats["rgb"] = {"verified": True}
            except Exception as error:
                stats["rgb"] = {"verified": False, "error_type": type(error).__name__, "error": str(error)}
                raise
            finally:
                write_json(folder / "probe-stats.json", stats)
            report = {
                "schema_version": SCHEMA, "rank": rank, "tolerance": TOLERANCE,
                "sample_pixels": list(SAMPLE_PIXELS), "include_geom_groups": [0, 2, 5], "bodyexclude": -1,
                "range_limit_m": float(sensor.cfg.max_distance), "phases": {},
            }
            reset_comparison, reset_frame = depth_comparison("reset-")
            report["phases"]["random_reset"] = reset_comparison
            report["phases"]["random_step"], _ = depth_comparison(
                "", previous_frame=reset_frame,
                done=np.logical_or(array(terminated), array(truncated)).reshape(-1),
            )
            _assign_training_tile(core, family="flat", difficulty_row=0)
            root_states = torch.zeros((2, 13), device=core.device)
            root_states[:, :3] = core.env_origins
            root_states[:, 2] += 0.75
            root_states[:, 6] = 1.0
            dof_states = torch.zeros((2, 23, 2), device=core.device)
            dof_states[..., 0] = core.default_dof_pos
            obs, _ = env.reset(
                to_numpy=False, target_states={"root_states": root_states, "dof_states": dof_states}
            )
            stats["positive_control"] = capture(obs, "positive-")
            write_json(folder / "probe-stats.json", stats)
            Image.fromarray(array(obs["depth_image"])[0, -1]).save(folder / "positive-network-depth.png")
            image = renderer.render_qpos(policy_qpos_from_env(env, expected_qpos_size=30))
            if image is None or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("P-BFM positive control robot/terrain RGB render is missing")
            Image.fromarray(image).save(folder / "positive-robot-terrain.png")
            report["phases"]["positive_control"], _ = depth_comparison("positive-", positive=True)
            write_json(folder / "depth-validation.json", report)
            validate_depth_report(report)
            contract["simulation_execution"] = simulation_execution_evidence(
                env, eager=cfg.online_parallel_envs == 2048,
            )
            return {
                **identity,
                "status": "probe_complete",
                "environment_reset_step": True,
                "egl_render_verified": True,
                "depth_verified": True,
                "depth_validation": report,
                "native_contract": contract,
                "process_group": identity["world_size"] > 1,
                "training_completed": False,
                "evidence_dir": str(folder.relative_to(OUTPUT_ROOT / "outputs")),
            }
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
