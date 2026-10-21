"""RLinf owns the rank; unmodified UFO owns every training operation."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import os
from datetime import timedelta
from pathlib import Path

from rlinf.scheduler import Worker

from eb.runners.rlinf_ufo_runtime import (
    NATIVE_ROOT,
    build_native_config,
    parameter_change,
    parameter_snapshot,
    verify_full_native_state_finite,
    write_json,
)


class UfoTrainWorker(Worker):
    def __init__(self, plan: dict):
        super().__init__()
        self.plan = plan
        self.used = False

    def describe_runtime(self) -> dict:
        import torch
        from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if world != self.plan["world_size"] or not 0 <= rank < world:
            raise RuntimeError("RLinf training rank mismatch")
        if int(os.environ["LOCAL_RANK"]) != 0 or torch.cuda.device_count() != 1 or "," in visible:
            raise RuntimeError("Worker must own exactly one isolated GPU")
        expected_egl = NvidiaGPUManager.get_egl_device_id(visible)
        egl = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        if expected_egl is None or egl != str(expected_egl):
            raise RuntimeError("RLinf EGL mapping is unavailable or mismatched")
        imports = {}
        for name in ("ufo", "rlinf"):
            module = importlib.import_module(name)
            source = Path(module.__file__).resolve()
            root = Path(self.plan["source_refs"][name]["path"]).resolve()
            if not source.is_relative_to(root):
                raise RuntimeError(f"{name} imported outside frozen source")
            imports[name] = str(source)
        identity = {
            "rank": rank,
            "world_size": world,
            "local_rank": 0,
            "pid": os.getpid(),
            "cuda_visible_devices": visible,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "mujoco_egl_device_id": egl,
            "imports": imports,
            "source_refs": self.plan["source_refs"],
            "python": os.sys.version,
            "worker_class": type(self).__name__,
            "worker_group": self._worker_address.root_group_name,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("torch", "ray", "numpy", "wandb", "mujoco", "mjlab", "nvidia-nccl-cu12")
            },
        }
        write_json(NATIVE_ROOT.parent / f"runtime/rank-{rank}.json", identity)
        return identity

    def run_segment(self, args: dict, probe: bool = False, probe_fail_rank: int | None = None) -> dict:
        import torch
        import torch.distributed as dist

        if self.used:
            raise RuntimeError("A training Worker can execute only one segment")
        self.used = True
        identity = self.describe_runtime()
        rank, world = identity["rank"], identity["world_size"]
        if dist.is_initialized():
            raise RuntimeError("A training default process group already exists")
        if probe_fail_rank is not None:
            if not probe:
                raise ValueError("Failure injection is restricted to diagnostic probes")
            if rank == probe_fail_rank:
                raise RuntimeError(f"Injected A0 failure on diagnostic rank {rank}")
        torch.cuda.set_device(0)
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
            cfg = build_native_config(args, rank, world)
            write_json(NATIVE_ROOT.parent / f"runtime/config-rank-{rank}.json", cfg.model_dump(mode="json"))
            if probe:
                if world > 1:
                    value = torch.tensor([rank + 1.0], device="cuda:0")
                    dist.all_reduce(value)
                    if value.item() != world * (world + 1) / 2:
                        raise RuntimeError("NCCL probe failed")
                env, _ = cfg.env.build(num_envs=2)
                try:
                    env.reset(to_numpy=False)
                    env.step(torch.zeros((2, env.single_action_space.shape[0]), device="cuda:0"))
                finally:
                    env.close()
                import mujoco
                from PIL import Image

                model = mujoco.MjModel.from_xml_path(cfg.env.mjcf_path)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                with mujoco.Renderer(model, height=64, width=64) as renderer:
                    renderer.update_scene(data)
                    frame = renderer.render()
                if frame.shape != (64, 64, 3):
                    raise RuntimeError("EGL renderer returned an invalid frame")
                Image.fromarray(frame).save(NATIVE_ROOT.parent / f"runtime/egl-rank-{rank}.png")
                return {
                    **identity,
                    "status": "probe_complete",
                    "seed": cfg.seed,
                    "process_group": world > 1,
                    "environment_reset_step": True,
                    "egl_render_verified": True,
                }
            trainer = cfg.build()
            spaces = {name: list(space.shape) for name, space in trainer.obs_space.spaces.items()}
            simulation = trainer.train_env_info["mjlab_env_cfg"]
            if (
                spaces
                != {"state": [52], "last_action": [23], "history_actor": [300], "privileged_state": [358]}
                or trainer.action_dim != 23
                or trainer.agent.cfg.model.archi.z_dim != 256
                or abs(trainer.train_env.base_env.dt - 1 / 50) > 1e-9
                or abs(simulation.sim.mujoco.timestep - 1 / 400) > 1e-9
                or simulation.decimation != 8
            ):
                raise RuntimeError("Native RPO observation/action/control contract differs")
            contract = {
                "observation_shapes": spaces,
                "actor_observation_dim": 375,
                "privileged_observation_dim": 358,
                "z_dim": 256,
                "action_dim": 23,
                "physics_hz": 400,
                "policy_hz": 50,
                "control_decimation": 8,
            }
            before = parameter_snapshot(trainer.agent._model)
            write_json(NATIVE_ROOT.parent / f"runtime/initial-parameters-rank-{rank}.json", before[0])
            trainer.train()
            checkpoint = NATIVE_ROOT / "checkpoint"
            status = json.loads((checkpoint / "train_status.json").read_text())
            if status["optimizer_steps"] != self.plan["target_updates"]:
                raise RuntimeError(f"Optimizer endpoint mismatch: {status['optimizer_steps']}")
            if status["global_time"] != args["num_env_steps"] or (checkpoint / ".saving").exists():
                raise RuntimeError("Incomplete or misaligned segment endpoint")
            counts = {key: status[key] for key in ("local_time", "global_time", "optimizer_steps")}
            verify_full_native_state_finite(trainer.agent, rank=rank, **counts)
            change = parameter_change(before, parameter_snapshot(trainer.agent._model))
            # Native loaders validate the serialized model, optimizers and rank Replay.
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            loaded = cfg.agent.object_class.load(checkpoint, device="cpu")
            verify_full_native_state_finite(loaded, rank=rank, **counts)
            del loaded
            from ufo.agents.common.buffers.trajectory import ReplayBuffer

            replay = ReplayBuffer.load(checkpoint / "buffers" / f"train_rank_{rank}", device="cpu")
            if not len(replay):
                raise RuntimeError("Empty native Replay checkpoint")
            del replay
            result = {
                **identity,
                **counts,
                "status": "succeeded",
                "loader_verified": True,
                "replay_loader_verified": True,
                "full_state_finite_verified": True,
                "parameter_change": change,
                "native_contract": contract,
                "seed": cfg.seed,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            write_json(NATIVE_ROOT.parent / f"runtime/result-rank-{rank}.json", result)
            return result
        finally:
            if world > 1 and dist.is_initialized():
                # No barrier here: failed peers cannot take part in cleanup collectives.
                dist.destroy_process_group()
