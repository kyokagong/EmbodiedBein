"""Task-image CPU check: execute the real Workspace loop with cheap I/O doubles.

This checks native scheduling/configuration only; it does not exercise GPU,
gradient synchronization, model updates or checkpoint tensor serialization.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


def native_config(root, *, num_envs, world, endpoint):
    from pbfm.perception.instinct_direct_depth import DirectDepthConfig
    from pbfm.training.config import _effective_batch_size

    from eb.runners.pbfm_source import rpo_robot_config
    from eb.runners.rlinf_pbfm_runtime import build_native_config

    cfg = build_native_config({
        "work_dir": str(root), "num_envs": num_envs, "num_env_steps": endpoint,
        "seed": 42, "checkpoint_every_steps": endpoint, "data_path": "/training-data/train.pkl",
        "buffer_size": 131072, "robot_config": str(rpo_robot_config(Path("/tmp/eb-pbfm-accounting-source"))),
    }, rank=0, world=world)
    assert cfg.online_parallel_envs == num_envs
    assert cfg.agent.train.batch_size == 1024 and _effective_batch_size(cfg) == 1024 * world
    assert cfg.num_seed_steps == 10240 and cfg.update_agent_every == 1024 and cfg.num_agent_updates == 16
    assert cfg.log_every_updates == 1024 * world
    assert cfg.prior_plane_envs == 0 and cfg.resume_replay_warmup_steps == 0
    assert cfg.checkpoint_buffer and cfg.checkpoint_rank_buffers
    assert cfg.buffer_size == 131072
    assert cfg.env.terrain_curriculum_start_global_step == 0
    assert cfg.env.terrain_curriculum_interval == 19200000
    assert list(DirectDepthConfig().sampled_ages) == [35, 30, 25, 20, 15, 10, 5, 0]
    rates = {key: value for key, value in cfg.agent.train.model_dump().items() if key.startswith("lr_")}
    assert rates == {
        "lr_f": 0.0003, "lr_b": 0.00001, "lr_actor": 0.0003, "lr_discriminator": 5e-6,
        "lr_critic": 0.0003, "lr_aux_critic": 0.0003, "lr_heading_critic": 0.0003,
        "lr_velocity_critic": 0.0003,
    }
    return cfg


def run_native_loop(root, cfg, *, start_updates, start_local, memory_handoff=False, saves_out=None):
    import numpy as np
    import torch
    from pbfm.agents.fb_cpr_aux.agent import RolloutContextState
    from pbfm.training import checkpoint
    from pbfm.training.workspace import Workspace

    class Replay:
        def __init__(self, *, capacity=64, **kwargs):
            self.capacity, self.rows = capacity, 0

        def __len__(self):
            return self.rows

        def extend(self, data):
            assert data["action"].shape[1] == cfg.online_parallel_envs
            self.rows = min(self.capacity, self.rows + 1)

        def end_episodes(self):
            pass

        @classmethod
        def load_for_training(cls, *args, **kwargs):
            replay = cls()
            replay.rows = 1
            return replay

    envs = cfg.online_parallel_envs
    times, saves, logs = [], [], []
    step = 0
    closed = []

    def observation():
        return {"time": np.full(envs, step, dtype=np.int64), "state": np.zeros((envs, 1), np.float32)}

    def env_step(action):
        nonlocal step
        assert np.asarray(action).shape == (envs, 1) and not np.asarray(action).any()
        step += 1
        return observation(), np.zeros(envs, np.float32), np.zeros(envs, bool), np.zeros(envs, bool), {}

    env = NS(
        _env=NS(), reset=lambda: (observation(), {}), step=env_step,
        action_space=NS(sample=lambda: np.zeros((envs, 1), np.float32)), close=lambda: closed.append(True),
    )

    def update(replay, local_time):
        assert len(replay["train"]) > 0
        times.append(local_time)
        return {"loss": torch.tensor(0.)}

    # Preserve native W/E/seed/checker values; isolate simulation and collectives.
    cfg = cfg.model_copy(update={
        "buffer_device": "cpu", "disable_tqdm": True, "fail_on_nonfinite": False,
        "distributed_average_metrics": False, "log_torso_contact_forces": False,
        "env": cfg.env.model_copy(update={"terrain_training_curriculum": False}),
    })
    workspace = Workspace.__new__(Workspace)
    workspace.cfg, workspace.work_dir = cfg, root
    workspace.train_env, workspace.train_env_info, workspace.prior_env = env, {}, None
    workspace.agent = NS(device="cpu", _model=NS(), act=lambda **kwargs: torch.zeros(envs, 1), update=update)
    workspace._advance_rollout_state = lambda *args, **kwargs: RolloutContextState(z=torch.ones(envs, 1))
    workspace._collector_root_heading_xy = lambda *args: None
    workspace._attach_rollout_heading = lambda *args: None
    workspace._heading_transition_metadata = lambda **kwargs: {}
    workspace._load_expert_buffer = lambda: None
    workspace.training_with_expert_data = workspace._write_shared_artifacts = False
    workspace.distributed_rank, workspace.distributed_world_size = 0, cfg.distributed_world_size
    workspace._checkpoint_local_time = start_local
    workspace._checkpoint_global_time = start_local * cfg.distributed_world_size
    workspace._optimizer_steps = start_updates
    workspace.evaluate, workspace.evaluations, workspace.train_logger = False, {}, NS(log=logs.append)
    workspace.save = lambda **kwargs: saves.append(kwargs)
    if start_local:
        checkpoint.checkpoint_buffer_path(
            root / checkpoint.CHECKPOINT_DIR_NAME, cfg, rank=0,
            world_size=cfg.distributed_world_size, name="train",
        ).mkdir(parents=True)
    from eb.runners import rlinf_pbfm_runtime as runtime

    handoffs, memory = [], []
    if memory_handoff:
        runtime.install_simulation_memory_handoff(workspace, 0, [], {"main": {}})
    with (patch("pbfm.training.workspace.TrajectoryDictBufferMultiDim", Replay),
          patch("pbfm.training.workspace.sync_floating_buffers", lambda model: None),
          patch.object(torch.cuda, "synchronize", side_effect=lambda: handoffs.append("sync")),
          patch.object(torch.cuda, "empty_cache", side_effect=lambda: handoffs.append("empty")),
          patch("gc.collect", side_effect=lambda: handoffs.append("gc")),
          patch.object(runtime, "write_json"),
          patch.object(runtime, "record_cuda_memory",
                       side_effect=lambda *a, **k: memory.append((a[1], k["sample"])))):
        workspace.train_online()
    if memory_handoff:
        assert handoffs == ["sync", "gc", "empty"] * step
        assert memory == [(stage, sample) for sample in range(1, step + 1)
                          for stage in ("before_sample", "after_sample_cleanup")]
    else:
        assert not handoffs and not memory
    assert closed == [True]
    assert saves[-1]["global_time"] == cfg.num_env_steps
    assert saves[-1]["local_time"] * cfg.distributed_world_size == cfg.num_env_steps
    if saves_out is not None:
        saves_out.extend(saves)
    return times, saves[-1], logs


def main():
    from eb.pbfm_training import segment_endpoint
    from eb.runners.pbfm_source import prepare_pbfm_source

    source = Path("/tmp/eb-pbfm-accounting-source")
    prepare_pbfm_source(Path("/opt/pbfm-upstream"), source)
    sys.path.insert(0, str(source))
    # Literal expectations are from the design; all scheduling runs upstream.
    cases = [
        (2048, 4, 112, 0, 0, 12288, 26624, 106496),
        (2048, 4, 368, 112, 26624, 28672, 61440, 245760),
        (2048, 4, 512, 368, 61440, 63488, 81920, 327680),
        (2048, 4, 10000, 512, 81920, 83968, 1298432, 5193728),
        (1024, 1, 112, 0, 0, 11264, 18432, 18432),
        (1024, 4, 512, 112, 18432, 19456, 45056, 180224),
    ]
    with tempfile.TemporaryDirectory(prefix="eb-pbfm-accounting-") as temporary:
        for index, (envs, world, target, start, local, first, saved_local, endpoint) in enumerate(cases):
            root = Path(temporary) / str(index)
            root.mkdir()
            cfg = native_config(root, num_envs=envs, world=world, endpoint=endpoint)
            times, saved, logs = run_native_loop(root, cfg, start_updates=start, start_local=local,
                                                memory_handoff=envs == 2048)
            assert len(times) == target - start and times[0] == first
            assert saved["optimizer_steps"] == target and saved["local_time"] == saved_local
            assert times[-1] + envs == saved_local
            assert logs[-1]["distributed/global_env_steps"] == (saved_local - envs) * world
            assert segment_endpoint(target, world, start_updates=start, start_local_time=local,
                                    num_envs=envs) == saved["global_time"]
            print(json.dumps({
                "envs": envs, "world": world, "target": target, "first_update_local": times[0],
                "saved_local": saved["local_time"], "saved_global": saved["global_time"],
            }), flush=True)
    print("NATIVE_WORKSPACE_E2048_AND_E1024_ACCOUNTING_AND_CONFIG_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
