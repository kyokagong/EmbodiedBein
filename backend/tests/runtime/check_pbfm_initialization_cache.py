"""CPU boundary check for initialization cache release, using the real worker."""

import gc
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

from eb.runners import rlinf_pbfm_runtime as runtime
from eb.runners import rlinf_pbfm_worker as worker


def check(write_failure=False, memory_failure=False, sync_failure=False):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    params = [(p, p.detach().clone()) for p in model.parameters()]
    states = {id(v): v.clone() for state in optimizer.state.values()
              for v in state.values() if isinstance(v, torch.Tensor)}
    random_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.arange(16, dtype=torch.uint8)
    calls, hashes = [], []
    original = RuntimeError("original first training error")

    def train():
        calls.append("train")
        assert (calls.index("synchronize") < calls.index("collect")
                < calls.index("empty_cache") < calls.index("train"))
        assert len(hashes) == 2 and hashes[0] == hashes[1]
        for param, value in params:
            assert any(param is live for live in model.parameters())
            assert torch.equal(param, value)
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, states[id(value)])
        assert random.getstate() == random_state
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        assert torch.equal(torch.get_rng_state(), cpu_state)
        raise original

    agent_cfg = NS(model_dump=lambda **kwargs: {})
    env = NS(base_env=NS(config={}), close=lambda: calls.append("close"))
    trainer = NS(train_env=env, agent=NS(cfg=agent_cfg, _model=model), train=train)
    cfg = NS(agent=agent_cfg, build=lambda: trainer, model_dump=lambda **kwargs: {})
    subject = NS(used=False, plan={"target_updates": 112, "identity": {"num_envs_per_rank": 1024}},
                 describe_runtime=lambda: {"rank": 0, "world_size": 1})

    def parameter_hash(value):
        digest = runtime.parameter_hash(value)
        hashes.append(digest)
        return digest

    real_collect = gc.collect

    def collect():
        calls.append("collect")
        return real_collect()

    def synchronize():
        calls.append("synchronize")
        if sync_failure:
            raise original

    with tempfile.TemporaryDirectory(prefix="eb-init-cache-") as directory:
        root = Path(directory)
        with (
            patch.object(worker, "OUTPUT_ROOT", root),
            patch.object(worker, "STAGING_ROOT", root / "staging"),
            patch.object(runtime, "OUTPUT_ROOT", root),
            patch.object(worker, "build_native_config", return_value=cfg),
            patch.object(worker, "verify_environment", return_value={"simulation_execution": {}}),
            patch.object(worker, "resolved_environment_config", return_value={}),
            patch.object(worker, "parameter_hash", side_effect=parameter_hash),
            patch.object(torch.distributed, "is_initialized", return_value=False),
            patch.multiple(torch.cuda, set_device=Mock(), get_rng_state=Mock(return_value=cuda_state),
                           synchronize=Mock(side_effect=synchronize),
                           memory_allocated=Mock(return_value=100), memory_reserved=Mock(return_value=500),
                           max_memory_allocated=Mock(return_value=300),
                           max_memory_reserved=Mock(return_value=800),
                           mem_get_info=Mock(side_effect=RuntimeError("CUDA stats unavailable"))
                           if memory_failure else Mock(return_value=(1000, 2000)),
                           empty_cache=Mock(side_effect=lambda: calls.append("empty_cache"))),
            patch.object(gc, "collect", side_effect=collect),
            patch.object(runtime, "write_json", side_effect=OSError("telemetry disk error"))
            if write_failure else patch.object(runtime, "write_json", wraps=runtime.write_json),
        ):
            try:
                worker.PbfmTrainWorker.run_segment(subject, {})
            except RuntimeError as exc:
                assert exc is original, repr(exc)
            else:
                raise AssertionError("training error was swallowed")
        assert calls == ["synchronize", "close"] if sync_failure else calls[-2:] == ["train", "close"]
        if not write_failure:
            report = json.loads((root / "outputs/runtime/memory-rank-0.json").read_text())
            snapshots = report["snapshots"]
            assert [s["stage"] for s in snapshots] == ([
                "before_build", "after_build", "failed",
            ] if sync_failure else [
                "before_build", "after_build", "before_cache_cleanup", "after_cache_cleanup", "failed",
            ])
            for entry in snapshots:
                if memory_failure:
                    assert entry["snapshot_error"] == "RuntimeError: CUDA stats unavailable"
                else:
                    assert entry["allocated_bytes"] == 100 and entry["reserved_bytes"] == 500
                    assert entry["max_allocated_bytes"] == 300 and entry["max_reserved_bytes"] == 800
                    assert entry["free_bytes"] == 1000 and entry["total_bytes"] == 2000
            if not sync_failure:
                assert snapshots[2]["parameter_sha256"] == snapshots[3]["parameter_sha256"]
                assert snapshots[2]["rng_sha256"] == snapshots[3]["rng_sha256"]


if __name__ == "__main__":
    check()
    check(write_failure=True)
    check(memory_failure=True)
    check(sync_failure=True)
    print("INITIALIZATION_CACHE_RELEASE_PRESERVES_LIVE_STATE_RNG_AND_FIRST_ERROR", flush=True)
