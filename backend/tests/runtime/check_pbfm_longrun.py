"""CPU-only source loader, RNG preservation and real native long-loop accounting.

Simulation, gradient updates and collectives are I/O doubles in the loop check;
the source agent/optimizer/Replay payload loaders below consume real frozen files.
"""
import gc
import json
import random
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from check_pbfm_2048_accounting import native_config, run_native_loop

from eb.pbfm_longrun import long_exploration_metadata, runtime_compatibility_receipt
from eb.pbfm_training import native_arguments, training_fingerprint, validate_frozen_plan
from eb.runners.pbfm_source import prepare_pbfm_source
from eb.runners.rlinf_pbfm_runtime import parameter_hash, verify_native_state


def main():
    source = Path("/tmp/eb-pbfm-accounting-source")
    prepare_pbfm_source(Path("/opt/pbfm-upstream"), source)
    sys.path.insert(0, str(source))
    from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
    from pbfm.agents.fb_cpr_aux.agent import FBcprAuxAgent

    from eb.runners.pbfm_longrun import preserve_rng, verify_resume_start

    receipt = json.loads(Path("/opt/pbfm-long-runtime.json").read_text())
    assert receipt == runtime_compatibility_receipt()
    plan = json.loads(Path("/source-plan.json").read_text())
    source_identity = deepcopy(plan["identity"])
    # The real producer image is supplied by the outside image-layer audit.
    image = sys.argv[1]
    plan["long_exploration"] = long_exploration_metadata(source_identity, image)
    plan["identity"]["image_digest"] = plan["image_digest"] = image
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan.update(start_updates=10000, start_local_time=1298432, target_updates=100000,
                segment_timeout_seconds=108000, communication_timeout_seconds=900)
    plan.pop("hash", None)
    plan["native_args"] = native_arguments(plan)
    validate_frozen_plan(plan)
    with tempfile.TemporaryDirectory(prefix="eb-long-accounting-") as temporary:
        cfg = native_config(Path(temporary), num_envs=2048, world=4, endpoint=51281920)
        cfg = cfg.model_copy(update={"checkpoint_every_steps": 5120000})
        saves = []
        times, final, logs = run_native_loop(Path(temporary), cfg,
            start_updates=10000, start_local=1298432, saves_out=saves)
        assert len(times) == 90000 and times[0] == 1300480
        assert final["optimizer_steps"] == 100000 and final["local_time"] == 12820480
        assert len(logs) == 5625
        assert [s["optimizer_steps"] for s in saves] == [*range(19984, 100000, 10000), 100000]
        assert [int(row["distributed/optimizer_steps"]) for row in logs] == list(range(10016, 100001, 16))
        print(json.dumps({"check": "native_loop_only", "new_updates": len(times), "rows": len(logs),
              "saves": [{k: s[k] for k in ("optimizer_steps", "local_time", "global_time")}
                        for s in saves]}), flush=True)

    checkpoint = Path("/source-native/checkpoint")
    # CPU-only stand-in for the CUDA RNG slot; real CUDA invariance is asserted at each actual save.
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with (patch.object(torch.cuda, "get_rng_state", side_effect=lambda device=None: torch.get_rng_state()),
          patch.object(torch.cuda, "set_rng_state", side_effect=lambda *a, **kw: None), preserve_rng()):
        loaded = FBcprAuxAgent.load(checkpoint, device="cpu")
        evidence = verify_native_state(loaded, checkpoint, expected_updates=10000)
        actual_hash = parameter_hash(loaded._model)
        assert actual_hash == "sha256:1859dc1706eaad310b2a535f305bccd37abe329df4b1c945eb850ef2b97dfc3b"
        verify_resume_start(loaded, plan, actual_hash, Path("/source-native/checkpoint-manifest.json"))
        del loaded
        gc.collect()
        for rank in range(4):
            replay = TrajectoryDictBufferMultiDim.load_for_training(
                checkpoint / f"buffers/train_rank_{rank}", device="cpu")
            assert len(replay) == 64 and not replay.empty()
            replay._validate_sparse_storage()
            del replay
            gc.collect()
        random.random(), np.random.random(), torch.rand(1)
    assert random.getstate() == python_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_state))
    assert torch.equal(torch.get_rng_state(), torch_state)
    print(json.dumps({"check": "actual_native_source_cpu_loaders", "parameter_sha256": actual_hash,
          "optimizer_state_steps": evidence["optimizer_state_steps"], "replay_slots_each_rank": 64,
          "cpu_rng_preserved": True, "gpu_execution_verified": False}), flush=True)


if __name__ == "__main__":
    main()
