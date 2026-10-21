"""Entry-only observation of the unmodified native Trainer in elastic children."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def install() -> None:
    if os.environ.get("EB_UFO_BASELINE_OBSERVER") != "1":
        return
    if sys.getprofile() is not None:
        raise RuntimeError("Baseline refuses an inherited profiler")
    expected_file = "/sources/ufo/ufo/training/trainer.py"

    def observe(frame, event, arg):
        if event != "call" or frame.f_code.co_name != "train" or frame.f_code.co_filename != expected_file:
            return
        # Disable first: hashing/config serialization and the entire native train
        # loop run without a profiler. No UFO function is replaced or wrapped.
        sys.setprofile(None)
        disabled_at = time.time_ns()
        rank = int(os.environ["RANK"])
        trainer = frame.f_locals["self"]
        from eb.runners.rlinf_ufo_runtime import file_hash, parameter_snapshot, write_json
        from eb.ufo_native_baseline import config_differences

        cfg = trainer.cfg.model_dump(mode="json")
        if (
            trainer.cfg.distributed_rank != rank
            or trainer.cfg.distributed_world_size != 4
            or trainer.cfg.seed != 42 + rank
            or trainer._optimizer_steps != 0
        ):
            raise RuntimeError("Native baseline rank/seed/initial update count differs")
        reference = json.loads(Path(f"/reference/outputs/runtime/config-rank-{rank}.json").read_text())
        differences = config_differences(reference, cfg, rank)
        if differences:
            raise RuntimeError(f"Native baseline configuration differs: {differences[:5]}")
        snapshot, samples = parameter_snapshot(trainer.agent._model)
        expected = json.loads(
            Path(f"/reference/outputs/runtime/initial-parameters-rank-{rank}.json").read_text()
        )
        if snapshot != expected:
            raise RuntimeError("Native baseline actual initial parameters differ from Worker")
        import torch

        properties = torch.cuda.get_device_properties(rank)
        identity = {
            "rank": rank,
            "world_size": 4,
            "pid": os.getpid(),
            "gpu_uuid": str(properties.uuid),
            "seed": 42 + rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "device": str(trainer.agent.device),
            "optimizer_steps_before_train": 0,
            "profiler_disabled_before_hashing": sys.getprofile() is None,
            "profiler_disabled_at_unix_ns": disabled_at,
            "observer_sha256": file_hash(Path(__file__)),
            "configuration_differences": differences,
            "config": cfg,
            "parameters": snapshot,
            "samples": {name: value.tolist() for name, value in samples.items()},
        }
        write_json(Path(f"/run/eb-output/outputs/runtime/baseline-initial-rank-{rank}.json"), identity)

    sys.setprofile(observe)
