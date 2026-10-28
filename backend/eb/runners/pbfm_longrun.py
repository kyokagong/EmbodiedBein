"""Complete periodic recovery generations for the bounded 100000-update run."""
from __future__ import annotations

import gc
import json
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from eb.pbfm_longrun import validate_long_plan
from eb.pbfm_training_checkpoint import _tree, build_pbfm_checkpoint_manifest, canonical_hash
from eb.runners.pbfm_source import file_sha256
from eb.runners.rlinf_pbfm_runtime import (
    OUTPUT_ROOT,
    STAGING_ROOT,
    parameter_hash,
    rng_state_hash,
    verify_native_state,
    write_json,
)


def publish_periodic_checkpoint(staging: Path, root: Path, plan: dict, rank_results: list[dict]) -> dict:
    validate_long_plan(plan)
    _tree(staging)  # Reject links/incomplete native saves before copying any payload.
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("P-BFM recovery root is symlinked")
    parent_hash = canonical_hash(plan)
    previous = []
    if (root / "index.json").exists():
        old = json.loads((root / "index.json").read_text())
        if old.get("parent_plan_hash") != parent_hash or old.get("run_id") != plan["run_id"]:
            raise ValueError("P-BFM recovery index belongs to a different run")
        previous = old["generations"]
        if len(previous) > 2 or any(not re.fullmatch(r"generations/u[0-9]+/native", e["path"])
                                    for e in previous):
            raise ValueError("P-BFM recovery index generation path differs")
    status = json.loads((staging / "checkpoint/train_status.json").read_text())
    update = status["optimizer_steps"]
    if previous and update <= previous[0]["counts"]["optimizer_steps"]:
        raise ValueError("P-BFM recovery generations must advance")
    destination = root / "generations" / f"u{update}"
    temporary = destination.with_name(destination.name + ".publishing")
    if destination.exists() or temporary.exists():
        raise ValueError("P-BFM recovery generation already exists")
    try:
        shutil.copytree(staging, temporary / "native")
        manifest = build_pbfm_checkpoint_manifest(
            temporary / "native", plan, plan["run_id"], rank_results=rank_results,
            loader_verified=True, allow_intermediate=True,
        )
        manifest_path = temporary / "native/checkpoint-manifest.json"
        write_json(manifest_path, manifest)
        entry = {"path": f"generations/u{update}/native", "counts": manifest["counts"],
                 "manifest_sha256": file_sha256(manifest_path)}
        write_json(temporary / "receipt.json", {
            "schema_version": "eb/pbfm-periodic-recovery/v1", "status": "complete",
            "training_complete": False, "run_id": plan["run_id"],
            "parent_plan_hash": parent_hash, **entry,
        })
        temporary.rename(destination)
        index = {"schema_version": "eb/pbfm-periodic-recovery-index/v1",
                 "run_id": plan["run_id"], "parent_plan_hash": parent_hash,
                 "parent_target_updates": plan["target_updates"], "latest": entry,
                 "generations": [entry, *previous][:2]}
        write_json(root / "index.json", index)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    for entry in previous[1:]:
        shutil.rmtree((root / entry["path"]).parent)
    return index


@contextmanager
def preserve_rng():
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def verify_resume_start(agent, plan: dict, parameter_sha256: str,
                        manifest_path: Path = Path("/resume-native/checkpoint-manifest.json")) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected = {"optimizer_steps": plan["start_updates"], "local_time": plan["start_local_time"],
                "global_time": plan["start_local_time"] * plan["world_size"]}
    source_hashes = {row.get("parameter_sha256") or row["parameter_change"]["after_sha256"]
                     for row in manifest["rank_results"]}
    if manifest["counts"] != expected or source_hashes != {parameter_sha256}:
        raise ValueError("P-BFM long resume parameter hash or source counts differ")
    return {"counts": expected, "parameter_sha256": parameter_sha256,
            **verify_native_state(agent, expected_updates=plan["start_updates"])}


def install_periodic_checkpoints(trainer, plan: dict, identity: dict) -> None:
    import torch.distributed as dist
    from pbfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim

    validate_long_plan(plan)
    original_save = trainer.save
    rank, world = identity["rank"], identity["world_size"]

    def save(**kwargs):
        if kwargs["optimizer_steps"] >= plan["target_updates"]:
            return original_save(**kwargs)
        before, rng_before = parameter_hash(trainer.agent._model), rng_state_hash()
        checkpoint = STAGING_ROOT / "checkpoint"
        proof_root = OUTPUT_ROOT / "runtime/periodic" / str(kwargs["optimizer_steps"])
        with preserve_rng():
            original_save(**kwargs)
            # Serialize CPU readback while all four simulator/learner processes remain resident.
            for current in range(world):
                if rank == current:
                    loaded = trainer.cfg.agent.object_class.load(checkpoint, device="cpu")
                    evidence = verify_native_state(
                        loaded, checkpoint, expected_updates=kwargs["optimizer_steps"])
                    if parameter_hash(loaded._model) != before:
                        raise ValueError("P-BFM periodic model loader changed parameters")
                    del loaded
                    gc.collect()
                    replay = TrajectoryDictBufferMultiDim.load_for_training(
                        checkpoint / "buffers" / f"train_rank_{rank}", device="cpu")
                    if replay.empty():
                        raise ValueError("P-BFM periodic Replay loader returned an empty buffer")
                    replay._validate_sparse_storage()
                    rows = len(replay)
                    del replay
                    gc.collect()
                    write_json(proof_root / f"rank-{rank}.json", {
                        **identity, **evidence,
                        **{key: kwargs[key] for key in ("local_time", "global_time", "optimizer_steps")},
                        "status": "succeeded", "loader_verified": True, "replay_loader_verified": True,
                        "full_state_finite_verified": True, "replay_rows": rows,
                        "parameter_sha256": before,
                    })
                dist.barrier()
        if parameter_hash(trainer.agent._model) != before or rng_state_hash() != rng_before:
            raise ValueError("P-BFM periodic save/readback changed running parameters or RNG")
        dist.barrier()
        if rank == 0:
            results = [json.loads((proof_root / f"rank-{i}.json").read_text()) for i in range(world)]
            if len({r["parameter_sha256"] for r in results}) != 1:
                raise ValueError("P-BFM periodic rank parameters diverged")
            publish_periodic_checkpoint(STAGING_ROOT, OUTPUT_ROOT / "outputs/recovery", plan, results)
        dist.barrier()

    trainer.save = save
