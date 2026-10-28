"""Long runs keep full metrics and publish only complete recovery generations."""
import csv
import json
import random
import shutil
import sys
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from test_pbfm_longrun import long_plan


def test_long_analysis_retains_all5625_rows_when_csv_exceeds_four_mib(tmp_path):
    from eb.runners.rlinf_pbfm_runtime import build_training_evidence

    path = tmp_path / "outputs/native/train_log.txt"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["distributed/optimizer_steps", "critic_loss", *[f"metric_{i}" for i in range(326)]])
        for update in range(10016, 100001, 16):
            writer.writerow([update, "0.123456", *["0.123456" for _ in range(326)]])
    assert path.stat().st_size > 4 * 1024**2
    plan = long_plan(tmp_path)
    plan["run_id"] = "run_long"
    index = build_training_evidence(tmp_path, plan, {"optimizer_steps": 100000})
    out = tmp_path / "outputs/analysis"
    assert (out / "train-timeseries.csv").read_bytes() == path.read_bytes()
    assert json.loads((out / "summary.json").read_text())["rows"] == 5625
    assert next(r for r in index["files"] if r["path"].endswith(".csv"))["bytes"] == path.stat().st_size


def test_memory_debug_window_is_bounded_and_keeps_dropped_peak(tmp_path, monkeypatch):
    from eb.runners import rlinf_pbfm_runtime as runtime

    snapshots_type = getattr(runtime, "MemorySnapshots", None)
    assert snapshots_type is not None, "long runtime needs bounded sampling history"
    snapshots = snapshots_type()
    marker = [0]
    cuda = NS(mem_get_info=lambda: (900, 100000),
              memory_allocated=lambda: 10000 if marker[0] == 20 else marker[0],
              memory_reserved=lambda: 20000 if marker[0] == 20 else marker[0],
              max_memory_allocated=lambda: 10000, max_memory_reserved=lambda: 20000)
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=cuda))
    monkeypatch.setattr(runtime, "OUTPUT_ROOT", tmp_path)
    for i in range(1024):
        marker[0] = i
        runtime.record_cuda_memory(0, "sampling", snapshots, marker=i)
    saved = json.loads((tmp_path / "outputs/runtime/memory-rank-0.json").read_text())
    assert len(saved["snapshots"]) == 256 and saved["total_snapshots"] == 1024
    assert saved["dropped_snapshots"] == 768
    assert [s["marker"] for s in saved["snapshots"][:8]] == list(range(8))
    assert saved["snapshots"][8]["marker"] == 776
    assert saved["peak_memory"]["allocated_bytes"] == 10000


def periodic_fixture(tmp_path):
    from test_pbfm_checkpoint import fixture

    staging, _, rank = fixture(tmp_path)
    plan = long_plan(tmp_path)
    plan["run_id"] = "run_long"
    for filename in ("config.json", "effective_config.json"):
        path = staging / filename
        value = json.loads(path.read_text())
        value["distributed_world_size"] = 4
        path.write_text(json.dumps(value))
    buffers = staging / "checkpoint/buffers"
    (buffers / "train").rename(buffers / "train_rank_0")
    for i in range(1, 4):
        shutil.copytree(buffers / "train_rank_0", buffers / f"train_rank_{i}")
    ranks = []
    for i in range(4):
        result = deepcopy(rank)
        result.update(rank=i, world_size=4, gpu_uuid=f"GPU-{i}", source_refs=plan["source_refs"])
        ranks.append(result)
    return staging, plan, ranks


def set_counts(staging, ranks, update, local):
    counts = {"optimizer_steps": update, "local_time": local, "global_time": local * 4}
    (staging / "checkpoint/train_status.json").write_text(json.dumps({**counts, "world_size": 4}))
    for rank in ranks:
        rank.update(counts)
        rank["optimizer_state_steps"] = {name: [update] for name in rank["optimizer_names"]}


def test_periodic_failure_preserves_last_complete_generation_then_retains_two(tmp_path, monkeypatch):
    from eb.runners import pbfm_longrun as runtime

    staging, plan, ranks = periodic_fixture(tmp_path)
    root = tmp_path / "recovery"
    set_counts(staging, ranks, 19984, 2578432)
    first = runtime.publish_periodic_checkpoint(staging, root, plan, ranks)
    assert first["latest"]["counts"]["optimizer_steps"] == 19984
    before = (root / "index.json").read_bytes()
    set_counts(staging, ranks, 29984, 3858432)
    with monkeypatch.context() as patch:
        def invalid(*args, **kwargs):
            raise ValueError("incomplete replay")
        patch.setattr(runtime, "build_pbfm_checkpoint_manifest", invalid)
        with pytest.raises(ValueError, match="incomplete replay"):
            runtime.publish_periodic_checkpoint(staging, root, plan, ranks)
    assert (root / "index.json").read_bytes() == before
    assert (root / first["latest"]["path"]).is_dir()
    runtime.publish_periodic_checkpoint(staging, root, plan, ranks)
    set_counts(staging, ranks, 39984, 5138432)
    final = runtime.publish_periodic_checkpoint(staging, root, plan, ranks)
    assert [r["counts"]["optimizer_steps"] for r in final["generations"]] == [39984, 29984]
    assert not (root / first["latest"]["path"]).exists()
    assert final["parent_target_updates"] == 100000


def test_intermediate_checkpoint_requires_all_rank_loader_proof_and_real_counts(tmp_path):
    from eb.pbfm_training_checkpoint import build_pbfm_checkpoint_manifest

    staging, plan, ranks = periodic_fixture(tmp_path)
    set_counts(staging, ranks, 19984, 2578432)
    manifest = build_pbfm_checkpoint_manifest(staging, plan, "run_long", rank_results=ranks,
                                             loader_verified=True, allow_intermediate=True)
    assert manifest["counts"]["optimizer_steps"] == 19984
    assert manifest["plan_hash"] != ""
    ranks[2]["replay_loader_verified"] = False
    with pytest.raises(ValueError):
        build_pbfm_checkpoint_manifest(staging, plan, "run_long", rank_results=ranks,
                                       loader_verified=True, allow_intermediate=True)


def test_intermediate_checkpoint_rejects_mismatched_optimizer_proof(tmp_path):
    from eb.pbfm_training_checkpoint import build_pbfm_checkpoint_manifest

    staging, plan, ranks = periodic_fixture(tmp_path)
    set_counts(staging, ranks, 19984, 2578432)
    ranks[1]["optimizer_state_steps"]["actor_optimizer"] = [10000]
    with pytest.raises(ValueError):
        build_pbfm_checkpoint_manifest(staging, plan, "run_long", rank_results=ranks,
                                       loader_verified=True, allow_intermediate=True)


def test_long_resume_start_checks_actual_optimizer_steps_and_source_parameter_hash(tmp_path, monkeypatch):
    from eb.runners import pbfm_longrun as runtime

    plan = long_plan(tmp_path)
    manifest = {"counts": {"optimizer_steps": 10000, "local_time": 1298432, "global_time": 5193728},
                "rank_results": [{"parameter_change": {"after_sha256": "original"}} for _ in range(4)]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    observed = []
    def verify(agent, *, expected_updates):
        observed.append(expected_updates)
        return {"optimizer_state_steps": {"actor_optimizer": [10000]}}
    monkeypatch.setattr(runtime, "verify_native_state", verify)
    assert runtime.verify_resume_start(None, plan, "original", path)["counts"] == manifest["counts"]
    assert observed == [10000]
    with pytest.raises(ValueError):
        runtime.verify_resume_start(None, plan, "different", path)


@pytest.mark.parametrize("failed", [False, True])
def test_periodic_hook_orders_save_readback_publish_and_preserves_rng_on_failure(
    tmp_path, monkeypatch, failed,
):
    from eb.runners import pbfm_longrun as runtime

    calls = []
    def barrier():
        calls.append("barrier")
        proof = tmp_path / "runtime/periodic/19984/rank-0.json"
        if proof.exists():
            for i in range(1, 4):
                target = proof.with_name(f"rank-{i}.json")
                if not target.exists():
                    target.write_bytes(proof.read_bytes())
    dist = NS(barrier=barrier)
    monkeypatch.setitem(sys.modules, "torch", NS(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    def load(*args, **kwargs):
        calls.append("load")
        random.random()
        return NS(_model={"sha": "unchanged"})
    def verify(*args, **kwargs):
        calls.append("verify")
        assert kwargs["expected_updates"] == 19984
        if failed:
            raise ValueError("bad optimizer")
        return {"strict_state_verified": True}
    class Replay:
        def __len__(self):
            return 64

        def empty(self):
            return False

        def _validate_sparse_storage(self):
            calls.append("replay")
    replay = Replay()
    for name in ("pbfm", "pbfm.agents", "pbfm.agents.buffers"):
        monkeypatch.setitem(sys.modules, name, NS())
    monkeypatch.setitem(sys.modules, "pbfm.agents.buffers.trajectory", NS(
        TrajectoryDictBufferMultiDim=NS(load_for_training=lambda *a, **k: replay)))
    @contextmanager
    def preserve():
        state = random.getstate()
        try:
            yield
        finally:
            random.setstate(state)
    def original_save(**kwargs):
        calls.append("save")
        random.random()
    monkeypatch.setattr(runtime, "preserve_rng", preserve)
    monkeypatch.setattr(runtime, "parameter_hash", lambda model: model["sha"])
    monkeypatch.setattr(runtime, "rng_state_hash", lambda: repr(random.getstate()))
    monkeypatch.setattr(runtime, "verify_native_state", verify)
    monkeypatch.setattr(runtime, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "STAGING_ROOT", tmp_path / "staging")
    monkeypatch.setattr(runtime, "publish_periodic_checkpoint", lambda *a: calls.append("publish"))
    trainer = NS(save=original_save, cfg=NS(agent=NS(object_class=NS(load=load))),
                 agent=NS(_model={"sha": "unchanged"}))
    runtime.install_periodic_checkpoints(trainer, long_plan(tmp_path), {"rank": 0, "world_size": 4})
    before = random.getstate()
    kwargs = {"optimizer_steps": 19984, "local_time": 2578432, "global_time": 10313728}
    if failed:
        with pytest.raises(ValueError, match="bad optimizer"):
            trainer.save(**kwargs)
        assert "publish" not in calls
    else:
        trainer.save(**kwargs)
        assert calls.index("save") < calls.index("load") < calls.index("replay") < calls.index("publish")
    assert random.getstate() == before
    calls.clear()
    trainer.save(**{**kwargs, "optimizer_steps": 100000})
    assert calls == ["save"]
