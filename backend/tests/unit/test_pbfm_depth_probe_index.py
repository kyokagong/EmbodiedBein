"""Use the real shared verifier for the 232-file four-rank emitted footprint."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from test_pbfm_depth_probe_receipt import fixture

from eb import pbfm_io
from eb.orchestrator.ufo_native import verify_index
from eb.pbfm_training import native_arguments, training_fingerprint
from eb.runners.pbfm_depth_probe import PHASES


def emitted_fixture(tmp_path, monkeypatch, *, world=4, legacy=False):
    read_json, safe_file = pbfm_io.read_json, pbfm_io.safe_file
    probe, spec, receipt, report, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pbfm_io, "read_json", read_json)
    monkeypatch.setattr(pbfm_io, "safe_file", safe_file)
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", verify_index)
    plan = spec.params["native_train_plan"]
    plan.update(world_size=world, log_every_steps=1024 * world)
    plan["identity"]["world_size"] = world
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["native_args"] = native_arguments(plan)
    spec.resources.gpus, spec.resources.cpus, spec.resources.memory_gb = world, 8 * world, 16 + 16 * world
    receipt.update(world_size=world, plan_hash=training_fingerprint({**plan, "run_id": "run_probe"}))
    if legacy:
        receipt["schema_version"] = "eb/pbfm-native-runtime-probe/v1"
    prototype = receipt["ranks"][0]
    receipt["ranks"], receipt["files"] = [], []
    # Frozen from run_f6610eb2ec4441139306: 57 folder files plus one descriptor per rank.
    captures = (
        "base-quat", "camera-pos-w", "camera-quat-w", "depth-delay-frames", "metric-depth",
        "motion-ids", "motion-start-times", "network-depth", "projected-gravity", "qpos", "qvel",
        "reference-network-depth", "robot-root-states", "sensor-cached-world-origins",
        "sensor-cached-world-rays", "sensor-ranges", "simulation-qpos",
    )
    names = {prefix + name + ".npy" for prefix in PHASES.values() for name in captures}
    names.update({"depth-validation.json", "probe-stats.json", "network-depth.png", "robot-terrain.png",
                  "positive-network-depth.png", "positive-robot-terrain.png"})
    assert len(names) == 57
    for rank in range(world):
        rank_report = deepcopy(report)
        rank_report["rank"] = rank
        row = deepcopy(prototype)
        row.update(rank=rank, world_size=world, pid=100 + rank, gpu_uuid=f"GPU-{rank}",
                   process_group=world > 1, depth_validation=rank_report)
        receipt["ranks"].append(row)
        paths = {f"runtime/probe-rank-{rank}/{name}" for name in names}
        paths.add(f"runtime/rank-{rank}.json")
        for relative in sorted(paths):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                json.dumps(rank_report).encode() if path.name == "depth-validation.json" else b"evidence"
            )
            path.write_bytes(payload)
            receipt["files"].append({"path": relative, "bytes": len(payload),
                                     "sha256": hashlib.sha256(payload).hexdigest()})
    assert len(receipt["files"]) == 58 * world

    async def scan(path):
        payload = Path(path).read_bytes()
        return {"files": 1, "samples": [{"sha256": hashlib.sha256(payload).hexdigest()}]}

    scanner = AsyncMock(side_effect=scan)
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient.scan", scanner)

    def save_receipt():
        (tmp_path / "probe-receipt.json").write_text(json.dumps(receipt))

    save_receipt()
    return probe, spec, receipt, scanner, save_receipt


@pytest.mark.asyncio
async def test_v2_collects_actual_four_rank_232_file_footprint_with_real_verifier(tmp_path, monkeypatch):
    probe, spec, receipt, scanner, _ = emitted_fixture(tmp_path, monkeypatch)
    run = NS(id="run_probe", metrics_summary={})
    await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), run, tmp_path)
    assert len(run.metrics_summary["runtime_probe"]["files"]) == 232
    assert scanner.await_count == 232


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["duplicate", "missing", "oversize", "excess"])
async def test_v2_real_verifier_rejects_invalid_or_unbounded_evidence(tmp_path, monkeypatch, failure):
    world = 1 if failure in {"oversize", "excess"} else 4
    probe, spec, receipt, _, save = emitted_fixture(tmp_path, monkeypatch, world=world)
    if failure == "duplicate":
        receipt["files"].append(deepcopy(receipt["files"][0]))
    elif failure == "missing":
        (tmp_path / receipt["files"][0]["path"]).unlink()
    elif failure == "oversize":
        row = next(item for item in receipt["files"] if item["path"].endswith("/sensor-ranges.npy"))
        payload = b"x" * (4 * 1024**2 + 1)
        (tmp_path / row["path"]).write_bytes(payload)
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    else:
        for index in range(71):
            relative = f"runtime/probe-rank-0/extra-{index}.npy"
            (tmp_path / relative).write_bytes(b"x")
            receipt["files"].append({
                "path": relative, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest(),
            })
        assert len(receipt["files"]) == 129
    save()
    with pytest.raises(ValueError):
        await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), NS(id="run_probe"), tmp_path)


@pytest.mark.asyncio
async def test_legacy_keeps_the_shared_200_file_default(tmp_path, monkeypatch):
    probe, spec, receipt, scanner, _ = emitted_fixture(tmp_path, monkeypatch, legacy=True)
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), NS(id="run_probe"), tmp_path)
    scanner.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_keeps_its_existing_larger_per_file_limit(tmp_path, monkeypatch):
    probe, spec, receipt, scanner, save = emitted_fixture(tmp_path, monkeypatch, world=1, legacy=True)
    row = next(item for item in receipt["files"] if item["path"].endswith("/sensor-ranges.npy"))
    payload = b"x" * (4 * 1024**2 + 1)
    (tmp_path / row["path"]).write_bytes(payload)
    row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    save()
    run = NS(id="run_probe", metrics_summary={})
    await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), run, tmp_path)
    assert scanner.await_count == 58
