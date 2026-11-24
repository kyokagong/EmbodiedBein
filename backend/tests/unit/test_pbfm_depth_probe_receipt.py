from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from test_pbfm_segment_diagnostic import segment_spec

from eb.pbfm_training import CAMERA_CONTRACT, training_fingerprint
from eb.runners.pbfm_depth_probe import (
    PHASES,
    RAW_NAMES,
    RECEIPT_SCHEMA,
    SAMPLE_PIXELS,
    SCHEMA,
    TOLERANCE,
    validate_range_rows,
)


def fixture(tmp_path, monkeypatch):
    from eb import pbfm_runtime_probe as probe

    spec = segment_spec(tmp_path)
    spec.params["probe_mode"] = "runtime"
    rows = [
        {"pixel": pixel, "cpu_range_m": 1.0 + index * 0.03, "gpu_range_m": 1.0 + index * 0.03,
         "geom_id": 0, "welded_body_id": 0, "collidable": True}
        for index, pixel in enumerate(SAMPLE_PIXELS)
    ]
    report = {
        "schema_version": SCHEMA, "rank": 0, "tolerance": TOLERANCE,
        "sample_pixels": list(SAMPLE_PIXELS), "include_geom_groups": [0, 2, 5], "bodyexclude": -1,
        "range_limit_m": 4.0,
        "phases": {
            name: {
                "lanes": [deepcopy(validate_range_rows(rows, 4.0, positive=True)) for _ in range(2)],
                "history": {"shape": [2, 8, 18, 32], "dtype": "uint8", "history_length": 37,
                            "sampled_ages": [35, 30, 25, 20, 15, 10, 5, 0],
                            "delay_frames": [0, 1], "max_abs_error": 0},
            } for name in PHASES
        },
    }
    names = {"depth-validation.json", "probe-stats.json", "network-depth.png", "robot-terrain.png",
             "positive-robot-terrain.png", "positive-network-depth.png"}
    names.update(prefix + name for prefix in PHASES.values() for name in RAW_NAMES)
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "run_id": "run_probe", "status": "complete", "world_size": 1,
        "plan_hash": training_fingerprint({**spec.params["native_train_plan"], "run_id": "run_probe"}),
        "training_completed": False,
        "ranks": [{
            "rank": 0, "world_size": 1, "pid": 100, "gpu_uuid": "GPU-probe",
            "worker_class": "PbfmTrainWorker",
            "status": "probe_complete", "training_completed": False, "environment_reset_step": True,
            "egl_render_verified": True, "depth_verified": True, "process_group": False,
            "source_refs": spec.params["native_train_plan"]["source_refs"], "depth_validation": report,
            "native_contract": {"actor_scalar_dim": 380, "action_dim": 23, "physics_hz": 400,
                                "policy_hz": 50, "camera_contract": CAMERA_CONTRACT},
        }],
        "files": [{"path": f"runtime/probe-rank-0/{name}", "bytes": 100, "sha256": "b" * 64}
                  for name in sorted(names)],
    }
    monkeypatch.setattr("eb.pbfm_io.safe_file", lambda root, name: Path(root) / name)
    monkeypatch.setattr(
        "eb.pbfm_io.read_json", lambda path: report if path.name == "depth-validation.json" else receipt
    )
    verify = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", verify)
    return probe, spec, receipt, report, verify


@pytest.mark.asyncio
async def test_v2_probe_collects_geometry_and_complete_raw_index(tmp_path, monkeypatch):
    probe, spec, receipt, report, verify = fixture(tmp_path, monkeypatch)
    run = NS(id="run_probe", metrics_summary={})
    await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), run, tmp_path)
    assert run.metrics_summary["runtime_probe"]["ranks"][0]["depth_validation"] == report
    verify.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["allmiss", "constant", "wrong_range", "nan", "rank", "missing", "history"]
)
async def test_v2_probe_rejects_bad_geometry_or_unindexed_raw_evidence(tmp_path, monkeypatch, failure):
    probe, spec, receipt, report, verify = fixture(tmp_path, monkeypatch)
    lane = report["phases"]["positive_control"]["lanes"][1]
    if failure == "allmiss":
        for row in lane["rows"]:
            row.update(cpu_range_m=-1.0, gpu_range_m=-1.0, geom_id=-1, welded_body_id=-1, collidable=False)
    elif failure == "constant":
        for row in lane["rows"]:
            row.update(cpu_range_m=1.0, gpu_range_m=1.0)
    elif failure == "wrong_range":
        lane["rows"][0]["gpu_range_m"] += 0.2
    elif failure == "nan":
        lane["rows"][0]["cpu_range_m"] = float("nan")
    elif failure == "rank":
        report["rank"] = 1
    elif failure == "missing":
        receipt["files"] = [
            row for row in receipt["files"] if not row["path"].endswith("simulation-qpos.npy")
        ]
    else:
        reference = next(
            row for row in receipt["files"] if row["path"].endswith("/reference-network-depth.npy")
        )
        reference["sha256"] = "c" * 64
    with pytest.raises(ValueError):
        await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), NS(id="run_probe"), tmp_path)
