from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from eb.orchestrator.ufo_live import validate_receipt


def evidence():
    plan = {
        "model_id": "mv_x",
        "source_artifact_sha256": "sha256:" + "a" * 64,
        "environment_id": "ep_x",
        "environment_hash": "sha256:" + "b" * 64,
        "duration_seconds": 30,
    }
    value = {
        "model_id": plan["model_id"],
        "payload_sha256": plan["source_artifact_sha256"],
        "environment_id": plan["environment_id"],
        "environment_hash": plan["environment_hash"],
        "diagnostic_only": True,
        "complete": True,
        "steps": 123,
        "episodes": 1,
        "command_count": 2,
        "termination": "operator_stop",
        "task_order": ["move-ego-0-0", "move-ego-0-0.3", "move-ego-0-0.7"],
        "limits": {"joint": 4, "velocity": 0, "torque": 0},
        "trace_chunks": ["traces/chunk-0000.npz"],
        "recording": {
            "path": "videos/session.mp4",
            "decodable": True,
            "frames": 13,
            "fps": 5,
            "seconds": 2.6,
        },
    }
    names = [
        "trace-index.json",
        "events.jsonl",
        "timeseries.svg",
        "videos/session.mp4",
        *value["trace_chunks"],
    ]
    value["files"] = [{"path": name} for name in names]
    return plan, value


def test_live_evidence_is_diagnostic_and_binds_exact_model_and_environment():
    plan, value = evidence()
    validate_receipt(plan, value)
    for field in (
        "model_id",
        "payload_sha256",
        "environment_id",
        "environment_hash",
        "diagnostic_only",
        "complete",
    ):
        bad = deepcopy(value)
        bad[field] = None
        with pytest.raises(ValueError):
            validate_receipt(plan, bad)


def test_live_evidence_rejects_missing_trace_video_chart_and_incomplete_coverage():
    plan, value = evidence()
    for name in ["traces/chunk-0000.npz", "videos/session.mp4", "timeseries.svg", "events.jsonl"]:
        bad = deepcopy(value)
        bad["files"] = [item for item in bad["files"] if item["path"] != name]
        with pytest.raises(ValueError):
            validate_receipt(plan, bad)
    for field, wrong in [("frames", 12), ("fps", 25), ("decodable", False), ("seconds", 22)]:
        bad = deepcopy(value)
        bad["recording"][field] = wrong
        with pytest.raises(ValueError):
            validate_receipt(plan, bad)


def test_live_evidence_requires_bounded_steps_and_no_fake_qualification():
    plan, value = evidence()
    for field, wrong in [
        ("steps", 1501),
        ("episodes", 9),
        ("command_count", 1025),
        ("qualification", "pass"),
    ]:
        bad = deepcopy(value)
        bad[field] = wrong
        with pytest.raises(ValueError):
            validate_receipt(plan, bad)


@pytest.mark.asyncio
async def test_failed_live_job_registers_verified_partial_traces_without_eval_report(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from test_ufo_live import live_plan

    from eb.api.routes_ufo_native import spec_for
    from eb.orchestrator.collect import archive_partial_eval
    from eb.store.models import Artifact, EvalReport
    from eb.ufo_native import sha256_file

    rid = "run_0123456789abcdef0123"
    output = tmp_path / "runs" / rid / "outputs"
    (output / "traces").mkdir(parents=True)
    (output / "events.jsonl").write_text('{"event":"error"}\n')
    (output / "traces/chunk-0000.npz").write_bytes(b"recorded partial data")
    (output / "trace-index.json").write_text(
        json.dumps({"chunks": ["traces/chunk-0000.npz"], "complete": False})
    )

    async def scan(path):
        path = Path(path)
        return {"files": 1, "samples": [{"size": path.stat().st_size, "sha256": sha256_file(path)}]}

    monkeypatch.setattr("eb.orchestrator.ufo_live.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("eb.store.ebd.EbdClient.scan", lambda self, path: scan(path))
    session = SimpleNamespace(scalar=AsyncMock(return_value=None), add=Mock())
    job = SimpleNamespace(kind="serve.policy", spec=spec_for(live_plan()).model_dump(), status="failed")
    run = SimpleNamespace(id=rid, metrics_summary={}, abort_reason="docker_error")
    await archive_partial_eval(session, job, run)
    added = [call.args[0] for call in session.add.call_args_list]
    artifacts = [row for row in added if isinstance(row, Artifact)]
    assert len(artifacts) == 1 and artifacts[0].meta["complete"] is False
    assert not any(isinstance(row, EvalReport) for row in added)
    assert {row["path"] for row in run.metrics_summary["files"]} == {
        "events.jsonl",
        "trace-index.json",
        "traces/chunk-0000.npz",
    }
