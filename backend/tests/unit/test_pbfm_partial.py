import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from test_pbfm_data_evaluation import plan, valid_rows

from eb.orchestrator.collect import archive_partial_eval
from eb.store.models import EvalReport


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mutated", "duplicate", "cancelled", "truncated", "missing"])
async def test_partial_pbfm_keeps_verified_cases_with_fixed_denominator(tmp_path, monkeypatch, failure):
    frozen = plan()
    output = tmp_path / "runs/run_partial/outputs"
    output.mkdir(parents=True)
    row = {
        **valid_rows()[0],
        "run_id": "run_partial",
        "plan_hash": frozen["hash"],
        "model_version_id": frozen["model_version_id"],
        "model_hash": frozen["model_hash"],
    }
    files = []
    for name in ("video", "depth_video", "depth", "trace"):
        path = output / row[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen")
        files.append({"path": row[name], "bytes": 6, "sha256": hashlib.sha256(b"frozen").hexdigest()})
    row["evidence_files"] = files
    (output / "cases.jsonl").write_text(
        json.dumps(row) + "\n" + (json.dumps(row) + "\n" if failure == "duplicate" else "")
    )
    if failure == "mutated":
        (output / row["video"]).write_bytes(b"broken")
    if failure == "truncated":
        with (output / "cases.jsonl").open("a") as stream:
            stream.write('{"case_id":')
    if failure == "missing":
        (output / "cases.jsonl").unlink()

    async def scan(name):
        from pathlib import Path

        path = Path(name)
        return {"files": 1, "samples": [{"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}

    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: NS(scan=scan))
    session = NS(scalar=AsyncMock(return_value=None), add=Mock())
    run = NS(id="run_partial", abort_reason=None, metrics_summary={})
    job = NS(
        kind="eval.sim",
        status="cancelled" if failure == "cancelled" else "failed",
        error="interrupted",
        spec={"params": {"operation": "evaluate_pbfm_terrain", "pbfm_plan": frozen}},
    )
    await archive_partial_eval(session, job, run)
    reports = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], EvalReport)]
    assert len(reports) == 1
    report = reports[0]
    expected_valid = 0 if failure in {"mutated", "duplicate", "missing"} else 1
    assert report.metrics["planned"] == 32
    assert report.metrics["valid"] == expected_valid
    assert report.gate_results["qualification"] == "incomplete"
    assert report.metrics["quality_passed"] is False
    assert len(report.metrics["cases"]) == expected_valid
    assert report.metrics["cancelled"] == (31 if failure == "cancelled" else 0)
    assert (
        len(report.gate_results["partial_files"]) == (0 if failure == "missing" else 1) + 4 * expected_valid
    )
    assert (
        sum(report.metrics[key] for key in ("valid", "invalid", "infra_failed", "cancelled", "not_started"))
        == 32
    )
    session.scalar.return_value = report
    session.add.reset_mock()
    await archive_partial_eval(session, job, run)
    session.add.assert_not_called()
