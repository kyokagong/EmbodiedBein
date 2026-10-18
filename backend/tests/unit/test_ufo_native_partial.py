import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from eb.orchestrator.ufo_native_partial import archive_partial_native
from eb.store.models import EvalReport
from eb.ufo_native import plan_hash, safe_file, suite_spec


@pytest.fixture
def context(tmp_path, monkeypatch):
    output = tmp_path / "runs/run_partial/outputs"
    output.mkdir(parents=True)
    plan = {
        **suite_spec("mjlab", "smoke"),
        "operation": "ufo_native_eval",
        "model_id": "mv_native",
        "source_artifact_sha256": "sha256:" + "a" * 64,
        "suite_id": "suite_native_smoke",
    }
    plan["hash"] = plan_hash(plan)
    job = SimpleNamespace(
        id="job_partial",
        kind="eval.sim",
        status="failed",
        error="worker exited",
        spec={"params": {"operation": "ufo_native_eval", "native_plan": plan}},
    )
    run = SimpleNamespace(id="run_partial", abort_reason="exit_1", metrics_summary={"kept": True})
    session = SimpleNamespace(scalar=AsyncMock(return_value=None), add=Mock())
    monkeypatch.setattr(
        "eb.orchestrator.ufo_native_partial.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )

    async def verify(root, files):
        for item in files:
            path = safe_file(root, item["path"])
            content = path.read_bytes()
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("damaged output hash")

    verifier = AsyncMock(side_effect=verify)
    monkeypatch.setattr("eb.orchestrator.ufo_native_partial.verify_index", verifier)
    return SimpleNamespace(output=output, plan=plan, job=job, run=run, session=session, verifier=verifier)


def completed_row(context, index=0, *, success=False):
    case = context.plan["cases"][index]
    files = []
    for name in (f"videos/{case['case_id']}.mp4", f"traces/{case['case_id']}.npz"):
        path = context.output / name
        path.parent.mkdir(exist_ok=True)
        content = name.encode()
        path.write_bytes(content)
        files.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return {
        **case,
        "run_id": context.run.id,
        "plan_hash": context.plan["hash"],
        "model_id": context.plan["model_id"],
        "payload_sha256": context.plan["source_artifact_sha256"],
        "status": "success" if success else "task_failure",
        "reason": "completed" if success else "fall",
        "steps": 1000 if success else 3,
        "video": files[0]["path"],
        "evidence_files": files,
        "media": {
            "decodable": True,
            "frames": 500 if success else 1,
            "fps": 25,
            "seconds": 20 if success else 0.04,
        },
        "metrics": {
            "min_height_m": 0.7,
            "max_tilt_rad": 0.1,
            "speed_rmse_m_s": 0.1,
            "drift_m": 0.1,
            "forward_displacement_m": 5,
            "joint_violation_fraction": 0,
            "velocity_violation_fraction": 0,
            "torque_violation_fraction": 0,
        },
    }


def write_rows(context, rows, tail=""):
    (context.output / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + tail)


def report_from(context):
    return next(
        call.args[0] for call in context.session.add.call_args_list if isinstance(call.args[0], EvalReport)
    )


@pytest.mark.parametrize("backend,kind", [("mjlab", "eval.sim"), ("mujoco", "eval.sim2sim")])
async def test_partial_native_keeps_verified_cases_with_planned_denominator(context, backend, kind):
    context.plan["backend"] = backend
    context.job.kind = kind
    row = completed_row(context, success=True)
    write_rows(context, [row])
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.model_version_id == "mv_native"
    assert report.metrics["planned"] == 3
    assert report.metrics["completed"] == report.metrics["valid"] == 1
    assert report.metrics["remaining"] == 2
    assert report.metrics["success_rate"] == pytest.approx(1 / 3)
    assert report.metrics["cases"] == [row]
    assert report.gate_results["qualification"] == report.gate_results["validity"] == "incomplete"
    assert report.gate_results["backend"] == backend
    assert report.gate_results["reason"] == "worker exited"
    assert {item["path"] for item in report.gate_results["partial_files"]} == {
        "cases.jsonl",
        row["video"],
        f"traces/{row['case_id']}.npz",
    }
    assert context.run.metrics_summary["eval_report_id"] == report.id
    assert context.run.metrics_summary["qualification"] == "incomplete"
    assert context.run.metrics_summary["kept"] is True
    assert context.verifier.await_count == 2


async def test_missing_case_file_still_records_incomplete_cancelled_run(context):
    context.job.status = "cancelled"
    context.run.abort_reason = "cancelled"
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.metrics["planned"] == report.metrics["remaining"] == report.metrics["cancelled"] == 3
    assert report.metrics["completed"] == report.metrics["successes"] == 0
    assert report.metrics["success_rate"] == 0
    assert report.gate_results["qualification"] == "incomplete"
    assert report.gate_results["partial_files"] == []


async def test_duplicate_and_unknown_rows_do_not_discard_unrelated_valid_cases(context):
    first, second = completed_row(context), completed_row(context, 1)
    unknown = {**deepcopy(first), "case_id": "unknown"}
    write_rows(context, [first, second, first, unknown])
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.metrics["cases"] == [second]
    assert report.metrics["invalid"] == 1
    assert report.metrics["remaining"] == 2
    assert "duplicate" in report.gate_results["evidence_error"].lower()
    assert "unknown" in report.gate_results["evidence_error"].lower()
    paths = {item["path"] for item in report.gate_results["partial_files"]}
    assert first["video"] not in paths and second["video"] in paths


async def test_truncated_json_preserves_preceding_complete_cases(context):
    row = completed_row(context)
    write_rows(context, [row], '{"case_id": "smoke-1"')
    await archive_partial_native(context.session, context.job, context.run, invalid=True)
    report = report_from(context)
    assert report.metrics["cases"] == [row]
    assert report.gate_results["validity"] == "incomplete"
    assert report.gate_results["evidence_error"]


@pytest.mark.parametrize("field", ["run_id", "plan_hash", "model_id", "payload_sha256"])
@pytest.mark.parametrize("missing", [False, True])
async def test_partial_case_rejects_foreign_or_missing_execution_identity(context, field, missing):
    row = completed_row(context)
    if missing:
        row.pop(field)
    else:
        row[field] = "another-execution"
    write_rows(context, [row])
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.metrics["completed"] == 0
    assert report.metrics["invalid"] == 1
    assert report.metrics["cases"] == []
    assert "identity" in report.gate_results["evidence_error"].lower()
    assert {item["path"] for item in report.gate_results["partial_files"]} == {"cases.jsonl"}


@pytest.mark.parametrize("damage", ["nan", "video", "hash", "seed", "success", "missing_trace"])
async def test_invalid_completed_case_is_excluded(context, damage):
    row = completed_row(context)
    if damage == "nan":
        row["metrics"]["speed_rmse_m_s"] = float("nan")
    elif damage == "video":
        row["media"]["decodable"] = False
    elif damage == "hash":
        (context.output / row["video"]).write_bytes(b"damaged")
    elif damage == "seed":
        row["seed"] += 1
    elif damage == "success":
        row["status"] = "success"
    else:
        row["evidence_files"] = row["evidence_files"][:1]
    write_rows(context, [row])
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.metrics["completed"] == report.metrics["valid"] == 0
    assert report.metrics["invalid"] == 1
    assert report.metrics["cases"] == []
    assert {item["path"] for item in report.gate_results["partial_files"]} == {"cases.jsonl"}


async def test_existing_complete_report_is_never_overwritten(context):
    existing = EvalReport(id="ev_existing", gate_results={"validity": "complete", "qualification": "pass"})
    context.session.scalar.return_value = existing
    await archive_partial_native(context.session, context.job, context.run, invalid=True)
    context.session.add.assert_not_called()
    context.verifier.assert_not_awaited()
    assert existing.gate_results["qualification"] == "pass"


async def test_repeated_partial_collection_does_not_duplicate_report(context):
    write_rows(context, [completed_row(context)])
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    count = context.session.add.call_count
    context.session.scalar.return_value = report
    await archive_partial_native(context.session, context.job, context.run)
    assert context.session.add.call_count == count
    assert context.run.metrics_summary["eval_report_id"] == report.id


async def test_unavailable_verifier_keeps_incomplete_report_without_publishing_files(context):
    write_rows(context, [completed_row(context)])
    context.verifier.side_effect = httpx.ConnectError("ebd socket unavailable")
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.metrics["completed"] == 0
    assert report.gate_results["partial_files"] == []
    assert "ebd socket unavailable" in report.gate_results["evidence_error"]


@pytest.mark.parametrize("kind,operation", [("train.rl", "ufo_native_eval"), ("eval.sim", "other")])
async def test_unrelated_jobs_are_ignored(context, kind, operation):
    context.job.kind = kind
    context.job.spec["params"]["operation"] = operation
    await archive_partial_native(context.session, context.job, context.run)
    context.session.scalar.assert_not_awaited()
    context.session.add.assert_not_called()


@pytest.mark.parametrize("unsafe", ["symlink", "oversize"])
async def test_untrusted_jsonl_is_not_published(context, tmp_path, unsafe):
    target = context.output / "cases.jsonl"
    if unsafe == "symlink":
        external = tmp_path / "external.jsonl"
        external.write_text("{}")
        target.symlink_to(external)
    else:
        target.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    await archive_partial_native(context.session, context.job, context.run)
    report = report_from(context)
    assert report.metrics["completed"] == 0
    assert report.gate_results["partial_files"] == []
    assert report.gate_results["evidence_error"]
