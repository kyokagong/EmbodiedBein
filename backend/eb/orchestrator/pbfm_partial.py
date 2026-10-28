"""Preserve verified P-BFM cases when an independent evaluation is interrupted."""

import hashlib
import json

import httpx
from sqlalchemy import select

from eb.ids import new_id
from eb.orchestrator.ufo_native import verify_index
from eb.pbfm_evaluation import EVAL_OPERATION, validate_case_rows, validate_eval_plan
from eb.pbfm_io import safe_file
from eb.store.models import EvalReport, LineageEdge


async def archive_partial_pbfm(session, job, run, *, invalid=False):
    from eb.settings import get_settings

    params = (getattr(job, "spec", None) or {}).get("params") or {}
    if job.kind != "eval.sim" or params.get("operation") != EVAL_OPERATION:
        return
    plan = validate_eval_plan(params["pbfm_plan"])
    if await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id)):
        return
    expected = {
        (case["id"], case["seed"], condition)
        for case in plan["cases"]
        for condition in ("normal", "occluded")
    }
    output = get_settings().data_root / "runs" / run.id / "outputs"
    identity = {
        "run_id": run.id,
        "plan_hash": plan["hash"],
        "model_version_id": plan["model_version_id"],
        "model_hash": plan["model_hash"],
    }
    found, indexes, seen, rejected, errors, journal = {}, {}, set(), set(), [], []
    try:
        path = safe_file(output, "cases.jsonl", maximum=4 * 1024**2)
        raw = path.read_bytes()
        lines = raw.splitlines()
        if len(lines) > 64:
            raise ValueError("P-BFM case journal exceeds bounded rollout count")
        journal = [{"path": "cases.jsonl", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
        await verify_index(output, journal)
        for number, line in enumerate(lines, 1):
            key = None
            try:
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("case_id"), str)
                    or type(row.get("seed")) is not int
                    or not isinstance(row.get("condition"), str)
                ):
                    raise ValueError("invalid P-BFM case identity")
                key = (row.get("case_id"), row.get("seed"), row.get("condition"))
                if key not in expected:
                    raise ValueError("unknown P-BFM case/seed/condition")
                if key in seen:
                    found.pop(key, None)
                    indexes.pop(key, None)
                    raise ValueError("duplicate P-BFM case")
                seen.add(key)
                if any(row.get(name) != value for name, value in identity.items()):
                    raise ValueError("P-BFM partial case producer differs")
                validate_case_rows(plan, [row], allow_partial=True)
                files = row.get("evidence_files")
                required = {row[name] for name in ("video", "depth_video", "depth", "trace")}
                if not isinstance(files, list) or len(files) != 4 or {f["path"] for f in files} != required:
                    raise ValueError("partial case lacks indexed RGB/depth/trace evidence")
                used = {item["path"] for group in indexes.values() for item in group}
                if required & used:
                    raise ValueError("partial case reuses another rollout's evidence")
                await verify_index(output, files)
                found[key], indexes[key] = row, files
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                AttributeError,
                IndexError,
                httpx.HTTPError,
            ) as exc:
                if key in expected:
                    rejected.add(key)
                errors.append(f"row {number}: {exc}")
    except (ValueError, OSError, KeyError, TypeError, httpx.HTTPError) as exc:
        errors.append(str(exc))
    rows = [found[key] for key in sorted(found)]
    remaining = len(expected) - len(rows) - len(rejected)
    cancelled = job.status == "cancelled" or getattr(run, "abort_reason", None) == "cancelled"
    failed = job.status in {"failed", "infra_failed", "timeout"} or invalid
    metrics = {
        "planned": len(expected),
        "completed": len(rows) + len(rejected),
        "valid": len(rows),
        "invalid": len(rejected),
        "remaining": remaining,
        "infra_failed": remaining if failed and not cancelled else 0,
        "cancelled": remaining if cancelled else 0,
        "not_started": remaining if not failed and not cancelled else 0,
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(expected),
        "quality_passed": False,
        "cases": rows,
    }
    report = EvalReport(
        id=new_id("ev"),
        suite_id=plan["suite_id"],
        model_version_id=plan["model_version_id"],
        run_id=run.id,
        metrics=metrics,
        gate_results={
            "qualification": "incomplete",
            "validity": "incomplete",
            "plan_hash": plan["hash"],
            "output_invalid": invalid,
            "formal_acceptance": False,
            "reason": getattr(job, "error", None) or "Evaluation interrupted",
            "evidence_error": "; ".join(errors)[:2000] or None,
            "partial_files": journal + [item for key in sorted(indexes) for item in indexes[key]],
        },
    )
    session.add(report)
    session.add(
        LineageEdge(
            src_type="model_version",
            src_id=plan["model_version_id"],
            dst_type="eval_report",
            dst_id=report.id,
            relation="evaluated",
            run_id=run.id,
        )
    )
    run.metrics_summary = {
        **(run.metrics_summary or {}),
        "eval_report_id": report.id,
        "qualification": "incomplete",
        "validity": "incomplete",
        **{key: value for key, value in metrics.items() if key != "cases"},
    }
