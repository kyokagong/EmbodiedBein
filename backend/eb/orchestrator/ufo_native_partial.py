"""Keep verified native simulation cases after an interrupted evaluation."""

from __future__ import annotations

import hashlib
import json

import httpx
from sqlalchemy import select

from eb.ids import new_id
from eb.orchestrator.ufo_native import verify_index
from eb.settings import get_settings
from eb.store.models import EvalReport, LineageEdge
from eb.ufo_native import safe_file, validate_cases


async def archive_partial_native(session, job, run, *, invalid: bool = False) -> None:
    params = (getattr(job, "spec", None) or {}).get("params") or {}
    plan = params.get("native_plan")
    if (
        getattr(job, "kind", None) not in ("eval.sim", "eval.sim2sim")
        or params.get("operation") != "ufo_native_eval"
        or not isinstance(plan, dict)
        or plan.get("operation") != "ufo_native_eval"
    ):
        return
    if await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id)):
        return

    expected = {case["case_id"]: case for case in plan["cases"]}
    identity = {
        "run_id": run.id,
        "plan_hash": plan["hash"],
        "model_id": plan["model_id"],
        "payload_sha256": plan["source_artifact_sha256"],
    }
    data_root = get_settings().data_root
    output = data_root / "runs" / run.id / "outputs"
    found, indexes = {}, {}
    seen, rejected = set(), set()
    errors = []
    partial_files = []
    try:
        if not output.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("Native case output leaves managed storage")
        path = safe_file(output, "cases.jsonl")
        with path.open("rb") as stream:
            raw = stream.read(2 * 1024 * 1024 + 1)
        if not 0 < len(raw) <= 2 * 1024 * 1024:
            raise ValueError("Native case evidence is empty or exceeds 2 MiB")
        lines = raw.splitlines()
        if len(lines) > 200:
            raise ValueError("Native case evidence exceeds 200 rows")
        index = {"path": "cases.jsonl", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        await verify_index(output, [index])
        partial_files.append(index)
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            case_id = None
            try:
                row = json.loads(line)
                case_id = row.get("case_id") if isinstance(row, dict) else None
                if not isinstance(case_id, str) or case_id not in expected:
                    raise ValueError("Unknown native case identity")
                if any(row.get(key) != value for key, value in identity.items()):
                    raise ValueError("Native case execution identity differs")
                if case_id in seen:
                    found.pop(case_id, None)
                    indexes.pop(case_id, None)
                    raise ValueError("Duplicate native case identity")
                seen.add(case_id)
                validate_cases({**plan, "cases": [expected[case_id]]}, [row])
                media = row.get("media") or {}
                if (
                    not isinstance(media, dict)
                    or media.get("decodable") is not True
                    or type(media.get("frames")) is not int
                    or media["frames"] < 1
                    or media.get("fps") != 25
                    or type(media.get("seconds")) not in (int, float)
                    or not 0 < media["seconds"] <= 21
                ):
                    raise ValueError("Native case video is not decodable")
                files = row.get("evidence_files")
                required = {row["video"], f"traces/{case_id}.npz"}
                if not isinstance(files, list) or len(files) != 2 or {f["path"] for f in files} != required:
                    raise ValueError("Native case video/trace index is incomplete")
                await verify_index(output, files)
                found[case_id], indexes[case_id] = row, files
            except (
                ValueError,
                OSError,
                TypeError,
                KeyError,
                IndexError,
                RecursionError,
                httpx.HTTPError,
            ) as exc:
                if isinstance(case_id, str) and case_id in expected:
                    rejected.add(case_id)
                errors.append(f"row {number}: {exc}")
    except (ValueError, OSError, TypeError, KeyError, IndexError, httpx.HTTPError) as exc:
        errors.append(str(exc))

    rows = [found[case_id] for case_id in expected if case_id in found]
    partial_files.extend(item for case_id in expected for item in indexes.get(case_id, []))
    planned, completed = len(expected), len(rows)
    remaining = planned - completed
    successes = sum(row["status"] == "success" for row in rows)
    cancelled = (
        getattr(job, "status", None) == "cancelled" or getattr(run, "abort_reason", None) == "cancelled"
    )
    infra_failed = getattr(job, "status", None) in ("failed", "infra_failed", "timeout")
    metrics = {
        "planned": planned,
        "completed": completed,
        "valid": completed,
        "invalid": len(rejected),
        "remaining": remaining,
        "infra_failed": remaining if infra_failed and not cancelled else 0,
        "cancelled": remaining if cancelled else 0,
        "successes": successes,
        "success_rate": successes / planned if planned else None,
        "cases": rows,
    }
    report = EvalReport(
        id=new_id("ev"),
        suite_id=plan["suite_id"],
        model_version_id=plan["model_id"],
        run_id=run.id,
        metrics=metrics,
        gate_results={
            "qualification": "incomplete",
            "validity": "incomplete",
            "backend": plan["backend"],
            "profile": plan["profile"],
            "plan_hash": plan["hash"],
            "reason": (
                getattr(job, "error", None) or getattr(run, "abort_reason", None) or "Evaluation incomplete"
            ),
            "evidence_error": "; ".join(errors)[:2000] or None,
            "output_invalid": invalid,
            "partial_files": partial_files,
            "scope": "simulation_only",
            "physics_family": "MuJoCo",
        },
    )
    session.add(report)
    session.add(
        LineageEdge(
            src_type="model_version",
            src_id=plan["model_id"],
            dst_type="eval_report",
            dst_id=report.id,
            relation="evaluated",
            run_id=run.id,
        )
    )
    run.metrics_summary = {
        **(getattr(run, "metrics_summary", None) or {}),
        "eval_report_id": report.id,
        "qualification": "incomplete",
        "validity": "incomplete",
        **{key: value for key, value in metrics.items() if key != "cases"},
    }
