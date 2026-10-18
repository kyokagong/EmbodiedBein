"""Register complete interactive diagnostics, without creating acceptance reports."""

from __future__ import annotations

import math
import re

import httpx
from sqlalchemy import select

from eb.ids import new_id
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Artifact, LineageEdge
from eb.ufo_native import TASKS, finite_tree, plan_hash, read_json, safe_file


def validate_receipt(plan, value):
    try:
        media, steps = value["recording"], value["steps"]
        chunks = value["trace_chunks"]
        filenames = [item["path"] for item in value["files"]]
        expected = {"events.jsonl", "trace-index.json", "timeseries.svg", "videos/session.mp4", *chunks}
        if (
            value.get("model_id") != plan["model_id"]
            or value.get("payload_sha256") != plan["source_artifact_sha256"]
            or value.get("environment_id") != plan["environment_id"]
            or value.get("environment_hash") != plan["environment_hash"]
            or value.get("diagnostic_only") is not True
            or value.get("complete") is not True
            or "qualification" in value
            or "success_rate" in value
            or type(steps) is not int
            or not 0 <= steps <= plan["duration_seconds"] * 50
            or type(value["command_count"]) is not int
            or not 0 <= value["command_count"] <= 1024
            or type(value["episodes"]) is not int
            or not 1 <= value["episodes"] <= value["command_count"] + 1
            or value["termination"] not in ("operator_stop", "lease_expired", "duration_limit")
            or value["task_order"] != [task["id"] for task in TASKS]
            or len(chunks) > 1100
            or len(set(chunks)) != len(chunks)
            or any(not re.fullmatch(r"traces/chunk-[0-9]{4}\.npz", name) for name in chunks)
            or (steps > 0) != bool(chunks)
            or len(set(filenames)) != len(filenames)
            or set(filenames) != expected
            or media.get("path") != "videos/session.mp4"
            or media.get("decodable") is not True
            or media.get("fps") != 5
            or media.get("frames") != max(1, math.ceil(steps / 10))
            or abs(media.get("seconds", 0) - media["frames"] / 5) > 0.01
            or set(value["limits"]) != {"joint", "velocity", "torque"}
            or any(type(v) is not int or not 0 <= v <= steps * 8 for v in value["limits"].values())
            or not finite_tree(value)
        ):
            raise ValueError("Interactive diagnostic evidence is incomplete or changed")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Interactive diagnostic receipt is malformed") from exc


async def collect_live(session, run, outputs, plan, value):
    from eb.orchestrator.ufo_native import verify_index

    validate_receipt(plan, value)
    await verify_index(outputs, value["files"], max_files=1105)
    artifact = await session.scalar(
        select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == "ufo_native_live_diagnostic")
    )
    if not artifact:
        artifact = Artifact(
            id=new_id("art"),
            kind="ufo_native_live_diagnostic",
            run_id=run.id,
            sha256=plan_hash(value),
            storage=str(outputs),
            pinned=True,
            size_bytes=sum(item["bytes"] for item in value["files"]),
            meta={"receipt": value, "diagnostic_only": True, "simulation_only": True},
        )
        session.add(artifact)
        session.add(
            LineageEdge(
                src_type="model_version",
                src_id=plan["model_id"],
                dst_type="artifact",
                dst_id=artifact.id,
                relation="diagnosed",
                run_id=run.id,
            )
        )
    run.metrics_summary = {
        "artifact_id": artifact.id,
        "diagnostic_only": True,
        "complete": True,
        "steps": value["steps"],
        "episodes": value["episodes"],
        "limits": value["limits"],
        "termination": value["termination"],
        "recording": value["recording"],
        "files": [
            {"path": name, "url": f"/api/v1/ufo-native/runs/{run.id}/files/{name}"}
            for name in ["receipt.json", *[item["path"] for item in value["files"]]]
        ],
    }


async def archive_partial_live(session, job, run, *, invalid=False):
    """Expose only bounded, scanned files after interruption; never claim a full recording."""
    from eb.ufo_live import validate_live_plan

    plan = validate_live_plan(job.spec["params"]["native_plan"])
    if await session.scalar(
        select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == "ufo_native_live_diagnostic")
    ):
        return
    output = get_settings().data_root / "runs" / run.id / "outputs"
    candidates = ["events.jsonl", "trace-index.json", "live/state.json", "live/scene.json"]
    errors, files = [], []
    try:
        index = read_json(safe_file(output, "trace-index.json"), limit=128 * 1024)
        chunks = index.get("chunks")
        if (
            not isinstance(chunks, list)
            or len(chunks) > 1100
            or len(set(chunks)) != len(chunks)
            or any(
                not isinstance(name, str) or not re.fullmatch(r"traces/chunk-[0-9]{4}\.npz", name)
                for name in chunks
            )
        ):
            raise ValueError("Partial live trace index is invalid")
        candidates += chunks
    except (ValueError, OSError, TypeError) as exc:
        errors.append(str(exc))
    for name in candidates:
        try:
            path = safe_file(output, name)
            limit = 64 * 1024**2 if name.endswith(".npz") else 2 * 1024**2
            if not 0 < path.stat().st_size <= limit:
                raise ValueError("Partial file exceeds bounded evidence size")
            info = await EbdClient().scan(str(path))
            samples = info.get("samples", [])
            if (
                info.get("files") != 1
                or len(samples) != 1
                or samples[0].get("size") != path.stat().st_size
                or not re.fullmatch(r"[a-f0-9]{64}", samples[0].get("sha256", ""))
            ):
                raise ValueError("Partial evidence could not be frozen by ebd")
            files.append({"path": name, "bytes": samples[0]["size"], "sha256": samples[0]["sha256"]})
        except (ValueError, OSError, TypeError, httpx.HTTPError) as exc:
            errors.append(f"{name}: {exc}")
    artifact = Artifact(
        id=new_id("art"),
        kind="ufo_native_live_diagnostic",
        run_id=run.id,
        pinned=True,
        sha256=plan_hash({"plan_hash": plan["hash"], "partial_files": files}),
        storage=str(output),
        size_bytes=sum(item["bytes"] for item in files),
        meta={
            "diagnostic_only": True,
            "simulation_only": True,
            "complete": False,
            "plan_hash": plan["hash"],
            "partial_files": files,
            "errors": errors,
            "invalid_output": invalid,
        },
    )
    session.add(artifact)
    session.add(
        LineageEdge(
            src_type="model_version",
            src_id=plan["model_id"],
            dst_type="artifact",
            dst_id=artifact.id,
            relation="diagnosed",
            run_id=run.id,
        )
    )
    run.metrics_summary = {
        "artifact_id": artifact.id,
        "diagnostic_only": True,
        "complete": False,
        "evidence_errors": errors,
        "files": [
            {"path": item["path"], "url": f"/api/v1/ufo-native/runs/{run.id}/files/{item['path']}"}
            for item in files
        ],
    }
