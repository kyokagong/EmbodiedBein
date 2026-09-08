"""Claim run outputs into CAS / models using measured ONNX I/O."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.contract import EmbodimentEnv
from eb.domain.fsm import JobStatus, RunStatus
from eb.eval.parse_play import parse_play
from eb.gates.engine import g3_model
from eb.ids import new_id
from eb.onnx_io import read_signature, sha256_file
from eb.settings import get_settings
from eb.store.models import (
    Artifact,
    EmbodimentEnvRow,
    EvalReport,
    Job,
    LineageEdge,
    ModelFamily,
    ModelVersion,
    Run,
)


async def claim_outputs(session: AsyncSession, job: Job, run: Run) -> None:
    root = get_settings().data_root / "runs" / run.id
    outputs = root / "outputs"
    cas_root = get_settings().data_root / "cas"
    cas_root.mkdir(parents=True, exist_ok=True)

    metrics_path = outputs / "metrics.json"
    if job.kind == "eval.sim" and metrics_path.exists():
        parsed = parse_play(metrics_path)
        report = EvalReport(
            id=new_id("ev"),
            suite_id=(job.spec.get("params") or {}).get("suite_id") or "suite_biped_locomotion_v2",
            model_version_id=(job.spec.get("inputs") or {}).get("base_model"),
            run_id=run.id,
            metrics={
                "success_rate": parsed["success_rate"],
                "per_seed": parsed["per_seed"],
                "mean_reward": parsed["mean_reward"],
            },
            gate_results={"g3": "pending"},
        )
        session.add(report)
        run.metrics_summary = {k: v for k, v in report.metrics.items() if k != "per_seed"}

    search_roots = [p for p in (outputs, root / "logs") if p.exists()]
    if not search_roots:
        return
    onnx_files: list[Path] = []
    for base in search_roots:
        onnx_files.extend(base.rglob("exported/policy.onnx"))
        onnx_files.extend(p for p in base.rglob("*.onnx") if "exported" not in p.parts or p.name != "policy.onnx")
    # unique
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in onnx_files:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    ckpt_files: list[Path] = []
    for base in search_roots:
        ckpt_files.extend(base.rglob("model_*.pt"))
    ckpt_seen: set[str] = set()
    ckpt_uniq: list[Path] = []
    for p in ckpt_files:
        k = str(p)
        if k not in ckpt_seen:
            ckpt_seen.add(k)
            ckpt_uniq.append(p)
    ckpt_files = ckpt_uniq
    if not uniq and not ckpt_files:
        return

    fam = await session.get(ModelFamily, "mf_run")
    if fam is None:
        fam = ModelFamily(id="mf_run", name="run policies", family="mlp")
        session.add(fam)
        await session.flush()
    ctr_row = await session.get(EmbodimentEnvRow, job.embodiment_env_id)
    ctr = EmbodimentEnv.model_validate(ctr_row.spec) if ctr_row else None

    for path in uniq:
        try:
            digest = sha256_file(path)
            sig = read_signature(path)
        except Exception:
            job.status = JobStatus.FAILED.value
            job.error = f"onnx unreadable: {path.name}"
            run.status = RunStatus.ABORTED.value
            run.abort_reason = "g3_unreadable"
            return
        if ctr:
            gate = g3_model(sig.get("obs"), sig.get("act"), ctr)
            if not gate.passed:
                job.status = JobStatus.FAILED.value
                job.error = "; ".join(v.hint or str(v.path) for v in gate.violations)
                run.status = RunStatus.ABORTED.value
                run.abort_reason = "g3_failed"
                run.metrics_summary = {"g3": gate.model_dump()}
                return
        dest_dir = cas_root / digest[:2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / digest
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        existing = await session.scalar(select(Artifact).where(Artifact.sha256 == digest))
        if existing:
            art = existing
        else:
            art = Artifact(
                id=new_id("art"),
                sha256=digest,
                kind="onnx",
                size_bytes=path.stat().st_size,
                storage=str(dest),
                run_id=run.id,
                meta={"name": path.name, **sig},
            )
            session.add(art)
            await session.flush()
        mv = ModelVersion(
            id=new_id("mv"),
            family_id=fam.id,
            name=path.stem,
            embodiment_env_id=job.embodiment_env_id,
            io_signature={"obs": sig.get("obs"), "act": sig.get("act")},
            provenance="full",
            artifact_id=art.id,
            tags=["policy", "run"],
            card={"source_path": str(path), "measured": True},
        )
        session.add(mv)
        await session.flush()
        session.add(
            LineageEdge(
                src_type="run",
                src_id=run.id,
                dst_type="model",
                dst_id=mv.id,
                relation="produced",
                run_id=run.id,
            )
        )
    for path in ckpt_files:
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        dest_dir = cas_root / digest[:2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / digest
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        session.add(
            Artifact(
                id=new_id("art"),
                sha256=digest,
                kind="checkpoint",
                size_bytes=path.stat().st_size,
                storage=str(dest),
                run_id=run.id,
                meta={"name": path.name, "folded": True},
            )
        )
