from __future__ import annotations

import hashlib
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.ids import new_id
from eb.store.models import Artifact, Experiment, Job, ModelFamily, ModelVersion, Run

CHECKPOINT_ROOTS = (
    Path("/data/kk/Berkeley-Humanoid-Lite/checkpoints"),
    Path("/data/kk/Berkeley-Humanoid-Lite/logs"),
)
TFEVENTS_ROOTS = (
    Path("/data/kk/Berkeley-Humanoid-Lite/logs/rsl_rl"),
    Path("/data/kk/Berkeley-Humanoid-Lite/logs/rlinf"),
)

_POLICY_SUFFIXES = {".onnx", ".pt", ".pth"}


def infer_contract(path: Path) -> tuple[str | None, dict[str, int]]:
    n = path.as_posix().lower()
    if "biped" in n:
        return "ec_bhl_biped_12dof_v1", {"obs": 45, "act": 12}
    if "wbc" in n:
        return "ec_bhl_wbc_22dof_v1", {"obs": 85, "act": 22}
    if "video" in n or "hand" in n or "fixedhand" in n or "ego" in n:
        return "ec_bhl_fixedhand_10dof_v1", {"obs": 10, "act": 10}
    if "humanoid" in n:
        return "ec_bhl_humanoid_22dof_v1", {"obs": 75, "act": 22}
    return None, {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


async def ingest_all(session: AsyncSession) -> dict[str, int]:
    n_ck = await ingest_checkpoints(session)
    n_log = await ingest_tfevents(session)
    return {"checkpoints": n_ck, "runs": n_log}


async def ingest_checkpoints(session: AsyncSession, root: Path | None = None) -> int:
    fam = await session.get(ModelFamily, "mf_legacy")
    if fam is None:
        fam = ModelFamily(id="mf_legacy", name="legacy checkpoints", family="mlp")
        session.add(fam)
        await session.flush()

    paths: list[Path] = []
    if root is not None:
        search = [root]
    else:
        search = [CHECKPOINT_ROOTS[0]]
        # exported onnx next to rsl_rl runs
        search.append(Path("/data/kk/Berkeley-Humanoid-Lite/logs/rsl_rl"))

    for base in search:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _POLICY_SUFFIXES:
                continue
            # prefer named policies; skip optimizer-like blobs
            if path.name in {"full_weights.pt"}:
                continue
            paths.append(path)

    # checkpoints/ first so the six ONNX policies get stable ids
    paths.sort(key=lambda p: (0 if "checkpoints" in p.parts else 1, str(p)))

    n = 0
    for path in paths:
        if n >= 80:
            break
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        existing = await session.scalar(select(Artifact).where(Artifact.sha256 == digest))
        if existing:
            continue
        size = path.stat().st_size
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        cid, io = infer_contract(path)
        art = Artifact(
            id=new_id("art"),
            sha256=digest,
            kind="onnx" if path.suffix.lower() == ".onnx" else "checkpoint",
            size_bytes=size,
            storage=str(path),
            meta={"mtime": mtime.isoformat(), "name": path.name, "discovered_at": str(path)},
            discovered_at=mtime.isoformat(),
        )
        session.add(art)
        await session.flush()
        session.add(
            ModelVersion(
                id=new_id("mv"),
                family_id=fam.id,
                name=path.stem,
                contract_id=cid,
                io_signature=io,
                provenance="partial",
                artifact_id=art.id,
                tags=["legacy", path.suffix.lstrip(".")],
                card={
                    "source_path": str(path),
                    "contract": cid or "待人工归属",
                    "intended_use": "遗留导入策略",
                    "limitations": "五元组不完整；I/O 由文件名推断，待 ONNX 实测核对",
                },
            )
        )
        n += 1
    return n


async def ingest_tfevents(session: AsyncSession) -> int:
    n = 0
    for base in TFEVENTS_ROOTS:
        if not base.exists():
            continue
        for ev in sorted(base.rglob("events.out.tfevents.*")):
            if n >= 40:
                return n
            run_dir = ev.parent
            rel = str(run_dir)
            idem = f"legacy:{hashlib.sha256(rel.encode()).hexdigest()[:20]}"
            exists = await session.scalar(select(Job).where(Job.idem_key == idem))
            if exists:
                continue
            cid, _ = infer_contract(run_dir)
            cid = cid or "ec_bhl_humanoid_22dof_v1"
            exp_id = f"exp_legacy_{cid.split('_')[2] if '_' in cid else 'misc'}"
            if await session.get(Experiment, exp_id) is None:
                session.add(Experiment(id=exp_id, name=f"legacy {cid}", contract_id=cid))
                await session.flush()
            series = downsample_scalars(parse_tfevents(ev))
            mtime = datetime.fromtimestamp(ev.stat().st_mtime, tz=timezone.utc)
            last = _last_values(series)
            job = Job(
                id=new_id("job"),
                experiment_id=exp_id,
                contract_id=cid,
                kind="train.rl",
                spec={
                    "kind": "train.rl",
                    "contract_id": cid,
                    "name": run_dir.name,
                    "params": {"source": rel},
                },
                spec_hash=idem,
                idem_key=idem,
                status="succeeded",
                created_by="ingest:legacy",
            )
            session.add(job)
            await session.flush()
            session.add(
                Run(
                    id=new_id("run"),
                    job_id=job.id,
                    attempt=1,
                    status="finished",
                    metrics_summary={
                        **last,
                        "source": str(ev),
                        "series": series,
                    },
                    provenance="partial",
                    started_at=mtime,
                    finished_at=mtime,
                )
            )
            n += 1
    return n


def parse_tfevents(path: Path, max_bytes: int = 8 * 1024 * 1024) -> dict[str, list[dict[str, float]]]:
    """Best-effort TFRecord + Event protobuf scalar extractor (no tensorflow)."""
    out: dict[str, list[dict[str, float]]] = {}
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError:
        return out
    i = 0
    n = len(data)
    while i + 12 <= n:
        length = struct.unpack_from("<Q", data, i)[0]
        i += 12  # length + crc
        if length <= 0 or i + length + 4 > n:
            break
        payload = data[i : i + length]
        i += length + 4
        step, values = _parse_event(payload)
        for tag, val in values:
            out.setdefault(tag, []).append({"step": float(step), "value": float(val)})
    return out


def downsample_scalars(series: dict[str, list[dict[str, float]]], cap: int = 120) -> dict[str, list[dict[str, float]]]:
    slim: dict[str, list[dict[str, float]]] = {}
    for tag, pts in series.items():
        if len(pts) <= cap:
            slim[tag] = pts
            continue
        step = max(1, len(pts) // cap)
        slim[tag] = pts[::step][:cap]
    return slim


def _last_values(series: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for tag, pts in series.items():
        if pts:
            last[tag] = pts[-1]["value"]
    return last


def _parse_event(buf: bytes) -> tuple[int, list[tuple[str, float]]]:
    step = 0
    values: list[tuple[str, float]] = []
    i = 0
    n = len(buf)
    while i < n:
        try:
            field, wire, i = _key(buf, i)
        except Exception:
            break
        if wire == 0:
            val, i = _varint(buf, i)
            if field == 2:
                step = val
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        elif wire == 2:
            ln, i = _varint(buf, i)
            inner = buf[i : i + ln]
            i += ln
            if field == 5:
                values.extend(_parse_summary(inner))
        else:
            break
    return step, values


def _parse_summary(buf: bytes) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    i = 0
    n = len(buf)
    while i < n:
        try:
            field, wire, i = _key(buf, i)
        except Exception:
            break
        if wire != 2:
            if wire == 0:
                _, i = _varint(buf, i)
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            else:
                break
            continue
        ln, i = _varint(buf, i)
        inner = buf[i : i + ln]
        i += ln
        if field == 1:
            tag, val = _parse_value(inner)
            if tag is not None and val is not None:
                out.append((tag, val))
    return out


def _parse_value(buf: bytes) -> tuple[str | None, float | None]:
    tag = None
    val = None
    i = 0
    n = len(buf)
    while i < n:
        try:
            field, wire, i = _key(buf, i)
        except Exception:
            break
        if field == 1 and wire == 2:
            ln, i = _varint(buf, i)
            tag = buf[i : i + ln].decode("utf-8", "replace")
            i += ln
        elif field == 2 and wire == 5:
            val = struct.unpack_from("<f", buf, i)[0]
            i += 4
        elif wire == 0:
            _, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        elif wire == 2:
            ln, i = _varint(buf, i)
            i += ln
        else:
            break
    return tag, val


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    x = 0
    s = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        x |= (b & 0x7F) << s
        if not (b & 0x80):
            return x, i
        s += 7
        if s > 70:
            break
    return x, i


def _key(buf: bytes, i: int) -> tuple[int, int, int]:
    k, i = _varint(buf, i)
    return k >> 3, k & 7, i
