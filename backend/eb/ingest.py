from __future__ import annotations

import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.orm.attributes import flag_modified

from eb.ids import new_id
from eb.onnx_io import read_signature, sha256_file
from eb.store.models import Artifact, AuditLog, ModelFamily, ModelVersion

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
        return "ee_bhl_biped_12dof_v1", {"obs": 45, "act": 12}
    if "wbc" in n:
        return "ee_bhl_wbc_22dof_v1", {"obs": 85, "act": 22}
    if "video" in n or "hand" in n or "fixedhand" in n or "ego" in n:
        return "ee_bhl_fixedhand_10dof_v1", {"obs": 10, "act": 10}
    if "humanoid" in n:
        return "ee_bhl_humanoid_22dof_v1", {"obs": 75, "act": 22}
    return None, {}


async def ingest_all(session: AsyncSession) -> dict[str, int]:
    n_ck = await ingest_checkpoints(session)
    n_log = await ingest_tfevents(session)
    n_fix = await refresh_measured_onnx_io(session)
    return {"checkpoints": n_ck, "runs": n_log, "onnx_refreshed": n_fix}


async def refresh_measured_onnx_io(session: AsyncSession) -> int:
    """Overwrite filename-hint I/O with ONNX graph dims for existing rows."""
    from eb.store.models import Artifact, ModelVersion

    n = 0
    rows = (await session.scalars(select(ModelVersion))).all()
    for mv in rows:
        if not mv.artifact_id:
            continue
        art = await session.get(Artifact, mv.artifact_id)
        if art is None or art.kind != "onnx":
            continue
        path = Path(art.storage)
        if not path.exists() or path.suffix.lower() != ".onnx":
            continue
        try:
            sig = read_signature(path)
        except Exception:
            continue
        io = {"obs": sig.get("obs"), "act": sig.get("act")}
        if mv.io_signature == io and (mv.card or {}).get("measured"):
            continue
        mv.io_signature = io
        card = dict(mv.card or {})
        card["measured"] = True
        card["filename_hint_only"] = False
        mv.card = card
        flag_modified(mv, "io_signature")
        flag_modified(mv, "card")
        n += 1
    return n


async def ingest_checkpoints(session: AsyncSession, root: Path | None = None) -> int:
    from eb.orchestrator.deletion import references
    tombstones = (await session.scalars(select(AuditLog).where(AuditLog.action == "storage.delete"))).all()
    deleted_sources = [path for row in tombstones if (row.before or {}).get("kind") == "model"
                       for path in row.before.get("paths", []) + row.before.get("preserved_paths", [])]
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
        if references(str(path), set(), deleted_sources):
            continue
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        existing = await session.scalar(select(Artifact).where(Artifact.sha256 == digest))
        if existing:
            continue
        size = path.stat().st_size
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        cid, _hint_io = infer_contract(path)
        io: dict[str, Any] = {}
        measured = False
        if path.suffix.lower() == ".onnx":
            try:
                sig = read_signature(path)
                io = {"obs": sig.get("obs"), "act": sig.get("act")}
                measured = True
            except Exception:
                io = {}
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
                embodiment_env_id=cid,
                io_signature=io,
                provenance="partial",
                artifact_id=art.id,
                tags=["legacy", path.suffix.lstrip("."), "policy" if path.suffix.lower() == ".onnx" else "checkpoint"],
                card={
                    "source_path": str(path),
                    "contract_hint": cid,
                    "intended_use": "遗留导入策略",
                    "limitations": (
                        "五元组不完整；I/O 为 ONNX 实测"
                        if measured
                        else "五元组不完整；未能实测 I/O，文件名仅作 UI hint"
                    ),
                    "filename_hint_only": True,
                    "measured": measured,
                },
            )
        )
        n += 1
    return n


async def ingest_tfevents(session: AsyncSession) -> int:
    """Do not mint succeeded Jobs from host logs.

    Historical tfevents stay on disk; the collector/TSDB path is for platform runs only.
    """
    return 0


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
