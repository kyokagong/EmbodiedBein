from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.disk import disk_preflight
from eb.ids import new_id
from eb.paths import PathNotAllowed, is_under, validate_allowed_path
from eb.store.models import Dataset
from eb.workbench import dir_size

router = APIRouter()


class DatasetIn(BaseModel):
    name: str = ""
    source_kind: str = "path"
    source_uri: str
    dest_path: str | None = None
    hf_token: str | None = None


def _ds_out(row: Dataset) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "source_kind": row.source_kind,
        "source_uri": row.source_uri,
        "dest_path": row.dest_path,
        "format": row.format,
        "status": row.status,
        "bytes": row.total_bytes,
        "total_bytes": row.total_bytes,
        "download_progress": row.download_progress or {},
        "summary_stats": row.summary_stats or {},
        "error_message": row.error_message,
        "layer": "L1",
        "episodes": (row.summary_stats or {}).get("episodes_count"),
        "license": row.license,
        "embodiment_env_id": row.embodiment_env_id,
        "g2": "—",
        "content_hash": None,
    }


@router.post("/datasets")
async def create_dataset(body: DatasetIn, session: SessionDep, principal: PrincipalDep) -> dict:
    kind = body.source_kind
    if kind not in {"path", "huggingface", "http"}:
        raise HTTPException(400, "source_kind 必须是 path | huggingface | http")
    dest = body.dest_path
    if kind == "path":
        try:
            dest_p = validate_allowed_path(body.source_uri)
        except PathNotAllowed as e:
            raise HTTPException(400, str(e)) from e
        dest = str(dest_p)
    else:
        dest = dest or f"/var/lib/eb/datasets/{new_id('ds')}"
        try:
            dest_p = validate_allowed_path(dest)
        except PathNotAllowed as e:
            raise HTTPException(400, str(e)) from e
        dest = str(dest_p)
    check = disk_preflight(dest, needed_bytes=0, archive=kind == "http")
    if not check.ok:
        raise HTTPException(400, check.reason)
    did = new_id("ds")
    row = Dataset(
        id=did,
        name=body.name or Path(dest).name,
        format="unknown",
        license=None,
        source_kind=kind,
        source_uri=body.source_uri,
        dest_path=dest,
        status="ready" if kind == "path" else "pending",
        total_bytes=dir_size(Path(dest)) if kind == "path" else 0,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    if kind != "path":
        asyncio.create_task(_download_later(did, body.hf_token))
    return _ds_out(row)


async def _download_later(did: str, token: str | None) -> None:
    from eb.store.db import SessionLocal

    if SessionLocal is None:
        return
    async with SessionLocal() as session:
        row = await session.get(Dataset, did)
        if not row:
            return
        row.status = "downloading"
        await session.commit()
        dest = Path(row.dest_path)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if row.source_kind == "http":
                tmp = dest / ".download.tmp"
                async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                    async with client.stream("GET", row.source_uri) as resp:
                        resp.raise_for_status()
                        with tmp.open("wb") as f:
                            async for chunk in resp.aiter_bytes(1024 * 256):
                                f.write(chunk)
                tmp.replace(dest / (Path(row.source_uri).name or "archive.bin"))
            elif row.source_kind == "huggingface":
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as e:
                    raise RuntimeError("未安装 huggingface_hub") from e
                snapshot_download(row.source_uri, local_dir=str(dest), token=token)
            row.status = "ready"
            row.total_bytes = dir_size(dest)
            row.error_message = None
        except Exception as e:
            row.status = "failed"
            row.error_message = str(e)
        await session.commit()


@router.get("/datasets/{did}/tree")
async def dataset_tree(
    did: str,
    session: SessionDep,
    principal: PrincipalDep,
    path: str = "",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    root = Path(row.dest_path or ((row.meta or {}).get("root") if row.meta else "") or "")
    if not root.exists():
        raise HTTPException(404, "dest_path 不存在")
    rel = path.strip("/")
    cur = (root / rel).resolve() if rel else root.resolve()
    if not is_under(root, cur) and cur != root.resolve():
        raise HTTPException(403, "path 越权")
    if not cur.is_dir():
        raise HTTPException(400, "不是目录")
    kids = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    start = max(page - 1, 0) * page_size
    chunk = kids[start : start + page_size]
    items = []
    for p in chunk:
        items.append(
            {
                "name": p.name,
                "path": str(p.relative_to(root)).replace("\\", "/"),
                "is_dir": p.is_dir(),
                "bytes": p.stat().st_size if p.is_file() else None,
                "mtime": int(p.stat().st_mtime),
            }
        )
    return {"path": rel, "page": page, "page_size": page_size, "total": len(kids), "items": items}


@router.get("/datasets/{did}/media")
async def dataset_media(did: str, session: SessionDep, principal: PrincipalDep, path: str = Query(...)):
    row = await session.get(Dataset, did)
    if not row:
        raise HTTPException(404)
    root = Path(row.dest_path or ((row.meta or {}).get("root") if row.meta else "") or "")
    dest = (root / path).resolve()
    if not is_under(root, dest) or not dest.is_file():
        raise HTTPException(404)
    return FileResponse(dest)
