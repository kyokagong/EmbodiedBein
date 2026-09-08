from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.ids import new_id
from eb.paths import PathNotAllowed, validate_allowed_path
from eb.store.models import LineageEdge, WeightModel
from eb.workbench import dir_size

router = APIRouter()


class BaseModelIn(BaseModel):
    name: str
    path: str
    embodiment_id: str | None = None
    framework: str = "pytorch"


def _out(row: WeightModel) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "path": row.path,
        "framework": row.framework,
        "size_bytes": row.size_bytes,
        "embodiment_id": row.embodiment_id,
        "train_job_id": row.train_job_id,
        "parent_model_id": row.parent_model_id,
        "metrics_at_checkpoint": row.metrics_at_checkpoint or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/weights")
async def list_weights(session: SessionDep, principal: PrincipalDep, kind: str | None = None) -> dict:
    rows = (await session.scalars(select(WeightModel))).all()
    items = [_out(r) for r in rows]
    if kind:
        items = [i for i in items if i["kind"] == kind]
    return {"items": items}


@router.post("/weights")
async def register_base(body: BaseModelIn, session: SessionDep, principal: PrincipalDep) -> dict:
    try:
        path = validate_allowed_path(body.path)
    except PathNotAllowed as e:
        raise HTTPException(400, str(e)) from e
    if not path.exists():
        raise HTTPException(400, "路径不存在")
    row = WeightModel(
        id=new_id("mdl"),
        name=body.name or path.name,
        kind="base",
        path=str(path),
        framework=body.framework,
        size_bytes=dir_size(path),
        embodiment_id=body.embodiment_id,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _out(row)


@router.get("/weights/{mid}")
async def get_weight(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(WeightModel, mid)
    if not row:
        raise HTTPException(404)
    edges = (
        await session.scalars(
            select(LineageEdge).where((LineageEdge.src_id == mid) | (LineageEdge.dst_id == mid))
        )
    ).all()
    return {
        **_out(row),
        "edges": [
            {"src_type": e.src_type, "src_id": e.src_id, "dst_type": e.dst_type, "dst_id": e.dst_id, "relation": e.relation}
            for e in edges
        ],
    }


@router.get("/weights/{mid}/lineage")
async def weight_lineage(mid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    return await get_weight(mid, session, principal)
