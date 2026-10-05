from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.paths import default_prefixes
from eb.settings import get_settings
from eb.store.models import SystemSettingsRow

router = APIRouter()


class SettingsIn(BaseModel):
    allowed_path_prefixes: list[str] | None = None
    rlinf_repo_path: str | None = None
    default_train_image: str | None = None
    disk_alert_threshold_gb: int | None = None


async def _row(session) -> SystemSettingsRow:
    row = await session.get(SystemSettingsRow, 1)
    if row:
        return row
    settings = get_settings()
    row = SystemSettingsRow(
        id=1,
        allowed_path_prefixes=default_prefixes(),
        rlinf_repo_path=str(settings.rlinf_repo),
        default_train_image="eb/rlinf:0.4.0-emb",
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@router.get("/workbench/settings")
async def get_wb_settings(session: SessionDep, principal: PrincipalDep) -> dict:
    row = await _row(session)
    return {
        "allowed_path_prefixes": row.allowed_path_prefixes,
        "rlinf_repo_path": row.rlinf_repo_path,
        "default_train_image": row.default_train_image,
        "disk_alert_threshold_gb": row.disk_alert_threshold_gb,
    }


@router.patch("/workbench/settings")
async def patch_wb_settings(body: SettingsIn, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await _row(session)
    if body.allowed_path_prefixes is not None:
        row.allowed_path_prefixes = body.allowed_path_prefixes
    if body.rlinf_repo_path is not None:
        row.rlinf_repo_path = body.rlinf_repo_path
    if body.default_train_image is not None:
        row.default_train_image = body.default_train_image
    if body.disk_alert_threshold_gb is not None:
        row.disk_alert_threshold_gb = body.disk_alert_threshold_gb
    await session.commit()
    return await get_wb_settings(session, principal)
