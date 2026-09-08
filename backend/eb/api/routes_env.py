from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.ids import new_id
from eb.store.ebd import EbdClient
from eb.store.models import SimImage

router = APIRouter()

_CATALOG = [
    {"name": "eb/rlinf:0.4.0-emb", "tag": "0.4.0-emb", "role": "train", "is_default": True},
    {"name": "eb/isaac-lab:2.3.2", "tag": "2.3.2", "role": "sim", "is_default": False},
]


async def _ensure_catalog(session) -> None:
    existing = {r.name: r for r in (await session.scalars(select(SimImage))).all()}
    for item in _CATALOG:
        if item["name"] in existing:
            continue
        session.add(
            SimImage(
                id=new_id("img"),
                name=item["name"],
                tag=item["tag"],
                role=item["role"],
                is_default=item["is_default"],
                status="missing",
            )
        )
    await session.commit()


def _img_out(row: SimImage) -> dict:
    digest = row.digest or ""
    return {
        "id": row.id,
        "name": row.name,
        "tag": row.tag,
        "digest": digest,
        "digest_short": digest[7:19] if digest.startswith("sha256:") else digest[:12],
        "size_bytes": row.size_bytes,
        "role": row.role,
        "is_default": row.is_default,
        "status": row.status,
        "pull_progress": row.pull_progress or {},
        "error_message": row.error_message,
    }


@router.get("/envs/stack")
async def env_stack(session: SessionDep, principal: PrincipalDep) -> dict:
    await _ensure_catalog(session)
    default = (await session.scalars(select(SimImage).where(SimImage.is_default.is_(True)))).first()
    return {
        "id": "isaaclab",
        "display_name": "Isaac Lab",
        "default_image": default.name if default else "eb/rlinf:0.4.0-emb",
        "cache": {"ov_data": "/var/lib/eb/cache/ov_data", "nucleus": "/var/lib/eb/cache/isaac_nucleus"},
    }


@router.get("/envs/images")
async def env_images(session: SessionDep, principal: PrincipalDep) -> dict:
    await _ensure_catalog(session)
    rows = (await session.scalars(select(SimImage))).all()
    return {"items": [_img_out(r) for r in rows]}


@router.post("/envs/images/{iid}/default")
async def set_default_image(iid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(SimImage))).all()
    found = False
    for r in rows:
        r.is_default = r.id == iid
        if r.id == iid:
            found = True
    if not found:
        raise HTTPException(404)
    await session.commit()
    return {"ok": True}


@router.get("/envs/doctor")
async def env_doctor(principal: PrincipalDep) -> dict:
    ebd = await EbdClient().doctor()
    vulkan = Path("/usr/share/vulkan/icd.d/nvidia_icd.json").exists()
    shm = Path("/dev/shm")
    shm_ok = shm.exists()
    checks = [
        {"id": "nvidia", "label": "NVIDIA Driver / GPU", "ok": bool(ebd.get("gpus")), "detail": ebd},
        {"id": "vulkan", "label": "Vulkan ICD", "ok": vulkan, "detail": "/usr/share/vulkan/icd.d/nvidia_icd.json"},
        {"id": "shm", "label": "/dev/shm", "ok": shm_ok, "detail": str(shm)},
        {"id": "ebd", "label": "ebd 可达", "ok": "error" not in ebd or bool(ebd.get("gpus")), "detail": ebd.get("error")},
    ]
    return {"checks": checks, "healthy": all(c["ok"] for c in checks if c["id"] != "vulkan")}
