from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.contract import EmbodimentEnv
from eb.domain.contract_view import contract_workspace, gate_matrix
from eb.domain.embodiment import BHL_ASSET_INDEX, EmbodimentSource
from eb.domain.job import JobSpec
from eb.domain.task import TaskSpec
from eb.domain.urdf import (
    collect_urdf_joints,
    iter_mesh_filenames,
    missing_joints,
    next_clone_id,
    parse_urdf_joint_tree,
    parse_urdf_joints,
    resolve_mesh_rel,
)
from eb.gates.engine import GateContext, g1_submit
from eb.ids import new_id, sha256_bytes, sha256_text
from eb.store.models import (
    Bundle,
    DatasetVersion,
    EmbodimentEnvRow,
    EmbodimentRow,
    Job,
    ModelVersion,
    Robot,
    RobotAssetRow,
    TaskRow,
)
from eb.worktree import git_is_dirty, resolve_commit

def _asset_rel_paths(assets: list[RobotAssetRow]) -> list[str]:
    rels: list[str] = []
    for a in assets:
        items = (a.paths or {}).get("items") or []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("path"):
                    rels.append(str(it["path"]))
                elif isinstance(it, str):
                    rels.append(it)
        for key in ("urdf", "mjcf", "usd"):
            p = (a.paths or {}).get(key)
            if p:
                rels.append(str(p))
    if not rels:
        rels = [str(it["path"]) for it in BHL_ASSET_INDEX if it.get("path")]
    return rels


def _env_asset_joints(emb: EmbodimentRow, assets: list[RobotAssetRow]) -> tuple[list[str], str | None]:
    root = Path((emb.source or {}).get("path") or "")
    if not root.exists():
        return [], None
    return collect_urdf_joints(root, _asset_rel_paths(assets))

_KIND = {"git_external", "git_snapshot", "managed_bundle", "in_image", "plugin"}


class EmbodimentIn(BaseModel):
    id: str | None = None
    display_name: str = ""
    source: dict[str, Any] | None = None
    root_path: str | None = None
    tags: list[str] = []


class EnvDraftIn(BaseModel):
    spec: dict[str, Any]


class TaskIn(BaseModel):
    spec: dict[str, Any]


router = APIRouter()


def _caps(envs: list[EmbodimentEnvRow], tasks: list[TaskRow]) -> dict[str, bool]:
    published = [e for e in envs if e.status == "active"]
    pub_ids = {e.id for e in published}
    train = False
    for t in tasks:
        spec = t.spec or {}
        compat = spec.get("compatible_env_ids") or []
        if t.status == "active" and spec.get("stack_bindings") and any(c in pub_ids for c in compat):
            train = True
            break
    preview = True
    return {"preview": preview, "train": train, "eval": train, "deploy": bool(published) and train}


def _env_out(row: EmbodimentEnvRow) -> dict:
    spec = row.spec or {}
    obs = spec.get("observation") or {}
    concat = obs.get("concat", True)
    irregular = []
    if spec.get("morphology") == "fixed_hand" or concat is False:
        irregular.append("多模态")
    action = spec.get("action") or {}
    if action.get("terms"):
        irregular.append("双 action")
    return {
        "id": row.id,
        "embodiment_id": row.embodiment_id,
        "status": row.status,
        "obs_dim": row.obs_dim,
        "action_dim": row.action_dim,
        "policy_hz": row.policy_hz,
        "spec": spec,
        "spec_hash": row.spec_hash,
        "morphology": spec.get("morphology"),
        "dof": spec.get("dof") or row.action_dim,
        "concat": concat,
        "irregular": irregular,
        "yaml_readonly": row.status != "draft",
    }


def _emb_out(row: EmbodimentRow, envs: list[EmbodimentEnvRow], tasks: list[TaskRow]) -> dict:
    caps = dict(row.capabilities or {})
    caps.update(_caps(envs, tasks))
    return {
        "id": row.id,
        "display_name": row.display_name,
        "source": row.source,
        "status": row.status,
        "seed": row.seed,
        "tags": row.tags,
        "capabilities": caps,
        "env_count": len(envs),
        "envs": [_env_out(e) for e in envs],
        "root_path": row.root_path or (row.source or {}).get("path"),
        "default_urdf": row.default_urdf,
        "urdf_variants": row.urdf_variants or [],
        "joint_tree": row.joint_tree or [],
        "gym_ids": row.gym_ids or [],
        "interface_snapshot": row.interface_snapshot,
        "scanned_at": row.scanned_at.isoformat() if row.scanned_at else None,
        "error_message": row.error_message,
        "variant_count": len(row.urdf_variants or []),
        "task_count": len(row.gym_ids or []),
    }


@router.get("/embodiments")
async def list_embodiments(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(EmbodimentRow))).all()
    envs = (await session.scalars(select(EmbodimentEnvRow))).all()
    tasks = (await session.scalars(select(TaskRow))).all()
    by_e: dict[str, list[EmbodimentEnvRow]] = {}
    for e in envs:
        by_e.setdefault(e.embodiment_id, []).append(e)
    return {"items": [_emb_out(r, by_e.get(r.id, []), tasks) for r in rows]}


@router.post("/embodiments")
async def create_embodiment(body: EmbodimentIn, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.paths import PathNotAllowed, validate_allowed_path
    from eb.workbench import apply_scan

    eid = body.id or new_id("emb")
    if await session.get(EmbodimentRow, eid):
        raise HTTPException(409, "id exists")
    caps = {"preview": True, "train": True, "eval": False, "deploy": False}
    if body.root_path:
        try:
            root = validate_allowed_path(body.root_path)
        except PathNotAllowed as e:
            raise HTTPException(400, str(e)) from e
        if not root.exists():
            raise HTTPException(400, f"path not readable: {body.root_path}")
        row = EmbodimentRow(
            id=eid,
            display_name=body.display_name or root.name,
            source={"kind": "git_external", "path": str(root), "writable": False},
            status="ready",
            seed=False,
            tags=body.tags,
            capabilities=caps,
            root_path=str(root),
        )
        apply_scan(row, root)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _emb_out(row, [], [])
    if not body.source:
        raise HTTPException(400, "需要 root_path 或 source")
    src = EmbodimentSource.model_validate(body.source)
    if src.kind not in _KIND:
        raise HTTPException(400, "unknown source.kind")
    if src.kind == "git_external" and src.path:
        root = Path(src.path)
        if not root.exists():
            raise HTTPException(400, f"path not readable: {src.path}")
    row = EmbodimentRow(
        id=eid,
        display_name=body.display_name or (Path(src.path).name if src.path else eid),
        source=src.model_dump(mode="json"),
        status="active",
        seed=False,
        tags=body.tags,
        capabilities=caps,
        root_path=src.path,
    )
    if src.path and Path(src.path).is_dir():
        apply_scan(row, Path(src.path))
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _emb_out(row, [], [])


@router.get("/embodiments/{eid}")
async def get_embodiment(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    envs = (await session.scalars(select(EmbodimentEnvRow).where(EmbodimentEnvRow.embodiment_id == eid))).all()
    tasks = (await session.scalars(select(TaskRow))).all()
    assets = (await session.scalars(select(RobotAssetRow).where(RobotAssetRow.embodiment_id == eid))).all()
    return {
        **_emb_out(row, list(envs), list(tasks)),
        "assets": [{"id": a.id, "version_label": a.version_label, "paths": a.paths} for a in assets],
    }


@router.patch("/embodiments/{eid}")
async def patch_embodiment(eid: str, body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    if "source" in body:
        src = dict(row.source or {})
        incoming = body["source"] or {}
        if incoming.get("kind") and incoming.get("kind") != src.get("kind"):
            raise HTTPException(400, "source.kind 不可改")
        if incoming.get("path") and incoming.get("path") != src.get("path"):
            raise HTTPException(400, "source.path 不可改")
        if "default_ref" in incoming:
            src["default_ref"] = incoming["default_ref"]
            row.source = src
    if "display_name" in body:
        row.display_name = body["display_name"]
    if "tags" in body:
        row.tags = body["tags"]
    await session.commit()
    return {"id": row.id, "display_name": row.display_name, "source": row.source, "tags": row.tags}


@router.post("/embodiments/{eid}/archive")
async def archive_embodiment(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    row.status = "archived"
    await session.commit()
    return {"id": row.id, "status": row.status}


@router.delete("/embodiments/{eid}")
async def delete_embodiment(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    if row.seed:
        raise HTTPException(403, "种子本体不可删除，只能归档")
    published = (
        await session.scalars(
            select(EmbodimentEnvRow).where(EmbodimentEnvRow.embodiment_id == eid, EmbodimentEnvRow.status == "active")
        )
    ).all()
    if published:
        raise HTTPException(409, "仍有已发布环境，只能归档")
    await session.delete(row)
    await session.commit()
    return {"ok": True}


@router.post("/embodiments/{eid}/scan")
async def scan_embodiment(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.workbench import apply_scan, embodiment_root

    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    src = row.source or {}
    root = embodiment_root(row)
    found: list[str] = []
    dirty = None
    head = None
    git_present = False
    result: dict[str, Any] = {}
    if root.exists():
        result = apply_scan(row, root)
        found = [v.get("path", "") for v in (result.get("urdf_variants") or [])]
        found.extend(result.get("meshes") or [])
        git = root / ".git"
        git_present = git.exists()
        dirty = git_is_dirty(root) if git_present else None
        if git_present:
            try:
                head = resolve_commit(root, src.get("default_ref") or "HEAD")
            except Exception:
                head = None
        found = found[:80]
    await session.commit()
    return {
        "ok": True,
        "wrote": False,
        "files": found,
        "git_present": git_present,
        "dirty": dirty,
        "head": head,
        "urdf_variants": result.get("urdf_variants") or [],
        "gym_ids": result.get("gym_ids") or [],
        "rlinf_envs": result.get("rlinf_envs") or [],
    }


@router.get("/embodiments/{eid}/assets")
async def list_assets(eid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    assets = (await session.scalars(select(RobotAssetRow).where(RobotAssetRow.embodiment_id == eid))).all()
    return {"items": [{"id": a.id, "version_label": a.version_label, "paths": a.paths} for a in assets]}


def _send_asset(row: EmbodimentRow, path: str) -> FileResponse:
    from eb.paths import is_under
    from eb.workbench import embodiment_root

    root = embodiment_root(row)
    if not root.exists():
        raise HTTPException(404, "asset root missing")
    if not path or Path(path).is_absolute() or path.startswith("/") or path.startswith("~"):
        raise HTTPException(403, "path must stay inside asset root")
    dest = (root / path).resolve()
    if not is_under(root, dest):
        raise HTTPException(403, "path must stay inside asset root")
    if not dest.is_file():
        raise HTTPException(404)
    media = {
        ".urdf": "application/xml",
        ".xml": "application/xml",
        ".stl": "model/stl",
        ".obj": "model/obj",
        ".dae": "model/vnd.collada+xml",
        ".usd": "application/octet-stream",
        ".usda": "text/plain",
    }.get(dest.suffix.lower())
    return FileResponse(dest, media_type=media)


@router.get("/embodiments/{eid}/assets/file")
async def asset_file(eid: str, session: SessionDep, principal: PrincipalDep, path: str = Query(...)):
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    return _send_asset(row, path)


@router.get("/embodiments/{eid}/assets/{asset_path:path}")
async def asset_by_path(eid: str, asset_path: str, session: SessionDep, principal: PrincipalDep):
    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    return _send_asset(row, asset_path)


@router.get("/embodiments/{eid}/preview")
async def embodiment_preview(
    eid: str,
    session: SessionDep,
    principal: PrincipalDep,
    urdf: str | None = Query(None),
) -> dict:
    from eb.domain.urdf import parse_urdf_joint_tree
    from eb.workbench import embodiment_root

    row = await session.get(EmbodimentRow, eid)
    if not row:
        raise HTTPException(404)
    root = embodiment_root(row)
    assets = (await session.scalars(select(RobotAssetRow).where(RobotAssetRow.embodiment_id == eid))).all()
    names, urdf_rel = _env_asset_joints(row, list(assets))
    if urdf:
        urdf_rel = urdf
    elif row.default_urdf and not urdf_rel:
        urdf_rel = row.default_urdf
    joints: list[dict[str, str]] = []
    meshes: list[dict[str, Any]] = []
    missing_meshes: list[str] = []
    usd: list[dict[str, Any]] = []
    if urdf_rel and root.exists():
        dest = root / urdf_rel
        if dest.is_file():
            text = dest.read_text(errors="replace")
            joints = parse_urdf_joint_tree(text)
            seen: set[str] = set()
            for filename in iter_mesh_filenames(text):
                rel = resolve_mesh_rel(urdf_rel, filename)
                if rel in seen:
                    continue
                seen.add(rel)
                mesh_path = root / rel
                item = {
                    "filename": filename,
                    "path": rel,
                    "url": f"/api/v1/embodiments/{eid}/assets/file?path={rel}",
                    "exists": mesh_path.is_file(),
                    "bytes": mesh_path.stat().st_size if mesh_path.is_file() else 0,
                }
                meshes.append(item)
                if not item["exists"]:
                    missing_meshes.append(rel)
            usd_dir = dest.parent.parent / "usd"
            if usd_dir.is_dir():
                for p in sorted(usd_dir.glob("*.usd"))[:8]:
                    rel = str(p.relative_to(root))
                    raw = p.read_bytes()
                    usd.append(
                        {
                            "path": rel,
                            "bytes": len(raw),
                            "sha256": sha256_bytes(raw),
                            "url": f"/api/v1/embodiments/{eid}/assets/file?path={rel}",
                        }
                    )
    if not joints and names:
        joints = [{"name": n, "type": "", "parent": "", "child": ""} for n in names]
    urdf_url = None
    if urdf_rel:
        urdf_url = f"/api/v1/embodiments/{eid}/assets/file?path={quote(urdf_rel, safe='/')}"
        for m in meshes:
            m["url"] = f"/api/v1/embodiments/{eid}/assets/file?path={quote(m['path'], safe='/')}"
        for u in usd:
            u["url"] = f"/api/v1/embodiments/{eid}/assets/file?path={quote(u['path'], safe='/')}"
    return {
        "kind": "urdf",
        "urdf_path": urdf_rel,
        "urdf_url": urdf_url,
        "meshes": meshes,
        "missing_meshes": missing_meshes,
        "usd": usd,
        "joints": [j["name"] for j in joints],
        "tree": joints,
        "variants": row.urdf_variants or [],
        "usd_only": not urdf_rel and bool(usd),
        "note": "浏览器 three.js 渲 URDF+STL；USD 只做卡片，不在浏览器解析",
    }


@router.post("/embodiments/{eid}/envs")
async def create_env_draft(eid: str, body: EnvDraftIn, session: SessionDep, principal: PrincipalDep) -> dict:
    emb = await session.get(EmbodimentRow, eid)
    if not emb:
        raise HTTPException(404)
    spec = dict(body.spec)
    spec.setdefault("embodiment_id", eid)
    spec.setdefault("status", "draft")
    env = EmbodimentEnv.model_validate(spec)
    if await session.get(EmbodimentEnvRow, env.id):
        raise HTTPException(409, "id exists")
    row = EmbodimentEnvRow(
        id=env.id,
        embodiment_id=eid,
        spec=env.model_dump(mode="json"),
        spec_hash=env.spec_hash(),
        status="draft",
        obs_dim=env.observation.dim,
        action_dim=env.action.dim,
        policy_hz=env.control.policy_hz,
    )
    session.add(row)
    await session.commit()
    return _env_out(row)


@router.get("/embodiment-envs/{eeid}")
async def get_env(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    env = EmbodimentEnv.model_validate(row.spec)
    return {**_env_out(row), "self_check": env.validate_self().model_dump(), **contract_workspace(env)}


@router.patch("/embodiment-envs/{eeid}")
async def patch_env(eeid: str, body: EnvDraftIn, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    if row.status != "draft":
        raise HTTPException(403, "已发布本体环境不可改数字，请克隆 _v2")
    spec = {**row.spec, **body.spec, "status": "draft", "id": row.id, "embodiment_id": row.embodiment_id}
    env = EmbodimentEnv.model_validate(spec)
    row.spec = env.model_dump(mode="json")
    row.spec_hash = env.spec_hash()
    row.obs_dim = env.observation.dim
    row.action_dim = env.action.dim
    row.policy_hz = env.control.policy_hz
    await session.commit()
    return _env_out(row)


@router.post("/embodiment-envs/{eeid}/publish")
async def publish_env(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    env = EmbodimentEnv.model_validate(row.spec)
    check = env.validate_self()
    if not check.passed:
        raise HTTPException(400, detail={"violations": [v.model_dump() for v in check.violations]})
    emb = await session.get(EmbodimentRow, row.embodiment_id)
    assets = (
        await session.scalars(select(RobotAssetRow).where(RobotAssetRow.embodiment_id == row.embodiment_id))
    ).all()
    asset_joints, _urdf = _env_asset_joints(emb, list(assets)) if emb else ([], None)
    missing = missing_joints(env.joint_order, asset_joints)
    if missing:
        raise HTTPException(
            400,
            detail={
                "violations": [
                    {
                        "path": "joint_order",
                        "expected": "subset of URDF joints",
                        "actual": missing,
                        "hint": "发布要求 joint_order ⊆ 资产关节名",
                    }
                ]
            },
        )
    row.status = "active"
    spec = dict(row.spec)
    spec["status"] = "active"
    row.spec = spec
    row.spec_hash = sha256_text(EmbodimentEnv.model_validate(spec).model_dump_json())
    await session.commit()
    return _env_out(row)


@router.post("/embodiment-envs/{eeid}/clone")
async def clone_env(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    taken = set((await session.scalars(select(EmbodimentEnvRow.id))).all())
    new_id_s = next_clone_id(row.id, taken)
    spec = dict(row.spec)
    spec["id"] = new_id_s
    spec["status"] = "draft"
    env = EmbodimentEnv.model_validate(spec)
    clone = EmbodimentEnvRow(
        id=new_id_s,
        embodiment_id=row.embodiment_id,
        spec=env.model_dump(mode="json"),
        spec_hash=env.spec_hash(),
        status="draft",
        obs_dim=env.observation.dim,
        action_dim=env.action.dim,
        policy_hz=env.control.policy_hz,
    )
    session.add(clone)
    await session.commit()
    return _env_out(clone)


@router.post("/embodiment-envs/{eeid}/deprecate")
async def deprecate_env(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    row.status = "deprecated"
    spec = dict(row.spec)
    spec["status"] = "deprecated"
    row.spec = spec
    await session.commit()
    return _env_out(row)


class YamlIn(BaseModel):
    yaml: str


@router.put("/embodiment-envs/{eeid}/yaml")
async def put_env_yaml(eeid: str, body: YamlIn, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    if row.status != "draft":
        raise HTTPException(403, "已发布本体环境不可改数字，请克隆 _v2")
    loaded = yaml.safe_load(body.yaml) or {}
    spec = {**loaded, "status": "draft", "id": row.id, "embodiment_id": row.embodiment_id}
    env = EmbodimentEnv.model_validate(spec)
    row.spec = env.model_dump(mode="json")
    row.spec_hash = env.spec_hash()
    row.obs_dim = env.observation.dim
    row.action_dim = env.action.dim
    row.policy_hz = env.control.policy_hz
    await session.commit()
    return _env_out(row)


@router.get("/embodiment-envs/{eeid}/yaml")
async def get_env_yaml(eeid: str, session: SessionDep, principal: PrincipalDep):
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    text = yaml.safe_dump(row.spec, allow_unicode=True, sort_keys=False)
    return Response(content=text, media_type="text/yaml")


@router.delete("/embodiment-envs/{eeid}")
async def delete_env(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    if row.status != "draft":
        raise HTTPException(403, "只能删除草稿")
    await session.delete(row)
    await session.commit()
    return {"ok": True}


@router.get("/embodiment-envs/{eeid}/validate")
async def validate_env(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    env = EmbodimentEnv.model_validate(row.spec)
    self_check = env.validate_self().model_dump()
    emb = await session.get(EmbodimentRow, row.embodiment_id)
    assets = (
        await session.scalars(select(RobotAssetRow).where(RobotAssetRow.embodiment_id == row.embodiment_id))
    ).all()
    asset_joints, urdf_rel = _env_asset_joints(emb, list(assets)) if emb else ([], None)
    missing = missing_joints(env.joint_order, asset_joints)
    return {
        **self_check,
        "urdf_path": urdf_rel,
        "asset_joint_count": len(asset_joints),
        "missing_joints": missing,
        "joints_ok": not missing,
    }


@router.post("/embodiment-envs/{eeid}/validate")
async def validate_job_against_env(eeid: str, body: dict, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    spec = JobSpec.model_validate({**(body.get("spec") or {}), "embodiment_env_id": eeid})
    task = None
    if spec.task_id:
        trow = await session.get(TaskRow, spec.task_id)
        if trow:
            task = TaskSpec.model_validate(trow.spec)
    result = g1_submit(spec, EmbodimentEnv.model_validate(row.spec), GateContext(dataset_versions=[], task=task))
    return result.model_dump()


@router.get("/embodiment-envs/{eeid}/refs")
async def env_refs(eeid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(EmbodimentEnvRow, eeid)
    if not row:
        raise HTTPException(404)
    env_rows = (await session.scalars(select(EmbodimentEnvRow))).all()
    envs = [EmbodimentEnv.model_validate(r.spec) for r in env_rows]
    jobs = (await session.scalars(select(Job).limit(200))).all()
    models = (await session.scalars(select(ModelVersion).limit(50))).all()
    dsvs = (await session.scalars(select(DatasetVersion).limit(50))).all()
    bundles = (await session.scalars(select(Bundle).limit(50))).all()
    robots = (await session.scalars(select(Robot).limit(20))).all()
    versions = [
        {"id": v.id, "dataset_id": v.dataset_id, "compatible_envs": v.compatible_envs, "stats": v.stats} for v in dsvs
    ]
    matrix = gate_matrix(
        envs,
        versions=versions,
        models=[{"embodiment_env_id": m.embodiment_env_id, "io_signature": m.io_signature} for m in models],
        jobs=[{"embodiment_env_id": j.embodiment_env_id, "spec": j.spec} for j in jobs],
        bundles=[{"embodiment_env_id": b.embodiment_env_id, "manifest": b.manifest} for b in bundles],
        robots=[{"embodiment_env_id": r.embodiment_env_id, "calibration": r.calibration} for r in robots],
    )
    return {
        "jobs": [{"id": j.id, "kind": j.kind, "status": j.status} for j in jobs if j.embodiment_env_id == eeid],
        "models": [
            {"id": m.id, "name": m.name, "io_signature": m.io_signature}
            for m in models
            if m.embodiment_env_id == eeid
        ],
        "datasets": [
            {"id": v.dataset_id, "version_id": v.id, "compatible_envs": v.compatible_envs}
            for v in dsvs
            if eeid in (v.compatible_envs or [])
        ],
        "matrix": matrix,
        "env_ids": [e.id for e in envs],
    }


@router.get("/tasks")
async def list_tasks(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(TaskRow))).all()
    return {"items": [{"id": r.id, "status": r.status, **(r.spec or {})} for r in rows]}


@router.post("/tasks")
async def create_task(body: TaskIn, session: SessionDep, principal: PrincipalDep) -> dict:
    tsk = TaskSpec.model_validate({**body.spec, "status": body.spec.get("status") or "draft"})
    if await session.get(TaskRow, tsk.id):
        raise HTTPException(409, "id exists")
    row = TaskRow(id=tsk.id, spec=tsk.model_dump(mode="json"), spec_hash=tsk.spec_hash(), status=tsk.status)
    session.add(row)
    await session.commit()
    return {"id": row.id, "status": row.status, **tsk.model_dump()}


@router.get("/tasks/{tid}")
async def get_task(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TaskRow, tid)
    if not row:
        raise HTTPException(404)
    return {"id": row.id, "status": row.status, **(row.spec or {})}


@router.patch("/tasks/{tid}")
async def patch_task(tid: str, body: TaskIn, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TaskRow, tid)
    if not row:
        raise HTTPException(404)
    spec = {**row.spec, **body.spec, "id": tid}
    if row.status != "draft":
        raise HTTPException(403, "已发布任务不可改，请新建草稿")
    tsk = TaskSpec.model_validate(spec)
    row.spec = tsk.model_dump(mode="json")
    row.spec_hash = tsk.spec_hash()
    await session.commit()
    return {"id": row.id, "status": row.status, **tsk.model_dump()}


@router.post("/tasks/{tid}/publish")
async def publish_task(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TaskRow, tid)
    if not row:
        raise HTTPException(404)
    row.status = "active"
    spec = dict(row.spec)
    spec["status"] = "active"
    row.spec = spec
    await session.commit()
    return {"id": row.id, "status": row.status}


@router.delete("/tasks/{tid}")
async def delete_task(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TaskRow, tid)
    if not row:
        raise HTTPException(404)
    if row.status != "draft":
        raise HTTPException(403, "只能删除草稿任务")
    await session.delete(row)
    await session.commit()
    return {"ok": True}
