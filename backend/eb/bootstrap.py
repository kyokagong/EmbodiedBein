from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from passlib.hash import pbkdf2_sha256
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.contract import load_env_dir
from eb.domain.embodiment import BHL_ASSET_INDEX, BHL_SEED
from eb.domain.task import BHL_SEED_TASKS
from eb.ids import new_id, sha256_bytes
from eb.ingest import ingest_all
from eb.settings import get_settings
from eb.store.models import (
    ApiKey,
    Bundle,
    Dataset,
    DatasetVersion,
    EmbodimentEnvRow,
    EmbodimentRow,
    EvalSuite,
    Job,
    Pipeline,
    Robot,
    RobotAssetRow,
    Run,
    Share,
    TaskRow,
    User,
)


async def seed_if_empty(session: AsyncSession) -> None:
    settings = get_settings()
    n = await session.scalar(select(func.count()).select_from(User))
    if not n:
        session.add(
            User(
                id=new_id("usr"),
                username=settings.admin_user,
                password_hash=pbkdf2_sha256.hash(settings.admin_password),
                actor_type="user",
            )
        )
        session.add(
            ApiKey(
                id=new_id("key"),
                principal=f"user:{settings.admin_user}",
                token_hash=sha256_bytes(b"eb-dev-key"),
                actor_type="user",
            )
        )
        session.add(
            ApiKey(
                id=new_id("key"),
                principal="agent:autopilot",
                token_hash=sha256_bytes(b"eb-agent-key"),
                actor_type="agent",
            )
        )

    if not await session.get(EmbodimentRow, BHL_SEED.id):
        src = BHL_SEED.source.model_dump(mode="json")
        session.add(
            EmbodimentRow(
                id=BHL_SEED.id,
                display_name=BHL_SEED.display_name,
                source=src,
                status=BHL_SEED.status,
                seed=BHL_SEED.seed,
                tags=BHL_SEED.tags,
                capabilities=BHL_SEED.capabilities,
                root_path=src.get("path"),
            )
        )
        session.add(
            RobotAssetRow(
                id="ras_bhl_v1",
                embodiment_id=BHL_SEED.id,
                version_label="bhl-hw-v1",
                paths={"items": BHL_ASSET_INDEX},
                content_hash="",
            )
        )
        await session.flush()
    contracts_dir = settings.contracts_dir
    if not contracts_dir.exists():
        here = Path(__file__).resolve().parents[2] / "contracts"
        if here.exists():
            contracts_dir = here
    for env in load_env_dir(contracts_dir):
        if await session.get(EmbodimentEnvRow, env.id):
            continue
        session.add(
            EmbodimentEnvRow(
                id=env.id,
                embodiment_id=env.embodiment_id,
                spec=env.model_dump(mode="json"),
                spec_hash=env.spec_hash(),
                status=env.status,
                obs_dim=env.observation.dim,
                action_dim=env.action.dim,
                policy_hz=env.control.policy_hz,
            )
        )
    for tsk in BHL_SEED_TASKS:
        if await session.get(TaskRow, tsk.id):
            continue
        session.add(
            TaskRow(
                id=tsk.id,
                spec=tsk.model_dump(mode="json"),
                spec_hash=tsk.spec_hash(),
                status=tsk.status,
            )
        )
    await session.flush()
    try:
        await ingest_all(session)
    except Exception:
        logging.getLogger(__name__).exception("legacy ingest skipped")

    if not await session.scalar(select(func.count()).select_from(Dataset)):
        ds = Dataset(
            id="ds_berkeley_ego",
            name="berkeley_ego_data",
            format="ego_jsonl",
            license="mixed/nc_tainted",
            embodiment_env_id="ee_bhl_fixedhand_10dof_v1",
            meta={"root": "/data/kk/berkeley_ego_data"},
            source_kind="path",
            source_uri="/data/kk/berkeley_ego_data",
            dest_path="/data/kk/berkeley_ego_data",
            status="ready",
        )
        session.add(ds)
        session.add(
            DatasetVersion(
                id="dsv_ego_raw",
                dataset_id=ds.id,
                version="raw",
                compatible_envs=[],
                stats={"note": "await data.scan"},
                funnel={
                    "source": {"n": 0, "rejected": []},
                    "raw": {"n": 0, "rejected": []},
                    "valid_episode": {"n": 0, "rejected": []},
                    "retargeted": {"n": 0, "rejected": []},
                    "canonical": {"n": 0, "rejected": []},
                    "slice": {"n": 0, "rejected": []},
                },
                content_hash="unscanned",
            )
        )
        session.add(
            EvalSuite(
                id="suite_biped_locomotion_v2",
                name="biped locomotion",
                embodiment_env_id="ee_bhl_biped_12dof_v1",
                seeds=[0, 1, 2, 3, 4, 5, 6, 7],
                frozen=True,
            )
        )
        await session.flush()
        session.add(
            Robot(
                id="robot_bhl_01",
                name="BHL-01",
                embodiment_env_id="ee_bhl_biped_12dof_v1",
                calibration={"expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat()},
            )
        )
        session.add(
            Bundle(
                id="bnd_biped_demo",
                embodiment_env_id="ee_bhl_biped_12dof_v1",
                stage="S0",
                signed=False,
                manifest={"note": "first-version placeholder bundle"},
            )
        )
        session.add(
            Share(
                token="bhl-progress-demo",
                title="Berkeley-Humanoid-Lite 项目进展",
                snapshot={
                    "kpis": [
                        {"label": "历史策略", "value": "6+", "foot": "遗留导入"},
                        {"label": "本体环境", "value": "4", "foot": "humanoid/biped/wbc/hand"},
                        {"label": "数据场景", "value": "待扫描", "foot": "漏斗来自 data.scan"},
                    ],
                    "milestones": [
                        {"t": "M0", "d": "平台骨架与只读纳管"},
                        {"t": "Now", "d": "compose 单机可部署"},
                    ],
                    "caveat": "部分指标仅仿真，尚未上机",
                },
            )
        )
    if not await session.get(Pipeline, "pipe_ego_scan_slice"):
        session.add(
            Pipeline(
                id="pipe_ego_scan_slice",
                name="scan then slice ego",
                spec={
                    "status": "idle",
                    "cursor": 0,
                    "job_ids": [],
                    "steps": [
                        {
                            "kind": "data.scan",
                            "embodiment_env_id": "ee_bhl_fixedhand_10dof_v1",
                            "params": {"path": "/data/kk/berkeley_ego_data"},
                            "resources": {"gpus": 0},
                        },
                        {
                            "kind": "data.slice",
                            "embodiment_env_id": "ee_bhl_fixedhand_10dof_v1",
                            "params": {
                                "dataset_id": "ds_berkeley_ego",
                                "query": {"limit": 1},
                            },
                            "resources": {"gpus": 0},
                        },
                    ],
                },
            )
        )
    await session.commit()
    await _scrub_stub_success(session)


async def _scrub_stub_success(session: AsyncSession) -> None:
    """Never leave skeleton stub runs marked succeeded."""
    runs = (await session.scalars(select(Run))).all()
    changed = False
    for run in runs:
        ms = run.metrics_summary or {}
        note = str(ms.get("note") or "")
        if "stub" not in note.lower() and "image missing" not in note.lower():
            continue
        job = await session.get(Job, run.job_id)
        if job is None:
            continue
        job.status = "infra_failed"
        job.error = note or "stub_forbidden"
        run.status = "aborted"
        run.abort_reason = "stub_forbidden"
        run.metrics_summary = {}
        changed = True
    if changed:
        await session.commit()
