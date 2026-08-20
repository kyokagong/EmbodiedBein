from __future__ import annotations

import logging
from pathlib import Path

from passlib.hash import pbkdf2_sha256
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import datetime, timedelta, timezone

from eb.domain.contract import load_contract_dir
from eb.ids import new_id, sha256_bytes
from eb.ingest import ingest_all
from eb.settings import get_settings
from eb.store.models import (
    ApiKey,
    Bundle,
    Dataset,
    DatasetVersion,
    EmbodimentContractRow,
    EvalReport,
    EvalSuite,
    Robot,
    Share,
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

    contracts_dir = settings.contracts_dir
    if not contracts_dir.exists():
        # repo-relative fallback
        here = Path(__file__).resolve().parents[2] / "contracts"
        if here.exists():
            contracts_dir = here
    for ctr in load_contract_dir(contracts_dir):
        if await session.get(EmbodimentContractRow, ctr.id):
            continue
        session.add(
            EmbodimentContractRow(
                id=ctr.id,
                spec=ctr.model_dump(mode="json"),
                spec_hash=ctr.spec_hash(),
                status=ctr.status,
                obs_dim=ctr.observation.dim,
                action_dim=ctr.action.dim,
                policy_hz=ctr.control.policy_hz,
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
            contract_id="ec_bhl_fixedhand_10dof_v1",
            meta={"root": "/data/kk/berkeley_ego_data"},
        )
        session.add(ds)
        session.add(
            DatasetVersion(
                id="dsv_ego_raw",
                dataset_id=ds.id,
                version="raw",
                compatible_contracts=[],
                stats={"events": 155000, "scenarios": 35},
                funnel={
                    "stages": [
                        {"name": "raw_events", "n": 155000},
                        {"name": "valid_episode", "n": 1200},
                        {"name": "scenario", "n": 35},
                    ]
                },
                content_hash="unknown",
            )
        )
        session.add(
            EvalSuite(
                id="suite_biped_locomotion_v2",
                name="biped locomotion",
                contract_id="ec_bhl_biped_12dof_v1",
                seeds=[0, 1, 2, 3, 4, 5, 6, 7],
                frozen=True,
            )
        )
        await session.flush()
        session.add(
            EvalReport(
                id="ev_biped_baseline",
                suite_id="suite_biped_locomotion_v2",
                model_version_id=None,
                metrics={"success_rate": 0.82, "mean_reward": 16.4},
                gate_results={"g1": "pass", "g3": "unknown"},
            )
        )
        session.add(
            Robot(
                id="robot_bhl_01",
                name="BHL-01",
                contract_id="ec_bhl_biped_12dof_v1",
                calibration={"expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()},
            )
        )
        session.add(
            Bundle(
                id="bnd_biped_demo",
                contract_id="ec_bhl_biped_12dof_v1",
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
                        {"label": "契约", "value": "4", "foot": "humanoid/biped/wbc/hand"},
                        {"label": "数据场景", "value": "35", "foot": "155k 事件漏斗"},
                    ],
                    "milestones": [
                        {"t": "M0", "d": "平台骨架与只读纳管"},
                        {"t": "Now", "d": "compose 单机可部署"},
                    ],
                    "caveat": "部分指标仅仿真，尚未上机",
                },
            )
        )
    await session.commit()
