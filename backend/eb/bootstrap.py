from __future__ import annotations

import logging
import hashlib
import json
import yaml
from datetime import UTC, datetime, timedelta
from pathlib import Path

from passlib.hash import pbkdf2_sha256
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.contract import load_env_dir
from eb.domain.embodiment import BHL_ASSET_INDEX, BHL_SEED
from eb.domain.task import BHL_SEED_TASKS, TaskSpec
from eb.ids import new_id, sha256_bytes
from eb.ingest import ingest_all
from eb.settings import get_settings
from eb.store.models import (
    ApiKey,
    AuditLog,
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
    rpo_root = settings.assets_root / "roboparty/rpo-v2"
    rpo_manifest = {}
    try:
        rpo_manifest = json.loads((rpo_root / "asset-manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    rpo = await session.get(EmbodimentRow, "emb_roboparty_rpo")
    if not rpo:
        rpo = EmbodimentRow(
            id="emb_roboparty_rpo", display_name="RoboParty RPO V2.0",
            source={"kind":"managed_bundle", "path":str(rpo_root), "writable":False},
            root_path=str(rpo_root), status="draft", seed=True,
            tags=["roboparty", "rpo", "v2", "humanoid", "wbc"],
            capabilities={"preview":True, "train_smoke":True, "train":False,
                          "eval":True, "deploy":False, "hardware_verified":False},
        )
        session.add(rpo)
    if rpo_manifest.get("status") == "verified":
        rpo.source = {"kind":"managed_bundle", "path":str(rpo_root), "writable":False}
        rpo.root_path = str(rpo_root)
        rpo.status = "active"
        if not await session.get(RobotAssetRow, "ras_rpo_v2_managed"):
            digest = hashlib.sha256(json.dumps(rpo_manifest, sort_keys=True).encode()).hexdigest()
            session.add(RobotAssetRow(
                id="ras_rpo_v2_managed", embodiment_id=rpo.id,
                version_label="RPO V2.0 managed bundle",
                paths={"urdf":"rpo_description/urdf/rpo.urdf",
                       "mjcf":"rpo_description/mjcf/rpo.xml",
                       "manifest":"asset-manifest.json"}, content_hash="sha256:" + digest,
            ))
    await session.flush()
    # Explicitly simulated benchmark identity, separate from user-owned real robots.
    if not await session.get(EmbodimentRow, "emb_libero_franka"):
        session.add(EmbodimentRow(id="emb_libero_franka", display_name="LIBERO Franka（仿真）",
            source={"kind":"in_image", "path":str(settings.rlinf_repo), "writable":False},
            root_path=str(settings.rlinf_repo), status="active", seed=True, tags=["libero", "simulation"],
            capabilities={"train":True, "eval":True, "deploy":False}))
        await session.flush()
    from eb.robotwin import ROBOTWIN_EMBODIMENTS, ROBOTWIN_HF_REPO, ROBOTWIN_HF_REVISION
    robotwin_asset_root = settings.datasets_root / "robotwin-assets" / ROBOTWIN_HF_REVISION / "embodiments" / "aloha-agilex"
    for profile in ROBOTWIN_EMBODIMENTS.values():
        embodiment = await session.get(EmbodimentRow, profile.embodiment_id)
        if not embodiment:
            embodiment = EmbodimentRow(id=profile.embodiment_id, display_name=f"RoboTwin {profile.label}（仿真）",
                source={"kind":"managed_bundle", "writable":False}, status="draft", seed=True,
                tags=["robotwin", "simulation", profile.id],
                capabilities={"train":True, "eval":True, "deploy":False})
            session.add(embodiment)
        if profile.id == "aloha-agilex" and robotwin_asset_root.is_dir():
            embodiment.source = {"kind":"managed_bundle", "path":str(robotwin_asset_root), "writable":False}
            embodiment.root_path = str(robotwin_asset_root)
            if not await session.get(RobotAssetRow, "ras_robotwin_aloha_981c92"):
                session.add(RobotAssetRow(id="ras_robotwin_aloha_981c92",
                    embodiment_id=profile.embodiment_id,
                    version_label=f"{ROBOTWIN_HF_REPO}@{ROBOTWIN_HF_REVISION}",
                    paths={"urdf":"urdf/arx5_description_isaac.urdf"},
                    content_hash="sha256:6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"))
    await session.flush()
    contracts_dir = settings.contracts_dir
    if not contracts_dir.exists():
        here = Path(__file__).resolve().parents[2] / "contracts"
        if here.exists():
            contracts_dir = here
    for env in load_env_dir(contracts_dir):
        if env.id == "ee_rpo_v2_ufo_native_23dof_v2" and rpo_manifest.get("status") == "verified":
            env = env.model_copy(update={"status": "active"})
        env_row = await session.get(EmbodimentEnvRow, env.id)
        if not env_row:
            session.add(EmbodimentEnvRow(
                id=env.id,
                embodiment_id=env.embodiment_id,
                spec=env.model_dump(mode="json"),
                spec_hash=env.spec_hash(),
                status=env.status,
                obs_dim=env.observation.dim,
                action_dim=env.action.dim,
                policy_hz=env.control.policy_hz,
            ))
        elif env.id in {"ee_rpo_v2_wbc_23dof_v1", "ee_rpo_v2_ufo_native_23dof_v2"} and rpo_manifest.get("status") == "verified" \
                and env_row.status == "draft":
            env_row.spec = env.model_dump(mode="json")
            env_row.spec_hash = env.spec_hash()
            env_row.status = "active"
            env_row.obs_dim = env.observation.dim
            env_row.action_dim = env.action.dim
            env_row.policy_hz = env.control.policy_hz
        elif env.id == "ee_robotwin_aloha_joint14_v1" and env_row.status == "draft" \
                and "left_waist" in (env_row.spec or {}).get("joint_order", []):
            env_row.spec = env.model_dump(mode="json")
            env_row.spec_hash = env.spec_hash()
    for profile in ROBOTWIN_EMBODIMENTS.values():
        if profile.id == "aloha-agilex" or await session.get(EmbodimentEnvRow, profile.embodiment_env_id):
            continue
        spec = {"id":profile.embodiment_env_id, "embodiment_id":profile.embodiment_id,
                "status":"draft", "source":"robotwin", "native_embodiment":profile.id,
                "blocked_reason":"等待 RoboTwin 资产探测后固化状态、动作维度和控制频率"}
        session.add(EmbodimentEnvRow(id=profile.embodiment_env_id,
            embodiment_id=profile.embodiment_id, spec=spec,
            spec_hash=sha256_bytes(str(sorted(spec.items())).encode()), status="draft",
            obs_dim=None, action_dim=None, policy_hz=None))
    pbfm_task = TaskSpec(
        id="tsk_rpo_pbfm_terrain_locomotion_v1", status="draft",
        compatible_env_ids=["ee_rpo_v2_pbfm_depth_23dof_v1"],
        stack_bindings={"rlinf": {"template_id": "rpo_pbfm_fb_depth_native",
                                  "config_name": "rpo_23dof"}},
    )
    for tsk in [*BHL_SEED_TASKS, pbfm_task]:
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
    from eb.rpo_flat import FLAT_ENV, FLAT_SUITE, FLAT_TASK, flat_suite_spec
    flat_task = TaskSpec(
        id=FLAT_TASK, status="active", compatible_env_ids=[FLAT_ENV],
        stack_bindings={"rlinf":{"config_name":"rpo_flat_ppo", "gym_id":"RPO-Flat-RLinf-v1"}},
    )
    if not await session.get(TaskRow, FLAT_TASK):
        session.add(TaskRow(id=FLAT_TASK, spec=flat_task.model_dump(mode="json"),
                            spec_hash=flat_task.spec_hash(), status=flat_task.status))
    if not await session.get(EvalSuite, FLAT_SUITE):
        session.add(EvalSuite(id=FLAT_SUITE, name="RPO-Flat 官方速度与转向 · 100 例/80 例通过",
            embodiment_env_id=FLAT_ENV, seeds=list(range(50000, 50100)),
            frozen=True, spec=flat_suite_spec()))
    rpo_task = TaskSpec(
        id="tsk_rpo_walk_upperbody_tracking", status="active",
        compatible_env_ids=["ee_rpo_v2_wbc_23dof_v1"],
        stack_bindings={"rlinf":{"config_name":"rpo_wbc_ppo", "gym_id":"RPO-WBC-WalkUpper-v0"}},
    )
    if not await session.get(TaskRow, rpo_task.id):
        session.add(TaskRow(id=rpo_task.id, spec=rpo_task.model_dump(mode="json"),
                            spec_hash=rpo_task.spec_hash(), status=rpo_task.status))
    basic_task = TaskSpec(
        id="tsk_rpo_stand_walk", status="active",
        compatible_env_ids=["ee_rpo_v2_wbc_23dof_v1"],
        stack_bindings={"rlinf":{"config_name":"rpo_wbc_stand_walk_ppo", "gym_id":"RPO-WBC-StandWalk-v1"}},
    )
    if not await session.get(TaskRow, basic_task.id):
        session.add(TaskRow(id=basic_task.id, spec=basic_task.model_dump(mode="json"),
                            spec_hash=basic_task.spec_hash(), status=basic_task.status))
    amp_task = TaskSpec(
        id="tsk_rpo_amp_stand_walk", status="draft",
        compatible_env_ids=["ee_rpo_v2_amp_locomotion_23dof_v1"],
        stack_bindings={"rlinf": {"config_name": "rpo_amp_stand_walk_ppo",
                                   "gym_id": "RPO-AMP-StandWalk-v1"}},
    )
    if not await session.get(TaskRow, amp_task.id):
        session.add(TaskRow(id=amp_task.id, spec=amp_task.model_dump(mode="json"),
                            spec_hash=amp_task.spec_hash(), status=amp_task.status))
    if not await session.get(EvalSuite, "suite_rpo_v2_wbc_sim2sim_v1"):
        session.add(EvalSuite(id="suite_rpo_v2_wbc_sim2sim_v1",
            name="RPO V2 WBC MuJoCo closed-loop", embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
            seeds=list(range(12)), frozen=True,
            spec={"benchmark":"rpo_wbc", "backend":"mujoco", "policy":"deterministic_onnx"}))
    if not await session.get(EvalSuite, "suite_rpo_v2_wbc_isaac_v1"):
        session.add(EvalSuite(id="suite_rpo_v2_wbc_isaac_v1",
            name="RPO V2 WBC Isaac closed-loop", embodiment_env_id="ee_rpo_v2_wbc_23dof_v1",
            seeds=list(range(12)), frozen=True,
            spec={"benchmark":"rpo_wbc", "backend":"isaac", "policy":"deterministic_onnx"}))
    for backend in ("isaac", "mujoco"):
        amp_suite_id = f"suite_rpo_v2_amp_{backend}_stand_walk_v1"
        if not await session.get(EvalSuite, amp_suite_id):
            session.add(EvalSuite(id=amp_suite_id,
                name=f"RPO V2 AMP {'Isaac Lab' if backend == 'isaac' else 'MuJoCo'} 直立＋直行诊断",
                embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                seeds=list(range(12)), frozen=True,
                spec={"benchmark":"rpo_amp", "backend":backend, "policy":"deterministic_onnx",
                      "criterion":"stand_walk", "required_steps":1000,
                      "cases":"even_seed_stand_odd_seed_walk", "stand_max_drift_m":0.5,
                      "walk_min_forward_m":4.0, "fall_height_m":0.5, "fall_tilt_rad":0.8,
                      "qualification":"diagnostic"}))
        suite_id = f"suite_rpo_v2_wbc_{backend}_stability_v1"
        if not await session.get(EvalSuite, suite_id):
            session.add(EvalSuite(id=suite_id,
                name=f"RPO V2 WBC {'Isaac Lab' if backend == 'isaac' else 'MuJoCo'} 身体稳定 · 存活 1000 步",
                embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", seeds=list(range(12)), frozen=True,
                spec={"benchmark":"rpo_wbc", "backend":backend, "policy":"deterministic_onnx",
                      "criterion":"stability", "required_steps":1000}))
        basic_suite_id = f"suite_rpo_v2_wbc_{backend}_stand_walk_v1"
        if not await session.get(EvalSuite, basic_suite_id):
            session.add(EvalSuite(id=basic_suite_id,
                name=f"RPO V2 {'Isaac Lab' if backend == 'isaac' else 'MuJoCo'} 直立＋0.3m/s 直行",
                embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", seeds=list(range(12)), frozen=True,
                spec={"benchmark":"rpo_wbc", "backend":backend, "policy":"deterministic_onnx",
                      "criterion":"stand_walk", "required_steps":1000,
                      "cases":"even_seed_stand_odd_seed_walk", "stand_max_drift_m":0.5,
                      "walk_min_forward_m":4.0, "fall_height_m":0.5, "fall_tilt_rad":0.8}))
        versioned_suite_id = f"suite_rpo_v2_wbc_{backend}_stand_walk_v2"
        if not await session.get(EvalSuite, versioned_suite_id):
            session.add(EvalSuite(id=versioned_suite_id,
                name=f"RPO V2 {'Isaac Lab' if backend == 'isaac' else 'MuJoCo'} 直立＋0.3m/s 直行 v2",
                embodiment_env_id="ee_rpo_v2_wbc_23dof_v1", seeds=list(range(12)), frozen=True,
                spec={"benchmark":"rpo_wbc", "backend":backend, "policy":"deterministic_onnx",
                      "criterion":"stand_walk", "required_steps":1000,
                      "cases":"even_seed_stand_odd_seed_walk",
                      "initial_joint_offset_rad":[-0.05, 0.05],
                      "initial_base_velocity_mps":[-0.05, 0.05],
                      "initial_yaw_rad":[-3.141592653589793, 3.141592653589793],
                      "stand_max_drift_m":0.5, "walk_min_forward_m":4.0,
                      "fall_height_m":0.5, "fall_tilt_rad":0.8}))
    robotwin_task = TaskSpec(id="tsk_robotwin_multitask", status="draft",
        compatible_env_ids=["ee_robotwin_aloha_joint14_v1"],
        stack_bindings={"rlinf":{"config_name":"robotwin_sft_openpi_pi05"}})
    task_row = await session.get(TaskRow, robotwin_task.id)
    if not task_row:
        session.add(TaskRow(id=robotwin_task.id, spec=robotwin_task.model_dump(mode="json"),
                            spec_hash=robotwin_task.spec_hash(), status=robotwin_task.status))
    elif "compatible_env_ids" not in (task_row.spec or {}):
        task_row.spec = robotwin_task.model_dump(mode="json")
        task_row.spec_hash = robotwin_task.spec_hash()
        task_row.status = robotwin_task.status
    for suite in ("spatial", "object", "goal", "10"):
        sid = f"suite_tsk_libero_{suite}"
        asset_source = {"repo": "RLinf/LIBERO-assets", "revision": "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"}
        existing_suite = await session.get(EvalSuite, sid)
        if not existing_suite:
            session.add(EvalSuite(id=sid, name=f"LIBERO {suite} independent evaluation",
                embodiment_env_id="ee_libero_franka_delta_v1", seeds=[], frozen=True,
                spec={"suite":f"libero_{suite}", "qualification":"not_qualified", "scope":"simulation_only",
                      "benchmark_assets": asset_source}))
        elif "benchmark_assets" not in (existing_suite.spec or {}):
            existing_suite.spec = {**(existing_suite.spec or {}), "benchmark_assets": asset_source}
    asset_manifest = settings.robotwin_assets_root / "asset-manifest.json"
    seed_file = settings.rlinf_repo / "rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
    try:
        from eb.robotwin_suite import (EVAL_SEEDS_SHA256, asset_manifest_sha256,
                                       build_robotwin_full_suite_spec, build_robotwin_suite_spec)
        receipt = json.loads(asset_manifest.read_text())
        if receipt.get("status") != "verified" or receipt.get("sha256") != asset_manifest_sha256(receipt):
            raise ValueError("RoboTwin asset manifest digest changed")
        seed_bytes = seed_file.read_bytes()
        if hashlib.sha256(seed_bytes).hexdigest() != EVAL_SEEDS_SHA256:
            raise ValueError("RoboTwin eval seed manifest revision changed")
        seed_map = json.loads(seed_bytes)
        seeds = seed_map["adjust_bottle"]["success_seeds"]
        spec = build_robotwin_suite_spec(seeds, receipt["sha256"])
        sid = "suite_robotwin_adjust_bottle_clean_v1"
        suite = await session.get(EvalSuite, sid)
        if not suite:
            session.add(EvalSuite(id=sid, name="RoboTwin adjust_bottle clean v1",
                embodiment_env_id=spec["embodiment_env_id"], seeds=seeds, frozen=True, spec=spec))
        elif suite.spec != spec:
            raise ValueError("已登记 RoboTwin Suite 与冻结输入不一致")
        base_config = settings.rlinf_repo / "examples/embodiment/config/env/robotwin_adjust_bottle.yaml"
        task_limits_path = settings.robotwin_repo / "task_config/_eval_step_limit.yml"
        task_limits = yaml.safe_load(task_limits_path.read_text())
        base_config_sha = "sha256:" + hashlib.sha256(base_config.read_bytes()).hexdigest()
        full_spec = build_robotwin_full_suite_spec(
            seed_map, task_limits, base_config_sha, receipt["sha256"])
        full_sid = "suite_robotwin_rlinf_full_clean_v1"
        full_suite = await session.get(EvalSuite, full_sid)
        full_seeds = [case["seed"] for case in full_spec["cases"]]
        if not full_suite:
            session.add(EvalSuite(id=full_sid, name="RoboTwin RLInf full clean · 22 tasks / 5000 cases",
                embodiment_env_id=full_spec["embodiment_env_id"], seeds=full_seeds,
                frozen=True, spec=full_spec))
        elif full_suite.spec != full_spec:
            raise ValueError("已登记 RoboTwin 全量 Suite 与冻结输入不一致")
        env_row = await session.get(EmbodimentEnvRow, "ee_robotwin_aloha_joint14_v1")
        if env_row and env_row.status == "draft":
            env_row.status = "active"
        if task_row and task_row.status == "draft":
            task_row.status = "active"
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        logging.getLogger(__name__).info("RoboTwin eval suite stays unavailable until pinned assets are verified")
    await session.flush()
    try:
        await ingest_all(session)
    except Exception:
        logging.getLogger(__name__).exception("legacy ingest skipped")
    robotwin_jobs = (await session.scalars(select(Job).where(
        Job.kind.in_(["train.sft", "train.rlft"]), Job.status == "succeeded"))).all()
    deletions = (await session.scalars(select(AuditLog).where(AuditLog.action == "storage.delete"))).all()
    deleted_model_runs = {rid for row in deletions for rid in (row.before or {}).get("suppressed_model_runs", [])}
    from eb.orchestrator.collect import claim_outputs
    for job in robotwin_jobs:
        if not ((job.spec or {}).get("params") or {}).get("template_id", "").startswith("robotwin_"):
            continue
        run = await session.scalar(select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc()))
        if not run:
            continue
        if run.id in deleted_model_runs:
            continue
        try:
            await claim_outputs(session, job, run)
        except (OSError, ValueError, KeyError):
            logging.getLogger(__name__).exception("RoboTwin model package promotion skipped for %s", run.id)

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
    from eb.environment_catalog import seed_environment_profiles
    from eb.orchestrator.ufo_native import seed_native
    await seed_native(session)
    await seed_environment_profiles(session)
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
