"""Frozen metadata for bounded interactive native UFO diagnostic sessions."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from types import SimpleNamespace

from eb.domain.environment_profile import profile_hash
from eb.ufo_native import ENV_ID, TASK_ID, TASKS, plan_hash

OPERATION = "ufo_native_live"
TASK_IDS = tuple(task["id"] for task in TASKS)


def suite_id(backend: str, *, version=2) -> str:
    if backend not in ("mujoco", "mjlab") or version not in (1, 2):
        raise ValueError("不支持的实时仿真后端")
    return f"suite_rpo_ufo_native_{backend}_live_v{version}"


def live_suite_spec(backend: str, *, version=2) -> dict:
    suite_id(backend, version=version)
    value = {
        "benchmark": "rpo_ufo_native_live",
        "backend": backend,
        "profile": "interactive",
        "policy": "onnx",
        "tasks": list(TASK_IDS),
        "policy_hz": 50,
        "physics_hz": 400,
        "lease_seconds": 15,
        "max_seconds": 600,
        "min_height_m": 0.5,
        "max_tilt_rad": 0.8,
        "record_fps": 5,
        "trace_chunk_seconds": 10,
        "scope": "simulation_only",
        "qualification": "diagnostic",
        "physics_family": "MuJoCo",
    }
    if version == 2:
        value["stream"] = {"transport": "websocket-jpeg", "width": 640, "height": 480,
                           "fps": 30, "max_frame_bytes": 2097152}
    return value


def environment_spec(backend: str, *, version=2) -> dict:
    value = {
        "id": f"ep_rpo_ufo_native_{backend}_live_v{version}",
        "name": f"UFO/RPO {backend} 平地实时诊断 v{version}",
        "backend": backend,
        "usage": "eval",
        "embodiment_env_id": ENV_ID,
        "task_id": TASK_ID,
        "suite_id": suite_id(backend, version=version),
        "template_id": None,
        "definition": {
            "task": "原生 UFO 站立与前行任务连续切换；固定 ONNX，50Hz 策略与 400Hz 物理。",
            "reward": "不训练、不重新生成任务 z，不计算训练奖励。",
            "failure": "跌倒暂停；非有限状态终止；15 秒无租约自动停止。",
            "success": "交互诊断，不产生正式评估通过资格或真机放行。",
            "settings": {},
        },
    }
    if version == 2:
        value["name"] = f"UFO/RPO {backend} GPU 原生画面推流 v2"
        value["definition"]["settings"] = {"render": "egl", "stream": "websocket-jpeg",
                                            "render_gpus": 1, "width": 640, "height": 480, "fps": 30}
    value["content_hash"] = profile_hash(SimpleNamespace(**value))
    return value


def validate_live_plan(plan: dict) -> dict:
    """Validate independently of database, numerical runtimes, or filesystem IO."""
    try:
        expected_fields = {
            "operation",
            "name",
            "model_id",
            "source_model_id",
            "input_path",
            "source_artifact_sha256",
            "native_source_path",
            "native_source_sha256",
            "image_digest",
            "backend",
            "environment_id",
            "environment_hash",
            "suite_id",
            "suite_spec",
            "tasks",
            "initial_task",
            "duration_seconds",
            "seed",
            "hash",
        }
        allowed_fields = expected_fields | ({"source_refs", "code_sources"} if plan.get("code_sources") else set())
        if not isinstance(plan, dict) or set(plan) != allowed_fields:
            raise ValueError("实时计划字段不完整或包含未知字段")
        version = 1 if plan["environment_id"].endswith("_v1") else 2
        environment = environment_spec(plan["backend"], version=version)
        if (
            plan["operation"] != OPERATION
            or plan["hash"] != plan_hash(plan)
            or plan["environment_id"] != environment["id"]
            or plan["environment_hash"] != environment["content_hash"]
            or plan["suite_id"] != environment["suite_id"]
            or plan["suite_spec"] != live_suite_spec(plan["backend"], version=version)
            or plan["tasks"] != list(TASK_IDS)
            or plan["initial_task"] not in TASK_IDS
            or type(plan["duration_seconds"]) is not int
            or not 30 <= plan["duration_seconds"] <= 600
            or type(plan["seed"]) is not int
            or not 0 <= plan["seed"] <= 2**31 - 1
            or not isinstance(plan["name"], str)
            or not 0 < len(plan["name"]) <= 256
        ):
            raise ValueError("实时计划、环境、任务或运行预算已变化")
        for key in ("image_digest", "source_artifact_sha256", "native_source_sha256"):
            if not isinstance(plan[key], str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", plan[key]):
                raise ValueError("实时模型或镜像摘要无效")
        for key in ("model_id", "source_model_id"):
            if not isinstance(plan[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", plan[key]):
                raise ValueError("实时模型标识无效")
        for key in ("input_path", "native_source_path"):
            path = plan[key]
            if (
                not isinstance(path, str)
                or not PurePosixPath(path).is_absolute()
                or any(part in ("", ".", "..") for part in path.split("/")[1:])
                or "\\" in path
                or any(ord(c) < 32 for c in path)
            ):
                raise ValueError("实时模型路径无效")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("实时计划结构无效") from exc
    return plan
