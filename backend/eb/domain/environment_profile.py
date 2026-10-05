"""Versioned operator-facing environments and their supported runtime knobs."""

from __future__ import annotations

import hashlib
import json
import math

RPO_TRAIN = "rpo_v2_wbc_stand_walk_ppo"
RPO_EVAL_SUFFIX = "_stand_walk_v2"
AMP_TRAIN = "rpo_v2_amp_stand_walk_ppo"
AMP_EVAL_PREFIX = "suite_rpo_v2_amp_"
AMP_INITIAL_POSE = "official_a1_stand_v1"
TRAIN_SETTINGS = {
    "target_speed_mps": (0.05, 1.0),
    "stand_fraction": (0.0, 1.0),
    "velocity_reward_weight": (0.0, 10.0),
    "upright_reward_weight": (0.0, 10.0),
}
EVAL_SETTINGS = {"stand_max_drift_m": (0.05, 2.0), "walk_min_forward_m": (0.5, 8.0)}


def validate_definition(usage: str, binding: str, value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("环境定义必须是对象")
    result = {}
    for key in ("task", "reward", "failure", "success"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError(f"{key} 必须填写 1–2000 字")
        result[key] = text.strip()
    settings = value.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings 必须是对象")
    allowed = (
        TRAIN_SETTINGS
        if usage == "train" and binding == RPO_TRAIN
        else (EVAL_SETTINGS if usage == "eval" and binding.endswith(RPO_EVAL_SUFFIX) else {})
    )
    if (usage == "train" and binding == AMP_TRAIN
            or usage == "eval" and binding.startswith(AMP_EVAL_PREFIX)):
        allowed = {**allowed, "initial_pose": AMP_INITIAL_POSE}
    if usage == "train" and binding == AMP_TRAIN:
        allowed = {**allowed, "initial_logstd": (-4.0, 0.0),
                   "velocity_tracking_std": (0.05, 0.5),
                   "velocity_reward_weight": TRAIN_SETTINGS["velocity_reward_weight"],
                   "actuator_delay_steps": (0, 2)}
    if usage == "eval" and binding.startswith("suite_rpo_v2_amp_isaac_"):
        allowed = {**allowed, "actuator_delay_steps": (0, 2)}
    if set(settings) - set(allowed):
        raise ValueError("环境参数不支持当前执行入口")
    for key, number in settings.items():
        if key == "initial_pose":
            if number != AMP_INITIAL_POSE:
                raise ValueError("不支持的 AMP 初始姿态")
            continue
        if key == "actuator_delay_steps":
            if type(number) is not int or not 0 <= number <= 2:
                raise ValueError("AMP 执行器延迟必须为 0–2 物理步")
            continue
        low, high = allowed[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not low <= number <= high
        ):
            raise ValueError(f"{key} 超出允许范围 {low}–{high}")
    result["settings"] = settings
    return result


def compatible_profile(row, usage: str, binding: str, env_id: str, task_id: str | None) -> None:
    if (
        row.status != "active"
        or row.usage != usage
        or row.embodiment_env_id != env_id
        or row.task_id != task_id
        or (row.template_id if usage == "train" else row.suite_id) != binding
    ):
        raise ValueError("环境与模板或套件不匹配，或未发布")
    if profile_hash(row) != row.content_hash:
        raise ValueError("环境内容哈希不一致")


def profile_hash(row) -> str:
    value = {
        key: getattr(row, key)
        for key in ("usage", "embodiment_env_id", "task_id", "template_id", "suite_id", "definition")
    }
    return (
        "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    )


def profile_snapshot(row) -> dict:
    return {"id": row.id, "hash": row.content_hash, "definition": row.definition}


def rpo_eval_arguments(plan: dict) -> list[str]:
    settings = plan.get("environment_settings") or {}
    if not settings:
        return []
    if plan.get("criterion") != "stand_walk":
        raise ValueError("当前评估任务不支持可编辑阈值")
    amp = str(plan.get("suite_id", "")).startswith(AMP_EVAL_PREFIX)
    amp_isaac = str(plan.get("suite_id", "")).startswith("suite_rpo_v2_amp_isaac_")
    allowed = {**EVAL_SETTINGS, **({"initial_pose": AMP_INITIAL_POSE} if amp else {}),
               **({"actuator_delay_steps": (0, 2)} if amp_isaac else {})}
    if set(settings) - set(allowed):
        raise ValueError("评估环境参数不支持")
    result = []
    for key, value in settings.items():
        if key == "initial_pose":
            if value != AMP_INITIAL_POSE:
                raise ValueError("不支持的 AMP 初始姿态")
            result += ["--initial-pose", value]
            continue
        if key == "actuator_delay_steps":
            if type(value) is not int or not 0 <= value <= 2:
                raise ValueError("AMP 执行器延迟必须为 0–2 物理步")
            result += ["--actuator-delay-steps", str(value)]
            continue
        low, high = EVAL_SETTINGS[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError("评估环境参数超出范围")
        result += ["--" + key.replace("_", "-"), str(value)]
    return result
