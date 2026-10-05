from __future__ import annotations

import hashlib
import json

ROBOTWIN_REVISION = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"
RLINF_REVISION = "018e6a9e0da60f96eeda8b11008543911027d414"
EVAL_SEEDS_SHA256 = "194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f"
TASK_CONFIG_SHA256 = "sha256:eecba407c0eb3a0f3e90b71ac4cb1aa78d46ed4da12fec1fd4965dc91c1005c8"
SUCCESS_CODE_SHA256 = "sha256:fc9288b497d588588d495540a7f42f18b5d50832f8b0169356b64925e26b0830"


def asset_manifest_sha256(receipt: dict) -> str:
    payload = {key:value for key, value in receipt.items() if key != "sha256"}
    return "sha256:" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def build_robotwin_suite_spec(seeds: list[int], assets_manifest_sha256: str) -> dict:
    if (not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds)
            or not assets_manifest_sha256.startswith("sha256:")):
        raise ValueError("RoboTwin suite 输入未冻结")
    cases = [{"case_id":f"adjust_bottle/clean/{index:06d}", "task_id":"adjust_bottle",
              "seed":seed, "scenario":"clean"} for index, seed in enumerate(seeds)]
    case_hash = "sha256:" + hashlib.sha256(json.dumps(
        cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"benchmark":"robotwin", "suite":"robotwin_adjust_bottle_clean_v1",
            "protocol":"platform_fixed", "robotwin_revision":ROBOTWIN_REVISION,
            "rlinf_revision":RLINF_REVISION, "assets_manifest_sha256":assets_manifest_sha256,
            "embodiment_env_id":"ee_robotwin_aloha_joint14_v1", "robot_platform":"ALOHA",
            "controller_profile":"aloha_joint14_position_v1",
            "observation_profile":"robotwin_aloha_3cam_joint14_v1", "domain":"clean",
            "tasks":[{"task_id":"adjust_bottle", "task_config_sha256":TASK_CONFIG_SHA256}],
            "case_manifest_sha256":case_hash, "cases":cases,
            "success_definition":{"name":"robotwin_success_once", "code_sha256":SUCCESS_CODE_SHA256},
            "max_episode_steps":200,
            "action_chunk":{"consume_rule":"stop_on_success_timeout_or_termination"},
            "precision":"bf16", "repeats":1,
            "qualification":{"metric":"success_rate", "threshold":0.0}}


def build_robotwin_full_suite_spec(seed_map: dict, task_limits: dict[str, int],
                                   base_config_sha256: str,
                                   assets_manifest_sha256: str) -> dict:
    if (not isinstance(seed_map, dict) or not seed_map
            or not base_config_sha256.startswith("sha256:")
            or not assets_manifest_sha256.startswith("sha256:")):
        raise ValueError("RoboTwin full suite 输入未冻结")
    tasks, per_task = [], {}
    for task_id, value in seed_map.items():
        seeds = value.get("success_seeds") if isinstance(value, dict) else None
        limit = task_limits.get(task_id)
        if (not isinstance(task_id, str) or not task_id or not isinstance(seeds, list) or not seeds
                or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds)
                or type(limit) is not int or limit <= 0):
            raise ValueError(f"RoboTwin full suite 任务未冻结: {task_id}")
        config = {"base_config_sha256":base_config_sha256, "task_id":task_id,
                  "scenario":"clean", "embodiment":"aloha-agilex",
                  "max_episode_steps":limit}
        task_hash = "sha256:" + hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        tasks.append({"task_id":task_id, "task_config_sha256":task_hash,
                      "max_episode_steps":limit})
        per_task[task_id] = seeds
    cases = []
    for index in range(max(map(len, per_task.values()))):
        for task_id, seeds in per_task.items():
            if index < len(seeds):
                cases.append({"case_id":f"{task_id}/clean/{index:06d}", "task_id":task_id,
                              "seed":seeds[index], "scenario":"clean"})
    case_hash = "sha256:" + hashlib.sha256(json.dumps(
        cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"benchmark":"robotwin", "suite":"robotwin_rlinf_full_clean_v1",
            "protocol":"platform_fixed", "evaluation_scope":"rlinf_supported_full",
            "robotwin_revision":ROBOTWIN_REVISION, "rlinf_revision":RLINF_REVISION,
            "assets_manifest_sha256":assets_manifest_sha256,
            "embodiment_env_id":"ee_robotwin_aloha_joint14_v1", "robot_platform":"ALOHA",
            "controller_profile":"aloha_joint14_position_v1",
            "observation_profile":"robotwin_aloha_3cam_joint14_v1", "domain":"clean",
            "tasks":tasks, "case_manifest_sha256":case_hash, "cases":cases,
            "success_definition":{"name":"robotwin_success_once", "code_sha256":SUCCESS_CODE_SHA256},
            "max_episode_steps":max(task_limits[task["task_id"]] for task in tasks),
            "action_chunk":{"consume_rule":"stop_on_success_timeout_or_termination"},
            "precision":"bf16", "repeats":1,
            "qualification":{"metric":"success_rate", "threshold":0.0}}
