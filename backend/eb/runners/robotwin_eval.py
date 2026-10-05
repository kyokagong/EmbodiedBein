"""RoboTwin evaluation-only entry owned by the platform evidence protocol."""
from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from eb.domain.robotwin_eval_runtime import RobotwinEvalRuntime, read_robotwin_runtime


@dataclass(frozen=True)
class TaskWave:
    task_id: str
    wave_id: int
    cases: tuple[dict, ...]
    active_slots: tuple[int, ...]
    max_episode_steps: int
    env_worker_count: int


class EvalBackend(Protocol):
    def start_policy(self, plan: dict, runtime: RobotwinEvalRuntime) -> dict: ...
    def start_task(self, task_id: str, max_episode_steps: int,
                   runtime: RobotwinEvalRuntime) -> None: ...
    def run_wave(self, wave: TaskWave) -> dict: ...
    def close_task(self): ...
    def close(self) -> None: ...


def wave_plan(plan: dict, runtime: RobotwinEvalRuntime) -> list[TaskWave]:
    """Partition frozen cases into in-process waves without changing placement."""
    by_task: dict[str, list[dict]] = {}
    for case in plan.get("cases", []):
        by_task.setdefault(case["task_id"], []).append(case)
    tasks = {item["task_id"]:item for item in (plan.get("suite_spec") or {}).get("tasks", [])}
    waves: list[TaskWave] = []
    wave_id = 0
    for task_id, cases in by_task.items():
        task = tasks.get(task_id) or {}
        limit = task.get("max_episode_steps") or (plan.get("suite_spec") or {}).get("max_episode_steps")
        if type(limit) is not int or limit <= 0:
            raise ValueError(f"RoboTwin 任务缺少步数上限: {task_id}")
        if len(cases) < runtime.env_worker_count:
            raise ValueError(f"RoboTwin 任务 case 少于 Env Worker 数: {task_id}")
        full, tail = divmod(len(cases), runtime.parallel_envs)
        sizes = [runtime.parallel_envs] * full
        if tail:
            if full and tail < runtime.env_worker_count:
                sizes[-1] -= runtime.env_worker_count - tail
                sizes.append(runtime.env_worker_count)
            else:
                sizes.append(tail)
        offset = 0
        for size in sizes:
            chunk = tuple(cases[offset:offset + size])
            waves.append(TaskWave(
                task_id=task_id, wave_id=wave_id, cases=chunk,
                active_slots=tuple(range(len(chunk))), max_episode_steps=limit,
                env_worker_count=runtime.env_worker_count,
            ))
            wave_id += 1
            offset += size
    if sum(len(wave.cases) for wave in waves) != plan.get("planned"):
        raise ValueError("RoboTwin 评估 wave 丢失 case")
    return waves


def evaluate_waves(plan: dict, runtime: RobotwinEvalRuntime, backend: EvalBackend) -> dict:
    """Execute deterministic waves while retaining one policy for the suite."""
    waves = wave_plan(plan, runtime)
    metrics: list[dict] = []
    current_task: str | None = None
    task_open = False
    backend.start_policy(plan, runtime)
    try:
        for wave in waves:
            if wave.task_id != current_task:
                if task_open:
                    _require_clean_task(backend.close_task())
                    task_open = False
                backend.start_task(wave.task_id, wave.max_episode_steps, runtime)
                current_task = wave.task_id
                task_open = True
            metrics.append(backend.run_wave(wave))
        if task_open:
            _require_clean_task(backend.close_task())
            task_open = False
    finally:
        if task_open:
            backend.close_task()
        backend.close()
    return merge_shard_metrics(metrics)


def _require_clean_task(receipt) -> None:
    rows = receipt if isinstance(receipt, list) else [receipt]
    for row in rows:
        if not isinstance(row, dict):
            continue
        if any(int(row.get(field, 0)) != 0
               for field in ("children_alive", "renderers_alive", "pending_messages")):
            raise RuntimeError("RoboTwin 任务环境清理未完成")


class RlinfEvalBackend:
    """One RLInf cluster and one Policy Worker for all in-process waves."""
    def __init__(self, cfg, runner, rollout, env):
        self.cfg, self.runner, self.rollout, self.env = cfg, runner, rollout, env
        self.policy_receipt = None

    def start_policy(self, plan, runtime):
        del plan, runtime
        self.runner.init_workers()
        receipts = self.rollout.runtime_receipt().wait()
        self.policy_receipt = next((value for value in receipts if value), None)
        if not self.policy_receipt or self.policy_receipt.get("model_load_count") != 1:
            raise RuntimeError("RoboTwin Policy 必须且只能加载一次")
        return self.policy_receipt

    def start_task(self, task_id, max_episode_steps, runtime):
        del runtime
        self.task_id, self.max_episode_steps = task_id, max_episode_steps

    def run_wave(self, wave: TaskWave):
        cases = [{**case, "wave_id":wave.wave_id} for case in wave.cases]
        configured = self.env.configure_eval_wave(
            cases=cases, active_slots=wave.active_slots, task_id=wave.task_id,
            max_episode_steps=wave.max_episode_steps).wait()
        if sum(int(value.get("envs", 0)) for value in configured if value) != len(cases):
            raise RuntimeError("RoboTwin wave 环境分配不完整")
        metrics = self.runner.evaluate_wave(
            steps=math.ceil(wave.max_episode_steps / int(self.cfg.rollout.model.num_action_chunks)),
            micro_batch_size=int(self.cfg.rollout.inference_micro_batch_size),
        )
        receipt = metrics.pop("runtime_receipt", None)
        if receipt != self.policy_receipt:
            raise RuntimeError("RoboTwin wave 使用了不同 Policy 实例")
        return {key:value.item() if hasattr(value, "item") else value
                for key, value in metrics.items()}

    def close_task(self):
        return self.env.close_eval_task(timeout_s=30.0).wait()

    def close(self):
        receipts = self.rollout.runtime_receipt().wait()
        final = next((value for value in receipts if value), None)
        if final != self.policy_receipt:
            raise RuntimeError("RoboTwin Policy 生命周期回执发生变化")
        self.rollout.close_policy().wait()


def runtime_parallelism(plan: dict) -> tuple[int, int]:
    runtime = ((plan.get("model") or {}).get("runtime") or {})
    gpu_count = runtime.get("gpu_count")
    per_gpu = runtime.get("per_gpu_concurrency")
    batch_size = runtime.get("eval_batch_size", 8)
    if gpu_count is None and per_gpu is None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("RoboTwin 历史评估并发无效")
        return 1, batch_size
    if (gpu_count not in {1, 2, 4} or type(per_gpu) is not int or per_gpu <= 0
            or type(batch_size) is not int or batch_size != gpu_count * per_gpu):
        raise ValueError("RoboTwin 多卡评估并发配置无效")
    return gpu_count, per_gpu


def shard_plan(plan: dict, gpu_count: int, per_gpu_concurrency: int) -> list[dict]:
    if (type(gpu_count) is not int or gpu_count <= 0
            or type(per_gpu_concurrency) is not int or per_gpu_concurrency <= 0):
        raise ValueError("RoboTwin 评估 GPU 数与每卡并发必须为正整数")
    by_task: dict[str, list[dict]] = {}
    for case in plan.get("cases", []):
        by_task.setdefault(case["task_id"], []).append(case)
    tasks = {item["task_id"]:item for item in (plan.get("suite_spec") or {}).get("tasks", [])}
    suite = {key:value for key, value in (plan.get("suite_spec") or {}).items()
             if key != "cases"}
    shards = []
    for task_id, cases in by_task.items():
        task = tasks.get(task_id) or {}
        limit = task.get("max_episode_steps") or (plan.get("suite_spec") or {}).get("max_episode_steps")
        if type(limit) is not int or limit <= 0:
            raise ValueError(f"RoboTwin 任务缺少步数上限: {task_id}")
        offset = 0
        while offset < len(cases):
            remaining = len(cases) - offset
            active = min(gpu_count, remaining)
            per_gpu = min(per_gpu_concurrency, remaining // active)
            planned = active * per_gpu
            chunk = cases[offset:offset + planned]
            shards.append({**plan, "suite_spec":suite, "planned":len(chunk), "cases":chunk,
                           "task_id":task_id, "max_episode_steps":limit,
                           "active_gpu_count":active, "per_gpu_envs":per_gpu})
            offset += planned
    if sum(item["planned"] for item in shards) != plan.get("planned"):
        raise ValueError("RoboTwin 评估分片丢失 case")
    return shards


def merge_shard_metrics(values: list[dict]) -> dict:
    if not values:
        raise ValueError("RoboTwin 评估没有分片指标")
    total = sum(value.get("num_trajectories", 0) for value in values)
    if type(total) is not int or total <= 0:
        raise ValueError("RoboTwin 分片轨迹数无效")
    keys = set(values[0]) - {"num_trajectories"}
    if any((set(value) - {"num_trajectories"}) != keys for value in values):
        raise ValueError("RoboTwin 分片指标字段不一致")
    merged = {"num_trajectories":total}
    for key in keys:
        numerator = 0.0
        for value in values:
            measured = value[key]
            if type(measured) not in (int, float) or not math.isfinite(measured):
                raise ValueError(f"RoboTwin 分片指标非法: {key}")
            numerator += measured * value["num_trajectories"]
        merged[key] = numerator / total
    return merged


def evaluate_shards(plan: dict, output: Path, shards: list[dict]) -> dict:
    """Run bounded single-task RLInf processes and merge their measured evidence."""
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "eval-plan.json", plan)
    replaced = ("env.eval.task_config.task_name=", "env.eval.task_config.step_lim=",
                "env.eval.total_num_envs=", "env.eval.max_episode_steps=",
                "env.eval.max_steps_per_rollout_epoch=", "env.eval.seeds_path=",
                "runner.logger.log_path=", "runner.logger.experiment_name=", "hydra.run.dir=",
                "cluster.component_placement=")
    base_args = [arg for arg in sys.argv[1:] if not arg.startswith(replaced)]
    expected, metrics, policy = set(), [], None
    evidence = output / "eval-episodes-rank0.jsonl"
    if evidence.exists():
        raise FileExistsError("RoboTwin 全量评估输出已存在")
    for index, shard in enumerate(shards):
        shard_output = output / "shards" / f"{index:04d}-{shard['task_id']}"
        if shard_output.exists():
            raise FileExistsError("RoboTwin 评估分片输出已存在")
        child_plan = {key:value for key, value in shard.items()
                      if key not in {"task_id", "max_episode_steps"}}
        env = {**os.environ, "EB_EVAL_PLAN":json.dumps(child_plan, ensure_ascii=False),
               "EB_OUTPUT_DIR":str(shard_output), "EB_ROBOTWIN_SHARD":"1"}
        active = shard["active_gpu_count"]
        command = [sys.executable, str(Path(__file__).with_name("rlinf_entry.py")), *base_args,
            f"cluster.component_placement={{env\\, rollout:0-{active - 1}}}",
            f"env.eval.task_config.task_name={shard['task_id']}",
            f"env.eval.task_config.step_lim={shard['max_episode_steps']}",
            f"env.eval.total_num_envs={shard['planned']}",
            f"env.eval.max_episode_steps={shard['max_episode_steps']}",
            f"env.eval.max_steps_per_rollout_epoch={shard['max_episode_steps']}",
            f"env.eval.seeds_path={shard_output / 'robotwin-eval-seeds.json'}",
            f"runner.logger.log_path={shard_output}",
            f"hydra.run.dir={shard_output / 'hydra'}",
            f"runner.logger.experiment_name=robotwin-{index:04d}-{shard['task_id']}"]
        subprocess.run(command, env=env, check=True)
        manifest = json.loads((shard_output / "eval-manifest.json").read_text())
        child_policy = (shard_output / "policy-receipt.json").read_bytes()
        if policy is None:
            policy = child_policy
            (output / "policy-receipt.json").write_bytes(policy)
        elif child_policy != policy:
            raise ValueError("RoboTwin 分片模型加载凭证不一致")
        rows = []
        for path in sorted(shard_output.glob("eval-episodes-rank*.jsonl")):
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        wanted = {case["case_id"] for case in shard["cases"]}
        found = {row.get("case_id") for row in rows}
        if (manifest.get("plan_hash") != plan["hash"] or found != wanted
                or len(rows) != len(wanted)
                or manifest.get("metrics", {}).get("num_trajectories") != len(wanted)
                or expected & found):
            raise ValueError("RoboTwin 分片证据与冻结计划不一致")
        with evidence.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        expected.update(found)
        metrics.append(manifest["metrics"])
    if expected != {case["case_id"] for case in plan["cases"]}:
        raise ValueError("RoboTwin 全量评估缺少 case")
    return merge_shard_metrics(metrics)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    temporary.replace(path)


def seal_policy_evidence(output: Path, receipt: dict) -> str:
    """Write the post-load receipt and bind every completed case to it."""
    receipt_path = output / "policy-receipt.json"
    _write_json(receipt_path, receipt)
    digest = "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    for path in sorted(output.glob("eval-episodes-rank*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(
            {**row, "policy_receipt_sha256":digest}, ensure_ascii=False,
            separators=(",", ":")) + "\n" for row in rows))
        temporary.replace(path)
    return digest


def validated_completed_case_ids(output: Path, plan: dict,
                                 receipt_sha256: str) -> set[str]:
    """Return only unique evidence rows bound to this exact plan and policy."""
    expected = {case["case_id"]:case for case in plan.get("cases", [])}
    task_hashes = {task["task_id"]:task.get("task_config_sha256")
                   for task in (plan.get("suite_spec") or {}).get("tasks", [])}
    valid, duplicates = set(), set()
    for path in sorted(output.glob("eval-episodes-rank*.jsonl")):
        if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                case = expected.get(row.get("case_id"))
            except json.JSONDecodeError:
                continue
            if (not case or row.get("plan_hash") != plan.get("hash")
                    or row.get("policy_receipt_sha256") != receipt_sha256
                    or row.get("task_id") != case.get("task_id")
                    or row.get("seed") != case.get("seed")
                    or row.get("scenario") != case.get("scenario")
                    or row.get("task_config_sha256") != task_hashes.get(case.get("task_id"))):
                continue
            case_id = case["case_id"]
            if case_id in valid:
                duplicates.add(case_id)
            valid.add(case_id)
    return valid - duplicates


def _load_eval_plan() -> dict | None:
    path = os.environ.get("EB_EVAL_PLAN_PATH")
    raw = Path(path).read_text() if path else os.environ.get("EB_EVAL_PLAN")
    return json.loads(raw) if raw else None


def prepare_plan(output: Path, task_name: str | None = None) -> dict:
    try:
        plan = _load_eval_plan()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("RoboTwin 评估缺少冻结计划") from exc
    if (not isinstance(plan, dict) or plan.get("schema_version") != "eb/robotwin-eval-plan/v1"
            or not isinstance(plan.get("cases"), list) or len(plan["cases"]) != plan.get("planned")):
        raise ValueError("RoboTwin 评估计划无效")
    suite = plan.get("suite_spec") or {}
    if suite.get("robotwin_revision") != os.environ.get("EB_ROBOTWIN_REVISION"):
        raise ValueError("RoboTwin revision 与冻结计划不一致")
    seeds_by_task, case_ids, seeds = {}, set(), set()
    known_tasks = {item.get("task_id") for item in suite.get("tasks", [])}
    for case in plan["cases"]:
        if (not isinstance(case, dict) or case.get("task_id") not in known_tasks
                or (task_name is not None and case.get("task_id") != task_name)
                or type(case.get("seed")) is not int or not isinstance(case.get("case_id"), str)):
            raise ValueError("RoboTwin 评估任务或 case 无效")
        seed_key = (case["task_id"], case["seed"])
        if case["case_id"] in case_ids or seed_key in seeds:
            raise ValueError("RoboTwin 评估包含重复 case 或 seed")
        case_ids.add(case["case_id"])
        seeds.add(seed_key)
        seeds_by_task.setdefault(case["task_id"], {"success_seeds":[]})[
            "success_seeds"].append(case["seed"])
    _write_json(output / "robotwin-eval-seeds.json", seeds_by_task)
    _write_json(output / "eval-plan.json", plan)
    return plan


def build_case_record(plan: dict, seed: int, success_once: bool, env_steps: int,
                      chunk_requested: int, chunk_executed: int, sim_steps: int, sim_seconds: float,
                      instruction: str, run_id: str, task_id: str | None = None) -> dict:
    case = next((item for item in plan.get("cases", [])
                 if item.get("seed") == seed and (task_id is None or item.get("task_id") == task_id)), None)
    if case is None:
        raise ValueError("RoboTwin 实测 seed 不在冻结计划中")
    if (type(success_once) is not bool or type(env_steps) is not int or env_steps < 0
            or type(chunk_executed) is not int or not 0 <= chunk_executed <= chunk_requested
            or type(sim_steps) is not int or sim_steps < 0
            or not isinstance(sim_seconds, (int, float)) or sim_seconds < 0
            or not isinstance(instruction, str) or not instruction.strip()):
        raise ValueError("RoboTwin case 指标无效")
    task_spec = next((item for item in (plan.get("suite_spec") or {}).get("tasks", [])
                      if item.get("task_id") == case["task_id"]), {})
    max_steps = int(task_spec.get("max_episode_steps")
                    or (plan.get("suite_spec") or {}).get("max_episode_steps", 0))
    if success_once:
        status = "success"
    elif env_steps >= max_steps:
        status = "timeout"
    else:
        status = "task_failure"
    task_config_sha256 = task_spec.get("task_config_sha256")
    if not isinstance(task_config_sha256, str) or not task_config_sha256.startswith("sha256:"):
        raise ValueError("RoboTwin case 缺少冻结任务配置")
    scene = {"task_id":case["task_id"], "scenario":case["scenario"], "seed":seed,
             "instruction":instruction, "task_config_sha256":task_config_sha256}
    scene_hash = "sha256:" + hashlib.sha256(json.dumps(
        scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return {**case, "run_id": run_id, "plan_hash": plan["hash"],
            "seed_planned": case["seed"], "seed_actual": seed,
            "scene_hash":scene_hash,
            "status": status, "success_once": success_once,
            "termination_reason": status, "env_steps": env_steps,
            "action_index":max(0, env_steps - 1), "sim_steps":sim_steps,
            "sim_seconds":float(sim_seconds), "task_config_sha256":task_config_sha256,
            "chunk_requested": chunk_requested, "chunk_executed": chunk_executed,
            "instruction":instruction, "media":[],
            "model_version_id": (plan.get("model") or {}).get("model_version_id")}


def evaluate(cfg) -> dict:
    """Run RLInf's evaluation scheduler with platform-owned case evidence."""
    output = Path(os.environ.get("EB_OUTPUT_DIR", "/workspace/output"))
    plan = prepare_plan(output)
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from robotwin_env import RobotwinEnvWorker

    runtime = read_robotwin_runtime(plan, submission=True)
    print(f"[RoboTwin eval] cases={plan['planned']} parallel_envs={runtime.parallel_envs}", flush=True)
    from robotwin_source import prepare_robotwin_source
    runtime_values = (plan.get("model") or {}).get("runtime") or {}
    patched_root, patched_source_sha = prepare_robotwin_source(
        Path(os.environ["ROBOTWIN_PATH"]), output,
        str(runtime_values.get("robotwin_evidence_patch_sha256", "")))
    sys.path.insert(0, str(patched_root))
    os.environ["PYTHONPATH"] = str(patched_root) + ":" + os.environ.get("PYTHONPATH", "")
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=placement.get_strategy("rollout"))
    env = RobotwinEnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement.get_strategy("env"))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    backend = RlinfEvalBackend(cfg, runner, rollout, env)
    metrics = evaluate_waves(plan, runtime, backend)
    normalized = {key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()}
    if normalized.get("num_trajectories") != plan["planned"]:
        raise ValueError("RoboTwin 实测轨迹数与冻结计划不一致")
    seal_policy_evidence(output, {
        "schema_version":"eb/robotwin-policy-receipt/v1", "status":"loaded",
        "model_version_id":(plan.get("model") or {}).get("model_version_id"),
        "artifact_sha256":(plan.get("model") or {}).get("artifact_sha256"),
        "model_path":str(cfg.rollout.model.model_path),
        "config_name":str(cfg.rollout.model.openpi.config_name),
        "observation_profile":(plan.get("suite_spec") or {}).get("observation_profile"),
        "controller_profile":(plan.get("suite_spec") or {}).get("controller_profile"),
        "num_action_chunks":int(cfg.rollout.model.num_action_chunks),
        "robotwin_revision":os.environ.get("EB_ROBOTWIN_REVISION"),
        "rlinf_revision":(plan.get("suite_spec") or {}).get("rlinf_revision"),
        "robotwin_content_hash":runtime_values.get("robotwin_content_hash"),
        "evidence_patch_sha256":runtime_values.get("robotwin_evidence_patch_sha256"),
        "patched_base_task_sha256":patched_source_sha,
        **(backend.policy_receipt or {}),
        "headless":runtime.headless,
        "offscreen_rendering":runtime.offscreen_rendering,
        "save_video":runtime.save_video,
        "inference_micro_batch_size":runtime.inference_micro_batch_size,
    })
    runner.metric_logger.log(step=0, data={f"eval/{key}": value for key, value in normalized.items()})
    runner.metric_logger.finish()
    return normalized
