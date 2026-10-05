import json
import hashlib

import pytest

from eb.runners.robotwin_eval import (
    TaskWave,
    build_case_record,
    evaluate_waves,
    evaluate_shards,
    merge_shard_metrics,
    prepare_plan,
    runtime_parallelism,
    seal_policy_evidence,
    shard_plan,
    wave_plan,
)
from eb.domain.robotwin_eval_runtime import derive_robotwin_runtime
from eb.runners import robotwin_source


def plan():
    cases = [{"case_id": f"adjust_bottle/clean/{index:06d}", "task_id": "adjust_bottle",
              "seed": 1000 + index, "scenario": "clean"} for index in range(5)]
    return {"schema_version":"eb/robotwin-eval-plan/v1", "planned":5, "cases":cases,
            "hash":"sha256:" + "a" * 64,
            "suite_spec":{"robotwin_revision":"abc123", "domain":"clean", "max_episode_steps":200,
                          "tasks":[{"task_id":"adjust_bottle",
                                    "task_config_sha256":"sha256:" + "b" * 64}]}}


def test_prepare_plan_writes_exact_nonrandom_seed_manifest(tmp_path, monkeypatch):
    value = plan()
    monkeypatch.setenv("EB_EVAL_PLAN", json.dumps(value))
    monkeypatch.setenv("EB_ROBOTWIN_REVISION", "abc123")

    prepared = prepare_plan(tmp_path, "adjust_bottle")

    assert prepared == value
    assert json.loads((tmp_path / "robotwin-eval-seeds.json").read_text()) == {
        "adjust_bottle": {"success_seeds": [1000, 1001, 1002, 1003, 1004]}}
    assert json.loads((tmp_path / "eval-plan.json").read_text())["hash"] == value["hash"]


def test_prepare_plan_rejects_task_revision_or_duplicate_seed(tmp_path, monkeypatch):
    value = plan()
    monkeypatch.setenv("EB_EVAL_PLAN", json.dumps(value))
    monkeypatch.setenv("EB_ROBOTWIN_REVISION", "different")
    with pytest.raises(ValueError, match="revision"):
        prepare_plan(tmp_path, "adjust_bottle")
    monkeypatch.setenv("EB_ROBOTWIN_REVISION", "abc123")
    value["cases"][1]["seed"] = value["cases"][0]["seed"]
    monkeypatch.setenv("EB_EVAL_PLAN", json.dumps(value))
    with pytest.raises(ValueError, match="重复"):
        prepare_plan(tmp_path, "adjust_bottle")
    monkeypatch.setenv("EB_EVAL_PLAN", json.dumps(plan()))
    with pytest.raises(ValueError, match="任务"):
        prepare_plan(tmp_path, "place_shoe")


def test_case_record_distinguishes_success_failure_and_timeout():
    value = plan()
    success = build_case_record(value, 1000, True, 73, 50, 23, 258, 8.65, "adjust the bottle", "run_1")
    failure = build_case_record(value, 1001, False, 120, 50, 20, 300, 10.0, "adjust the bottle", "run_1")
    timeout = build_case_record(value, 1002, False, 200, 50, 50, 360, 12.0, "adjust the bottle", "run_1")
    assert success["status"] == "success" and success["chunk_executed"] == 23
    assert success["scene_hash"].startswith("sha256:") and success["sim_seconds"] == 8.65
    assert success["sim_steps"] == 258
    assert success["task_config_sha256"] == "sha256:" + "b" * 64
    assert failure["status"] == "task_failure" and failure["success_once"] is False
    assert timeout["status"] == "timeout" and timeout["env_steps"] == 200
    with pytest.raises(ValueError, match="seed"):
        build_case_record(value, 9999, False, 1, 50, 1, 1, 1.0, "adjust", "run_1")


def test_case_record_disambiguates_reused_seed_by_task():
    value = plan()
    value["cases"].append({"case_id":"other/clean/000000", "task_id":"other",
                           "seed":1000, "scenario":"clean"})
    value["suite_spec"]["tasks"].append({"task_id":"other",
        "task_config_sha256":"sha256:" + "c" * 64})

    row = build_case_record(value, 1000, False, 10, 50, 10, 10, 1.0,
                            "other task", "run_1", "other")

    assert row["case_id"] == "other/clean/000000" and row["task_id"] == "other"


def test_policy_receipt_is_hashed_into_every_case(tmp_path):
    row = build_case_record(plan(), 1000, False, 50, 50, 50, 50, 1.0, "adjust", "run_1")
    evidence = tmp_path / "eval-episodes-rank0.jsonl"
    evidence.write_text(json.dumps(row) + "\n")
    digest = seal_policy_evidence(tmp_path, {"schema_version":"eb/robotwin-policy-receipt/v1",
                                             "status":"loaded", "model_version_id":"mv_1"})
    assert digest.startswith("sha256:")
    assert json.loads(evidence.read_text())["policy_receipt_sha256"] == digest
    assert json.loads((tmp_path / "policy-receipt.json").read_text())["status"] == "loaded"


def test_full_plan_shards_by_task_and_batch_without_losing_cases():
    value = plan()
    value["cases"] = [
        {"case_id":f"task_a/clean/{i:06d}", "task_id":"task_a", "seed":100 + i,
         "scenario":"clean"} for i in range(3)
    ] + [
        {"case_id":f"task_b/clean/{i:06d}", "task_id":"task_b", "seed":200 + i,
         "scenario":"clean"} for i in range(2)
    ]
    value["planned"] = 5
    value["suite_spec"]["tasks"] = [
        {"task_id":"task_a", "task_config_sha256":"sha256:" + "a" * 64,
         "max_episode_steps":400},
        {"task_id":"task_b", "task_config_sha256":"sha256:" + "b" * 64,
         "max_episode_steps":500},
    ]
    value["suite_spec"]["cases"] = value["cases"] * 3000
    shards = shard_plan(value, gpu_count=1, per_gpu_concurrency=2)
    assert [len(item["cases"]) for item in shards] == [2, 1, 2]
    assert [item["task_id"] for item in shards] == ["task_a", "task_a", "task_b"]
    assert [case["case_id"] for item in shards for case in item["cases"]] == [
        case["case_id"] for case in value["cases"]]
    assert shards[-1]["max_episode_steps"] == 500
    assert all("cases" not in item["suite_spec"] for item in shards)


def test_tail_wave_uses_exact_slots_without_reducing_workers():
    value = plan()
    value["cases"] = [
        {"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
         "seed":1000 + i, "scenario":"clean"} for i in range(22)
    ]
    value["planned"] = len(value["cases"])
    runtime, _ = derive_robotwin_runtime(4, 16)

    waves = wave_plan(value, runtime)

    assert [len(wave.cases) for wave in waves] == [16, 6]
    assert waves[0].active_slots == tuple(range(16))
    assert waves[1].active_slots == tuple(range(6))
    assert all(wave.env_worker_count == 4 for wave in waves)
    assert [case["case_id"] for wave in waves for case in wave.cases] == [
        case["case_id"] for case in value["cases"]]


def test_in_process_waves_start_policy_once_without_subprocess(monkeypatch):
    value = plan()
    value["cases"] = [
        {"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
         "seed":1000 + i, "scenario":"clean"} for i in range(5)
    ]
    value["planned"] = len(value["cases"])
    runtime, _ = derive_robotwin_runtime(1, 4)
    events = []

    class Backend:
        def start_policy(self, _plan, _runtime):
            events.append("policy")
            return {"model_load_id":"load-1"}
        def start_task(self, task_id, max_episode_steps, _runtime):
            events.append(("task", task_id, max_episode_steps))
        def run_wave(self, wave: TaskWave):
            events.append(("wave", wave.wave_id, len(wave.cases)))
            return {"num_trajectories":len(wave.cases), "success_once":1.0}
        def close_task(self):
            events.append("close-task")
        def close(self):
            events.append("close")

    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("subprocess"))
    result = evaluate_waves(value, runtime, Backend())

    assert result == {"num_trajectories":5, "success_once":1.0}
    assert events.count("policy") == 1
    assert [event for event in events if isinstance(event, tuple) and event[0] == "wave"] == [
        ("wave", 0, 4), ("wave", 1, 1)]


def test_task_switch_requires_clean_barrier():
    value = plan()
    value["cases"] = [
        {"case_id":"task_a/clean/000000", "task_id":"task_a", "seed":1, "scenario":"clean"},
        {"case_id":"task_b/clean/000000", "task_id":"task_b", "seed":2, "scenario":"clean"},
    ]
    value["planned"] = 2
    value["suite_spec"]["tasks"] = [
        {"task_id":"task_a", "max_episode_steps":50},
        {"task_id":"task_b", "max_episode_steps":50},
    ]
    runtime, _ = derive_robotwin_runtime(1, 4)

    class Backend:
        started_tasks = []
        def start_policy(self, *_): return {}
        def start_task(self, task_id, *_): self.started_tasks.append(task_id)
        def run_wave(self, wave):
            return {"num_trajectories":len(wave.cases), "success_once":1.0}
        def close_task(self): return {"children_alive":1, "pending_messages":0}
        def close(self): pass

    backend = Backend()
    with pytest.raises(RuntimeError, match="清理"):
        evaluate_waves(value, runtime, backend)
    assert backend.started_tasks == ["task_a"]


def test_small_tail_is_rebalanced_across_all_env_workers():
    value = plan()
    value["cases"] = [
        {"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
         "seed":1000 + i, "scenario":"clean"} for i in range(17)
    ]
    value["planned"] = 17
    runtime, _ = derive_robotwin_runtime(4, 16)

    waves = wave_plan(value, runtime)

    assert [len(wave.cases) for wave in waves] == [13, 4]
    assert all(len(wave.cases) >= runtime.env_worker_count for wave in waves)


@pytest.mark.parametrize("count,expected", [
    (1, [(1, 1, 1)]),
    (3, [(3, 3, 1)]),
    (5, [(4, 4, 1), (1, 1, 1)]),
    (6, [(4, 4, 1), (2, 2, 1)]),
    (15, [(12, 4, 3), (3, 3, 1)]),
    (22, [(16, 4, 4), (4, 4, 1), (2, 2, 1)]),
])
def test_multigpu_shards_are_divisible_and_exact(count, expected):
    value = plan()
    value["cases"] = [
        {"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
         "seed":1000 + i, "scenario":"clean"}
        for i in range(count)
    ]
    value["planned"] = count
    shards = shard_plan(value, gpu_count=4, per_gpu_concurrency=4)
    assert [(s["planned"], s["active_gpu_count"], s["per_gpu_envs"])
            for s in shards] == expected
    assert [case["case_id"] for shard in shards for case in shard["cases"]] == [
        case["case_id"] for case in value["cases"]]
    assert all(shard["planned"] == shard["active_gpu_count"] * shard["per_gpu_envs"]
               for shard in shards)


def test_runtime_parallelism_accepts_new_and_legacy_plans():
    value = plan()
    value["model"] = {"runtime":{"gpu_count":4, "per_gpu_concurrency":3,
                                  "eval_batch_size":12}}
    assert runtime_parallelism(value) == (4, 3)
    value["model"] = {"runtime":{"eval_batch_size":5}}
    assert runtime_parallelism(value) == (1, 5)
    value["model"] = {"runtime":{"gpu_count":4, "per_gpu_concurrency":3,
                                  "eval_batch_size":8}}
    with pytest.raises(ValueError, match="并发配置"):
        runtime_parallelism(value)


def test_runtime_overlay_silences_unused_optional_planner_tracebacks(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "envs/robot").mkdir(parents=True)
    (source / "envs/camera").mkdir(parents=True)
    original = "before\n"
    patched = "after\n"
    (source / "envs/_base_task.py").write_text(original)
    (source / "envs/robot/planner.py").write_text(
        "except Exception as e:\n    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')\n"
        "    print('Exception traceback:')\n    traceback.print_exc()\n    CuroboPlanner = None\n")
    (source / "envs/camera/camera.py").write_text(
        'except:\n    print("missing pytorch3d")\n\n    def fps(points, num_points=1024, use_cuda=True):\n'
        '        print("fps error: missing pytorch3d")\n        exit()\n')
    monkeypatch.setattr(robotwin_source, "ORIGINAL_BASE_TASK_SHA256",
                        hashlib.sha256(original.encode()).hexdigest())
    monkeypatch.setattr(robotwin_source, "REPLACEMENTS", ((original, patched),))
    monkeypatch.setattr(robotwin_source, "PATCHED_BASE_TASK_SHA256",
                        "sha256:" + hashlib.sha256(patched.encode()).hexdigest())

    target, _ = robotwin_source.prepare_robotwin_source(
        source, tmp_path / "output", robotwin_source.PATCH_SHA256)

    planner = (target / "envs/robot/planner.py").read_text()
    camera = (target / "envs/camera/camera.py").read_text()
    assert "traceback.print_exc()" not in planner
    assert "Something wrong happened" not in planner
    assert 'print("missing pytorch3d")' not in camera
    assert "RuntimeError" in camera


def test_shard_metrics_are_weighted_by_measured_trajectories():
    merged = merge_shard_metrics([
        {"num_trajectories":2, "success_once":0.5, "return":1.0, "episode_len":10.0},
        {"num_trajectories":1, "success_once":1.0, "return":4.0, "episode_len":20.0},
    ])
    assert merged == {"num_trajectories":3, "success_once":2 / 3,
                      "return":2.0, "episode_len":40 / 3}


def test_sharded_evaluation_merges_one_job_evidence(tmp_path, monkeypatch):
    value = plan()
    value["cases"] = [
        {"case_id":"task_a/clean/000000", "task_id":"task_a", "seed":1, "scenario":"clean"},
        {"case_id":"task_b/clean/000000", "task_id":"task_b", "seed":2, "scenario":"clean"},
    ]
    value["planned"] = 2
    value["suite_spec"]["tasks"] = [
        {"task_id":"task_a", "task_config_sha256":"sha256:" + "a" * 64,
         "max_episode_steps":400},
        {"task_id":"task_b", "task_config_sha256":"sha256:" + "b" * 64,
         "max_episode_steps":500},
    ]
    commands = []
    monkeypatch.setattr("sys.argv", ["rlinf_entry.py", "--script", "entry.py",
                                    "env.eval.total_num_envs=2"])

    def run(command, env, check):
        commands.append(command)
        child = json.loads(env["EB_EVAL_PLAN"])
        assert "cases" not in child["suite_spec"]
        out = __import__("pathlib").Path(env["EB_OUTPUT_DIR"])
        out.mkdir(parents=True)
        (out / "policy-receipt.json").write_text('{"status":"loaded"}')
        (out / "eval-episodes-rank0.jsonl").write_text(
            json.dumps({"case_id":child["cases"][0]["case_id"]}) + "\n")
        (out / "eval-manifest.json").write_text(json.dumps({
            "plan_hash":value["hash"],
            "metrics":{"num_trajectories":1, "success_once":1.0}}))

    monkeypatch.setattr("eb.runners.robotwin_eval.subprocess.run", run)
    result = evaluate_shards(value, tmp_path, shard_plan(
        value, gpu_count=1, per_gpu_concurrency=1))
    assert result == {"num_trajectories":2, "success_once":1.0}
    assert len((tmp_path / "eval-episodes-rank0.jsonl").read_text().splitlines()) == 2
    assert any("env.eval.task_config.task_name=task_b" in arg for arg in commands[1])
    assert not any(arg == "env.eval.total_num_envs=2" for arg in commands[0])


def test_multigpu_shards_replace_parent_placement(tmp_path, monkeypatch):
    value = plan()
    value["cases"] = [
        {"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
         "seed":1000 + i, "scenario":"clean"}
        for i in range(6)
    ]
    value["planned"] = 6
    commands = []
    monkeypatch.setattr("sys.argv", ["rlinf_entry.py", "--script", "entry.py",
        "cluster.component_placement={env\\, rollout:0-0}", "env.eval.total_num_envs=6"])

    def run(command, env, check):
        commands.append(command)
        child = json.loads(env["EB_EVAL_PLAN"])
        out = __import__("pathlib").Path(env["EB_OUTPUT_DIR"])
        out.mkdir(parents=True)
        (out / "policy-receipt.json").write_text('{"status":"loaded"}')
        (out / "eval-episodes-rank0.jsonl").write_text("".join(
            json.dumps({"case_id":case["case_id"]}) + "\n" for case in child["cases"]))
        (out / "eval-manifest.json").write_text(json.dumps({
            "plan_hash":value["hash"],
            "metrics":{"num_trajectories":child["planned"], "success_once":1.0}}))

    monkeypatch.setattr("eb.runners.robotwin_eval.subprocess.run", run)
    result = evaluate_shards(value, tmp_path, shard_plan(
        value, gpu_count=4, per_gpu_concurrency=4))
    assert result == {"num_trajectories":6, "success_once":1.0}
    assert "cluster.component_placement={env\\, rollout:0-3}" in commands[0]
    assert "env.eval.total_num_envs=4" in commands[0]
    assert "cluster.component_placement={env\\, rollout:0-1}" in commands[1]
    assert "env.eval.total_num_envs=2" in commands[1]
    assert all("cluster.component_placement={env\\, rollout:0-0}" not in command
               for command in commands)
