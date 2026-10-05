import pytest
import asyncio
import json
from types import SimpleNamespace

from eb.api.routes_eval import EvalIn, build_eval_plan
from eb.orchestrator.collect import archive_partial_eval, validate_eval_episodes, validate_libero_episodes
from eb.runners.robotwin_eval import build_case_record


def test_libero_plan_is_frozen_and_distinguishes_profiles():
    suite = {"suite": "libero_spatial", "scope": "simulation_only"}
    source = {"run_id": "run_source", "step": 5800, "bytes": 7, "base_model_id": "base"}
    smoke = build_eval_plan("suite_tsk_libero_spatial", suite, source, "smoke")
    standard = build_eval_plan("suite_tsk_libero_spatial", suite, source, "standard")
    assert smoke["planned"] == 4 and standard["planned"] == 100
    assert len(smoke["cases"]) == 4 and len(standard["cases"]) == 100
    assert len({c["case_id"] for c in standard["cases"]}) == 100
    assert smoke["hash"] != standard["hash"]
    assert build_eval_plan("suite_tsk_libero_spatial", suite, source, "standard")["hash"] == standard["hash"]
    with pytest.raises(ValueError):
        build_eval_plan("suite_tsk_libero_spatial", suite, source, "arbitrary")


def test_plan_rejects_suite_mismatch():
    with pytest.raises(ValueError):
        build_eval_plan("suite_tsk_libero_spatial", {"suite": "libero_goal"},
                        {"run_id": "run_1", "step": 2, "bytes": 7}, "standard")


def robotwin_suite():
    return {
        "benchmark": "robotwin",
        "protocol": "platform_fixed",
        "suite": "robotwin_adjust_bottle_clean_v1",
        "robotwin_revision": "abc123",
        "rlinf_revision": "def456",
        "assets_manifest_sha256": "sha256:" + "a" * 64,
        "controller_profile": "aloha_joint14_position_v1",
        "observation_profile": "robotwin_aloha_3cam_joint14_v1",
        "domain": "clean",
        "max_episode_steps": 200,
        "tasks": [{"task_id":"adjust_bottle", "task_config_sha256":"sha256:" + "c" * 64}],
        "cases": [
            {"case_id": f"adjust_bottle/clean/{index:06d}", "task_id": "adjust_bottle",
             "seed": 1000 + index, "scenario": "clean"}
            for index in range(6)
        ],
    }


def test_robotwin_plan_uses_frozen_unique_cases_and_five_case_smoke():
    source = {"model_version_id": "mv_pi05", "artifact_sha256": "sha256:" + "b" * 64}
    smoke = build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(), source, "smoke")
    standard = build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(), source, "standard")
    assert smoke["schema_version"] == "eb/robotwin-eval-plan/v1"
    assert smoke["planned"] == 5 and standard["planned"] == 6
    assert [case["seed"] for case in smoke["cases"]] == [1000, 1001, 1002, 1003, 1004]
    assert smoke["model"]["model_version_id"] == "mv_pi05"
    assert build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(), source, "smoke")["hash"] == smoke["hash"]


def test_robotwin_parallelism_defaults_and_bounds():
    body = {"name":"eval", "suite_id":"suite_1", "model_version_id":"mv_1"}
    assert EvalIn(**body).gpus == 1
    assert EvalIn(**body).parallel_envs == 4
    assert EvalIn(**body, gpus=4, parallel_envs=16).gpus == 4
    for invalid in (0, 3, 8):
        with pytest.raises(ValueError):
            EvalIn(**body, gpus=invalid)
    for invalid in (0, 21):
        with pytest.raises(ValueError):
            EvalIn(**body, parallel_envs=invalid)


def test_robotwin_runtime_profile_request_rejects_raw_container_resources():
    body = {"name":"eval", "suite_id":"suite_1", "model_version_id":"mv_1",
            "runtime_profile_id":"memory_saver_sync",
            "resource_variant_id":"sync_4gpu_16env_80g"}
    parsed = EvalIn(**body)
    assert parsed.runtime_profile_id == "memory_saver_sync"
    assert parsed.resource_variant_id == "sync_4gpu_16env_80g"
    with pytest.raises(ValueError):
        EvalIn(**body, memory_gb=120)


def test_robotwin_parallelism_is_frozen_into_plan_hash():
    base = {"model_version_id":"mv_pi05", "runtime":{
        "gpu_count":1, "policy_worker_count":1, "env_worker_count":1,
        "envs_per_worker":4, "parallel_envs":4, "inference_micro_batch_size":4}}
    multi = {"model_version_id":"mv_pi05", "runtime":{
        "gpu_count":4, "policy_worker_count":1, "env_worker_count":4,
        "envs_per_worker":4, "parallel_envs":16, "inference_micro_batch_size":4}}
    single_plan = build_eval_plan(
        "suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(), base, "standard")
    multi_plan = build_eval_plan(
        "suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(), multi, "standard")
    assert single_plan["hash"] != multi_plan["hash"]
    runtime = multi_plan["model"]["runtime"]
    assert runtime["parallel_envs"] == runtime["env_worker_count"] * runtime["envs_per_worker"]
    assert runtime["policy_worker_count"] == 1


def test_robotwin_plan_rejects_duplicate_or_unfrozen_cases():
    suite = robotwin_suite()
    suite["cases"][1]["seed"] = suite["cases"][0]["seed"]
    with pytest.raises(ValueError, match="重复"):
        build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", suite,
                        {"model_version_id": "mv_pi05"}, "standard")


def test_full_robotwin_smoke_spans_five_tasks():
    suite = robotwin_suite()
    suite["suite"] = "robotwin_rlinf_full_clean_v1"
    suite["cases"] = [{"case_id":f"task_{i}/clean/000000", "task_id":f"task_{i}",
                       "seed":1000 + i, "scenario":"clean"} for i in range(6)]
    plan = build_eval_plan("suite_robotwin_rlinf_full_clean_v1", suite,
                           {"model_version_id":"mv_pi05"}, "smoke")
    assert plan["planned"] == 5
    assert len({case["task_id"] for case in plan["cases"]}) == 5
    suite = robotwin_suite()
    del suite["robotwin_revision"]
    with pytest.raises(ValueError, match="robotwin_revision"):
        build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", suite,
                        {"model_version_id": "mv_pi05"}, "standard")


def test_robotwin_episode_receipt_counts_timeout_as_model_failure():
    plan = build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(),
                           {"model_version_id": "mv_pi05"}, "smoke")
    statuses = ["success", "success", "task_failure", "timeout", "success"]
    episodes = [build_case_record(plan, case["seed"], status == "success",
                200 if status == "timeout" else 10, 50, 50 if status == "timeout" else 10,
                200 if status == "timeout" else 10, 1.0, "adjust the bottle", "run_eval")
                for case, status in zip(plan["cases"], statuses, strict=True)]
    result = validate_eval_episodes(plan, episodes, {"n_trials": 5, "success_rate": 0.6}, "run_eval")
    assert result["successes"] == 3 and result["task_failures"] == 1 and result["timeouts"] == 1
    assert result["valid"] == 5 and result["success_rate"] == 0.6
    episodes[0]["seed_actual"] = 999
    with pytest.raises(ValueError, match="计划"):
        validate_eval_episodes(plan, episodes, {"n_trials": 5, "success_rate": 0.6}, "run_eval")


def test_episode_receipt_requires_exact_coverage_and_measured_success():
    plan = build_eval_plan("suite_tsk_libero_spatial", {"suite": "libero_spatial"},
                           {"run_id": "run_1", "step": 2, "bytes": 7}, "smoke")
    episodes = [{**c, "success": index < 3, "status": "success" if index < 3 else "task_failure"}
                for index, c in enumerate(plan["cases"])]
    result = validate_libero_episodes(plan, episodes, {"n_trials": 4, "success_rate": 0.75})
    assert result["completed"] == 4 and result["successes"] == 3
    assert result["per_task"]["3"]["successes"] == 0
    with pytest.raises(ValueError):
        validate_libero_episodes(plan, episodes[:-1], {"n_trials": 4, "success_rate": 0.75})
    with pytest.raises(ValueError):
        validate_libero_episodes(plan, [*episodes[:-1], episodes[0]], {"n_trials": 4, "success_rate": 0.75})
    with pytest.raises(ValueError):
        validate_libero_episodes(plan, episodes, {"n_trials": 4, "success_rate": 1.0})


def test_aborted_eval_keeps_measured_cases_without_promoting_partial_result(tmp_path, monkeypatch):
    plan = build_eval_plan("suite_tsk_libero_spatial", {"suite": "libero_spatial"},
                           {"run_id": "run_source", "step": 2, "sha256": "a" * 64}, "smoke")
    output = tmp_path / "runs/run_eval/output"
    output.mkdir(parents=True)
    (output / "eval-episodes-rank0.jsonl").write_text(json.dumps({**plan["cases"][0],
        "run_id": "run_eval", "plan_hash": plan["hash"],
        "success": True, "status": "success"}) + "\n")
    monkeypatch.setattr("eb.orchestrator.collect.get_settings", lambda: SimpleNamespace(data_root=tmp_path))

    class Session:
        reports = []
        async def scalar(self, _query): return None
        def add(self, report): self.reports.append(report)

    session = Session()
    job = SimpleNamespace(id="job_eval", kind="eval.sim", spec={"params": {"eval_plan": plan}}, error="exit_1")
    run = SimpleNamespace(id="run_eval", abort_reason="exit_1", metrics_summary={})
    asyncio.run(archive_partial_eval(session, job, run))
    report = session.reports[0]
    assert report.gate_results["validity"] == "incomplete"
    assert report.metrics["completed"] == 1 and report.metrics["not_started"] == 3
    assert report.metrics["success_rate"] == 1.0
    assert run.metrics_summary["eval_report_id"] == report.id


def test_aborted_robotwin_eval_keeps_valid_measured_cases(tmp_path, monkeypatch):
    plan = build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(),
                           {"model_version_id":"mv_pi05"}, "smoke")
    case = plan["cases"][0]
    output = tmp_path / "runs/run_eval/output"
    output.mkdir(parents=True)
    row = build_case_record(plan, case["seed"], False, 200, 50, 50, 200, 8.0,
                            "adjust the bottle", "run_eval")
    (output / "eval-episodes-rank0.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr("eb.orchestrator.collect.get_settings", lambda: SimpleNamespace(data_root=tmp_path))

    class Session:
        reports = []
        async def scalar(self, _query): return None
        def add(self, report): self.reports.append(report)
    session = Session()
    job = SimpleNamespace(id="job_eval", kind="eval.sim", spec={"params":{"eval_plan":plan}}, error="exit_1")
    run = SimpleNamespace(id="run_eval", abort_reason="exit_1", metrics_summary={})
    asyncio.run(archive_partial_eval(session, job, run))
    report = session.reports[0]
    assert report.model_version_id == "mv_pi05"
    assert report.metrics["completed"] == 1 and report.metrics["timeouts"] == 1


def test_aborted_robotwin_eval_rejects_inexact_partial_evidence(tmp_path, monkeypatch):
    plan = build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", robotwin_suite(),
                           {"model_version_id":"mv_pi05"}, "smoke")
    case = plan["cases"][0]
    row = build_case_record(plan, case["seed"], False, 200, 50, 50, 200, 8.0,
                            "adjust the bottle", "run_eval")
    del row["sim_steps"]
    output = tmp_path / "runs/run_eval/output"
    output.mkdir(parents=True)
    (output / "eval-episodes-rank0.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr("eb.orchestrator.collect.get_settings", lambda: SimpleNamespace(data_root=tmp_path))

    class Session:
        reports = []
        async def scalar(self, _query): return None
        def add(self, report): self.reports.append(report)
    session = Session()
    job = SimpleNamespace(id="job_eval", kind="eval.sim", spec={"params":{"eval_plan":plan}}, error="exit_1")
    run = SimpleNamespace(id="run_eval", abort_reason="exit_1", metrics_summary={})
    asyncio.run(archive_partial_eval(session, job, run))
    report = session.reports[0]
    assert report.gate_results["validity"] == "invalid"
    assert report.metrics["completed"] == 0
