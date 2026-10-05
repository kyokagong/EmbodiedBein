import math
import json
from types import SimpleNamespace

import pytest

from eb.orchestrator.collect import claim_outputs, libero_eval_metrics
from eb.api.routes_eval import build_eval_plan
from eb.runners.robotwin_eval import (build_case_record, seal_policy_evidence,
                                      validated_completed_case_ids)
from eb.runners.robotwin_source import PATCH_SHA256, PATCHED_BASE_TASK_SHA256


def test_libero_metrics_use_completed_trajectory_count_and_success_once():
    raw = {"success_once": 0.75, "success_at_end": 0.25, "num_trajectories": 4, "return": 1.5}
    receipt = {"run_id": "run_eval", "status": "not_qualified", "metrics": raw}
    result = libero_eval_metrics(receipt, "run_eval")
    assert result["success_rate"] == 0.75 and result["n_trials"] == 4
    assert result["success_at_end"] == 0.25
    assert "success_rate" not in raw and "n_trials" not in raw
    for invalid in ({}, {"success_once": math.nan, "num_trajectories": 4},
                    {"success_once": 0.0, "num_trajectories": 0},
                    {"success_once": 1.1, "num_trajectories": 4},
                    {"success_at_end": 0.5, "num_trajectories": 4}):
        with pytest.raises(ValueError):
            libero_eval_metrics({**receipt, "metrics": invalid}, "run_eval")
    with pytest.raises(ValueError):
        libero_eval_metrics({**receipt, "status": "qualified"}, "run_eval")


@pytest.mark.asyncio
async def test_collect_preserves_raw_evidence_and_rejects_empty_evaluation(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr("eb.orchestrator.collect.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    output = tmp_path / "runs/run_eval/output"
    output.mkdir(parents=True)
    manifest = output / "eval-manifest.json"
    receipt = {"run_id": "run_eval", "status": "not_qualified", "metrics": {}}
    manifest.write_text(json.dumps(receipt))
    reports = []
    session = SimpleNamespace(add=reports.append)
    job = SimpleNamespace(kind="eval.sim", spec={"task_id": "tsk_libero_spatial",
                          "params": {"template_id": "libero_spatial_pi05_eval",
                                     "eval_checkpoint_run_id": "run_source", "eval_checkpoint_step": 5800},
                          "inputs": {"base_model": "/managed/ppo/model"}})
    run = SimpleNamespace(id="run_eval", metrics_summary={})
    with pytest.raises(ValueError, match="metrics"):
        await claim_outputs(session, job, run)
    assert not reports
    receipt["metrics"] = {"success_once": 0.5, "num_trajectories": 4}
    manifest.write_text(json.dumps(receipt))
    await claim_outputs(session, job, run)
    assert run.metrics_summary["n_trials"] == 4
    assert run.metrics_summary["qualification"] == "not_qualified"
    assert reports[0].gate_results["receipt"] == receipt
    assert reports[0].gate_results["model_input"] == "/managed/ppo/model"
    assert reports[0].gate_results["checkpoint"] == {"run_id": "run_source", "step": 5800}


@pytest.mark.asyncio
async def test_collect_robotwin_evaluation_binds_model_version(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr("eb.orchestrator.collect.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    suite = {"benchmark":"robotwin", "suite":"robotwin_adjust_bottle_clean_v1",
             "robotwin_revision":"abc123", "rlinf_revision":"def456",
             "assets_manifest_sha256":"sha256:" + "a" * 64,
             "controller_profile":"aloha_joint14_position_v1",
             "observation_profile":"robotwin_aloha_3cam_joint14_v1", "domain":"clean",
             "max_episode_steps":200,
             "tasks":[{"task_id":"adjust_bottle", "task_config_sha256":"sha256:" + "d" * 64}],
             "cases":[{"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
                       "seed":100 + i, "scenario":"clean"} for i in range(5)]}
    model = {"model_version_id":"mv_pi05", "artifact_sha256":"sha256:" + "b" * 64,
             "runtime":{"robotwin_content_hash":"sha256:" + "c" * 64,
                        "robotwin_evidence_patch_sha256":PATCH_SHA256,
                        "patched_base_task_sha256":PATCHED_BASE_TASK_SHA256}}
    plan = build_eval_plan("suite_robotwin_adjust_bottle_clean_v1", suite, model, "smoke")
    output = tmp_path / "runs/run_eval/output"
    output.mkdir(parents=True)
    rows = [build_case_record(plan, case["seed"], i == 0, 10 if i == 0 else 200,
                              50, 10 if i == 0 else 50, 10 if i == 0 else 200,
                              1.0, "adjust the bottle",
                              "run_eval") for i, case in enumerate(plan["cases"])]
    (output / "eval-episodes-rank0.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    seal_policy_evidence(output, {"schema_version":"eb/robotwin-policy-receipt/v1",
        "status":"loaded", "model_version_id":"mv_pi05", "artifact_sha256":model["artifact_sha256"],
        "robotwin_content_hash":model["runtime"]["robotwin_content_hash"],
        "evidence_patch_sha256":PATCH_SHA256,
        "patched_base_task_sha256":PATCHED_BASE_TASK_SHA256,
        "policy_worker_count":1, "model_load_count":1, "headless":True,
        "offscreen_rendering":True, "save_video":False, "inference_micro_batch_size":4})
    (output / "eval-manifest.json").write_text(json.dumps({"run_id":"run_eval",
        "status":"not_qualified", "plan_hash":plan["hash"],
        "metrics":{"success_once":0.2, "num_trajectories":5}}))

    class Session:
        reports = []
        model = SimpleNamespace(card={"readiness":"adapter_ready"})
        async def scalar(self, _query): return None
        async def get(self, kind, identity):
            return self.model if kind.__name__ == "ModelVersion" and identity == "mv_pi05" else None
        def add(self, row): self.reports.append(row)
    session = Session()
    job = SimpleNamespace(id="job_eval", kind="eval.sim", spec={"image":"sha256:x",
        "inputs":{"base_model":"/managed/model"}, "overrides":[], "params":{
            "template_id":"robotwin_pi05_eval", "suite_id":plan["suite_id"], "eval_plan":plan}})
    run = SimpleNamespace(id="run_eval", metrics_summary={})
    await claim_outputs(session, job, run)
    report = session.reports[0]
    assert report.model_version_id == "mv_pi05"
    assert report.metrics["success_rate"] == 0.2
    assert report.metrics["timeouts"] == 4
    assert session.model.card["readiness"] == "smoke_passed"
    assert session.model.card["closed_loop_evidence"]["report_id"] == report.id


def test_resume_only_skips_valid_completed_cases(tmp_path):
    value = build_eval_plan("suite_suite", {
        "benchmark":"robotwin", "suite":"suite", "robotwin_revision":"r",
        "rlinf_revision":"q", "assets_manifest_sha256":"sha256:" + "a" * 64,
        "controller_profile":"a", "observation_profile":"o", "domain":"clean",
        "max_episode_steps":10, "tasks":[{"task_id":"t",
            "task_config_sha256":"sha256:" + "b" * 64}],
        "cases":[{"case_id":"t/clean/000000", "task_id":"t", "seed":1, "scenario":"clean"},
                 {"case_id":"t/clean/000001", "task_id":"t", "seed":2, "scenario":"clean"}],
    }, {"model_version_id":"m", "runtime":{}}, "smoke")
    rows = [{**case, "plan_hash":value["hash"], "task_config_sha256":"sha256:" + "b" * 64,
             "policy_receipt_sha256":"sha256:receipt"} for case in value["cases"]]
    rows[1]["plan_hash"] = "sha256:wrong"
    (tmp_path / "eval-episodes-rank0.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")

    assert validated_completed_case_ids(tmp_path, value, "sha256:receipt") == {
        "t/clean/000000"}
