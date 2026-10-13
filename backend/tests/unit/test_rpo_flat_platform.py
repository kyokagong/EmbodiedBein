import copy
import hashlib
import json

import pytest

from eb.rpo_flat import FLAT_SUITE, flat_suite_spec, validate_flat_eval
from eb.runners.rpo_flat_artifact import summarize_cases


def test_hd_argument_preserves_old_frozen_runtime_commands(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eb.adapters.rlinf import RlinfAdapter
    from eb.domain.job import JobKind, JobSpec
    from eb.rpo_flat import FLAT_ENV, FLAT_TASK

    assets = tmp_path / "roboparty/rpo-v2"
    assets.mkdir(parents=True)
    (assets / "asset-manifest.json").write_text("{}")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setattr("eb.adapters.rlinf.get_settings", lambda: SimpleNamespace(
        assets_root=tmp_path, data_root=tmp_path))
    plan = {"schema_version":"eb/rpo-flat-eval-plan/v1", "suite_spec":flat_suite_spec(),
        "suite_id":FLAT_SUITE, "backend":"isaac", "profile":"smoke",
        "seeds":[60000, 60001, 60002], "hash":"sha256:frozen"}
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id=FLAT_ENV, task_id=FLAT_TASK,
        inputs={"base_model":str(weights)}, params={"template_id":"rpo_v2_flat_eval",
            "seeds":plan["seeds"], "rpo_eval_plan":plan})
    assert "--video-quality" not in RlinfAdapter().render(spec, "run_old").command
    spec.params.update(record_video=True, video_quality="hd")
    spec.params["rpo_eval_plan"]["video_quality"] = "hd"
    argv = RlinfAdapter().render(spec, "run_hd").command
    assert argv[argv.index("--video-quality") + 1] == "hd"
    spec.params["video_quality"] = "standard"
    with pytest.raises(ValueError, match="冻结"):
        RlinfAdapter().render(spec, "run_mismatch")


def test_flat_acceptance_requires_80_real_cases_and_unchanged_evidence(tmp_path):
    seeds = list(range(50000, 50100))
    plan = {"schema_version":"eb/rpo-flat-eval-plan/v1", "suite_id":FLAT_SUITE,
            "suite_spec":flat_suite_spec(), "seeds":seeds, "profile":"standard",
            "seed_override":False, "record_video":False, "checkpoint_step":500}
    plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def evidence(relative):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen evidence")
        return {"path":relative, "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

    onnx = {**evidence("exported/policy.onnx"), "path":"policy.onnx", "obs":780,
            "act":23, "max_abs_error":0.0, "error_atol":1e-4, "error_rtol":1e-5,
            "max_error_ratio":0.0, "max_clipped_action_error":0.0}
    exported = {"schema_version":"eb/rpo-flat-export/v2", "status":"verified", "onnx":onnx,
                "golden":{**evidence("exported/golden-traces.npz"), "path":"golden-traces.npz"},
                "normalization":{**evidence("exported/actor-normalization.npz"), "path":"actor-normalization.npz"}}
    (tmp_path / "exported/export-manifest.json").write_text(json.dumps(exported))
    rows = []
    for index, seed in enumerate(seeds):
        rows.append({"run_id":"run_flat", "plan_hash":plan["hash"], "seed":seed,
            "case_id":f"rpo-flat-seed-{seed}", "task_id":"tsk_rpo_flat",
            "completed":True, "actual_steps":1000, "episode_len":1000, "metric_steps":1000,
            "fall":False, "nan":False, "hard_limit":False, "velocity_rmse":0.20,
            "yaw_rmse":0.30 if index < 80 else 0.31, "success":index < 80,
            "status":"success" if index < 80 else "task_failure",
            "trace":{**evidence(f"traces/seed_{seed}.jsonl.gz"), "policy_steps":1000, "metric_steps":1000},
            "initial_state":evidence(f"traces/seed_{seed}.initial.json")})
    receipt = {"schema_version":"eb/rpo-flat-eval/v1", "run_id":"run_flat",
        "plan_hash":plan["hash"], "seeds":seeds, "backend":"isaac",
        "criterion":"velocity_commands", "required_steps":1000,
        "official_revision":flat_suite_spec()["robolab_revision"], "record_video":False,
        "velocity_frame":"yaw_aligned", "yaw_rate_frame":"world_z", "onnx":onnx}

    def publish(step=500):
        plan["checkpoint_step"] = step
        plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
            {k:v for k,v in plan.items() if k != "hash"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt["plan_hash"] = plan["hash"]
        for row in rows:
            row["plan_hash"] = plan["hash"]
        (tmp_path / "eval-cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (tmp_path / "eval-manifest.json").write_text(json.dumps({**receipt, **summarize_cases(rows, 100)}))

    publish(499)
    assert validate_flat_eval(tmp_path, plan, "run_flat")["qualification"] == "fail"
    publish()
    assert validate_flat_eval(tmp_path, plan, "run_flat")["qualification"] == "pass"
    manifest = tmp_path / "eval-manifest.json"
    original = manifest.read_text()
    large = json.loads(original)
    large["physical_case_settings"] = [{"settings": "x" * 25000} for _ in seeds]
    manifest.write_text(json.dumps(large))
    assert manifest.stat().st_size > 1024 * 1024
    assert validate_flat_eval(tmp_path, plan, "run_flat")["qualification"] == "pass"
    large["physical_case_settings"] = "x" * (7 * 1024 * 1024)
    manifest.write_text(json.dumps(large))
    with pytest.raises(ValueError, match="evidence unavailable or unsafe"):
        validate_flat_eval(tmp_path, plan, "run_flat")
    oversized_plan = {**plan, "seeds": list(range(50000, 51000))}
    oversized_plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        {k: v for k, v in oversized_plan.items() if k != "hash"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="evidence unavailable or unsafe"):
        validate_flat_eval(tmp_path, oversized_plan, "run_flat")
    manifest.write_text(original)
    plan["video_quality"] = "hd"
    publish()
    with pytest.raises(ValueError, match="receipt differs"):
        validate_flat_eval(tmp_path, plan, "run_flat")
    del plan["video_quality"]
    publish()
    exported["schema_version"] = "eb/rpo-flat-export/v3"
    exported["golden"]["batch_sizes"] = [1, 11]
    onnx["max_joint_target_error_rad"] = 0.0
    (tmp_path / "exported/export-manifest.json").write_text(json.dumps(exported))
    publish()
    assert validate_flat_eval(tmp_path, plan, "run_flat")["qualification"] == "pass"
    onnx["max_joint_target_error_rad"] = 0.001
    (tmp_path / "exported/export-manifest.json").write_text(json.dumps(exported))
    publish()
    with pytest.raises(ValueError, match="policy export changed"):
        validate_flat_eval(tmp_path, plan, "run_flat")
    onnx["max_joint_target_error_rad"] = 0.0
    (tmp_path / "exported/export-manifest.json").write_text(json.dumps(exported))
    rows[0].update(yaw_rmse=0.31, success=False, status="task_failure")
    publish()
    assert validate_flat_eval(tmp_path, plan, "run_flat")["qualification"] == "fail"
    rows[0]["success"] = True
    with pytest.raises(ValueError, match="success"):
        publish()
    rows[0]["success"] = False
    publish()
    saved = dict(rows[1])
    rows[1].update(completed=False, success=False, status="infra_failure",
                   actual_steps=0, episode_len=0, metric_steps=0)
    publish()
    assert validate_flat_eval(tmp_path, plan, "run_flat")["qualification"] == "incomplete"
    rows[1] = saved
    publish()
    (tmp_path / "traces/seed_50000.jsonl.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_flat_eval(tmp_path, plan, "run_flat")
    changed = copy.deepcopy(plan)
    changed["suite_spec"]["yaw_rmse_max"] = 2.0
    with pytest.raises(ValueError, match="plan changed"):
        validate_flat_eval(tmp_path, changed, "run_flat")
