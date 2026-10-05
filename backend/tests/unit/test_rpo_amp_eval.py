import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

import pytest

from eb.orchestrator.collect import claim_outputs, validate_amp_case_evidence
from eb.api.routes_eval import RpoEvalIn, _prepare_rpo


def test_mujoco_policy_joint_order_matches_the_managed_rpo_mjcf():
    from eb.rpo_wbc import JOINT_ORDER
    from eb.runners.rpo_wbc_artifact import POLICY_TO_URDF

    path = Path("/data/kk/assets/roboparty/rpo-v2/rpo_description/mjcf/rpo.xml")
    if not path.is_file():
        pytest.skip("managed RPO MJCF is unavailable")
    xml = ET.parse(path)
    joints = [node.attrib["name"] for node in xml.findall(".//worldbody//joint")]
    assert tuple(joints[-23:]) == JOINT_ORDER
    np.testing.assert_array_equal(POLICY_TO_URDF, np.arange(23))


def test_mujoco_pd_effort_limits_match_the_isaac_rpo_actuators():
    from eb.runners.rpo_wbc_artifact import EFFORT_LIMIT, pd_torque

    np.testing.assert_array_equal(EFFORT_LIMIT, [
        120, 120, 120, 120, 27, 27,
        120, 120, 120, 120, 27, 27,
        120, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    ])
    torque = pd_torque(np.zeros(23), np.zeros(23), np.ones(23))
    assert torque[0] == 100
    assert torque[4] == 27
    assert torque[13] == 27


def test_mujoco_control_period_respects_the_asset_timestep():
    from eb.runners.rpo_wbc_artifact import control_decimation

    assert control_decimation(0.001) == 20
    assert control_decimation(0.005) == 4


def test_isaac_eval_records_actual_random_actuator_delays():
    from eb.runners.rpo_wbc_artifact import actuator_delay_steps

    robot = SimpleNamespace(actuators={
        "legs": SimpleNamespace(positions_delay_buffer=SimpleNamespace(time_lags=np.array([2]))),
        "feet": SimpleNamespace(positions_delay_buffer=SimpleNamespace(time_lags=np.array([0]))),
    })
    assert actuator_delay_steps(robot) == {"legs": 2, "feet": 0}


def test_amp_case_evidence_recomputes_success_and_summary(tmp_path):
    rows = [
        {"seed": seed, "case_id": f"rpo-seed-{seed}",
         "task_id": "tsk_rpo_amp_stand_walk", "scenario": "stand" if seed == 0 else "walk",
         "completed": True, "episode_len": 1000, "success": True, "status": "success",
         "fall": False, "nan": False, "hard_limit": False,
         "distance_m": 0.1 if seed == 0 else 5.0, "velocity_rmse": 0.1}
        for seed in (0, 1)
    ]
    receipt = {"cases": {"planned": 2}, "environment_settings": {},
               "metrics": {"n_trials": 2, "successes": 2, "success_rate": 1.0,
                           "fall_count": 0, "nan_count": 0, "hard_limit_count": 0,
                           "velocity_rmse": 0.1}}
    validate_amp_case_evidence(rows, [0, 1], receipt, tmp_path, False)

    changed = copy.deepcopy(rows)
    changed[1]["distance_m"] = 0.1
    with pytest.raises(ValueError, match="success"):
        validate_amp_case_evidence(changed, [0, 1], receipt, tmp_path, False)
    changed = copy.deepcopy(rows)
    changed[1]["scenario"] = "stand"
    with pytest.raises(ValueError, match="malformed"):
        validate_amp_case_evidence(changed, [0, 1], receipt, tmp_path, False)
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["metrics"]["successes"] = 1
    with pytest.raises(ValueError, match="summary"):
        validate_amp_case_evidence(rows, [0, 1], changed_receipt, tmp_path, False)

    for row in rows:
        row["video_rel"] = f"seed_{row['seed']}/0.mp4"
    with pytest.raises(ValueError, match="video"):
        validate_amp_case_evidence(rows, [0, 1], receipt, tmp_path, True)
    for row in rows:
        path = tmp_path / "video/eval" / row["video_rel"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
    validate_amp_case_evidence(rows, [0, 1], receipt, tmp_path, True)


@pytest.mark.asyncio
async def test_amp_eval_collector_keeps_one_diagnostic_report_per_run(tmp_path, monkeypatch):
    monkeypatch.setattr("eb.orchestrator.collect.get_settings",
                        lambda: SimpleNamespace(data_root=tmp_path))
    output = tmp_path / "runs/run_amp/output"
    (output / "exported").mkdir(parents=True)
    onnx = output / "exported/policy.onnx"
    onnx.write_bytes(b"measured-policy")
    rows = [{"seed": seed, "case_id": f"rpo-seed-{seed}", "task_id": "tsk_rpo_amp_stand_walk",
             "scenario": "stand" if seed == 0 else "walk", "completed": True,
             "episode_len": 1000, "success": True, "status": "success", "fall": False,
             "nan": False, "hard_limit": False, "distance_m": 0.1 if seed == 0 else 5.0,
             "velocity_rmse": 0.1} for seed in (0, 1)]
    (output / "eval-cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {"schema_version": "eb/rpo-amp-eval/v1", "backend": "mujoco",
               "criterion": "stand_walk", "required_steps": 1000, "record_video": False,
               "environment_settings": {},
               "onnx": {"sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
                        "obs": 780, "act": 23, "max_abs_error": 0.0},
               "cases": {"planned": 2, "completed": 2, "valid": 2, "invalid": 0, "infra_failed": 0},
               "metrics": {"n_trials": 2, "successes": 2, "success_rate": 1.0,
                           "fall_count": 0, "hard_limit_count": 0, "nan_count": 0,
                           "upper_rmse": None, "velocity_rmse": 0.1}}
    (output / "eval-manifest.json").write_text(json.dumps(receipt))
    plan = {"schema_version": "eb/rpo-amp-eval-plan/v1", "backend": "mujoco",
            "seeds": [0, 1], "record_video": False, "profile": "smoke"}
    job = SimpleNamespace(kind="eval.sim2sim", embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                          spec={"params": {"suite_id": "suite_rpo_v2_amp_mujoco_stand_walk_v1",
                                           "model_version_id": "mv_amp", "rpo_eval_plan": plan}})
    run = SimpleNamespace(id="run_amp", metrics_summary={})

    class Session:
        def __init__(self):
            self.added = []
            self.existing = None

        async def scalar(self, query):
            return self.existing

        def add(self, row):
            self.added.append(row)

    session = Session()
    await claim_outputs(session, job, run)
    assert len(session.added) == 2
    report = session.added[0]
    assert report.model_version_id == "mv_amp"
    assert report.gate_results["qualification"] == "not_requested"
    session.existing = report
    await claim_outputs(session, job, run)
    assert len(session.added) == 2
    assert run.metrics_summary["eval_report_id"] == report.id


@pytest.mark.asyncio
async def test_amp_eval_plan_requires_unchanged_frozen_suite(tmp_path, monkeypatch):
    manifest = {"step": 10, "files": []}
    digest = "sha256:" + hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint = {"path": "/managed/amp-checkpoint", "step": 10, "manifest": manifest}
    monkeypatch.setattr("eb.checkpoints.checkpoint_model", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr("eb.api.routes_eval.get_settings", lambda: SimpleNamespace(data_root=tmp_path))

    async def image_info(self, image):
        return {"ok": True, "digest": "sha256:" + "a" * 64}

    monkeypatch.setattr("eb.store.ebd.EbdClient.image_info", image_info)
    suite_spec = {"benchmark":"rpo_amp", "backend":"mujoco", "policy":"deterministic_onnx",
                  "criterion":"stand_walk", "qualification":"diagnostic", "required_steps":1000,
                  "cases":"even_seed_stand_odd_seed_walk", "stand_max_drift_m":0.5,
                  "walk_min_forward_m":4.0, "fall_height_m":0.5, "fall_tilt_rad":0.8}
    suite = SimpleNamespace(id="suite_rpo_v2_amp_mujoco_stand_walk_v1", spec=suite_spec,
                            frozen=True, embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1")
    model = SimpleNamespace(id="mv_amp", artifact_id="art_amp", name="amp",
                            embodiment_env_id="ee_rpo_v2_amp_locomotion_23dof_v1",
                            card={"schema_version":"eb/rpo-amp-model/v1", "run_id":"run_train",
                                  "step":10, "template_id":"rpo_v2_amp_stand_walk_ppo"})
    artifact = SimpleNamespace(id="art_amp", kind="rlinf_checkpoint",
                               storage=checkpoint["path"], sha256=digest)
    run = SimpleNamespace(id="run_train", status="finished")

    class Session:
        async def get(self, kind, identity):
            return {"ModelVersion":model, "Artifact":artifact, "Run":run,
                    "EvalSuite":suite}.get(kind.__name__)

    body = RpoEvalIn(model_version_id="mv_amp", backend="mujoco")
    spec, plan = await _prepare_rpo(body, Session())
    assert spec.embodiment_env_id == model.embodiment_env_id
    assert plan["suite_spec"] == suite_spec
    suite.spec = {**suite_spec, "walk_min_forward_m":0.1}
    with pytest.raises(ValueError, match="未冻结"):
        await _prepare_rpo(body, Session())
