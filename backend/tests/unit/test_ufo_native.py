import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_native_bundle_cannot_use_legacy_builder_or_signing():
    from fastapi import HTTPException

    from eb.api.routes import build_bundle, sign_bundle
    from eb.ufo_native import ENV_ID

    session = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(
                embodiment_env_id=ENV_ID, manifest={"gates_passed": ["G4", "sim2sim"]}, signed=False
            )
        ),
        commit=AsyncMock(),
    )
    with pytest.raises(HTTPException) as built:
        await build_bundle({"embodiment_env_id": ENV_ID}, session, None)
    assert built.value.status_code == 403
    with pytest.raises(HTTPException) as signed:
        await sign_bundle("bnd_example", session, None, SimpleNamespace(headers={"x-confirm": "bnd_example"}))
    assert signed.value.status_code == 403


def test_native_contract_passes_platform_self_assertions():
    import yaml

    from eb.domain.contract import EmbodimentEnv
    from eb.ufo_native import ENV_ID

    path = Path(__file__).parents[3] / "contracts" / f"{ENV_ID}.yaml"
    contract = EmbodimentEnv.model_validate(yaml.safe_load(path.read_text()))
    assert not contract.self_assertions()


def test_native_contract_preserves_history_joint_and_action_semantics():
    from eb.ufo_native import JOINTS, validate_config

    fixture = json.loads((Path(__file__).parents[1] / "fixtures/ufo-native-config.json").read_text())
    source, effective, model, kwargs = (fixture[k] for k in ("saved", "effective", "model", "kwargs"))
    codec = validate_config(source, effective, model, kwargs)
    assert codec["joint_order"] == list(JOINTS)
    assert codec["history_order"] == ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
    assert codec["history_oldest_first"] is False
    assert codec["actor_dim"] == 631
    assert codec["action_gain"][0] == pytest.approx(0.6 * 120 / 100 * 5)
    assert codec["physics_dt"] == 0.0025
    broken = deepcopy(source)
    broken["env"]["robot_training"]["robot"]["control_joint_names"].reverse()
    with pytest.raises(ValueError, match="joint"):
        validate_config(broken, effective, model, kwargs)
    broken = deepcopy(effective)
    broken["task"]["obs"]["history_oldest_first"] = True
    with pytest.raises(ValueError, match="history"):
        validate_config(source, broken, model, kwargs)


def test_native_safe_file_rejects_symlinks_and_path_traversal(tmp_path):
    from eb.ufo_native import safe_file

    (tmp_path / "ok.json").write_text("{}")
    assert safe_file(tmp_path, "ok.json").name == "ok.json"
    (tmp_path / "linked").symlink_to(tmp_path / "ok.json")
    for name in ("../outside", "/etc/passwd", "linked", "./ok.json"):
        with pytest.raises(ValueError):
            safe_file(tmp_path, name)


def test_frozen_native_cases_are_complete_and_task_failure_is_valid():
    from eb.ufo_native import eval_cases, validate_cases

    plan = {"backend": "mujoco", "profile": "smoke", "cases": eval_cases("smoke")}
    rows = [
        {
            **case,
            "status": "task_failure",
            "reason": "fall",
            "steps": 3,
            "video": f"videos/{case['case_id']}.mp4",
            "metrics": {"speed_rmse_m_s": 0.5},
        }
        for case in plan["cases"]
    ]
    result = validate_cases(plan, rows)
    assert result["success_rate"] == 0
    assert result["qualification"] == "diagnostic"
    assert result["completed"] == result["planned"] == 3
    for bad in (rows[:-1], rows + [rows[0]], [{**row, "metrics": {"x": float("nan")}} for row in rows]):
        with pytest.raises(ValueError):
            validate_cases(plan, bad)


def test_acceptance_cannot_pass_from_claimed_success_without_full_horizon_metrics():
    from eb.ufo_native import eval_cases, validate_cases

    plan = {"backend": "mjlab", "profile": "acceptance", "cases": eval_cases("acceptance")}
    rows = [
        {**case, "status": "success", "steps": 1, "video": f"videos/{case['case_id']}.mp4"}
        for case in plan["cases"]
    ]
    with pytest.raises(ValueError, match="success|metric"):
        validate_cases(plan, rows)


def test_native_job_guard_rejects_unbounded_or_forged_plan():
    from eb.domain.job import JobSpec
    from eb.ufo_native import ENV_ID, plan_hash, validate_job

    plan = {"operation": "ufo_native_import", "image_digest": "sha256:" + "a" * 64, "source": {}}
    plan["hash"] = plan_hash(plan)
    spec = JobSpec(
        kind="data.convert",
        embodiment_env_id=ENV_ID,
        image=plan["image_digest"],
        params={"operation": plan["operation"], "native_plan": plan},
        runtime={"network": "none"},
        resources={"gpus": 0, "cpus": 4, "memory_gb": 16, "shm_gb": 2},
        policy={"timeout_h": 0.5, "max_retries": 0},
    )
    validate_job(spec)
    for patch in (
        {"runtime": {"network": "bridge"}},
        {"overrides": ["x=1"]},
        {"image": "arbitrary:latest"},
        {"params": {**spec.params, "native_plan": {**plan, "hash": "bad"}}},
    ):
        with pytest.raises(ValueError):
            validate_job(spec.model_copy(update=patch))


def test_native_import_does_not_use_training_gpu_memory_estimate():
    from eb.capacity import estimate_job
    from eb.domain.job import JobSpec
    from eb.ufo_native import ENV_ID

    spec = JobSpec(
        kind="data.convert",
        embodiment_env_id=ENV_ID,
        params={"operation": "ufo_native_import"},
        resources={"gpus": 0},
    )
    estimate = estimate_job(spec)
    assert estimate.verdict == "fits"
    assert estimate.per_gpu_peak_gb == 0


def test_native_cpu_jobs_hide_gpus_from_cuda_base_image(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from eb.adapters.ufo_native import UfoNativeAdapter
    from eb.api.routes_ufo_native import spec_for
    from eb.ufo_native import plan_hash

    plan = {
        "name": "native",
        "operation": "ufo_native_import",
        "image_digest": "sha256:" + "a" * 64,
        "source": {"path": "/source", "extras": []},
    }
    plan["hash"] = plan_hash(plan)
    monkeypatch.setattr("eb.adapters.ufo_native.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    rendered = UfoNativeAdapter().render(spec_for(plan), "run_preview")
    assert rendered.gpus == 0
    assert rendered.env["CUDA_VISIBLE_DEVICES"] == ""
    assert rendered.env["NVIDIA_VISIBLE_DEVICES"] == "void"


@pytest.mark.parametrize("kind", ["eval.sim2sim", "export.policy", "bundle.build", "data.convert"])
async def test_native_cpu_jobs_are_admitted_without_any_gpu(kind):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from eb.domain.job import JobSpec
    from eb.scheduler.loop import Scheduler
    from eb.ufo_native import ENV_ID, TASK_ID

    operation = {
        "eval.sim2sim": "eval",
        "export.policy": "export",
        "bundle.build": "bundle",
        "data.convert": "import",
    }[kind]
    spec = JobSpec(
        kind=kind,
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID if kind.startswith("eval.") else None,
        params={"operation": "ufo_native_" + operation},
        resources={"gpus": 0, "memory_gb": 16},
    )
    job = SimpleNamespace(
        id="job_cpu", kind=kind, embodiment_env_id=ENV_ID, spec=spec.model_dump(mode="json")
    )
    session = SimpleNamespace(
        scalars=AsyncMock(side_effect=[SimpleNamespace(all=lambda: []), SimpleNamespace(all=lambda: [job])])
    )
    orch = SimpleNamespace(admit_and_start=AsyncMock())
    scheduler = Scheduler(orch)
    scheduler.reap_running = AsyncMock()
    scheduler.advance_pipelines = AsyncMock()
    scheduler.ebd = SimpleNamespace(doctor=AsyncMock(return_value={"memory_available_mb": 80000, "gpus": []}))
    await scheduler.tick(session)
    orch.admit_and_start.assert_awaited_once_with(session, job, [])


def test_inference_overrides_preserve_saved_semantics_without_derived_struct_keys():
    from eb.ufo_native import inference_overrides

    fixture = json.loads((Path(__file__).parents[1] / "fixtures/ufo-native-config.json").read_text())
    result = inference_overrides(fixture["effective"])
    assert "robot" not in result
    assert "obs_dims" not in result["obs"]
    assert result["obs"]["history_oldest_first"] is False
    assert result["simulator"]["config"]["sim"]["fps"] == 400


@pytest.mark.parametrize("damage", [None, "source", "model", "payload", "missing"])
def test_offline_bundle_keeps_verified_original_source_identity(damage):
    from eb.ufo_native import plan_hash, validate_bundle_source

    source = {"schema_version": "eb/ufo-native-import/v1", "source": {"weights_sha256": "a" * 64}}
    payload = {"source_model_id": "mv_source", "source_artifact_sha256": plan_hash(source)}
    plan = {
        "source_model_id": "mv_source",
        "source_import_manifest": source,
        "source_artifact_sha256": plan_hash(payload),
    }
    if damage == "source":
        plan["source_import_manifest"] = {**source, "source": {"weights_sha256": "b" * 64}}
    elif damage == "model":
        plan["source_model_id"] = "mv_other"
    elif damage == "payload":
        payload["extra"] = "modified"
    elif damage == "missing":
        plan.pop("source_import_manifest")
    if damage:
        with pytest.raises(ValueError, match="source"):
            validate_bundle_source(payload, plan)
    else:
        assert validate_bundle_source(payload, plan) == source
