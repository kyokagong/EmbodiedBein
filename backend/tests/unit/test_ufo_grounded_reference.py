import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import numpy as np
import pytest

from eb.runners.ufo_grounded_reference import grounded_positions, playback_targets, quality_errors
from eb.ufo_grounded_reference import OPERATION, preview_hash, spec_for, validate_submission


def plan():
    return {
        "schema_version": "eb/ufo-grounded-a1-plan/v1",
        "operation": OPERATION,
        "parent_dataset_id": "ds_parent",
        "image_digest": "sha256:" + "a" * 64,
        "hash": "sha256:" + "b" * 64,
        "robot_config_sha256": "config_hash",
        "robot_xml_sha256": "xml_hash",
    }


def test_grounding_changes_only_z_and_never_duplicates_frames():
    root = np.arange(457 * 3, dtype=np.float64).reshape(457, 3) / 1000
    original = root.copy()
    result = grounded_positions(root, 0.04917634611991695)
    np.testing.assert_array_equal(root, original)
    np.testing.assert_array_equal(result[:, :2], root[:, :2])
    np.testing.assert_allclose(result[:, 2] - root[:, 2], 0.04917634611991695, atol=1e-15)
    assert result.shape == root.shape
    for invalid in (np.nan, -0.01, 0.11):
        with pytest.raises(ValueError):
            grounded_positions(root, invalid)


def test_reference_interpolation_keeps_real_duration_and_original_variation():
    q = np.column_stack((np.arange(457) / 120, np.sin(np.arange(457) / 120)))
    times, targets = playback_targets(q, 120, 50)
    assert len(times) == len(targets) == 190 and times[0] == 0 and times[-1] == 3.78
    np.testing.assert_allclose(targets[:, 0], times)
    assert np.std(targets[:, 1]) > 0.1
    with pytest.raises(ValueError):
        playback_targets(q[:-1], 120, 50)  # No truncation/padding to a control interval.


def passing_quality():
    return {
        "completed_steps": 190,
        "expected_steps": 190,
        "finite": True,
        "min_height_m": 0.70,
        "max_tilt_rad": 0.03,
        "drift_m": 0.01,
        "max_joint_excess_rad": 0.0,
        "max_velocity_excess_rad_s": 0.0,
        "max_torque_excess_nm": 0.0,
        "source_joint_excess_rad": 0.0,
        "source_velocity_excess_rad_s": 0.0,
        "floor_max_m": 0.0023,
        "self_max_m": 0.0,
        "ground_contact_frames": 180,
        "physics_seconds": 3.8,
        "simulation_time_monotonic": True,
        "mujoco_warning_count": 0,
        "grounding_delta_error_m": 0.0,
        "video": {"decodable": True, "frames": 96, "fps": 25.0, "seconds": 3.84},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_height_m", 0.49),
        ("max_tilt_rad", 0.81),
        ("max_joint_excess_rad", 0.02),
        ("max_velocity_excess_rad_s", 0.02),
        ("max_torque_excess_nm", 0.02),
        ("floor_max_m", 0.006),
        ("self_max_m", 0.03),
        ("finite", False),
        ("ground_contact_frames", 0),
        ("completed_steps", 18),
        ("mujoco_warning_count", 1),
        ("physics_seconds", 0.02),
        ("simulation_time_monotonic", False),
    ],
)
def test_failed_pd_candidate_cannot_be_training_ready(field, value):
    metrics = passing_quality()
    assert quality_errors(metrics) == []
    metrics[field] = value
    assert quality_errors(metrics)


def test_missing_or_nonfinite_media_and_physics_fail_closed():
    metrics = passing_quality()
    metrics["video"]["decodable"] = False
    assert quality_errors(metrics)
    metrics = passing_quality()
    metrics["max_tilt_rad"] = float("nan")
    assert quality_errors(metrics)
    metrics = passing_quality()
    del metrics["max_torque_excess_nm"]
    assert quality_errors(metrics)


def test_preview_binds_all_inputs_without_candidate_id_and_cpu_estimator():
    from eb.capacity.estimator import estimate_job

    spec = spec_for(plan(), "ds_new", "candidate")
    assert preview_hash(spec) == preview_hash(spec_for(plan(), "ds_other", "other name"))
    bad = spec.model_copy(deep=True)
    bad.params["grounded_plan"]["image_digest"] = "sha256:" + "c" * 64
    assert preview_hash(bad) != preview_hash(spec)
    assert spec.kind == "data.convert" and spec.resources.gpus == 0
    assert spec.resources.cpus == 4 and spec.resources.memory_gb == 16
    assert spec.runtime == {"network": "none"} and spec.policy.max_retries == 0
    assert spec.policy.timeout_h == 0.5
    assert estimate_job(spec).per_gpu_peak_gb == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["image", "network", "dataset", "holdout", "plan", "refs"])
async def test_raw_job_canonical_validation_rejects_tampering(monkeypatch, tamper):
    frozen = plan()
    spec = spec_for(frozen, "ds_new", "candidate")
    row = NS(
        id="ds_new",
        source_kind="ufo_native_motion",
        source_uri="ufo-grounded-a1:ds_parent",
        meta={"operation": OPERATION, "parent_dataset_id": "ds_parent", "plan_hash": frozen["hash"]},
    )
    session = NS(get=AsyncMock(return_value=row))
    monkeypatch.setattr("eb.ufo_grounded_reference.build_plan", AsyncMock(return_value=frozen))
    await validate_submission(session, spec)
    bad = spec.model_copy(deep=True)
    if tamper == "image":
        bad.image = "sha256:" + "e" * 64
    elif tamper == "network":
        bad.runtime["network"] = "host"
    elif tamper == "dataset":
        row.source_uri = "unregistered"
    elif tamper == "holdout":
        bad.inputs.datasets = [{"id": "ds_val", "path": "/holdout"}]
    elif tamper == "plan":
        bad.params["grounded_plan"]["root_z_shift_m"] = 0
    else:
        from eb.domain.job import CodeRefs

        bad.code_refs = CodeRefs(embodiment={"path": "/holdout"})
    with pytest.raises(ValueError):
        await validate_submission(session, bad)


def test_adapter_inputs_readonly_and_no_test_or_host_socket(tmp_path, monkeypatch):
    from eb.ufo_grounded_reference import render

    frozen = plan()
    frozen.update(
        ufo_source_ref={"path": str(tmp_path / "ufo")},
        asset_source_ref={"path": str(tmp_path / "assets")},
        parent_files=[{"source": str(tmp_path / "train.pkl"), "path": "train.pkl"}],
    )
    monkeypatch.setattr(
        "eb.settings.get_settings", lambda: NS(data_root=tmp_path, assets_root=tmp_path / "assetroot")
    )
    rendered = render(spec_for(frozen, "ds_new"), "run_candidate")
    assert rendered.gpus == 0 and rendered.devices == []
    assert all(mode == "ro" or target == "/outputs" for _, target, mode in rendered.mounts)
    assert not any("test.pkl" in source or "docker.sock" in source for source, _, _ in rendered.mounts)
    assert rendered.command[:3] == ["python", "-m", "eb.runners.ufo_grounded_reference"]
    assert json.loads(rendered.command[4])["parent_dataset_id"] == "ds_parent"


def test_free_base_playback_initializes_once_and_keeps_fall_frames(tmp_path, monkeypatch):
    import contextlib
    import sys

    from eb.runners import ufo_grounded_reference as worker
    from eb.runners import ufo_native_sim

    images = []

    @contextlib.contextmanager
    def video_writer(*args, **kwargs):
        yield NS(add_image=images.append)

    monkeypatch.setitem(sys.modules, "ufo.inference.runtime", NS(video_writer=video_writer))
    monkeypatch.setattr(ufo_native_sim, "probe_video", lambda _: passing_quality()["video"])
    monkeypatch.setattr(worker, "plane_contacts", lambda _: np.ones(2))
    data = NS(qpos=np.zeros(30), qvel=np.zeros(29), time=0.0, warning=NS(number=np.zeros(8)))
    counts = {"reset": 0, "step": 0}

    def reset(model, state):
        counts["reset"] += 1
        state.qpos[:] = 0
        state.time = 0.0

    def step(raw, target):
        counts["step"] += 1
        data.qpos[2] -= 0.002  # Falls before the horizon; it must never be teleported back.
        data.time += 0.02

    backend = NS(
        mj=NS(
            mj_resetData=reset,
            mj_forward=lambda *a: None,
            mj_differentiatePos=lambda model, velocity, dt, a, b: velocity.fill(0),
        ),
        model=None,
        data=data,
        qadr=np.arange(7, 30),
        step=step,
        codec={"actuator": {"min_delay": 0, "max_delay": 2}},
        substeps=np.zeros((8, 6)),
        render=lambda: np.zeros((2, 2, 3)),
        state=lambda: {
            "qpos": data.qpos.copy(),
            "q": data.qpos[7:].copy(),
            "dq": data.qvel[6:].copy(),
            "torque": np.zeros(23),
            "height": data.qpos[2],
            "tilt": 0.0,
            "com_position_w": np.zeros(3),
            "foot_position_w": np.zeros((2, 3)),
        },
    )
    root = np.zeros((457, 3))
    root[:, 2] = 0.71
    quat = np.tile([1.0, 0, 0, 0], (457, 1))
    metrics = worker.reference_playback(backend, root, quat, np.zeros((457, 23)), 120, tmp_path)
    assert counts == {"reset": 1, "step": 190} and len(images) == 96
    assert metrics["completed_steps"] == 190 and metrics["min_height_m"] < 0.5
    with np.load(tmp_path / "reference/trace.npz") as trace:
        assert len(trace["qpos"]) == 191 and len(trace["substep_limits"]) == 1520
        np.testing.assert_allclose(trace["height"], 0.71 - np.arange(191) * 0.002)


@pytest.mark.asyncio
async def test_operator_preview_binding_and_idempotent_reuse(monkeypatch):
    from fastapi import HTTPException

    from eb.api import routes_ufo_grounded_reference as routes
    from eb.api.deps import Principal

    frozen = plan()
    monkeypatch.setattr(routes, "build_plan", AsyncMock(return_value=frozen))
    session = NS(execute=AsyncMock(), scalar=AsyncMock(), get=AsyncMock())
    principal = Principal("operator", "user")
    with pytest.raises(HTTPException) as exc:
        await routes.preview("ds_parent", routes.GroundedA1In(), session, Principal("agent", "agent"))
    assert exc.value.status_code == 403
    value = await routes.preview("ds_parent", routes.GroundedA1In(), session, principal)
    body = routes.GroundedA1In(preview_hash=value["preview_hash"])
    previous = NS(
        id="job_existing",
        status="running",
        created_by=principal.id,
        spec=spec_for(frozen, "ds_existing").model_dump(mode="json"),
    )
    session.scalar.return_value = previous
    session.get.return_value = NS(id="ds_existing")
    result = await routes.submit("ds_parent", body, session, principal, "same-key")
    assert result["reused"] is True and result["dataset_id"] == "ds_existing"
    assert "pg_advisory_xact_lock" in str(session.execute.await_args.args[0])
    stale = routes.GroundedA1In(preview_hash="sha256:" + "f" * 64)
    with pytest.raises(HTTPException) as exc:
        await routes.submit("ds_parent", stale, session, principal, "same-key")
    assert exc.value.status_code == 409


def diagnostic_fixture(tmp_path, *, failed):
    from eb.ufo_grounded_reference import PROTOCOL, RECEIPT_SCHEMA

    metrics = passing_quality()
    if failed:
        metrics["min_height_m"] = 0.31
    errors = quality_errors(metrics)
    documents = {
        "dataset-index.json": {
            "dataset_id": "ds_candidate",
            "format": "ufo_native_motion",
            "storage_ref": str(tmp_path / "data"),
            "validation": {
                "scope": "ufo_native_motion_v1",
                "storage_ref": str(tmp_path / "data"),
                "status": "invalid" if failed else "validated",
                "training_ready": not failed,
                "errors": errors,
            },
        },
        "reference/quality.json": {"metrics": metrics, "errors": errors},
        "native-config-receipt.json": {
            "robot_source_sha256": "config_hash",
            "native_robot_xml_sha256": "xml_hash",
            "actuator_mapping": {
                "verified": True,
                "joints": [{"fixture_joint": i} for i in range(23)],
                "dc_motor_grid_samples": 1127,
                "max_clipping_error_nm": 0.0,
            },
        },
    }
    files = []
    for relative in (
        "dataset-index.json",
        "reference/quality.json",
        "reference/trace.npz",
        "reference/replay.mp4",
        "robot-state/A1-grounded.npz",
        "native-motion.yaml",
        "native-config-receipt.json",
        "diagnostic-input/codec.json",
        "diagnostic-input/physics.json",
        "diagnostic-input/effective_config.json",
        "diagnostic-input/assets/rpo.xml",
        "geometry.npz",
        "provenance/qualification.json",
        "provenance/import_report.json",
        "provenance/mask_feasibility.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(documents.get(relative, {"fixture": relative})))
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "run_id": "run_candidate",
        "dataset_id": "ds_candidate",
        "operation": OPERATION,
        "plan_hash": plan()["hash"],
        "image_digest": plan()["image_digest"],
        "protocol": PROTOCOL,
        "reference_pd_only": True,
        "training_ready": not failed,
        "files": files,
    }
    (tmp_path / "candidate-receipt.json").write_text(json.dumps(receipt))
    return receipt


@pytest.mark.asyncio
async def test_failed_candidate_keeps_verified_media_but_publishes_no_version(tmp_path, monkeypatch):
    from eb.ufo_grounded_reference import collect_candidate, verified_report
    from eb.ufo_training_dataset import publish_native_dataset_version

    receipt = diagnostic_fixture(tmp_path, failed=True)

    async def scan(path):
        return {
            "files": 1,
            "samples": [
                {"sha256": hashlib.sha256(__import__("pathlib").Path(path).read_bytes()).hexdigest()}
            ],
        }

    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: NS(scan=scan))
    _, report = await verified_report(tmp_path, plan(), "run_candidate", "ds_candidate")
    assert report["validation"]["training_ready"] is False
    assert not (tmp_path / "data/train.pkl").exists()
    rows = []
    session = NS(scalar=AsyncMock(return_value=None), add=rows.append, flush=AsyncMock())
    frozen = {**plan(), "parent_version_id": "dsv_parent"}
    job = NS(spec={"params": {"grounded_plan": frozen, "dataset_id": "ds_candidate"}})
    run = NS(id="run_candidate", metrics_summary={})
    await collect_candidate(session, job, run, tmp_path)
    assert run.metrics_summary["grounded_reference"]["training_ready"] is False
    assert len(rows) == 3 and rows[0].kind == "ufo_grounded_reference_diagnostic"
    assert rows[0].meta["receipt"] == receipt
    assert (
        await publish_native_dataset_version(
            session, NS(source_kind="ufo_native_motion", status="invalid"), None, None
        )
        is False
    )


@pytest.mark.asyncio
async def test_collector_rejects_fake_ready_missing_split_or_changed_media(tmp_path, monkeypatch):
    from pathlib import Path

    from eb.ufo_grounded_reference import verified_report

    async def scan(path):
        return {"files": 1, "samples": [{"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}]}

    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: NS(scan=scan))
    diagnostic_fixture(tmp_path, failed=False)
    with pytest.raises(ValueError, match="separately frozen splits"):
        await verified_report(tmp_path, plan(), "run_candidate", "ds_candidate")
    receipt = diagnostic_fixture(tmp_path, failed=True)
    receipt["training_ready"] = True
    (tmp_path / "candidate-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="readiness differs"):
        await verified_report(tmp_path, plan(), "run_candidate", "ds_candidate")
    diagnostic_fixture(tmp_path, failed=True)
    video = tmp_path / "reference/replay.mp4"
    old = video.read_bytes()
    video.write_bytes(b"x" * len(old))
    with pytest.raises(ValueError, match="hash differs"):
        await verified_report(tmp_path, plan(), "run_candidate", "ds_candidate")


@pytest.mark.asyncio
async def test_derived_dataset_version_records_parent_and_transform(tmp_path):
    from test_ufo_training_dataset import frozen_dataset

    from eb.store.models import DatasetVersion, LineageEdge
    from eb.ufo_grounded_reference import PROTOCOL
    from eb.ufo_training_dataset import publish_native_dataset_version

    row, job, run, root = frozen_dataset(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(parent_version_id="dsv_parent", source_transform=PROTOCOL)
    (root / "manifest.json").write_text(json.dumps(manifest))
    row.summary_stats["validation"]["manifest_sha256"] = hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()
    row.meta.update(parent_version_id="dsv_parent", parent_dataset_id="ds_parent")
    job.spec["params"].update(
        operation=OPERATION, grounded_plan={"parent_version_id": "dsv_parent", "protocol": PROTOCOL}
    )
    rows = []
    session = NS(
        execute=AsyncMock(),
        scalar=AsyncMock(return_value=None),
        add=rows.append,
        flush=AsyncMock(),
        get=AsyncMock(return_value=NS(id="dsv_parent", dataset_id="ds_parent")),
    )
    assert await publish_native_dataset_version(session, row, job, run)
    version = next(r for r in rows if isinstance(r, DatasetVersion))
    assert version.dataset_id == row.id and version.dataset_id != "ds_parent"
    edge = next(r for r in rows if isinstance(r, LineageEdge) and r.relation == "derived")
    assert edge.src_id == "dsv_parent" and edge.dst_id == version.id


@pytest.mark.asyncio
async def test_canonical_plan_uses_only_frozen_producer_files_and_original_a1(tmp_path, monkeypatch):
    from eb import ufo_grounded_reference as protocol
    from eb.runners.ufo_training_data import digest
    from eb.store.models import Dataset, Job, Run

    state = tmp_path / "state"
    outputs = state / "runs/run_parent/outputs"
    for relative in (
        "data/train.pkl",
        "data/manifest.json",
        "holdout/validation.pkl",
        "holdout/manifest.json",
        "provenance/qualification.json",
        "provenance/import_report.json",
        "provenance/mask_feasibility.json",
    ):
        path = outputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "episodes_count": 58 if relative.startswith("data/") else 15,
            "robot_config_sha256": "config_hash",
            "robot_xml_sha256": "xml_hash",
        }
        path.write_text(json.dumps(value))
    refs = {}
    for name, character in (("ufo", "a"), ("asset", "b")):
        root = state / "snapshots" / (character * 64) / "code"
        root.mkdir(parents=True)
        refs[name + "_source_ref"] = {
            "kind": "content_snapshot",
            "commit": "sha256:" + character * 64,
            "path": str(root),
        }
    a1 = __import__("pathlib").Path(refs["asset_source_ref"]["path"]) / protocol.A1_PATH
    a1.parent.mkdir(parents=True)
    a1.write_bytes(b"frozen known source; no pickle parsing in API")
    monkeypatch.setattr(protocol, "PARENT_MANIFEST", digest(outputs / "data/manifest.json"))
    monkeypatch.setattr(protocol, "TRAIN_SHA", digest(outputs / "data/train.pkl"))
    monkeypatch.setattr(protocol, "VALIDATION_SHA", digest(outputs / "holdout/validation.pkl"))
    monkeypatch.setattr(protocol, "A1_SHA", digest(a1))
    monkeypatch.setattr(
        "eb.settings.get_settings", lambda: NS(data_root=state, assets_root=tmp_path / "assets")
    )
    monkeypatch.setattr(
        "eb.ufo_training.require_native_dataset", lambda row: {"manifest_sha256": protocol.PARENT_MANIFEST}
    )
    parent = NS(
        id=protocol.PARENT_ID,
        dest_path=str(outputs / "data"),
        meta={"validation_job_id": "job_parent", "validation_run_id": "run_parent"},
    )
    producer = NS(
        id="job_parent",
        status="succeeded",
        kind="data.convert",
        spec={"params": {"operation": "prepare_ufo_native_motion", **refs}},
    )
    run = NS(id="run_parent", job_id=producer.id, status="finished")
    version = NS(
        id="dsv_parent", content_hash="sha256:" + protocol.PARENT_MANIFEST, stats={"producer_run": run.id}
    )
    session = NS(
        get=AsyncMock(side_effect=lambda model, identity: {Dataset: parent, Job: producer, Run: run}[model]),
        scalar=AsyncMock(return_value=version),
    )

    async def scanned(path):
        return {"source": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}

    monkeypatch.setattr("eb.api.routes_ufo_native.scanned_file", scanned)
    monkeypatch.setattr(
        "eb.store.ebd.EbdClient",
        lambda: NS(image_info=AsyncMock(return_value={"ok": True, "digest": "sha256:" + "c" * 64})),
    )
    frozen = await protocol.build_plan(session, protocol.PARENT_ID)
    assert len(frozen["parent_files"]) == 7 and frozen["source"]["source"] == str(a1)
    assert frozen["base_training_image"] == protocol.BASE_IMAGE
    assert frozen["protocol"]["source_arrays_unchanged"] is False
    assert not any("test.pkl" in item["source"] for item in frozen["parent_files"])
    (outputs / "holdout/validation.pkl").write_bytes(b"different holdout")
    with pytest.raises(ValueError, match="Approved parent or A1 bytes changed"):
        await protocol.build_plan(session, protocol.PARENT_ID)


@pytest.mark.asyncio
async def test_shared_orchestrator_dispatches_candidate_and_denies_agent(monkeypatch):
    from eb.orchestrator import service

    monkeypatch.setattr("eb.orchestrator.deletion.guard_storage_inputs", AsyncMock())
    monkeypatch.setattr(service, "validate_flat_warm_start_submission", AsyncMock())
    validator = AsyncMock()
    generic = AsyncMock(side_effect=AssertionError("candidate dispatched to deployment adapter"))
    monkeypatch.setattr("eb.ufo_grounded_reference.validate_submission", validator)
    monkeypatch.setattr("eb.orchestrator.ufo_native.validate_submission", generic)
    session = NS(get=AsyncMock(return_value=None))
    spec = spec_for(plan(), "ds_new")
    with pytest.raises(PermissionError, match="operator"):
        await service.Orchestrator(NS()).submit(session, spec, "agent:test", None)
    validator.assert_not_awaited()
    with pytest.raises(ValueError, match="unknown embodiment env"):
        await service.Orchestrator(NS()).submit(session, spec, "user:test", None)
    validator.assert_awaited_once_with(session, spec)
    generic.assert_not_awaited()


@pytest.mark.asyncio
async def test_candidate_submission_creates_separate_dataset_and_single_transaction(monkeypatch, tmp_path):
    from eb.api import routes_ufo_grounded_reference as routes
    from eb.api.deps import Principal
    from eb.orchestrator.service import Orchestrator
    from eb.store.models import Dataset, LineageEdge

    frozen = {**plan(), "parent_version_id": "dsv_parent"}
    monkeypatch.setattr(routes, "build_plan", AsyncMock(return_value=frozen))
    monkeypatch.setattr(routes, "get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr("eb.domain.disk.disk_preflight", lambda *a, **kw: NS(ok=True))
    submit_job = AsyncMock(return_value=NS(id="job_new", status="queued"))
    monkeypatch.setattr(Orchestrator, "submit", submit_job)
    rows = []
    session = NS(
        execute=AsyncMock(),
        scalar=AsyncMock(return_value=None),
        add=rows.append,
        flush=AsyncMock(),
        commit=AsyncMock(),
    )
    body = routes.GroundedA1In(preview_hash=preview_hash(spec_for(frozen, "ds_preview")))
    result = await routes.submit("ds_parent", body, session, Principal("operator", "user"), "new-key")
    dataset = next(row for row in rows if isinstance(row, Dataset))
    assert dataset.id == result["dataset_id"] and dataset.id != "ds_parent"
    assert dataset.status == "queued" and dataset.dest_path == ""
    assert dataset.meta["validation_job_id"] == "job_new"
    assert dataset.meta["parent_version_id"] == "dsv_parent"
    assert dataset.summary_stats["validation"]["training_ready"] is False
    assert submit_job.await_args.kwargs["commit"] is False
    assert any(isinstance(row, LineageEdge) and row.src_id == "ds_parent" for row in rows)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_incomplete_candidate_exposes_failure_without_publishing(tmp_path, monkeypatch):
    from test_ufo_training_dataset import frozen_dataset

    from eb.api.routes_data import reconcile_dataset

    row, job, run, root = frozen_dataset(tmp_path)
    job.spec["params"].update(operation=OPERATION, grounded_plan=plan())
    row.meta.update(operation=OPERATION)
    row.meta.pop("validation_run_id")
    row.status = "queued"
    (root.parent / "dataset-index.json").write_text(
        json.dumps({"validation": {"status": "validated", "training_ready": True}})
    )
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    session = NS(get=AsyncMock(return_value=job), scalar=AsyncMock(return_value=run), commit=AsyncMock())
    await reconcile_dataset(row, session)
    assert row.status == "invalid" and row.format == "unknown"
    assert row.summary_stats["validation"]["training_ready"] is False
    assert "incomplete or invalid" in row.error_message
    assert row.meta["candidate_evidence_verified"] is False
    session.commit.assert_awaited_once()


def test_candidate_routes_are_mounted_on_the_existing_dataset_router():
    from fastapi import FastAPI

    from eb.api.routes_data import router
    app = FastAPI()
    app.include_router(router)
    paths = set(app.openapi()["paths"])
    assert {
        "/datasets/{parent_id}/grounded-a1/preview",
        "/datasets/{parent_id}/grounded-a1",
        "/datasets/{dataset_id}/grounded-a1/files/{path}",
    } <= paths
