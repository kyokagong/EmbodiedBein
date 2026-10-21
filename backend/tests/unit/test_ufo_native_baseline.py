import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from eb.domain.job import JobSpec
from eb.ufo_native import ENV_ID, plan_hash, sha256_file
from eb.ufo_native_baseline import (
    ARTIFACT_KIND,
    BASE_IMAGE,
    DATASET,
    OPERATION,
    RECEIPT_SCHEMA,
    RESOURCES,
    RULES,
    SCHEMA,
    SOURCE_RUN,
    config_differences,
    metric_differences,
)
from eb.ufo_native_baseline_service import collect, preview_hash, read_receipt, spec_for, validate_submission


def plan():
    return {
        "schema_version": SCHEMA,
        "image_digest": "sha256:" + "d" * 64,
        "base_image_digest": BASE_IMAGE,
        "source_run_id": SOURCE_RUN,
        "source_spec": JobSpec(
            kind="train.rl",
            embodiment_env_id=ENV_ID,
            task_id="task",
            image=BASE_IMAGE,
            code_refs={"stack": {"path": "/frozen/rlinf"}},
            inputs={"datasets": [{"id": DATASET, "path": "/frozen/train"}]},
        ).model_dump(mode="json"),
        "native_train_plan": {"identity": {"buffer_size": 262144}},
        "comparison_rules": copy.deepcopy(RULES),
        "reference_files": {},
    }


def test_only_launch_device_mapping_is_ignored_and_all_other_leaves_are_exact():
    worker = {
        "env": {"device": "cuda:0", "joint_order": ["left", "right"]},
        "work_dir": "/run/eb-output/outputs/native",
        "seed": 45,
    }
    native = copy.deepcopy(worker)
    native["env"]["device"] = "cuda:3"
    assert config_differences(worker, native, 3) == []
    native["env"]["joint_order"].reverse()
    assert len(config_differences(worker, native, 3)) == 2
    native = copy.deepcopy(worker)
    native["work_dir"] = "/other-output"
    assert config_differences(worker, native, 3)[0]["path"] == "work_dir"


def test_metric_report_preserves_differences_and_never_claims_equivalence(tmp_path):
    worker, native = tmp_path / "worker.csv", tmp_path / "native.csv"
    worker.write_text("distributed/optimizer_steps,loss,only_worker\n96,10,2\n112,9,3\n")
    native.write_text("distributed/optimizer_steps,loss,only_native\n80,20,2\n112,900,7\n")
    result = metric_differences(worker, native)
    assert result["worker_only_steps"] == [96] and result["native_only_steps"] == [80]
    assert result["aligned"][0]["metrics"]["loss"]["native_minus_worker"] == 891
    assert result["aligned"][0]["metrics"]["only_worker"]["native"] is None
    assert result["numerical_equivalence_claimed"] is False
    native.write_text("distributed/optimizer_steps,loss\n112,nan\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        metric_differences(worker, native)


def test_preview_binds_full_spec_name_image_and_diagnostic_budget():
    value = plan()
    spec = spec_for(value)
    assert spec.kind == "asset.validate" and spec.params["operation"] == OPERATION
    assert spec.resources.model_dump() == RESOURCES and spec.policy.max_retries == 0
    assert spec.policy.timeout_h == 0.5 and spec.runtime == {"network": "none", "hostname": "localhost"}
    assert spec_for(value, "other name").name == "other name"
    assert preview_hash(spec) != preview_hash(spec_for(value, "other name"))
    assert preview_hash(spec) != preview_hash(spec.model_copy(update={"image": BASE_IMAGE}))
    from eb.adapters import get_adapter
    from eb.capacity import estimate_job

    assert get_adapter(spec).name == "ufo_native_baseline"
    assert estimate_job(spec).per_gpu_peak_gb > 12


def test_execution_image_identity_is_consistent_and_source_plan_is_unchanged():
    from eb.ufo_native_baseline import execution_training_plan
    from eb.ufo_training import training_fingerprint

    value = plan()
    value["native_train_plan"]["identity"]["image_digest"] = BASE_IMAGE
    before = copy.deepcopy(value)
    execution = execution_training_plan(value, "run_baseline")
    assert value == before
    assert execution["image_digest"] == execution["identity"]["image_digest"] == value["image_digest"]
    assert execution["fingerprint"] == training_fingerprint(execution["identity"])
    assert execution["full_state_finite_required"] is False
    assert execution["hash"] == plan_hash(execution)


def test_native_optimizer_comparison_requires_actual_steps_and_all_five_schemas():
    from eb.runners.ufo_native_baseline import optimizer_layout

    states = {
        str(index): {"state": {0: {"step": 112.0}}, "param_groups": [{"params": [0], "lr": 0.0001}]}
        for index in range(5)
    }
    layout = optimizer_layout(states)
    assert layout["0"]["step_minimum"] == 112 and layout["0"]["groups"][0]["params"] == [0]
    states["0"]["state"][0]["step"] = 111
    with pytest.raises(ValueError, match="internal step"):
        optimizer_layout(states)
    states["0"]["state"][0]["step"] = 112
    del states["4"]
    with pytest.raises(ValueError, match="five"):
        optimizer_layout(states)


def test_render_uses_only_image_owned_script_and_matching_reference_mounts(tmp_path, monkeypatch):
    from eb.adapters.base import RenderedJob
    from eb.ufo_native_baseline_service import render

    value = plan()
    value["reference_files"] = {
        f"outputs/runtime/{name}-rank-{rank}.json": "sha256:" + "a" * 64
        for rank in range(4)
        for name in ("config", "initial-parameters")
    }
    captured = []

    def native_render(spec, rid):
        captured.append(spec)
        return RenderedJob(
            spec.image,
            ["old-command"],
            {},
            ["outputs", "runtime", "inputs"],
            mounts=[
                ("/frozen/train.pkl", "/training-data/train.pkl", "ro"),
                ("/old/plan", "/plan.json", "ro"),
            ],
            gpus=4,
            shm_gb=8,
        )

    monkeypatch.setattr("eb.ufo_training.render_training", native_render)
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    result = render(spec_for(value), "run_preview")
    assert result.command == [
        "python",
        "-m",
        "eb.runners.ufo_native_baseline",
        "--plan",
        "/baseline-plan.json",
    ]
    assert result.image == captured[0].image == value["image_digest"]
    assert captured[0].params["native_train_plan"]["identity"]["image_digest"] == result.image
    assert all(mode == "ro" for _, target, mode in result.mounts if target.startswith("/reference/"))
    assert (
        str(tmp_path / f"runs/{SOURCE_RUN}/outputs/runtime/config-rank-3.json"),
        "/reference/outputs/runtime/config-rank-3.json",
        "ro",
    ) in result.mounts
    assert not any(target == "/plan.json" for _, target, _ in result.mounts)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("resources", {"gpus": 1}),
        ("runtime", {"network": "host"}),
        ("overrides", ["--arbitrary-code"]),
        ("outputs", {"path": "/host"}),
    ],
)
async def test_raw_baseline_cannot_change_execution_boundary(monkeypatch, field, value):
    current = plan()
    monkeypatch.setattr("eb.ufo_native_baseline_service.build_plan", AsyncMock(return_value=current))
    spec = spec_for(current).model_dump(mode="json")
    spec[field] = value
    with pytest.raises(ValueError, match="limits differ"):
        await validate_submission(None, JobSpec.model_validate(spec))


@pytest.mark.asyncio
async def test_source_run_must_be_the_fixed_successful_run_before_image_lookup():
    from eb.ufo_native_baseline_service import build_plan

    session = NS(get=AsyncMock(return_value=None))
    with pytest.raises(ValueError, match="Only the frozen"):
        await build_plan(session, "run_another")
    session.get.assert_not_awaited()
    with pytest.raises(ValueError, match="successfully completed"):
        await build_plan(session)


@pytest.mark.asyncio
async def test_operator_guard_applies_to_raw_job_before_any_compute(monkeypatch):
    from eb.orchestrator.service import Orchestrator

    session = NS(execute=AsyncMock())
    monkeypatch.setattr("eb.orchestrator.deletion.guard_storage_inputs", AsyncMock())
    monkeypatch.setattr("eb.orchestrator.service.validate_flat_warm_start_submission", AsyncMock())
    with pytest.raises(PermissionError, match="operator"):
        await Orchestrator().submit(session, spec_for(plan()), "agent:test", None)
    session.execute.assert_not_awaited()


def receipt_fixture(tmp_path):
    value = plan()
    runtime = {**value, "run_id": "run_baseline"}
    runtime["hash"] = plan_hash(runtime)
    names = {
        "baseline-report.json",
        "metrics-differences.json",
        "native/checkpoint-manifest.json",
        "native/train_log.txt",
        "analysis/summary.json",
        "metrics-index.json",
    }
    names |= {f"runtime/baseline-initial-rank-{rank}.json" for rank in range(4)}
    rows = []
    for name in sorted(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = {}
        if name == "baseline-report.json":
            contents = {
                "counts": RULES["counts"],
                "ranks": [
                    {"rank": rank, "initial_hash_exact": True, "configuration_leaf_values_equal": True}
                    for rank in range(4)
                ],
                "optimizer_internal_steps": {
                    str(index): {"step_minimum": 112, "step_maximum": 112} for index in range(5)
                },
                "native_synchronization": {
                    "world_size": 4,
                    "rank": 0,
                    "max_abs_diff_from_rank0": 0.0,
                    "rank_param_buffer_sum": [1.0] * 4,
                    "rank_param_buffer_sqsum": [1.0] * 4,
                    "rank_param_buffer_numel": [10] * 4,
                },
            }
        if name == "native/checkpoint-manifest.json":
            contents = {
                "schema_version": "eb/ufo-native-checkpoint/v1",
                "rank_results": [
                    {
                        "rank": rank,
                        "replay_loader_verified": True,
                        "full_state_finite_verified": True,
                        "loader_verified": True,
                    }
                    for rank in range(4)
                ],
            }
        path.write_text(json.dumps(contents))
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "run_id": "run_baseline",
        "operation": OPERATION,
        "plan_hash": runtime["hash"],
        "source_run_id": SOURCE_RUN,
        "image_digest": value["image_digest"],
        "base_image_digest": BASE_IMAGE,
        "counts": RULES["counts"],
        "comparison_rules": RULES,
        "structural_checks_passed": True,
        "numerical_equivalence_claimed": False,
        "quality_qualification": "diagnostic",
        "files": rows,
    }
    (tmp_path / "baseline-receipt.json").write_text(json.dumps(receipt))
    return value, receipt


@pytest.mark.parametrize(
    "change", ["missing_rank", "traversal", "wrong_plan", "false_equivalence", "changed_size"]
)
def test_receipt_rejects_partial_or_forged_diagnostics(tmp_path, change):
    value, receipt = receipt_fixture(tmp_path)
    if change == "missing_rank":
        receipt["files"] = [
            row for row in receipt["files"] if row["path"] != "runtime/baseline-initial-rank-3.json"
        ]
    elif change == "traversal":
        receipt["files"][0]["path"] = "../secret"
    elif change == "wrong_plan":
        receipt["plan_hash"] = "sha256:" + "a" * 64
    elif change == "false_equivalence":
        receipt["numerical_equivalence_claimed"] = True
    else:
        (tmp_path / receipt["files"][0]["path"]).write_text("changed")
    (tmp_path / "baseline-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        read_receipt(tmp_path, value, "run_baseline")


@pytest.mark.asyncio
async def test_collection_is_idempotent_and_publishes_only_diagnostic_artifact(tmp_path, monkeypatch):
    from eb.store.models import Artifact, LineageEdge

    value, receipt = receipt_fixture(tmp_path)
    verifier = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", verifier)
    added = []

    def existing(statement):
        kind = statement.compile().params["kind_1"]
        return next((row for row in added if isinstance(row, Artifact) and row.kind == kind), None)

    session = NS(
        scalar=AsyncMock(side_effect=existing),
        execute=AsyncMock(),
        add=added.append,
        flush=AsyncMock(),
    )
    job = NS(spec=spec_for(value).model_dump(mode="json"))
    run = NS(id="run_baseline", metrics_summary={})
    await collect(session, job, run, tmp_path)
    await collect(session, job, run, tmp_path)
    assert len(added) == 6 and all(isinstance(row, (Artifact, LineageEdge)) for row in added)
    assert added[0].kind == ARTIFACT_KIND and added[0].meta["diagnostic_only"]
    assert verifier.await_count == 4
    assert run.metrics_summary["native_baseline"]["actual_optimizer_updates"] == 112
    receipt["files"][0]["sha256"] = "b" * 64
    (tmp_path / "baseline-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Immutable"):
        await collect(session, job, run, tmp_path)


@pytest.mark.asyncio
async def test_ebd_hash_failure_cannot_publish_baseline_artifact(tmp_path, monkeypatch):
    value, _ = receipt_fixture(tmp_path)
    monkeypatch.setattr(
        "eb.orchestrator.ufo_native.verify_index", AsyncMock(side_effect=ValueError("hash mismatch"))
    )
    session = NS(scalar=AsyncMock(), add=Mock())
    with pytest.raises(ValueError, match="hash mismatch"):
        await collect(
            session,
            NS(spec=spec_for(value).model_dump(mode="json")),
            NS(id="run_baseline", metrics_summary={}),
            tmp_path,
        )
    session.scalar.assert_not_awaited()
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_scope_is_additive_immutable_and_independent_of_current_image(tmp_path, monkeypatch):
    from eb.api.deps import Principal
    from eb.api.routes_ufo_native_baseline import publish_scope, verification_scope
    from eb.store.models import Artifact, Run
    from eb.ufo_native_baseline import SCOPED_ARTIFACT_KIND, VERIFICATION_SCOPE

    outputs = tmp_path / "runs/run_baseline/outputs"
    value, receipt = receipt_fixture(outputs)
    raw = Artifact(
        id="art_raw_v1",
        kind=ARTIFACT_KIND,
        sha256=plan_hash(receipt),
        storage=str(outputs),
        run_id="run_baseline",
        meta={"receipt": receipt, "diagnostic_only": True},
    )
    original_meta, original_sha = copy.deepcopy(raw.meta), raw.sha256
    records, added = [raw], []

    def add(row):
        records.append(row)
        added.append(row)

    def existing(statement):
        kind = statement.compile().params["kind_1"]
        return next((row for row in records if isinstance(row, Artifact) and row.kind == kind), None)

    job = NS(
        id="job_baseline",
        kind="asset.validate",
        status="succeeded",
        spec=spec_for(value).model_dump(mode="json"),
    )
    run = NS(id="run_baseline", job_id=job.id, status="finished", metrics_summary={})
    session = NS(
        scalar=AsyncMock(side_effect=existing),
        execute=AsyncMock(),
        add=add,
        flush=AsyncMock(),
        commit=AsyncMock(),
        get=AsyncMock(side_effect=lambda cls, key: run if cls is Run else job),
    )
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", AsyncMock())
    monkeypatch.setattr(
        "eb.ufo_native_baseline_service.build_plan",
        AsyncMock(side_effect=AssertionError("No current image/source resolution for old Job")),
    )
    principal = Principal("test", "user")
    first = await publish_scope(run.id, session, principal)
    second = await publish_scope(run.id, session, principal)
    assert first == second == await verification_scope(run.id, session, principal)
    assert len(added) == 3  # only new scope Artifact and two lineage edges
    scoped = next(row for row in added if isinstance(row, Artifact))
    assert scoped.kind == SCOPED_ARTIFACT_KIND and scoped.storage == raw.storage
    assert raw.meta == original_meta and raw.sha256 == original_sha
    payload = scoped.meta["verification"]
    assert payload["legacy_manifest_overstates_rank_optimizer_verification"] is True
    assert payload["legacy_rank_optimizer_claims_accepted"] is False
    assert payload["verification_scope"] == VERIFICATION_SCOPE
    assert payload["verification_scope"]["live_optimizer_finite_per_rank"]["3"] == "not_observed"
    assert payload["verification_scope"]["shared_checkpoint_five_optimizers_finite"] == "observed"
    assert payload["raw_files"]["baseline-report.json"]["sha256"] == sha256_file(
        outputs / "baseline-report.json"
    )
    assert not any(path.name.startswith("scoped") for path in outputs.iterdir())
    assert run.metrics_summary["native_baseline"]["scoped_verification_artifact_id"] == scoped.id
    scoped.meta["verification"]["verification_scope"]["training_restore_supported"] = True
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await verification_scope(run.id, session, principal)


@pytest.mark.asyncio
async def test_scope_backfill_requires_operator_and_successful_baseline(tmp_path):
    from fastapi import HTTPException

    from eb.api.deps import Principal
    from eb.api.routes_ufo_native_baseline import publish_scope
    from eb.ufo_native_baseline_service import backfill_scoped_verification

    session = NS(get=AsyncMock())
    with pytest.raises(HTTPException) as exc:
        await publish_scope("run_baseline", session, Principal("test", "agent"))
    assert exc.value.status_code == 403
    session.get.assert_not_awaited()
    with pytest.raises(ValueError, match="successful completed"):
        await backfill_scoped_verification(session, NS(kind="asset.validate", status="failed"), None)


def test_scoped_payload_rejects_changed_report_or_unobserved_replay(tmp_path):
    from eb.ufo_native_baseline_service import scoped_payload

    value, receipt = receipt_fixture(tmp_path)
    raw = NS(id="art_raw", sha256=plan_hash(receipt))
    assert scoped_payload(tmp_path, receipt, raw)["legacy_rank_optimizer_claims_accepted"] is False
    path = tmp_path / "baseline-report.json"
    report = json.loads(path.read_text())
    report["optimizer_internal_steps"]["0"]["step_maximum"] = 113
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        scoped_payload(tmp_path, receipt, raw)


def test_future_manifest_cannot_claim_per_rank_optimizer_checks_or_training_restore(tmp_path, monkeypatch):
    from eb import ufo_training_checkpoint as checkpoint
    from eb.ufo_native_baseline import CHECKPOINT_SCHEMA, build_baseline_checkpoint_manifest

    metadata = {"counts": RULES["counts"], "world_size": 4}
    monkeypatch.setattr(checkpoint, "_native_metadata", lambda *args: metadata)
    (tmp_path / "raw.json").write_text("{}")
    observations = [
        {"rank": rank, "gpu_uuid": str(rank), "replay_loader_verified": True} for rank in range(4)
    ]
    training = {"image_digest": "sha256:" + "a" * 64, "source_refs": {}}
    manifest = build_baseline_checkpoint_manifest(
        tmp_path, training, "run_baseline", rank_observations=observations
    )
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA
    assert manifest["training_restore_supported"] is False
    assert "rank_results" not in manifest and "full_state_finite_verified" not in manifest
    with pytest.raises(ValueError, match="completion"):
        checkpoint.validate_native_checkpoint(tmp_path, manifest, run_id="run_baseline")
    observations[1]["full_state_finite_verified"] = True
    with pytest.raises(ValueError, match="optimizer verification"):
        build_baseline_checkpoint_manifest(tmp_path, training, "run_baseline", rank_observations=observations)


@pytest.mark.parametrize("operation", ["resume", "model"])
def test_legacy_raw_baseline_is_never_a_training_restore_source(tmp_path, operation):
    from eb.checkpoints import checkpoint_model, resume_checkpoint

    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/baseline-receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="not training"):
        (resume_checkpoint if operation == "resume" else checkpoint_model)(tmp_path, "run_baseline")


@pytest.mark.asyncio
async def test_submit_rejects_stale_preview_before_lock_and_reuses_cancelled_job(monkeypatch):
    from fastapi import HTTPException

    from eb.api.deps import Principal
    from eb.api.routes_ufo_native_baseline import BaselineIn, submit

    value = plan()
    monkeypatch.setattr("eb.api.routes_ufo_native_baseline.build_plan", AsyncMock(return_value=value))
    session = NS(execute=AsyncMock(), scalar=AsyncMock())
    principal = Principal("test", "user")
    with pytest.raises(HTTPException) as exc:
        await submit(SOURCE_RUN, BaselineIn(preview_hash="sha256:" + "0" * 64), session, principal, "idem")
    assert exc.value.status_code == 409
    session.execute.assert_not_awaited()
    body = BaselineIn()
    spec = spec_for(value, body.name)
    body.preview_hash = preview_hash(spec)
    session.scalar.return_value = NS(
        id="job_existing", status="cancelled", created_by=principal.id, spec=spec.model_dump(mode="json")
    )
    result = await submit(SOURCE_RUN, body, session, principal, "idem")
    assert result == {"id": "job_existing", "status": "cancelled", "reused": True}
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,allowed",
    [
        ("baseline-report.json", True),
        ("native/train_log.txt", True),
        ("native/checkpoint/model/model.safetensors", False),
        ("native/checkpoint/buffers/train_rank_0/buffer.hdf5", False),
    ],
)
async def test_download_exposes_only_hashed_diagnostic_evidence_not_replay(
    tmp_path, monkeypatch, path, allowed
):
    from fastapi import HTTPException

    from eb.api.routes_ufo_native_baseline import output_file
    from eb.store.models import Run

    outputs = tmp_path / "runs/run_baseline/outputs"
    value, receipt = receipt_fixture(outputs)
    if not (outputs / path).exists():
        private = outputs / path
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_bytes(b"private native state")
        receipt["files"].append(
            {"path": path, "bytes": private.stat().st_size, "sha256": sha256_file(private)}
        )
        (outputs / "baseline-receipt.json").write_text(json.dumps(receipt))
    run = NS(job_id="job_baseline")
    job = NS(status="succeeded", spec=spec_for(value).model_dump(mode="json"))
    session = NS(
        get=AsyncMock(side_effect=lambda cls, key: run if cls is Run else job),
        scalar=AsyncMock(return_value=NS(sha256=plan_hash(receipt))),
    )
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    response = NS(headers={})
    stream = AsyncMock(return_value=response)
    monkeypatch.setattr("eb.api.routes_ufo_native.stream_output", stream)
    if allowed:
        assert await output_file("run_baseline", path, None, session, None) is response
        assert response.headers["X-EB-Live-Rank-Optimizer-Verification"] == "not_observed"
        assert "/verification-scope" in response.headers["Link"]
    else:
        with pytest.raises(HTTPException) as exc:
            await output_file("run_baseline", path, None, session, None)
        assert exc.value.status_code == 404
        stream.assert_not_awaited()


def test_actual_entry_observer_disables_itself_before_hashing_and_train_body(tmp_path, monkeypatch):
    from eb.runners import rlinf_ufo_runtime as runtime
    from eb.runners import ufo_native_baseline_observer as observer

    snapshot = {"sha256": "sha256:" + "1" * 64, "parameters": [], "sample_definition": "test"}
    expected = {"env": {"device": "cuda:0"}, "seed": 42}
    for name, contents in (("config-rank-0.json", expected), ("initial-parameters-rank-0.json", snapshot)):
        (tmp_path / name).write_text(json.dumps(contents))

    def routed_path(path):
        if str(path).startswith("/reference/outputs/runtime/"):
            return tmp_path / Path(path).name
        return Path(path)

    monkeypatch.setattr(observer, "Path", routed_path)
    monkeypatch.setattr(runtime, "parameter_snapshot", lambda _: (snapshot, {}))
    writes = []
    monkeypatch.setattr(runtime, "write_json", lambda path, payload: writes.append((path, payload)))
    monkeypatch.setitem(
        sys.modules, "torch", NS(cuda=NS(get_device_properties=lambda rank: NS(uuid="unique-gpu")))
    )
    monkeypatch.setenv("EB_UFO_BASELINE_OBSERVER", "1")
    monkeypatch.delenv("RANK", raising=False)  # elastic assigns RANK after Python/sitecustomize startup
    namespace = {"sys": sys}
    exec(
        compile(
            "def train(self):\n assert sys.getprofile() is None\n self.ran = True\n",
            "/sources/ufo/ufo/training/trainer.py",
            "exec",
        ),
        namespace,
    )
    trainer = NS(
        cfg=NS(model_dump=lambda **kw: expected, distributed_rank=0, distributed_world_size=4, seed=42),
        _optimizer_steps=0,
        agent=NS(_model=None, device="cuda:0"),
        ran=False,
    )
    previous = sys.getprofile()
    try:
        sys.setprofile(None)
        observer.install()
        monkeypatch.setenv("RANK", "0")
        namespace["train"](trainer)
        assert trainer.ran and sys.getprofile() is None
        assert writes[0][1]["profiler_disabled_before_hashing"] is True
        assert writes[0][1]["optimizer_steps_before_train"] == 0
    finally:
        sys.setprofile(previous)
