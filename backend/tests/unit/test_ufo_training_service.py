import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from eb.domain.job import GitCodeRef, JobSpec
from eb.store.models import Dataset, EmbodimentEnvRow, EnvironmentProfile, Job, Run, TrainJob
from eb.ufo_native import ENV_ID, TASK_ID, plan_hash
from eb.ufo_training import native_arguments, training_fingerprint
from eb.ufo_training_service import validate_submission


def resume_fixture(tmp_path, monkeypatch):
    from eb import ufo_training_service as service

    digest = "sha256:" + "a" * 64
    state = tmp_path / "state"
    code = state / "snapshots" / ("a" * 64) / "code"
    code.mkdir(parents=True)
    ref = {"kind": "content_snapshot", "path": str(code), "commit": digest}
    profile = {"id": "profile_native", "version": 1}
    report = {"scope": "ufo_native_motion_v1", "manifest_sha256": "b" * 64}
    identity = {
        "image_digest": digest,
        "world_size": 1,
        "num_envs_per_rank": 1024,
        "buffer_size": 262144,
        "seed": 42,
        "dataset_id": "ds_native",
        "dataset_manifest": report["manifest_sha256"],
        "contract_hash": "contract_native",
        "environment_profile": profile,
        "ufo_snapshot": digest,
        "rlinf_snapshot": digest,
        "asset_snapshot": digest,
    }
    native_path = state / "runs/run_source/outputs/native"
    plan = {
        "schema_version": "eb/ufo-native-train-plan/v1",
        "identity": identity,
        "fingerprint": training_fingerprint(identity),
        "image_digest": digest,
        "world_size": 1,
        "target_updates": 512,
        "start_updates": 112,
        "start_local_time": 18432,
        "full_state_finite_required": True,
        "log_every_steps": 1024,
        "source_refs": {
            name: {
                "path": f"/sources/{name}",
                "manifest_path": f"/sources/{name}-manifest.json",
                "snapshot_hash": digest,
            }
            for name in ("ufo", "rlinf")
        },
        "resume": {
            "host_path": str(native_path),
            "checkpoint_path": "/resume-native/checkpoint",
            "step": 112,
        },
    }
    plan["native_args"] = native_arguments(plan)
    dataset = SimpleNamespace(dest_path=str(tmp_path / "training-data"))
    spec = JobSpec(
        kind="train.rl",
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID,
        image=digest,
        params={
            "template_id": "rpo_ufo_fb_native",
            "stack": "rlinf",
            "native_train_plan": plan,
            "ufo_source_ref": ref,
            "asset_source_ref": ref,
            "source_train_job_id": "trn_source",
        },
        inputs={
            "datasets": [{"id": "ds_native", "path": dataset.dest_path, "validation": report}],
            "resume_from": "run_source",
        },
        code_refs={"stack": ref},
        resources={"gpus": 1, "cpus": 8, "memory_gb": 32, "shm_gb": 8},
        runtime={"network": "none", "hostname": "localhost"},
        policy={"timeout_h": 6, "max_retries": 0, "retry_on": []},
    )
    original = fresh_spec(spec)
    rows = {
        Dataset: dataset,
        EmbodimentEnvRow: SimpleNamespace(status="active", spec_hash="contract_native"),
        EnvironmentProfile: SimpleNamespace(snapshot=profile),
        Run: SimpleNamespace(id="run_source", job_id="job_source", status="finished"),
        Job: SimpleNamespace(id="job_source", status="succeeded", spec=original.model_dump(mode="json")),
        TrainJob: SimpleNamespace(id="trn_source", job_id="job_source"),
    }
    artifacts = []
    session = SimpleNamespace(
        get=AsyncMock(side_effect=lambda model, key: rows[model]),
        rows=rows,
        artifacts=artifacts,
        scalars=AsyncMock(side_effect=lambda *_: SimpleNamespace(all=lambda: artifacts)),
    )
    settings = SimpleNamespace(
        data_root=state,
        ufo_native_root=tmp_path / "UFO",
        rlinf_repo=tmp_path / "RLinf",
        assets_root=tmp_path / "assets",
    )
    client = SimpleNamespace(
        get_run=AsyncMock(return_value={"status": "exited"}),
        image_info=AsyncMock(return_value={"ok": True, "digest": digest}),
        snapshot=AsyncMock(return_value={"path": str(code), "content_hash": digest}),
    )
    session.client = client
    monkeypatch.setattr(service, "EbdClient", lambda: client)
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    monkeypatch.setattr(service, "require_native_dataset", lambda row: report)
    monkeypatch.setattr(service, "compatible_profile", lambda *args: None)
    monkeypatch.setattr(service, "profile_snapshot", lambda row: row.snapshot)
    monkeypatch.setattr("eb.paths.validate_allowed_path", lambda value: Path(value).resolve())
    checkpoint = {
        "run_id": "run_source",
        "step": 112,
        "compatibility_fingerprint": plan["fingerprint"],
        "mode": "ufo_native_state",
        "native_path": str(native_path),
        "counts": {"optimizer_steps": 112, "local_time": 18432},
    }
    checkpoint["manifest"] = {"run_id": "run_source", "counts": checkpoint["counts"]}
    native_path.mkdir(parents=True)
    manifest_bytes = json.dumps(checkpoint["manifest"]).encode()
    (native_path / "checkpoint-manifest.json").write_bytes(manifest_bytes)
    checkpoint["receipt"] = {
        "run_id": "run_source",
        "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    receipt_bytes = json.dumps(checkpoint["receipt"]).encode()
    (native_path.parent / "receipt.json").write_bytes(receipt_bytes)
    spec.params["native_resume_source"] = {
        "schema_version": "eb/ufo-native-resume-source/v1",
        "run_id": "run_source",
        "manifest_sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        "receipt_sha256": "sha256:" + hashlib.sha256(receipt_bytes).hexdigest(),
        "artifact": None,
    }
    calls = []

    def checkpoint_reader(run_root, source_run_id: str, *, step=None):
        # Keep the production reader signature: step is keyword-only, never run_id.
        assert run_root == state / "runs/run_source"
        assert source_run_id == "run_source"
        assert step == 112
        calls.append((run_root, source_run_id, step))
        return checkpoint

    monkeypatch.setattr("eb.checkpoints.resume_checkpoint", checkpoint_reader)
    return spec, session, checkpoint, calls


def fresh_spec(resume):
    spec = resume.model_copy(deep=True)
    spec.inputs.resume_from = None
    spec.params.pop("source_train_job_id", None)
    spec.params.pop("native_resume_source", None)
    plan = spec.params["native_train_plan"]
    plan.pop("resume", None)
    plan.update(target_updates=112, start_updates=0, start_local_time=0)
    plan["native_args"] = native_arguments(plan)
    return spec


@pytest.mark.asyncio
async def test_native_resume_reads_exact_source_run_and_keyword_step(tmp_path, monkeypatch):
    spec, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    await validate_submission(session, spec)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "trainer_state"),
        ("native_path", "/other/run/outputs/native"),
        ("compatibility_fingerprint", "sha256:" + "c" * 64),
        ("counts", {"optimizer_steps": 96, "local_time": 17408}),
    ],
)
async def test_native_resume_rejects_checkpoint_binding_drift(tmp_path, monkeypatch, field, value):
    spec, session, checkpoint, calls = resume_fixture(tmp_path, monkeypatch)
    checkpoint[field] = value
    with pytest.raises(ValueError, match="恢复来源不兼容"):
        await validate_submission(session, spec)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_fresh_native_submission_checks_approved_image_and_current_source_origins(
    tmp_path, monkeypatch
):
    resume, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    await validate_submission(session, fresh_spec(resume))
    session.client.image_info.assert_awaited_once_with("eb/ufo-native:rpo-train-v3")
    assert session.client.snapshot.await_count == 3
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["image", "source_hash", "source_path"])
async def test_fresh_native_submission_rejects_changed_recipe_origins(tmp_path, monkeypatch, drift):
    resume, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    if drift == "image":
        session.client.image_info.return_value = {"ok": True, "digest": "sha256:" + "c" * 64}
    else:
        changed = dict(session.client.snapshot.return_value)
        changed["content_hash" if drift == "source_hash" else "path"] = (
            "sha256:" + "c" * 64 if drift == "source_hash" else str(tmp_path / "different-snapshot/code")
        )
        session.client.snapshot.return_value = changed
    with pytest.raises(ValueError, match="recipe (image|sources) changed"):
        await validate_submission(session, fresh_spec(resume))
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["wrong_source_job", "live_container", "source_image", "host_path"])
async def test_resume_cannot_substitute_producer_or_checkpoint_path(tmp_path, monkeypatch, drift):
    spec, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    if drift == "wrong_source_job":
        session.rows[TrainJob].job_id = "job_other"
    elif drift == "live_container":
        session.client.get_run.return_value = {"status": "running"}
    elif drift == "source_image":
        original = JobSpec.model_validate(session.rows[Job].spec)
        original.image = "sha256:" + "c" * 64
        original.params["native_train_plan"]["image_digest"] = original.image
        original.params["native_train_plan"]["identity"]["image_digest"] = original.image
        original.params["native_train_plan"]["fingerprint"] = training_fingerprint(
            original.params["native_train_plan"]["identity"]
        )
        session.rows[Job].spec = original.model_dump(mode="json")
    else:
        spec.params["native_train_plan"]["resume"]["host_path"] = str(tmp_path / "other-native")
    with pytest.raises(ValueError):
        await validate_submission(session, spec)
    assert len(calls) == (1 if drift == "host_path" else 0)


@pytest.mark.asyncio
async def test_native_job_rejects_extra_embodiment_snapshot_that_could_expose_holdout(tmp_path, monkeypatch):
    resume, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    spec = fresh_spec(resume)
    source = spec.params["ufo_source_ref"]
    (Path(source["path"]) / "validation.pkl").write_bytes(b"holdout fixture; API must not deserialize")
    spec.code_refs.embodiment = GitCodeRef.model_validate(source)
    with pytest.raises(ValueError):
        await validate_submission(session, spec)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("target_updates", [16, 1000016])
async def test_raw_native_job_cannot_bypass_operator_budget_bounds(tmp_path, monkeypatch, target_updates):
    resume, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    spec = fresh_spec(resume)
    plan = spec.params["native_train_plan"]
    plan["target_updates"] = target_updates
    # Supply the internally consistent argv an API caller could submit directly.
    endpoint = 11264 + target_updates // 16 * 1024
    for flag in ("--num-env-steps", "--checkpoint-every-steps", "--buffer-every-steps"):
        plan["native_args"][plan["native_args"].index(flag) + 1] = str(endpoint)
    with pytest.raises(ValueError):
        await validate_submission(session, spec)
    assert not calls
    session.client.image_info.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("start_updates", 16), ("start_local_time", 1024)])
async def test_raw_fresh_job_cannot_claim_resume_counters(tmp_path, monkeypatch, field, value):
    resume, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    spec = fresh_spec(resume)
    plan = spec.params["native_train_plan"]
    plan[field] = value
    plan["native_args"] = native_arguments(plan)
    with pytest.raises(ValueError):
        await validate_submission(session, spec)
    assert not calls
    session.client.image_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_plan_cannot_start_training_without_full_state_verification(tmp_path, monkeypatch):
    resume, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    spec = fresh_spec(resume)
    spec.params["native_train_plan"].pop("full_state_finite_required")
    with pytest.raises(ValueError, match="finite-state"):
        await validate_submission(session, spec)
    assert not calls
    session.client.image_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_cannot_add_finite_verification_to_an_unverified_source(tmp_path, monkeypatch):
    spec, session, _, calls = resume_fixture(tmp_path, monkeypatch)
    session.rows[Job].spec["params"]["native_train_plan"].pop("full_state_finite_required")
    with pytest.raises(ValueError, match="producer Job"):
        await validate_submission(session, spec)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "manifest_hash", "receipt_hash", "run_id", "artifact"])
async def test_new_native_resume_rejects_missing_or_changed_source_binding(tmp_path, monkeypatch, change):
    spec, session, _, _ = resume_fixture(tmp_path, monkeypatch)
    if change == "missing":
        spec.params.pop("native_resume_source")
    else:
        key = {"manifest_hash": "manifest_sha256", "receipt_hash": "receipt_sha256"}.get(change, change)
        spec.params["native_resume_source"][key] = "modified"
    with pytest.raises(ValueError, match="binding"):
        await validate_submission(session, spec)


@pytest.mark.asyncio
@pytest.mark.parametrize("file", ["checkpoint-manifest.json", "receipt.json"])
async def test_native_resume_rejects_source_bytes_changed_after_freezing_binding(tmp_path, monkeypatch, file):
    spec, session, checkpoint, _ = resume_fixture(tmp_path, monkeypatch)
    native = Path(checkpoint["native_path"])
    path = native / file if file == "checkpoint-manifest.json" else native.parent / file
    path.write_bytes(path.read_bytes() + b"\n")  # Same JSON object, different frozen bytes.
    with pytest.raises(ValueError, match="binding|changed"):
        await validate_submission(session, spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch", ["none", "canonical_hash", "manifest", "storage", "run_id", "pointer", "kind"]
)
async def test_resume_binding_checks_registered_artifact_canonical_manifest_and_identity(
    tmp_path, monkeypatch, mismatch
):
    from eb.ufo_training_service import build_resume_source_binding

    spec, session, checkpoint, _ = resume_fixture(tmp_path, monkeypatch)
    artifact = SimpleNamespace(
        id="art_source",
        kind="ufo_native_resume",
        run_id="run_source",
        sha256=plan_hash(checkpoint["manifest"]),
        storage=checkpoint["native_path"],
        meta={"manifest": checkpoint["manifest"]},
    )
    session.artifacts.append(artifact)
    session.rows[Run].metrics_summary = {"resume_artifact_id": artifact.id}
    if mismatch == "canonical_hash":
        artifact.sha256 = "sha256:" + "0" * 64
    elif mismatch == "manifest":
        artifact.meta = {"manifest": {"run_id": "different"}}
    elif mismatch in {"storage", "run_id", "kind"}:
        setattr(artifact, mismatch, "different")
    elif mismatch == "pointer":
        session.rows[Run].metrics_summary["resume_artifact_id"] = "art_other"
    if mismatch == "none":
        binding = await build_resume_source_binding(session, session.rows[Run], checkpoint)
        assert binding["artifact"] == {"id": artifact.id, "sha256": artifact.sha256}
        spec.params["native_resume_source"] = binding
        await validate_submission(session, spec)
    else:
        with pytest.raises(ValueError, match="Artifact"):
            await build_resume_source_binding(session, session.rows[Run], checkpoint)


@pytest.mark.asyncio
async def test_fresh_native_submission_cannot_carry_a_resume_source_binding(tmp_path, monkeypatch):
    resume, session, _, _ = resume_fixture(tmp_path, monkeypatch)
    spec = fresh_spec(resume)
    spec.params["native_resume_source"] = resume.params["native_resume_source"]
    with pytest.raises(ValueError, match="binding"):
        await validate_submission(session, spec)


@pytest.mark.asyncio
async def test_queued_resume_rechecks_source_before_creating_run_or_starting_container(tmp_path, monkeypatch):
    from eb.orchestrator.service import Orchestrator

    spec, session, checkpoint, _ = resume_fixture(tmp_path, monkeypatch)
    session.refresh, session.commit, session.add = AsyncMock(), AsyncMock(), Mock()
    job = SimpleNamespace(
        id="job_child",
        status="queued",
        kind="train.rl",
        attempts=0,
        version=0,
        embodiment_env_id=ENV_ID,
        spec=spec.model_dump(mode="json"),
    )
    path = Path(checkpoint["native_path"]) / "checkpoint-manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    client = SimpleNamespace(submit_run=AsyncMock())
    with pytest.raises(ValueError, match="binding|changed"):
        await Orchestrator(client).admit_and_start(session, job, [0])
    assert job.status == "blocked" and "source verification" in job.error
    assert job.attempts == 0
    session.add.assert_not_called()
    client.submit_run.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_resume_binding_detects_later_publication_or_missing_published_artifact(
    tmp_path, monkeypatch
):
    from eb.ufo_training_service import build_resume_source_binding

    spec, session, checkpoint, _ = resume_fixture(tmp_path, monkeypatch)
    artifact = SimpleNamespace(
        id="art_source",
        kind="ufo_native_resume",
        run_id="run_source",
        sha256=plan_hash(checkpoint["manifest"]),
        storage=checkpoint["native_path"],
        meta={"manifest": checkpoint["manifest"]},
    )
    session.artifacts.append(artifact)
    with pytest.raises(ValueError, match="binding"):
        await validate_submission(session, spec)
    binding = await build_resume_source_binding(session, session.rows[Run], checkpoint)
    spec.params["native_resume_source"] = binding
    session.rows[Run].metrics_summary = {"resume_artifact_id": artifact.id}
    session.artifacts.clear()
    with pytest.raises(ValueError, match="Artifact"):
        await validate_submission(session, spec)


@pytest.mark.asyncio
async def test_preview_to_submit_reads_actual_source_receipt_bytes_and_rejects_replacement(
    tmp_path, monkeypatch
):
    from fastapi import HTTPException

    from eb.api import routes_train as routes

    spec, session, checkpoint, _ = resume_fixture(tmp_path, monkeypatch)
    source_run, source_job = session.rows[Run], session.rows[Job]
    source_run.provenance = json.dumps({"image_digest": spec.image})
    source_job.kind, source_job.embodiment_env_id = "train.rl", ENV_ID
    checkpoint["state_coverage"] = {"model": "restored"}
    monkeypatch.setattr("eb.checkpoints.latest_resume_checkpoint", lambda *args, **kwargs: checkpoint)
    root = Path(checkpoint["native_path"]).parents[3]
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_root=root))
    render = SimpleNamespace(command=["native-worker"], mounts=[], image=spec.image)
    monkeypatch.setattr("eb.adapters.get_adapter", lambda _: SimpleNamespace(render=lambda *_: render))
    session.scalars = AsyncMock(
        side_effect=lambda query: SimpleNamespace(
            all=lambda: [source_run] if query.column_descriptions[0]["entity"] is Run else session.artifacts
        )
    )
    session.execute, session.scalar, session.commit = AsyncMock(), AsyncMock(), AsyncMock()
    submit = AsyncMock()
    monkeypatch.setattr(routes, "orch", SimpleNamespace(ebd=session.client, submit=submit))
    body = routes.ResumeTrainIn(name="continued", target_updates=512, checkpoint_step=112)
    preview = await routes.preview_resume("trn_source", body, session, SimpleNamespace(id="user:test"))
    binding = preview["native_resume_source"]
    assert binding["run_id"] == source_run.id and binding["artifact"] is None
    body.preview_hash = preview["preview_hash"]
    receipt_path = Path(checkpoint["native_path"]).parent / "receipt.json"
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    with pytest.raises(HTTPException) as expired:
        await routes.submit_resume("trn_source", body, session, SimpleNamespace(id="user:test"), "key")
    assert expired.value.status_code == 409
    submit.assert_not_awaited()
    session.execute.assert_not_awaited()
    session.scalar.assert_not_awaited()
    session.commit.assert_not_awaited()
