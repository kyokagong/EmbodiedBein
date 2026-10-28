import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from eb.api import routes_train
from eb.domain.job import JobKind, JobSpec
from eb.domain.rlinf_form import get_template, require_template_ready, template_image
from eb.native_training import recipe_for, service_for
from eb.pbfm_training import (
    CAMERA_CONTRACT,
    ENV_ID,
    SCHEMA,
    TASK_ID,
    TEMPLATE,
    TERRAIN_CONTRACT,
    native_arguments,
    training_fingerprint,
)
from eb.store.models import Artifact, Dataset, EmbodimentEnvRow, EnvironmentProfile, ModelVersion


def pbfm_spec(root=Path("/frozen")):
    digest = "sha256:" + "a" * 64
    identity = {
        "image_digest": digest,
        "world_size": 1,
        "num_envs_per_rank": 1024,
        "buffer_size": 262144,
        "seed": 42,
        "dataset_id": "ds_pbfm",
        "dataset_manifest": "b" * 64,
        "contract_hash": "a" * 64,
        "environment_profile": {"id": "env_train_pbfm", "hash": digest},
        "pbfm_snapshot": digest,
        "rlinf_snapshot": digest,
        "asset_snapshot": digest,
        "camera_contract": deepcopy(CAMERA_CONTRACT),
        "terrain_contract": deepcopy(TERRAIN_CONTRACT),
    }
    plan = {
        "schema_version": SCHEMA,
        "identity": identity,
        "fingerprint": training_fingerprint(identity),
        "image_digest": digest,
        "world_size": 1,
        "target_updates": 112,
        "start_updates": 0,
        "start_local_time": 0,
        "full_state_finite_required": True,
        "log_every_steps": 1024,
        "communication_timeout_seconds": 300,
        "segment_timeout_seconds": 21600,
        "source_refs": {
            name: {
                "path": f"/sources/{name}",
                "snapshot_hash": digest,
                "manifest_path": f"/sources/{name}-manifest.json",
            }
            for name in ("pbfm", "rlinf")
        },
    }
    plan["source_refs"]["asset"] = {
        "path": "/robot-assets",
        "snapshot_hash": digest,
        "manifest_path": "/sources/asset-manifest.json",
    }
    plan["native_args"] = native_arguments(plan)
    ref = {
        "kind": "content_snapshot",
        "path": str(root / "snapshots" / ("a" * 64) / "code"),
        "commit": digest,
    }
    return JobSpec(
        kind="train.rl",
        name="depth112",
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID,
        image=digest,
        params={
            "template_id": TEMPLATE,
            "stack": "rlinf",
            "native_train_plan": plan,
            "model_name": "depth112",
            "pbfm_source_ref": ref,
            "asset_source_ref": ref,
        },
        code_refs={"stack": ref},
        inputs={
            "datasets": [
                {"id": "ds_pbfm", "path": str(root / "train"), "validation": {"manifest_sha256": "b" * 64}}
            ]
        },
        resources={"gpus": 1, "cpus": 8, "memory_gb": 32, "shm_gb": 8},
        runtime={"network": "none", "hostname": "localhost"},
        policy={"timeout_h": 6, "max_retries": 0, "retry_on": []},
    )


def checkpoint_for(spec, root):
    native = root / "runs/run_source/outputs/native"
    return {
        "run_id": "run_source",
        "native_path": str(native),
        "path": str(native / "checkpoint"),
        "step": 112,
        "bytes": 10,
        "mode": "pbfm_native_state",
        "image_digest": spec.image,
        "compatibility_fingerprint": spec.params["native_train_plan"]["fingerprint"],
        "counts": {"optimizer_steps": 112, "local_time": 18432},
        "native_resume_source": {"schema_version": "eb/pbfm-native-resume-source/v1", "run_id": "run_source"},
    }


def test_native_dispatch_is_registered_and_old_ufo_stays_on_its_service():
    assert service_for(TEMPLATE).__name__ == "eb.pbfm_training_service"
    assert recipe_for(TEMPLATE).__name__ == "eb.pbfm_training"
    assert service_for("rpo_ufo_fb_native").__name__ == "eb.ufo_training_service"
    with pytest.raises(ValueError, match="Unregistered"):
        service_for("unknown_native_rl")
    tpl = get_template(TEMPLATE)
    assert tpl.ready is False and tpl.dependency_ready is False
    assert template_image(tpl) == "eb/rlinf-pbfm:rpo-depth-v1"
    with pytest.raises(ValueError, match="P-BFM"):
        require_template_ready(tpl)


@pytest.mark.asyncio
async def test_not_ready_template_reaches_preview_dependency_diagnostics(monkeypatch):
    from eb import pbfm_training_service

    body = routes_train.TrainIn(
        name="depth112",
        model_name="depth112",
        template_id=TEMPLATE,
        embodiment_id="emb_roboparty_rpo",
        pbfm_training={"target_updates": 112},
        gpus=1,
    )
    build = AsyncMock(return_value=pbfm_spec())
    monkeypatch.setattr(pbfm_training_service, "build_spec", build)
    assert (await routes_train._build_spec(body, NS())).params["template_id"] == TEMPLATE
    build.assert_awaited_once()
    with pytest.raises(ValueError, match="P-BFM"):
        await pbfm_training_service.validate_submission(NS(), pbfm_spec())


@pytest.mark.asyncio
async def test_pbfm_options_cannot_enter_ufo(monkeypatch):
    body = routes_train.TrainIn(
        name="depth",
        model_name="depth",
        template_id="rpo_ufo_fb_native",
        embodiment_id="emb_roboparty_rpo",
        pbfm_training={},
    )
    with pytest.raises(ValueError, match="pbfm_training"):
        await routes_train._build_spec(body, NS())
    with pytest.raises(ValidationError):
        routes_train.PbfmTrainingIn(prior_plane_envs=1)


def test_pbfm_preview_adapter_and_capacity_are_separate_and_have_no_writes(tmp_path, monkeypatch):
    from eb.adapters import get_adapter
    from eb.capacity.estimator import estimate_job

    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    spec = pbfm_spec(tmp_path)
    adapter = get_adapter(spec)
    rendered = adapter.render(spec, "run_preview")
    assert rendered.command[2] == "eb.runners.rlinf_pbfm_entry"
    assert not list(tmp_path.iterdir())
    assert all("holdout" not in src and "acceptance" not in src for src, _, _ in rendered.mounts)
    assert {dst for _, dst, _ in rendered.mounts if dst.startswith("/training-data/")} == {
        "/training-data/train.pkl",
        "/training-data/manifest.json",
    }
    one = estimate_job(spec)
    four = spec.model_copy(deep=True)
    four.resources.gpus = 4
    assert estimate_job(four).per_gpu_peak_gb == one.per_gpu_peak_gb
    assert one.verdict == "unknown" and one.basis["inputs"]["gpu_peak_measured"] is False


def test_pbfm_resume_preserves_identity_and_refuses_ufo_reconfigure(tmp_path):
    source = pbfm_spec(tmp_path)
    original = source.model_dump(mode="json")
    checkpoint = checkpoint_for(source, tmp_path)
    body = routes_train.ResumeTrainIn(name="depth512", target_updates=512)
    resumed = routes_train.build_resume_spec(source, "trn_source", checkpoint, body)
    assert resumed.params["resume_mode"] == "pbfm_native_state"
    assert resumed.params["native_train_plan"]["start_local_time"] == 18432
    assert resumed.params["native_train_plan"]["identity"] == source.params["native_train_plan"]["identity"]
    assert source.model_dump(mode="json") == original
    for change in (
        {"native_cuda_allocator": "expandable_segments"},
        {"native_new_budget": True},
        {"native_experiment_target_updates": 1008},
        {"native_reconfigure": {"num_envs_per_rank": 2048, "buffer_size": 262144, "replay_policy": "fresh"}},
    ):
        with pytest.raises(ValueError, match="P-BFM"):
            routes_train.build_resume_spec(
                source,
                "trn_source",
                checkpoint,
                routes_train.ResumeTrainIn(name="depth512", target_updates=512, **change),
            )


@pytest.mark.parametrize("change", ["mode", "binding", "image"])
def test_pbfm_resume_rejects_incompatible_source(tmp_path, change):
    source = pbfm_spec(tmp_path)
    checkpoint = checkpoint_for(source, tmp_path)
    if change == "mode":
        checkpoint["mode"] = "ufo_native_state"
    elif change == "binding":
        checkpoint["native_resume_source"]["schema_version"] = "eb/ufo-native-resume-source/v1"
    else:
        checkpoint["image_digest"] = "sha256:" + "c" * 64
    with pytest.raises(ValueError, match="P-BFM"):
        routes_train.build_resume_spec(
            source, "trn_source", checkpoint, routes_train.ResumeTrainIn(name="depth512", target_updates=512)
        )


def test_pbfm_checkpoint_never_calls_ufo_reader(tmp_path, monkeypatch):
    from eb.checkpoints import resume_checkpoint

    path = tmp_path / "outputs/native"
    path.mkdir(parents=True)
    (path / "checkpoint-manifest.json").write_text(
        json.dumps({"schema_version": "eb/pbfm-native-checkpoint/v1"})
    )
    source = pbfm_spec(tmp_path)
    checkpoint = checkpoint_for(source, tmp_path)
    reader = Mock(return_value=checkpoint)
    monkeypatch.setitem(sys.modules, "eb.pbfm_training_checkpoint", NS(read_pbfm_checkpoint=reader))
    assert resume_checkpoint(tmp_path, "run_source", spec=source)["mode"] == "pbfm_native_state"
    reader.assert_called_once_with(tmp_path, source, "run_source", verify_hashes=False)
    with pytest.raises(ValueError, match="frozen producer JobSpec"):
        resume_checkpoint(tmp_path, "run_source")


@pytest.mark.asyncio
async def test_pbfm_binding_ties_exact_owned_run_receipt_and_manifest(tmp_path, monkeypatch):
    from eb import pbfm_training_service as service

    source = pbfm_spec(tmp_path)
    checkpoint = checkpoint_for(source, tmp_path)
    native = Path(checkpoint["native_path"])
    native.mkdir(parents=True)
    checkpoint["manifest"] = {"schema_version": "eb/pbfm-native-checkpoint/v1", "run_id": "run_source"}
    raw = json.dumps(checkpoint["manifest"]).encode()
    (native / "checkpoint-manifest.json").write_bytes(raw)
    checkpoint["receipt"] = {
        "run_id": "run_source",
        "native_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (native.parent / "receipt.json").write_text(json.dumps(checkpoint["receipt"]))
    session = NS(scalars=AsyncMock(return_value=NS(all=lambda: [])))
    monkeypatch.setattr(service, "get_settings", lambda: NS(data_root=tmp_path))
    binding = await service.build_resume_source_binding(session, NS(id="run_source"), checkpoint)
    assert binding["schema_version"] == "eb/pbfm-native-resume-source/v1" and binding["artifact"] is None
    (native.parent / "receipt.json").write_text(json.dumps({**checkpoint["receipt"], "run_id": "run_other"}))
    with pytest.raises(ValueError, match="changed"):
        await service.build_resume_source_binding(session, NS(id="run_source"), checkpoint)


@pytest.mark.asyncio
@pytest.mark.parametrize("num_envs", [1024, 2048])
async def test_pbfm_build_freezes_sources_data_contract_and_image(tmp_path, monkeypatch, num_envs):
    from eb import pbfm_training_service as service

    spec = pbfm_spec(tmp_path)
    digest = spec.image
    ref = spec.params["pbfm_source_ref"]
    dataset = NS(id="ds_pbfm", dest_path=str(tmp_path / "train"))
    rows = {
        Dataset: dataset,
        EmbodimentEnvRow: NS(
            status="active", spec_hash=spec.params["native_train_plan"]["identity"]["contract_hash"]
        ),
        EnvironmentProfile: NS(snapshot=spec.params["native_train_plan"]["identity"]["environment_profile"]),
    }
    session = NS(get=AsyncMock(side_effect=lambda model, key: rows[model]))
    client = NS(
        image_info=AsyncMock(return_value={"ok": True, "digest": digest}),
        snapshot=AsyncMock(return_value={"content_hash": digest, "path": ref["path"]}),
    )
    monkeypatch.setattr(service, "EbdClient", lambda: client)
    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: NS(
            data_root=tmp_path,
            pbfm_root=tmp_path / "P-BFM",
            rlinf_repo=tmp_path / "RLinf",
            assets_root=tmp_path / "assets",
        ),
    )
    monkeypatch.setattr(service, "compatible_profile", lambda *args: None)
    monkeypatch.setattr(service, "profile_snapshot", lambda row: row.snapshot)
    monkeypatch.setattr(service, "require_pbfm_dataset", lambda row: {"manifest_sha256": "b" * 64})
    monkeypatch.setattr("eb.api.routes_data.reconcile_dataset", AsyncMock())
    body = routes_train.TrainIn(
        name="depth",
        model_name="depth",
        template_id=TEMPLATE,
        embodiment_id="emb_roboparty_rpo",
        gpus=4,
        dataset_ids=["ds_pbfm"],
        environment_id="env_train_pbfm",
        pbfm_training={"num_envs_per_rank": num_envs, "buffer_size": 131072},
    )
    frozen = await service.build_spec(body, session)
    assert frozen.params["native_train_plan"]["identity"]["camera_contract"] == CAMERA_CONTRACT
    plan = frozen.params["native_train_plan"]
    assert plan["identity"]["num_envs_per_rank"] == num_envs
    assert plan["identity"]["buffer_size"] == 131072
    assert plan["log_every_steps"] == 4096
    assert plan["fingerprint"] == training_fingerprint(plan["identity"])
    argv = dict(zip(plan["native_args"][::2], plan["native_args"][1::2]))
    assert argv["--num-envs"] == str(num_envs)
    assert argv["--num-env-steps"] == ("73728" if num_envs == 1024 else "106496")
    assert frozen.code_refs.stack.kind == "content_snapshot"
    assert client.snapshot.await_count == 3
    body.native_training = routes_train.NativeTrainingIn()
    with pytest.raises(ValueError, match="pbfm_training"):
        await service.build_spec(body, session)


def test_pbfm_environment_count_default_and_restricted_options():
    assert routes_train.PbfmTrainingIn().num_envs_per_rank == 1024
    assert routes_train.PbfmTrainingIn(num_envs_per_rank=2048).num_envs_per_rank == 2048
    for value in (True, False, 512, 4096, "2048"):
        with pytest.raises(ValidationError):
            routes_train.PbfmTrainingIn(num_envs_per_rank=value)


def test_pbfm_2048_resume_freezes_topology_and_uses_each_native_endpoint(tmp_path):
    from eb.pbfm_training_service import build_resume_spec

    source = pbfm_spec(tmp_path)
    plan = source.params["native_train_plan"]
    plan.update(world_size=4, log_every_steps=4096)
    plan["identity"].update(world_size=4, num_envs_per_rank=2048, buffer_size=131072)
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["native_args"] = native_arguments(plan)
    source.resources.gpus, source.resources.cpus, source.resources.memory_gb = 4, 32, 80
    identity = deepcopy(plan["identity"])
    checkpoint = checkpoint_for(source, tmp_path)
    checkpoint["counts"]["local_time"] = 26624
    for target, start, start_local, endpoint in ((368, 112, 26624, "245760"), (512, 368, 61440, "327680")):
        checkpoint.update(step=start)
        checkpoint["counts"].update(optimizer_steps=start, local_time=start_local)
        resumed = build_resume_spec(
            source, "trn_source", checkpoint,
            routes_train.ResumeTrainIn(name=f"depth{target}", target_updates=target),
        )
        resumed_plan = resumed.params["native_train_plan"]
        assert resumed_plan["identity"] == identity
        assert resumed_plan["start_updates"] == start
        assert resumed_plan["start_local_time"] == start_local
        argv = dict(zip(resumed_plan["native_args"][::2], resumed_plan["native_args"][1::2]))
        assert argv["--num-envs"] == "2048"
        assert argv["--num-env-steps"] == argv["--checkpoint-every-steps"] == endpoint
        source = resumed

    checkpoint["counts"]["local_time"] += 1024
    with pytest.raises(ValueError):
        build_resume_spec(
            source, "trn_source", checkpoint, routes_train.ResumeTrainIn(name="bad", target_updates=512)
        )


@pytest.mark.asyncio
async def test_pbfm_submit_rechecks_frozen_sources_and_profile(tmp_path, monkeypatch):
    from eb import pbfm_training_service as service

    spec = pbfm_spec(tmp_path)
    code = Path(spec.params["pbfm_source_ref"]["path"])
    code.mkdir(parents=True)
    identity = spec.params["native_train_plan"]["identity"]
    rows = {
        Dataset: NS(dest_path=spec.inputs.datasets[0]["path"]),
        EmbodimentEnvRow: NS(status="active", spec_hash=identity["contract_hash"]),
        EnvironmentProfile: NS(snapshot=identity["environment_profile"]),
    }
    session = NS(get=AsyncMock(side_effect=lambda model, key: rows[model]))
    client = NS(
        image_info=AsyncMock(return_value={"ok": True, "digest": spec.image}),
        snapshot=AsyncMock(return_value={"path": str(code), "content_hash": spec.image}),
    )
    monkeypatch.setattr(service, "EbdClient", lambda: client)
    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: NS(
            data_root=tmp_path,
            pbfm_root=tmp_path / "P-BFM",
            rlinf_repo=tmp_path / "RLinf",
            assets_root=tmp_path / "assets",
        ),
    )
    monkeypatch.setattr(service, "require_pbfm_dataset", lambda row: spec.inputs.datasets[0]["validation"])
    monkeypatch.setattr(service, "compatible_profile", lambda *args: None)
    monkeypatch.setattr(service, "profile_snapshot", lambda row: row.snapshot)
    monkeypatch.setattr("eb.paths.validate_allowed_path", lambda value: Path(value))
    template = get_template(TEMPLATE)
    monkeypatch.setattr(
        "eb.domain.rlinf_form.get_template",
        lambda value: replace(template, ready=True, dependency_ready=True),
    )
    await service.validate_submission(session, spec)
    client.snapshot.return_value = {"path": str(code), "content_hash": "sha256:" + "c" * 64}
    with pytest.raises(ValueError, match="sources changed"):
        await service.validate_submission(session, spec)
    client.snapshot.return_value = {"path": str(code), "content_hash": spec.image}
    rows[EnvironmentProfile].snapshot = {**identity["environment_profile"], "hash": "changed"}
    with pytest.raises(ValueError, match="Profile"):
        await service.validate_submission(session, spec)


@pytest.mark.asyncio
async def test_pbfm_probe_uses_asset_job_and_bypasses_only_readiness(tmp_path, monkeypatch):
    from eb import pbfm_runtime_probe as probe
    from eb import pbfm_training_service as service

    source = pbfm_spec(tmp_path)
    source.params["native_train_plan"]["identity"]["buffer_size"] = 32768
    source.params["native_train_plan"]["fingerprint"] = training_fingerprint(
        source.params["native_train_plan"]["identity"]
    )
    source.params["native_train_plan"]["segment_timeout_seconds"] = 600
    source.params["native_train_plan"]["native_args"] = native_arguments(source.params["native_train_plan"])
    source.params["environment_profile"] = source.params["native_train_plan"]["identity"][
        "environment_profile"
    ]
    source.policy.timeout_h = 600 / 3600
    build = AsyncMock(return_value=source)
    validate = AsyncMock()
    monkeypatch.setattr(service, "_build_spec", build)
    monkeypatch.setattr(service, "_validate_submission", validate)
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    spec = await probe.build_spec(
        probe.ProbeIn(name="runtime", dataset_id="ds_pbfm", environment_id="env_draft"), NS()
    )
    assert spec.kind == "asset.validate" and "template_id" not in spec.params
    assert build.await_args.kwargs == {"diagnostic": True}
    await probe.validate_submission(NS(), spec)
    assert validate.await_args.kwargs == {"diagnostic": True}
    from eb.adapters import get_adapter

    rendered = get_adapter(spec).render(spec, "run_preview")
    assert rendered.command[-1] == "--probe" and rendered.metrics_glob == ""
    assert not list(tmp_path.iterdir())
    spec.params["template_id"] = TEMPLATE
    with pytest.raises(ValueError, match="protected diagnostic"):
        probe.render(spec, "run_preview")
    spec.params.pop("template_id")
    plan = spec.params["native_train_plan"]
    receipt = {
        "schema_version": "eb/pbfm-native-runtime-probe/v1",
        "run_id": "run_probe",
        "status": "complete",
        "plan_hash": training_fingerprint({**plan, "run_id": "run_probe"}),
        "world_size": 1,
        "training_completed": False,
        "ranks": [
            {
                "rank": 0,
                "world_size": 1,
                "pid": 100,
                "gpu_uuid": "GPU-probe",
                "worker_class": "PbfmTrainWorker",
                "status": "probe_complete",
                "training_completed": False,
                "environment_reset_step": True,
                "egl_render_verified": True,
                "depth_verified": True,
                "process_group": False,
                "source_refs": plan["source_refs"],
                "native_contract": {
                    "actor_scalar_dim": 380,
                    "action_dim": 23,
                    "physics_hz": 400,
                    "policy_hz": 50,
                    "camera_contract": CAMERA_CONTRACT,
                },
            }
        ],
        "files": [
            {"path": f"runtime/probe-rank-0/{name}", "bytes": 1, "sha256": "b" * 64}
            for name in ("network-depth.png", "network-depth.npy", "metric-depth.npy", "robot-terrain.png")
        ],
    }
    monkeypatch.setattr("eb.pbfm_io.safe_file", lambda *args: Path("receipt"))
    monkeypatch.setattr("eb.pbfm_io.read_json", lambda *args: receipt)
    verify = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", verify)
    run = NS(id="run_probe", metrics_summary={})
    await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), run, tmp_path)
    assert run.metrics_summary["training_completed"] is False
    assert "model_version_id" not in run.metrics_summary
    verify.assert_awaited_once()
    receipt["ranks"][0]["depth_verified"] = False
    with pytest.raises(ValueError, match="rank failed"):
        await probe.collect(NS(), NS(spec=spec.model_dump(mode="json")), run, tmp_path)
    failure = await probe.build_spec(
        probe.ProbeIn(name="failure", dataset_id="ds_pbfm", environment_id="env_draft", failure_rank=0), NS()
    )
    rendered_failure = probe.render(failure, "run_preview")
    assert rendered_failure.command[-3:] == ["--probe", "--probe-fail-rank", "0"]
    assert failure.canonical_hash() != spec.canonical_hash()
    with pytest.raises(ValueError, match="must fail the Job"):
        await probe.collect(NS(), NS(spec=failure.model_dump(mode="json")), run, tmp_path)
    with pytest.raises(ValidationError, match="leased probe rank"):
        probe.ProbeIn(
            name="failure", dataset_id="ds_pbfm", environment_id="env_draft", gpus=1, failure_rank=1
        )
    with pytest.raises(ValidationError):
        routes_train.PbfmTrainingIn(failure_rank=0)


@pytest.mark.asyncio
async def test_pbfm_collector_publishes_separate_model_family_and_resume_artifact(tmp_path, monkeypatch):
    from eb.orchestrator import pbfm_training as collector

    source = pbfm_spec(tmp_path)
    checkpoint = checkpoint_for(source, tmp_path)
    checkpoint.update(
        full_state_finite_verified=True,
        limitations=["environment is reset"],
        state_coverage={"model": "restored", "replay_per_rank": "restored", "environment": "not_available"},
        manifest={
            "files": [{"path": "checkpoint/model/model.safetensors", "bytes": 10, "sha256": "b" * 64}],
            "image_digest": source.image,
            "source_refs": source.params["native_train_plan"]["source_refs"],
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "eb.pbfm_training_checkpoint",
        NS(read_pbfm_checkpoint=Mock(return_value=checkpoint), MAX_FILES=10000),
    )
    monkeypatch.setattr(collector, "read_pbfm_training_evidence", AsyncMock(return_value={"available": True}))
    verify = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", verify)
    session = NS(
        scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=None), add=Mock(), flush=AsyncMock()
    )
    run = NS(id="run_source", metrics_summary={})
    await collector.collect_pbfm_training(
        session, NS(id="job_source", spec=source.model_dump(mode="json")), run, tmp_path
    )
    added = [call.args[0] for call in session.add.call_args_list]
    artifacts = [row for row in added if isinstance(row, Artifact)]
    model = next(row for row in added if isinstance(row, ModelVersion))
    assert {row.kind for row in artifacts} == {"pbfm_native_resume", "pbfm_native_trained_model"}
    assert model.family_id == "mf_rpo_pbfm_depth" and model.card["qualification"] == "incomplete"
    assert model.card["train_plan"] == source.params["native_train_plan"]
    assert run.metrics_summary["resume_mode"] == "pbfm_native_state"
    verify.assert_awaited_once_with(
        Path(checkpoint["native_path"]),
        checkpoint["manifest"]["files"],
        max_files=10000,
        max_file_bytes=1024**4,
    )
    session.add.reset_mock()
    session.scalar.side_effect = [artifacts[0], artifacts[1], model]
    await collector.collect_pbfm_training(
        session, NS(id="job_source", spec=source.model_dump(mode="json")), run, tmp_path
    )
    session.add.assert_not_called()
    verify.side_effect = ValueError("Native output hash differs")
    with pytest.raises(ValueError, match="hash differs"):
        await collector.collect_pbfm_training(
            session, NS(id="job_source", spec=source.model_dump(mode="json")), run, tmp_path
        )
    session.add.assert_not_called()
    verify.side_effect = None
    checkpoint["full_state_finite_verified"] = False
    with pytest.raises(ValueError, match="finite-state"):
        await collector.collect_pbfm_training(
            session, NS(id="job_source", spec=source.model_dump(mode="json")), run, tmp_path
        )


@pytest.mark.asyncio
async def test_pbfm_preview_hash_and_probe_submit_are_bound_without_trainjob(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from eb import pbfm_runtime_probe as probe
    from eb.api import routes_pbfm

    spec = pbfm_spec(tmp_path)
    monkeypatch.setattr(routes_train, "_build_spec", AsyncMock(return_value=spec))
    body = routes_train.TrainIn(
        name="depth112",
        model_name="depth112",
        template_id=TEMPLATE,
        embodiment_id="emb_roboparty_rpo",
        gpus=1,
        pbfm_training={},
    )
    principal = NS(id="user:kk", actor_type="user")
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    preview = await routes_train.preview_train(body, NS(), principal)
    assert preview["preview_hash"] == "sha256:" + spec.canonical_hash()
    with pytest.raises(HTTPException) as stale:
        await routes_train.submit_train(body, NS(), principal, "idempotent")
    assert stale.value.status_code == 400 and "预览" in stale.value.detail
    params = {
        **spec.params,
        "operation": probe.OPERATION,
        "probe_recipe_id": TEMPLATE,
        "environment_profile": spec.params["native_train_plan"]["identity"]["environment_profile"],
    }
    params.pop("template_id")
    diagnostic = spec.model_copy(update={"kind": JobKind.ASSET_VALIDATE, "params": params})
    monkeypatch.setattr(probe, "build_spec", AsyncMock(return_value=diagnostic))
    monkeypatch.setattr(
        probe, "render", lambda *args: NS(image=spec.image, command=["python", "--probe"], mounts=[])
    )
    request = probe.ProbeIn(name="A0", dataset_id="ds_pbfm", environment_id="env_draft")
    session = NS(execute=AsyncMock(), scalar=AsyncMock(return_value=None), add=Mock())
    preview = await routes_pbfm.preview_runtime_probe(request, session, principal)
    assert preview["diagnostic_only"] is True and preview["training_ready"] is False
    with pytest.raises(HTTPException) as stale:
        await routes_pbfm.submit_runtime_probe(request, session, principal, "idempotent")
    assert stale.value.status_code == 409
    request.preview_hash = preview["preview_hash"]
    submit = AsyncMock(return_value=NS(id="job_probe", status="queued"))
    monkeypatch.setattr(routes_pbfm, "Orchestrator", lambda: NS(submit=submit))
    result = await routes_pbfm.submit_runtime_probe(request, session, principal, "idempotent")
    assert result["job_id"] == "job_probe" and result["diagnostic_only"] is True
    assert submit.await_args.args[1].kind == "asset.validate"
    session.add.assert_not_called()
