import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from eb.api import routes_pbfm as routes
from eb.pbfm_data import ENV_ID, EXPORT_OPERATION, OPERATION
from eb.pbfm_evaluation import EVAL_OPERATION, action_spec, validate_action_plan
from eb.pbfm_io import fingerprint, sha256, write_json
from eb.store.ebd import EbdClient
from eb.store.models import Artifact, Dataset, Job, ModelVersion, Run


@pytest.mark.asyncio
@pytest.mark.parametrize("periodic", [False, True])
async def test_export_plan_binds_published_hashes_and_preserves_model_card(tmp_path, monkeypatch, periodic):
    dataset_root = tmp_path / "motion/data"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train.pkl").write_bytes(b"train")
    manifest = {
        "schema_version": "eb/pbfm-rpo-motion/v1",
        "dataset_id": "ds_motion",
        "split": "train",
        "train_file": "train.pkl",
        "file_sha256": sha256(dataset_root / "train.pkl"),
        "native_loader": {"all_motions_loaded": True},
        "fk": {"verified": True},
        "joint_order": [f"j{x}" for x in range(23)],
        "parent_version_id": "dsv_original",
        "robot_xml_sha256": "f" * 64,
        "pbfm_patch_sha256": "a" * 64,
    }
    write_json(dataset_root / "manifest.json", manifest)
    validation = {
        "scope": "pbfm_rpo_motion_v1",
        "status": "validated",
        "training_ready": True,
        "manifest_sha256": sha256(dataset_root / "manifest.json"),
        "file_sha256": manifest["file_sha256"],
    }
    dataset = Dataset(
        id="ds_motion",
        name="motion",
        source_kind="pbfm_rpo_motion",
        format="pbfm_rpo_motion",
        embodiment_env_id=ENV_ID,
        status="validated",
        dest_path=str(dataset_root),
        summary_stats={"validation": validation},
        meta={"usage": "train"},
    )
    holdout = dataset_root.parent / "holdout"
    holdout.mkdir()
    write_json(
        holdout / "manifest.json",
        {"split": "validation", "parent_version_id": "dsv_original", "file_sha256": "b" * 64},
    )
    manifest["holdout_manifests"] = {"validation": sha256(holdout / "manifest.json")}
    (dataset_root / "manifest.json").unlink()
    write_json(dataset_root / "manifest.json", manifest)
    validation["manifest_sha256"] = sha256(dataset_root / "manifest.json")
    native = tmp_path / "runs/run_parent/outputs" / (
        "recovery/generations/u59968/native" if periodic else "native")
    files = []
    for name in (
        "config.json",
        "effective_config.json",
        "source-adaptation.json",
        "checkpoint/model/config.json",
        "checkpoint/model/init_kwargs.json",
        "checkpoint/model/model.safetensors",
    ):
        path = native / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque frozen checkpoint")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    resume_manifest = {"files": files}
    network_manifest = {"files": [row for row in files if "checkpoint/model/" in row["path"]]}
    card = {
        "schema_version": "eb/rpo-pbfm-trained-model/v1",
        "resume_artifact_id": "art_resume",
        "run_id": "run_parent",
    }
    model = ModelVersion(
        id="mv_pbfm",
        family_id="mf_rpo_pbfm_depth",
        embodiment_env_id=ENV_ID,
        artifact_id="art_model",
        card=card,
    )
    network = Artifact(
        id="art_model",
        kind="pbfm_native_trained_model",
        run_id="run_parent",
        sha256="sha256:" + fingerprint(network_manifest),
        meta={"manifest": network_manifest},
    )
    resume = Artifact(
        id="art_resume",
        kind="pbfm_native_resume",
        run_id="run_parent",
        storage=str(native),
        sha256="sha256:" + fingerprint(resume_manifest),
        meta={"manifest": resume_manifest},
    )
    run = Run(id="run_parent", job_id="job_train", status="finished")
    job = Job(
        id="job_train",
        status="succeeded",
        spec={
            "image": "sha256:" + "e" * 64,
            "params": {
                "native_train_plan": {
                    "identity": {
                        "dataset_id": "ds_motion",
                        "dataset_manifest": validation["manifest_sha256"],
                        "camera_contract": {"mount_body": "torso_link"},
                        "terrain_contract": {"mode": "training_mix"},
                        "pbfm_snapshot": "sha256:" + "c" * 64,
                        "asset_snapshot": "sha256:" + "d" * 64,
                    }
                }
            },
        },
    )
    if periodic:
        run.status = "aborted"
        job.status = "cancelled"
        card["periodic_checkpoint"] = {"step": 59968}
        from unittest.mock import AsyncMock

        monkeypatch.setattr("eb.pbfm_periodic.periodic_model_source", AsyncMock(return_value={
            "native_path": str(native), "manifest": resume_manifest}))
    values = {
        (ModelVersion, model.id): model,
        (Artifact, network.id): network,
        (Artifact, resume.id): resume,
        (Run, run.id): run,
        (Job, job.id): job,
        (Dataset, dataset.id): dataset,
    }

    async def get(cls, identity):
        return values.get((cls, identity))

    async def scan(_, path):
        from pathlib import Path

        file = Path(path)
        return {"files": 1, "samples": [{"size": file.stat().st_size, "sha256": sha256(file)}]}

    async def image_info(_, name):
        return {"ok": True, "digest": "sha256:" + "e" * 64}

    monkeypatch.setattr(EbdClient, "scan", scan)
    monkeypatch.setattr(EbdClient, "image_info", image_info)
    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    before = deepcopy(model.card)
    plan = await routes.build_action_plan(
        model.id, EXPORT_OPERATION, routes.ActionIn(), SimpleNamespace(get=get)
    )
    validate_action_plan(plan)
    assert action_spec(plan).kind == "data.convert"
    assert model.card == before
    for field, value in (
        ("camera_contract", {"mount_body": "wrong"}),
        ("model_hash", "f" * 64),
        ("motion_id", 1),
        ("prompt_manifest_sha256", "a" * 64),
    ):
        changed = deepcopy(plan)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_action_plan(changed)
    (native / "config.json").write_bytes(b"modified after immutable publication")
    with pytest.raises(ValueError, match="changed after collection"):
        await routes.build_action_plan(
            model.id, EXPORT_OPERATION, routes.ActionIn(), SimpleNamespace(get=get)
        )
    assert model.card == before


def test_pbfm_operator_routes_are_registered_without_ufo_dispatch():
    paths = {route.path for route in routes.router.routes}
    assert {
        "/pbfm/models/{mid}/export/preview",
        "/pbfm/models/{mid}/export",
        "/pbfm/models/{mid}/evals/preview",
        "/pbfm/models/{mid}/evals",
        "/pbfm/datasets/prepare/preview",
        "/pbfm/datasets/prepare",
        "/pbfm/datasets/from-ufo/preview",
        "/pbfm/datasets/from-ufo",
    } <= paths


@pytest.mark.asyncio
async def test_prepare_retry_preserves_existing_dataset_identity(monkeypatch):
    from eb import pbfm_data

    plan = {"hash": "a" * 64}
    previous = Job(id="job_data", status="running", spec={"params": {"dataset_id": "ds_existing"}})

    async def build(body, session):
        return plan

    async def scalar(query):
        return previous

    async def submit(*args):
        return {"id": previous.id, "status": previous.status}

    monkeypatch.setattr(routes, "build_data_plan", build)
    monkeypatch.setattr(routes, "submit", submit)
    monkeypatch.setattr(pbfm_data, "spec_for", lambda plan, did: {"dataset_id": did})
    result = await routes.prepare_data(
        routes.DataIn(
            name="motion",
            parent_version_id="dsv_source",
            source_manifest="manifest.json",
            preview_hash=plan["hash"],
        ),
        SimpleNamespace(scalar=scalar),
        SimpleNamespace(actor_type="user"),
        "retry-key",
    )
    assert result == {"id": "job_data", "status": "running", "dataset_id": "ds_existing"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_status,run_status",
    [("failed", "failed"), ("timeout", "aborted"), ("infra_failed", "aborted"), ("cancelled", "aborted")],
)
async def test_failed_eval_only_serves_indexed_partial_media(tmp_path, monkeypatch, job_status, run_status):
    from fastapi import HTTPException

    from eb.api import routes_ufo_native
    from eb.store.models import EvalReport

    root = tmp_path / "runs/run_failed/outputs"
    (root / "videos").mkdir(parents=True)
    media = root / "videos/first.mp4"
    media.write_bytes(b"verified partial video")
    row = {"path": "videos/first.mp4", "bytes": media.stat().st_size, "sha256": sha256(media)}
    run = Run(id="run_failed", job_id="job_failed", status=run_status)
    job = Job(
        id="job_failed",
        status=job_status,
        spec={
            "embodiment_env_id": ENV_ID,
            "params": {"operation": "evaluate_pbfm_terrain", "pbfm_plan": {"hash": "a" * 64}},
        },
    )
    report = EvalReport(
        id="ev_partial",
        run_id=run.id,
        gate_results={"qualification": "incomplete", "plan_hash": "a" * 64, "partial_files": [row]},
    )

    async def get(cls, key):
        return run if cls is Run else job if cls is Job else None

    async def scalar(query):
        return report

    async def scan(_, path):
        return {"files": 1, "samples": [{"size": row["bytes"], "sha256": row["sha256"]}]}

    async def stream(request, rid, path):
        return {"rid": rid, "path": path}

    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(EbdClient, "scan", scan)
    monkeypatch.setattr(routes_ufo_native, "stream_output", stream)
    session = SimpleNamespace(get=get, scalar=scalar)
    served = await routes.output_file(run.id, row["path"], None, session, None)
    assert served["path"] == row["path"]
    with pytest.raises(HTTPException) as error:
        await routes.output_file(run.id, "unindexed.mp4", None, session, None)
    assert error.value.status_code == 404
    run.status = "running"
    with pytest.raises(HTTPException):
        await routes.output_file(run.id, row["path"], None, session, None)


@pytest.mark.asyncio
async def test_segment112_serves_only_verified_indexed_training_analysis(tmp_path, monkeypatch):
    from pathlib import Path

    from fastapi import HTTPException

    from eb.api import routes_ufo_native

    root = tmp_path / "runs/run_segment/outputs"
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    chart = analysis / "chart.svg"
    chart.write_bytes(b"<svg>frozen</svg>")
    (analysis / "unindexed.svg").write_bytes(b"<svg>unindexed</svg>")
    entry = {"path": "analysis/chart.svg", "bytes": chart.stat().st_size, "sha256": sha256(chart)}
    index = analysis / "index.json"
    write_json(index, {
        "schema_version": "eb/pbfm-training-evidence/v1", "run_id": "run_segment", "files": [entry],
    })
    index_entry = {"path": "analysis/index.json", "bytes": index.stat().st_size, "sha256": sha256(index)}
    run = Run(id="run_segment", job_id="job_segment", status="finished",
              metrics_summary={"training_evidence": {
                  "available": True, "run_id": "run_segment", "files": [index_entry, entry],
              }})
    job = Job(id="job_segment", kind="asset.validate", status="succeeded",
              spec={"embodiment_env_id": ENV_ID,
                    "params": {"operation": "probe_pbfm_runtime", "probe_mode": "segment112"}})

    async def get(cls, key):
        return run if cls is Run else job if cls is Job else None

    async def scalar(query):
        return None

    async def scan(_, name):
        path = Path(name)
        return {"files": 1, "samples": [{"size": path.stat().st_size, "sha256": sha256(path)}]}

    async def stream(request, rid, path):
        return {"rid": rid, "path": path}

    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(EbdClient, "scan", scan)
    monkeypatch.setattr(routes_ufo_native, "stream_output", stream)
    session = SimpleNamespace(get=get, scalar=scalar)
    for path in (entry["path"], index_entry["path"]):
        assert await routes.output_file(run.id, path, None, session, None) == {"rid": run.id, "path": path}
    with pytest.raises(HTTPException) as error:
        await routes.output_file(run.id, "analysis/unindexed.svg", None, session, None)
    assert error.value.status_code == 404
    chart.write_bytes(b"<svg>broken</svg>")
    with pytest.raises(HTTPException) as error:
        await routes.output_file(run.id, entry["path"], None, session, None)
    assert error.value.status_code == 404
    # Updating the mutable disk index must not replace the collected hash.
    index.write_text(json.dumps({
        "schema_version": "eb/pbfm-training-evidence/v1", "run_id": run.id,
        "files": [{**entry, "sha256": sha256(chart)}],
    }))
    with pytest.raises(HTTPException) as error:
        await routes.output_file(run.id, entry["path"], None, session, None)
    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,relative", [
    (EVAL_OPERATION, "videos/case.mp4"),
    (EXPORT_OPERATION, "policy.zip"),
    (EXPORT_OPERATION, "policy/source/pbfm/__init__.py"),
    (OPERATION, "data/train.pkl"),
])
async def test_output_binds_collected_receipt_when_file_and_disk_receipt_change(
    tmp_path, monkeypatch, operation, relative,
):
    from pathlib import Path

    from fastapi import HTTPException

    from eb.api import routes_ufo_native
    from eb.store.models import EvalReport

    is_eval = operation == EVAL_OPERATION
    root = tmp_path / "runs/run_frozen/outputs"
    payload = root / relative
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"" if relative.endswith("__init__.py") else b"original evidence")
    row = {"path": relative, "bytes": payload.stat().st_size, "sha256": sha256(payload)}
    receipt = {"run_id": "run_frozen", "plan_hash": "a" * 64, "files": [row]}
    receipt_name = "eval-receipt.json" if is_eval else (
        "data-receipt.json" if operation == OPERATION else "export-receipt.json"
    )
    receipt_path = root / receipt_name
    receipt_path.write_text(json.dumps(receipt))
    run = Run(id="run_frozen", job_id="job_frozen", status="finished",
              metrics_summary={"artifact_id": "art_frozen"})
    job = Job(id=run.job_id, status="succeeded", spec={"embodiment_env_id": ENV_ID,
              "params": {"operation": operation, "pbfm_plan": {"hash": "a" * 64}}})
    report = EvalReport(id="ev_frozen", run_id=run.id,
                        gate_results={"receipt": deepcopy(receipt)}) if is_eval else None
    artifact = Artifact(id="art_frozen", run_id=run.id, sha256=fingerprint(receipt),
                        kind="pbfm_motion" if operation == OPERATION else "pbfm_policy",
                        meta={"receipt": deepcopy(receipt)})

    async def get(cls, key):
        return {Run: run, Job: job, Artifact: artifact}.get(cls)

    async def scalar(query):
        return report

    async def scan(_, name):
        actual = Path(name)
        return {"files": 1, "samples": [{"size": actual.stat().st_size, "sha256": sha256(actual)}]}

    async def stream(request, rid, path):
        return {"rid": rid, "path": path}

    monkeypatch.setattr(routes, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(EbdClient, "scan", scan)
    monkeypatch.setattr(routes_ufo_native, "stream_output", stream)
    session = SimpleNamespace(get=get, scalar=scalar)
    assert await routes.output_file(run.id, relative, None, session, None) == {
        "rid": run.id, "path": relative,
    }
    payload.write_bytes(b"mutated evidence!" if row["bytes"] else b"changed")
    receipt["files"] = [{**row, "bytes": payload.stat().st_size, "sha256": sha256(payload)}]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(HTTPException) as error:
        await routes.output_file(run.id, relative, None, session, None)
    assert error.value.status_code == 404
