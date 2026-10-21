import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def native_checkpoint(tmp_path, world_size=2):
    native = tmp_path / "run_native/outputs/native"
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/ufo-native-config.json").read_text())
    config, kwargs = fixture["model"], fixture["kwargs"]
    agent = {"name": "FBcprAuxAgent", "model": config}
    status = {"local_time": 18432, "global_time": 18432 * world_size,
              "optimizer_steps": 112, "world_size": world_size, "replay_id": "snapshot-identity",
              "buffer_global_time": 18432 * world_size,
              "loss_mode": "local_loss_average" if world_size > 1 else "single_rank",
              "effective_batch_size": 256 * world_size}
    values = {"config.json": {**fixture["saved"], "agent": agent}, "config.yaml": b"agent: FBcprAuxAgent\n",
              "effective_config.json": {**fixture["effective"], "agent": agent,
                                        "training": {"distributed_world_size": world_size}},
              "checkpoint/config.json": agent, "checkpoint/init_kwargs.json": kwargs,
              "checkpoint/optimizers.pth": b"native-optimizers",
              "checkpoint/model/config.json": config, "checkpoint/model/init_kwargs.json": kwargs,
              "checkpoint/model/model.safetensors": b"native-model",
              "checkpoint/train_status.json": status, "train_log.txt": b"global_time,fb_loss\n0,1.0\n"}
    if world_size > 1:
        values["checkpoint/distributed_sync.json"] = {
            "world_size": world_size, "rank": 0, "max_abs_diff_from_rank0": 0.0}
    for rank in range(world_size):
        prefix = f"checkpoint/buffers/train_rank_{rank}"
        values[prefix + "/snapshot.json"] = {"id": status["replay_id"], "global_time": status["global_time"],
                                                "world_size": world_size, "transition_format": 2}
        values[prefix + "/config.json"] = {"capacity": 64, "_idx": 32, "_is_full": False,
                                               "n_dim": 2, "seq_length": 1}
        values[prefix + "/buffer.hdf5"] = f"native-replay-rank-{rank}".encode()
    for relative, value in values.items():
        path = native / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    source_refs = {key: {"path": f"/sources/{key}", "manifest_path": f"/sources/{key}-manifest.json",
                         "snapshot_hash": "sha256:" + char * 64}
                   for key, char in (("ufo", "a"), ("rlinf", "b"))}
    plan = {"schema_version": "eb/ufo-native-train-plan/v1", "fingerprint": "sha256:" + "c" * 64,
            "image_digest": "sha256:" + "d" * 64, "source_refs": source_refs,
            "world_size": world_size, "target_updates": 112, "native_args": {}}
    ranks = [{"rank": rank, "world_size": world_size, "status": "succeeded", "pid": 1000 + rank,
              "local_time": status["local_time"], "global_time": status["global_time"],
              "optimizer_steps": 112, "loader_verified": True, "replay_loader_verified": True,
              "gpu_uuid": f"GPU-{rank}", "source_refs": source_refs} for rank in range(world_size)]
    return native, plan, ranks


def publish(native, plan, ranks):
    from eb.ufo_training_checkpoint import write_native_checkpoint_receipt
    return write_native_checkpoint_receipt(native, {**plan, "run_id": "run_native"}, "run_native",
                                           rank_results=ranks, loader_verified=True)


@pytest.mark.parametrize("missing", [True, False])
def test_finite_required_publication_rejects_any_unverified_rank(tmp_path, missing):
    native, plan, ranks = native_checkpoint(tmp_path)
    plan["full_state_finite_required"] = True
    for result in ranks:
        result["full_state_finite_verified"] = True
    if missing:
        ranks[1].pop("full_state_finite_verified")
    else:
        ranks[1]["full_state_finite_verified"] = False
    with pytest.raises(ValueError, match="finite"):
        publish(native, plan, ranks)
    assert not (native / "checkpoint-manifest.json").exists()
    assert not (native.parent / "receipt.json").exists()


def test_finite_required_publication_binds_all_rank_evidence(tmp_path):
    from eb.ufo_training_checkpoint import read_native_checkpoint

    native, plan, ranks = native_checkpoint(tmp_path)
    plan["full_state_finite_required"] = True
    for result in ranks:
        result["full_state_finite_verified"] = True
    receipt = publish(native, plan, ranks)
    checkpoint = read_native_checkpoint(native.parents[1], "run_native", plan=plan, verify_hashes=True)
    for value in (receipt, checkpoint, checkpoint["manifest"]):
        assert value["full_state_finite_verified"] is True
        assert value["full_state_finite_required"] is True


@pytest.mark.parametrize("with_plan", [True, False])
@pytest.mark.parametrize("corrupt", ["manifest_missing", "manifest_false", "rank", "receipt_missing",
                                      "receipt_false", "receipt_policy"])
def test_finite_required_read_rejects_missing_or_forged_evidence(tmp_path, with_plan, corrupt):
    from eb.ufo_training_checkpoint import read_native_checkpoint

    native, plan, ranks = native_checkpoint(tmp_path)
    plan["full_state_finite_required"] = True
    for result in ranks:
        result["full_state_finite_verified"] = True
    receipt = publish(native, plan, ranks)
    manifest_path = native / "checkpoint-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corrupt == "manifest_missing":
        manifest.pop("full_state_finite_verified")
    elif corrupt == "manifest_false":
        manifest["full_state_finite_verified"] = False
    elif corrupt == "rank":
        manifest["rank_results"][1].pop("full_state_finite_verified")
        receipt["rank_results"] = manifest["rank_results"]
    elif corrupt == "receipt_missing":
        receipt.pop("full_state_finite_verified")
    elif corrupt == "receipt_false":
        receipt["full_state_finite_verified"] = False
    else:
        receipt["full_state_finite_required"] = False
    manifest_path.write_text(json.dumps(manifest))
    receipt["native_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (native.parent / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="finite"):
        read_native_checkpoint(native.parents[1], "run_native", plan=plan if with_plan else None)


def test_legacy_checkpoint_stays_readable_without_claiming_finite_verification(tmp_path):
    from eb.ufo_training_checkpoint import read_native_checkpoint

    native, plan, ranks = native_checkpoint(tmp_path)
    receipt = publish(native, plan, ranks)
    assert receipt["full_state_finite_verified"] is False
    manifest_path = native / "checkpoint-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for field in ("full_state_finite_verified", "full_state_finite_required"):
        manifest.pop(field)
        receipt.pop(field)
    manifest_path.write_text(json.dumps(manifest))
    receipt["native_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (native.parent / "receipt.json").write_text(json.dumps(receipt))
    checkpoint = read_native_checkpoint(native.parents[1], "run_native", plan=plan, verify_hashes=True)
    assert checkpoint["full_state_finite_verified"] is False
    assert "full_state_finite_verified" not in checkpoint["manifest"]
    with pytest.raises(ValueError, match="finite"):
        read_native_checkpoint(native.parents[1], "run_native",
                               plan={**plan, "full_state_finite_required": True})


def test_legacy_manifest_cannot_claim_full_finite_verification_without_all_ranks(tmp_path):
    from eb.ufo_training_checkpoint import validate_native_checkpoint

    native, plan, ranks = native_checkpoint(tmp_path)
    publish(native, plan, ranks)
    manifest = json.loads((native / "checkpoint-manifest.json").read_text())
    manifest["full_state_finite_verified"] = True
    with pytest.raises(ValueError, match="finite"):
        validate_native_checkpoint(native, manifest, run_id="run_native", plan=plan)


def test_native_checkpoint_is_separate_resumable_state_with_original_counts(tmp_path):
    from eb.checkpoints import checkpoint_model, resume_checkpoint
    from eb.ufo_training_checkpoint import read_native_checkpoint
    native, plan, ranks = native_checkpoint(tmp_path)
    receipt = publish(native, plan, ranks)
    checkpoint = read_native_checkpoint(native.parents[1], "run_native", plan=plan, verify_hashes=True)
    assert receipt["status"] == "complete"
    assert checkpoint["mode"] == "ufo_native_state"
    assert checkpoint["step"] == 112
    assert checkpoint["counts"] == {"local_time": 18432, "global_time": 36864, "optimizer_steps": 112}
    assert checkpoint["state_coverage"]["rng_per_rank"] == "not_available"
    assert resume_checkpoint(native.parents[1], "run_native")["path"] == str(native / "checkpoint")
    assert checkpoint_model(native.parents[1], "run_native")["manifest"] == checkpoint["manifest"]
    manifest_bytes = (native / "checkpoint-manifest.json").read_bytes()
    assert receipt["native_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()


@pytest.mark.parametrize("relative,mutate", [
    ("checkpoint/.saving", lambda p: p.write_text("")),
    ("checkpoint/optimizers.pth", lambda p: p.unlink()),
    ("checkpoint/buffers/train_rank_1/buffer.hdf5", lambda p: p.unlink()),
    ("checkpoint/buffers/train_rank_1/snapshot.json", lambda p: p.write_text(json.dumps({"id": "wrong"}))),
    ("checkpoint/buffers/train_rank_1/config.json", lambda p: p.write_text(json.dumps({"capacity": 0}))),
    ("checkpoint/model/model.safetensors", lambda p: p.write_bytes(b"")),
    ("checkpoint/model/init_kwargs.json", lambda p: p.write_text("{}")),
])
def test_native_publish_rejects_partial_or_mismatched_native_state(tmp_path, relative, mutate):
    native, plan, ranks = native_checkpoint(tmp_path)
    mutate(native / relative)
    with pytest.raises(ValueError):
        publish(native, plan, ranks)
    assert not (native.parent / "receipt.json").exists()


@pytest.mark.parametrize("change", ["rank", "target", "loader", "replay_loader", "source", "world_size"])
def test_native_publish_requires_all_successful_identical_rank_evidence(tmp_path, change):
    native, plan, ranks = native_checkpoint(tmp_path)
    if change == "rank":
        ranks.pop()
    elif change == "target":
        plan["target_updates"] = 128
    elif change == "loader":
        ranks[1]["loader_verified"] = False
    elif change == "replay_loader":
        ranks[1]["replay_loader_verified"] = False
    elif change == "source":
        ranks[1]["source_refs"] = {}
    else:
        plan["world_size"] = 4
    with pytest.raises(ValueError):
        publish(native, plan, ranks)
    assert not (native.parent / "receipt.json").exists()


def test_native_read_checks_receipt_paths_hashes_and_resume_identity(tmp_path):
    from eb.ufo_training_checkpoint import read_native_checkpoint
    native, plan, ranks = native_checkpoint(tmp_path)
    publish(native, plan, ranks)
    run_root = native.parents[1]
    for key, value in (("fingerprint", "sha256:" + "f" * 64), ("image_digest", "sha256:" + "e" * 64),
                       ("source_refs", {}), ("world_size", 1)):
        with pytest.raises(ValueError):
            read_native_checkpoint(run_root, "run_native", plan={**plan, key: value})
    weight = native / "checkpoint/model/model.safetensors"
    weight.write_bytes(b"broken-model")
    with pytest.raises(ValueError, match="digest|hash"):
        read_native_checkpoint(run_root, "run_native", verify_hashes=True)
    weight.unlink()
    weight.symlink_to(native / "checkpoint/optimizers.pth")
    with pytest.raises(ValueError, match="symlink"):
        read_native_checkpoint(run_root, "run_native")


def test_native_read_rejects_unindexed_files_and_forged_completion(tmp_path):
    from eb.ufo_training_checkpoint import read_native_checkpoint
    native, plan, ranks = native_checkpoint(tmp_path)
    receipt = publish(native, plan, ranks)
    extra = native / "checkpoint/unindexed.pth"
    extra.write_bytes(b"new file")
    with pytest.raises(ValueError, match="index|manifest"):
        read_native_checkpoint(native.parents[1], "run_native")
    extra.unlink()
    receipt["rank_results"] = ranks[:1]
    (native.parent / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="rank"):
        read_native_checkpoint(native.parents[1], "run_native")


@pytest.mark.asyncio
async def test_native_collection_separates_resume_replay_from_model_and_is_idempotent(tmp_path, monkeypatch):
    from eb.orchestrator.collect import claim_outputs
    from eb.store.models import Artifact, ModelVersion
    native, plan, ranks = native_checkpoint(tmp_path)
    publish(native, plan, ranks)
    job = SimpleNamespace(id="job_native", kind="train.rl", spec={"name": "native model",
        "embodiment_env_id": "ee_rpo_v2_ufo_native_23dof_v2", "params": {
            "template_id": "rpo_ufo_fb_native", "native_train_plan": plan},
        "inputs": {"datasets": [{"id": "ds_train", "version": 1}]}})
    run = SimpleNamespace(id="run_native", metrics_summary={})
    rows = []
    session = SimpleNamespace(scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=None),
                              add=rows.append, flush=AsyncMock())
    monkeypatch.setattr("eb.orchestrator.collect.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("eb.orchestrator.collect.read_native_training_evidence", AsyncMock(return_value={
        "available": True, "run_id": run.id, "decision": "wait_for_independent_validation", "files": []}))
    # The fixture's run root is placed in the real collector layout.
    (tmp_path / "runs").mkdir()
    native.parents[1].rename(tmp_path / "runs/run_native")
    await claim_outputs(session, job, run)
    artifacts = [row for row in rows if isinstance(row, Artifact)]
    model = next(row for row in rows if isinstance(row, ModelVersion))
    assert {a.kind for a in artifacts} == {"ufo_native_resume", "ufo_native_trained_model"}
    inference = next(a for a in artifacts if a.kind == "ufo_native_trained_model")
    assert inference.storage.endswith("checkpoint/model")
    assert all("buffers/" not in item["path"] and "optimizers" not in item["path"]
               for item in inference.meta["manifest"]["files"])
    assert model.card["training_provenance"] == "platform"
    assert model.card["qualification"] == "incomplete"
    assert model.card["prompt_derivation"] == "required"
    assert run.metrics_summary["optimizer_steps"] == 112
    assert run.metrics_summary["global_time"] == 36864
    session.scalar = AsyncMock(side_effect=artifacts + [model])
    rows.clear()
    await claim_outputs(session, job, run)
    assert not rows
