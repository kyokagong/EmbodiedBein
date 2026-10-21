import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from eb.ufo_native import TASKS, plan_hash
from eb.ufo_training_prompt import OPERATION, PLAN_SCHEMA, PROMPT_SCHEMA


def prompt_plan():
    refs = {name: {"path": f"/sources/{name}", "manifest_path": f"/sources/{name}-manifest.json",
                   "snapshot_hash": "sha256:" + character * 64}
            for name, character in (("ufo", "a"), ("rlinf", "b"))}
    sources = {name: {"kind": "content_snapshot", "path": f"/state/snapshots/{name}/content",
                      "commit": ref["snapshot_hash"]} for name, ref in refs.items()}
    plan = {"schema_version": PLAN_SCHEMA, "operation": OPERATION, "name": "new trained model",
        "model_id": "mv_trained", "source_run_id": "run_train",
        "input_path": "/state/runs/run_train/outputs/native",
        "image_digest": "sha256:" + "c" * 64, "source_artifact_sha256": "sha256:" + "d" * 64,
        "source_resume_artifact_sha256": "sha256:" + "e" * 64,
        "source_fingerprint": "sha256:" + "f" * 64, "source_model_sha256": "a" * 64,
        "native_manifest_sha256": "b" * 64, "source_refs": refs,
        "ufo_source_ref": sources["ufo"], "rlinf_source_ref": sources["rlinf"],
        "buffer_rank": 0, "replay_id": "snapshot-new-model", "snapshot": {
            "id": "snapshot-new-model", "transition_format": 2, "world_size": 2, "global_time": 36864},
        "seed": 71000, "num_samples": 16384, "n_inferences": 1, "cpu_workers": 4,
        "tasks": [task["id"] for task in TASKS], "files": [], "extras": [], "codec": {"actor_dim": 631}}
    plan["hash"] = plan_hash(plan)
    return plan


def test_prompt_uses_original_native_entrypoint_with_explicit_matched_replay_and_limits():
    from eb.api.routes_ufo_native import spec_for
    from eb.runners.ufo_training_prompt import reward_inference_arguments
    from eb.ufo_native import validate_job

    plan = prompt_plan()
    spec = spec_for(plan)
    assert spec.kind == "data.convert"
    assert spec.policy.timeout_h == 1 and spec.policy.max_retries == 0
    assert validate_job(spec) == plan
    buffer = Path("/training-source/checkpoint/buffers/train_rank_0")
    args = reward_inference_arguments(Path("/outputs/model"), buffer, plan)
    assert args["buffer_rank"] == 0 and args["buffer_path"].name == "train_rank_0"
    assert args["num_samples"] == 16384 and args["n_inferences"] == 1
    assert args["skip_rollouts"] is True and args["export_onnx"] is False
    assert args["device"] == "cpu" and args["max_workers"] == 4 and args["process_executor"] is True
    assert args["model_folder"].is_relative_to("/outputs")


@pytest.mark.parametrize("key,value", [("buffer_rank", 1), ("replay_id", "old-model-snapshot"),
    ("num_samples", 263168), ("num_samples", 257), ("n_inferences", 2), ("cpu_workers", 24),
    ("seed", True), ("tasks", ["move-ego-0-0"]), ("image_digest", "latest")])
def test_prompt_rejects_unfrozen_tasks_replay_or_unbounded_compute(key, value):
    from eb.ufo_training_prompt import validate_prompt_plan
    plan = {**prompt_plan(), key: value}
    plan["hash"] = plan_hash(plan)
    with pytest.raises(ValueError):
        validate_prompt_plan(plan)


def test_prompt_adapter_mounts_sources_read_only_and_stages_outputs_in_new_run(tmp_path, monkeypatch):
    from eb.adapters.ufo_native import UfoNativeAdapter
    from eb.api.routes_ufo_native import spec_for
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    plan = prompt_plan()
    rendered = UfoNativeAdapter().render(spec_for(plan), "run_prompt")
    assert (plan["input_path"], "/training-source", "ro") in rendered.mounts
    assert (plan["ufo_source_ref"]["path"], "/sources/ufo", "ro") in rendered.mounts
    assert rendered.command == ["python", "-m", "eb.runners.ufo_training_prompt", "--plan", "/plan.json"]
    assert rendered.gpus == 0 and rendered.workdir == "/outputs"
    assert [mode for _, destination, mode in rendered.mounts if destination != "/outputs"] == ["ro"] * 6
    assert rendered.env["OMP_NUM_THREADS"] == "1"
    assert rendered.env["UFO_CACHE_DIR"].startswith("/outputs/")


@pytest.mark.asyncio
async def test_prompt_preview_is_operator_only_and_submission_reuses_existing_idempotent_job(monkeypatch):
    from fastapi import HTTPException

    from eb.api.deps import Principal
    from eb.api.routes_ufo_native import PromptIn, prompt_action
    plan = prompt_plan()
    build = AsyncMock(return_value=plan)
    monkeypatch.setattr("eb.ufo_training_prompt.prompt_plan", build)
    previous = NS(id="job_prompt", status="cancelled", spec={"params": {"native_plan": plan}})
    session = NS(scalar=AsyncMock(return_value=previous))
    with pytest.raises(HTTPException) as denied:
        await prompt_action("mv_trained", PromptIn(), session, Principal("test", "agent"))
    assert denied.value.status_code == 403
    build.assert_not_awaited()
    with pytest.raises(HTTPException) as stale:
        await prompt_action("mv_trained", PromptIn(preview_hash="changed"), session,
                            Principal("kk", "user"), "key")
    assert stale.value.status_code == 409
    result = await prompt_action("mv_trained", PromptIn(preview_hash=plan["hash"]), session,
                                 Principal("kk", "user"), "key")
    assert result == {"id": "job_prompt", "status": "cancelled"}


def write_prompt_outputs(tmp_path, plan):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    weight = write("model/checkpoint/model/model.safetensors", b"new trained weights")
    plan["source_model_sha256"] = weight["sha256"]
    prompts = {"source_model_sha256": plan["source_model_sha256"], "buffer_snapshot_id": plan["replay_id"],
               "seed": plan["seed"], "tasks": {task: [0.5] * 256 for task in plan["tasks"]}}
    prompt_record = write("prompts/prompts.json", prompts)
    raw_record = write("prompts/reward_locomotion.pkl", b"native freshly derived pickle")
    derivation = {"schema_version": PROMPT_SCHEMA, "operation": OPERATION,
        **{key: plan[key] for key in ("source_model_sha256", "source_artifact_sha256",
            "source_resume_artifact_sha256", "source_run_id", "snapshot", "buffer_rank", "seed",
            "n_inferences", "cpu_workers", "tasks", "source_refs", "image_digest")},
        "source_model_id": plan["model_id"], "num_samples_per_inference": plan["num_samples"],
        "process_executor": True, "prompts_sha256": prompt_record["sha256"],
        "raw_native_prompt_sha256": raw_record["sha256"]}
    derivation_record = write("prompts/derivation.json", derivation)
    prompt_manifest = {"derivation": derivation, "files": [prompt_record, raw_record, derivation_record]}
    write("prompts/manifest.json", prompt_manifest)
    write("model/prompts.json", prompts)
    model_manifest = {"schema_version": "eb/ufo-native-import/v1", "loader_verified": True,
        "codec": plan["codec"], "source_model_id": plan["model_id"], "training_provenance": "platform",
        "prompt_derivation": "native_reward_inference", "prompt_binding": derivation,
        "prompt_artifact_sha256": plan_hash(prompt_manifest), "source_refs": plan["source_refs"],
        "code_sources": {"ufo": plan["ufo_source_ref"], "rlinf": plan["rlinf_source_ref"]},
        "files": [{**weight, "path": "checkpoint/model/model.safetensors"}, prompt_record]}
    write("model/manifest.json", model_manifest)
    return {"manifest": model_manifest, "prompt_manifest": prompt_manifest, "package": "model"}


@pytest.mark.asyncio
async def test_prompt_collection_registers_derived_artifact_lineage(tmp_path, monkeypatch):
    from eb.store.models import Artifact, LineageEdge, ModelVersion
    from eb.ufo_training_prompt import collect_prompt
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", AsyncMock())
    plan = prompt_plan()
    value = write_prompt_outputs(tmp_path, plan)
    source = NS(id="mv_trained", family_id="mf_rpo_ufo_native",
                card={"schema_version": "eb/rpo-ufo-native-trained-model/v1"})
    rows = []
    session = NS(scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=source), add=rows.append,
                 flush=AsyncMock())
    run = NS(id="run_prompt", metrics_summary={})
    await collect_prompt(session, NS(), run, tmp_path, plan, value)
    artifacts = [row for row in rows if isinstance(row, Artifact)]
    model = next(row for row in rows if isinstance(row, ModelVersion))
    assert {row.kind for row in artifacts} == {"ufo_native_prompt", "ufo_native_model"}
    assert model.provenance == "full" and model.card["training_provenance"] == "platform"
    assert model.card["qualification"] == "diagnostic" and model.card["source_model_id"] == source.id
    assert model.card["prompt_binding"]["snapshot"] == plan["snapshot"]
    relations = {row.relation for row in rows if isinstance(row, LineageEdge)}
    assert relations >= {"conditioned_by", "prompt_derived_from"}
    rows.clear()
    session.scalar = AsyncMock(side_effect=artifacts + [model])
    await collect_prompt(session, NS(), run, tmp_path, plan, value)
    assert not rows
    wrong = deepcopy(value)
    wrong["prompt_manifest"]["derivation"]["source_model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bind"):
        await collect_prompt(session, NS(), run, tmp_path, plan, wrong)


@pytest.mark.asyncio
async def test_trained_exports_and_live_keep_the_frozen_runtime_and_source_manifests(tmp_path, monkeypatch):
    from eb.api.routes_ufo_native import execution_runtime

    plan = prompt_plan()
    sources, refs = {}, {}
    for name in ("ufo", "rlinf"):
        root = tmp_path / name / "content"
        root.mkdir(parents=True)
        manifest = root.parent / "manifest.json"
        manifest.write_text('{"entries":[]}')
        digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
        sources[name] = {"path": str(root), "commit": digest, "kind": "content_snapshot"}
        refs[name] = {"path": f"/sources/{name}", "snapshot_hash": digest,
                      "manifest_path": f"/sources/{name}-manifest.json"}
    monkeypatch.setattr("eb.api.routes_ufo_native.validate_allowed_path", Path)
    image_info = AsyncMock(return_value={"ok": True, "digest": plan["image_digest"]})
    monkeypatch.setattr("eb.api.routes_ufo_native.EbdClient", lambda: NS(image_info=image_info))
    manifest = {"training_provenance": "platform", "image_digest": plan["image_digest"],
                "code_sources": sources, "source_refs": refs}
    runtime = await execution_runtime(manifest)
    assert runtime == {"image_digest": plan["image_digest"], "code_sources": sources, "source_refs": refs}
    image_info.assert_awaited_once_with(plan["image_digest"])
    (tmp_path / "ufo/manifest.json").write_text('{"entries":["tampered"]}')
    with pytest.raises(ValueError, match="manifest"):
        await execution_runtime(manifest)
