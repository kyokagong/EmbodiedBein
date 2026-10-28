"""P-BFM publication checks; retain the shared Job/Run collection lifecycle."""

import re
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import select

from eb.domain.job import JobSpec
from eb.ids import new_id
from eb.onnx_io import sha256_file
from eb.store.models import Artifact, LineageEdge, ModelFamily, ModelVersion


async def read_pbfm_training_evidence(root: Path, run_id: str, *, target_updates: int) -> dict:
    from eb.orchestrator.ufo_native import verify_index
    from eb.ufo_native import finite_tree, read_json, safe_file

    outputs = root / "outputs"
    index_path = safe_file(outputs, "analysis/index.json")
    index = read_json(index_path)
    files = index.get("files")
    if (
        index.get("schema_version") != "eb/pbfm-training-evidence/v1"
        or index.get("run_id") != run_id
        or index.get("target_updates") != target_updates
        or not isinstance(files, list)
        or not 3 <= len(files) <= 200
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not re.fullmatch(r"analysis/[\w.-]+\.(?:svg|csv|json)", item["path"])
            for item in files
        )
    ):
        raise ValueError("P-BFM training evidence index identity or scope differs")
    names = {item["path"] for item in files}
    if not {
        "analysis/summary.json",
        "analysis/train-timeseries.svg",
        "analysis/train-timeseries.csv",
    }.issubset(names):
        raise ValueError("P-BFM training evidence lacks raw time series, plot or analysis")
    await verify_index(outputs, files)
    summary = read_json(safe_file(outputs, "analysis/summary.json"))
    if (
        summary.get("schema_version") != "eb/pbfm-training-analysis/v1"
        or summary.get("run_id") != run_id
        or (summary.get("counts") or {}).get("optimizer_steps") != target_updates
        or summary.get("decision") != "wait_for_independent_validation"
        or not finite_tree(summary)
    ):
        raise ValueError("P-BFM training analysis identity or decision differs")
    entries = [
        {
            "path": "analysis/index.json",
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
        },
        *files,
    ]
    return {
        "available": True,
        "run_id": run_id,
        "decision": summary["decision"],
        "summary": summary,
        "files": [
            {**item, "url": f"/api/v1/pbfm/runs/{run_id}/files/{quote(item['path'], safe='/')}"}
            for item in entries
        ],
    }


async def collect_pbfm_training(
    session, job, run, root: Path, *, diagnostic_spec: JobSpec | None = None,
    periodic_checkpoint: dict | None = None, periodic_binding: dict | None = None,
) -> str:
    """Publish depth networks separately from complete all-rank training state."""
    from eb.orchestrator.ufo_native import verify_index
    from eb.pbfm_training import training_fingerprint, validate_train_job
    from eb.pbfm_training_checkpoint import MAX_FILES, read_pbfm_checkpoint

    spec = JobSpec.model_validate(job.spec)
    diagnostic = diagnostic_spec is not None
    if diagnostic:
        from eb.pbfm_runtime_probe import TRAINING_MODES, _training_view

        if (
            spec.params.get("probe_mode") not in TRAINING_MODES | {"segment112"}
            or diagnostic_spec != _training_view(spec)
        ):
            raise ValueError("P-BFM diagnostic publication requires the exact protected training spec")
        diagnostic_mode = spec.params["probe_mode"]
        spec = diagnostic_spec
    producer = (
        {"diagnostic": True, "diagnostic_mode": diagnostic_mode, "diagnostic_job_id": job.id,
         "producer_job_kind": "asset.validate"}
        if diagnostic else {}
    )
    plan = validate_train_job(spec)
    periodic = periodic_checkpoint is not None
    if periodic:
        if (not diagnostic or job.status != "cancelled" or run.status != "aborted"
                or not periodic_binding or periodic_binding.get("job_id") != job.id
                or periodic_binding.get("run_id") != run.id
                or periodic_binding.get("step") != periodic_checkpoint.get("step")):
            raise ValueError("P-BFM periodic publication requires a verified stopped source binding")
        producer = {**producer, "periodic_checkpoint": periodic_binding, "training_completed": False}
    checkpoint = periodic_checkpoint if periodic else read_pbfm_checkpoint(root, spec, run.id)
    if checkpoint is None:
        raise ValueError("P-BFM producer has no complete checkpoint")
    manifest, counts = checkpoint["manifest"], checkpoint["counts"]
    if (
        not periodic and counts["optimizer_steps"] != plan["target_updates"]
        or checkpoint.get("full_state_finite_verified") is not True
    ):
        raise ValueError("P-BFM producer target or complete finite-state verification differs")
    native = Path(checkpoint["native_path"])
    # Native Replay shards can exceed the inference/media index's 2 GiB bound.
    # ebd hashes every payload before any immutable model or resume publication.
    await verify_index(native, manifest["files"], max_files=MAX_FILES, max_file_bytes=1024**4)
    evidence = (None if periodic else
                await read_pbfm_training_evidence(root, run.id, target_updates=plan["target_updates"]))
    model_files = [
        {**item, "path": item["path"].removeprefix("checkpoint/model/")}
        for item in manifest["files"]
        if item["path"].startswith("checkpoint/model/")
    ]
    if not model_files:
        raise ValueError("P-BFM checkpoint lacks an inference network package")
    network = {
        "schema_version": "eb/pbfm-native-trained-network/v1",
        "run_id": run.id,
        "training_provenance": "platform",
        **producer,
        "source_refs": manifest["source_refs"],
        "image_digest": manifest["image_digest"],
        "counts": counts,
        "camera_contract": plan["identity"]["camera_contract"],
        "terrain_contract": plan["identity"]["terrain_contract"],
        "files": model_files,
    }
    artifacts = []
    for kind, value, storage in (
        ("pbfm_native_resume", manifest, native),
        ("pbfm_native_trained_model", network, native / "checkpoint/model"),
    ):
        digest = training_fingerprint(value)
        artifact = await session.scalar(
            select(Artifact).where(Artifact.sha256 == digest, Artifact.kind == kind)
        )
        if artifact is not None and (
            artifact.run_id != run.id
            or artifact.storage != str(storage)
            or (artifact.meta or {}).get("manifest") != value
        ):
            raise ValueError("Published P-BFM Artifact differs from its immutable producer")
        if artifact is None:
            artifact = Artifact(
                id=new_id("art"),
                sha256=digest,
                kind=kind,
                run_id=run.id,
                pinned=True,
                storage=str(storage),
                size_bytes=sum(item["bytes"] for item in value["files"]),
                meta={
                    "manifest": value,
                    "step": checkpoint["step"],
                    "counts": counts,
                    "simulation_only": True,
                    "qualification": "incomplete",
                    **producer,
                },
            )
            session.add(artifact)
            await session.flush()
            session.add(
                LineageEdge(
                    src_type="run",
                    src_id=run.id,
                    dst_type="artifact",
                    dst_id=artifact.id,
                    relation="produced",
                    run_id=run.id,
                )
            )
        artifacts.append(artifact)
    resume_artifact, inference_artifact = artifacts
    model = await session.scalar(
        select(ModelVersion).where(ModelVersion.artifact_id == inference_artifact.id)
    )
    if model is None:
        family_id = "mf_rpo_pbfm_depth"
        if await session.get(ModelFamily, family_id) is None:
            session.add(ModelFamily(id=family_id, name="RPO P-BFM depth FB-CPR-Aux", family="mlp"))
            await session.flush()
        model = ModelVersion(
            id=new_id("mv"),
            family_id=family_id,
            artifact_id=inference_artifact.id,
            name=(spec.params.get("model_name") or spec.name) +
                 (f" [U{counts['optimizer_steps']}]" if periodic else ""),
            embodiment_env_id=spec.embodiment_env_id,
            provenance="full",
            io_signature={"obs": 380, "depth": [8, 18, 32], "latent": 256, "act": 23},
            tags=[
                "rpo-v2", "pbfm", "depth", "rlinf", "simulation-only",
                *(["diagnostic"] if diagnostic else []),
                *(["periodic-checkpoint"] if periodic else []),
            ],
            card={
                "schema_version": "eb/rpo-pbfm-trained-model/v1",
                "training_provenance": "platform",
                **({"diagnostic_job_id": job.id} if diagnostic else {"train_job_id": job.id}),
                **producer,
                "run_id": run.id,
                "template_id": spec.params["template_id"],
                "native_path": str(native),
                "train_plan": plan,
                "dataset_id": plan["identity"]["dataset_id"],
                "package_path": str(native / "checkpoint/model"),
                "checkpoint_path": checkpoint["path"],
                "resume_artifact_id": resume_artifact.id,
                "resume_mode": checkpoint["mode"],
                "full_state_finite_verified": True,
                "resume_limitations": checkpoint["limitations"],
                "state_coverage": checkpoint["state_coverage"],
                "counts": counts,
                "source_refs": manifest["source_refs"],
                "image_digest": manifest["image_digest"],
                "compatibility_fingerprint": plan["fingerprint"],
                "model_sha256": next(
                    (item["sha256"] for item in model_files if item["path"] == "model.safetensors"), None
                ),
                "camera_contract": plan["identity"]["camera_contract"],
                "terrain_contract": plan["identity"]["terrain_contract"],
                "datasets": spec.inputs.datasets,
                "prompt_derivation": "required",
                "readiness": "prompt_required",
                "qualification": "incomplete",
                "simulation_only": True,
                "hardware_verified": False,
            },
        )
        session.add(model)
        session.add(
            LineageEdge(
                src_type="run",
                src_id=run.id,
                dst_type="model_version",
                dst_id=model.id,
                relation="produced",
                run_id=run.id,
            )
        )
        for dataset in spec.inputs.datasets:
            session.add(
                LineageEdge(
                    src_type="dataset",
                    src_id=dataset["id"],
                    dst_type="model_version",
                    dst_id=model.id,
                    relation="trained_on",
                    run_id=run.id,
                )
            )
    if periodic:
        entries = (run.metrics_summary or {}).get("periodic_checkpoints", [])
        item = {key: checkpoint[key] for key in ("path", "bytes", "step", "mode")}
        item.update(registered_model_id=model.id, complete=True, weights_only=False,
                    periodic_checkpoint=periodic_binding, full_state_finite_verified=True)
        run.metrics_summary = {**(run.metrics_summary or {}), "periodic_checkpoints":
            [item, *[row for row in entries if row["step"] != checkpoint["step"]]]}
        return model.id
    run.metrics_summary = {
        **(run.metrics_summary or {}),
        "model_version_id": model.id,
        "resume_artifact_id": resume_artifact.id,
        "artifact_id": inference_artifact.id,
        "native_operation": "probe_pbfm_runtime" if diagnostic else "pbfm_native_train",
        **producer,
        **({"training_completed": True} if diagnostic else {}),
        "resume_mode": checkpoint["mode"],
        "full_state_finite_verified": True,
        "qualification": "incomplete",
        **counts,
        "segment_optimizer_steps": counts["optimizer_steps"] - plan.get("start_updates", 0),
        "segment_local_env_steps": counts["local_time"] - plan.get("start_local_time", 0),
        "checkpoint": {key: checkpoint[key] for key in ("path", "bytes", "step", "mode")},
        "training_evidence": evidence,
    }
    return model.id
