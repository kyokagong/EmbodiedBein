"""Control-plane binding for one approved native-launch baseline diagnostic."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from eb.ufo_native_baseline import (
    ARTIFACT_KIND,
    BASE_IMAGE,
    DATASET,
    IMAGE,
    OPERATION,
    RECEIPT_SCHEMA,
    RESOURCES,
    RULES,
    SCHEMA,
    SCOPED_ARTIFACT_KIND,
    SCOPED_SCHEMA,
    SOURCE_JOB,
    SOURCE_RUN,
    VERIFICATION_SCOPE,
)


def preview_hash(spec):
    from eb.ufo_native import plan_hash

    return plan_hash(spec.model_dump(mode="json"))


def spec_for(plan, name="UFO v2 native launch 112-update baseline"):
    from eb.domain.job import JobSpec
    from eb.ufo_native import ENV_ID

    source = plan["source_spec"]
    return JobSpec(
        kind="asset.validate",
        embodiment_env_id=ENV_ID,
        name=name,
        code_refs=source["code_refs"],
        inputs=source["inputs"],
        image=plan["image_digest"],
        params={"operation": OPERATION, "baseline_plan": plan},
        resources=RESOURCES,
        runtime={"network": "none", "hostname": "localhost"},
        policy={"timeout_h": 0.5, "max_retries": 0, "retry_on": []},
    )


async def build_plan(session, source_run_id=SOURCE_RUN):
    from eb.domain.job import JobSpec
    from eb.settings import get_settings
    from eb.store.ebd import EbdClient
    from eb.store.models import Job, Run
    from eb.ufo_native import plan_hash, read_json, safe_file, sha256_file
    from eb.ufo_training import validate_train_job
    from eb.ufo_training_checkpoint import read_native_checkpoint

    if source_run_id != SOURCE_RUN:
        raise ValueError("Only the frozen v2 four-rank 112-update Run may be compared")
    run = await session.get(Run, SOURCE_RUN)
    job = await session.get(Job, SOURCE_JOB)
    if (
        not run
        or not job
        or run.job_id != job.id
        or run.status != "finished"
        or job.status != "succeeded"
        or job.kind != "train.rl"
    ):
        raise ValueError("The frozen source must be a successfully completed native training Run")
    source = JobSpec.model_validate(job.spec)
    training = validate_train_job(source)
    if (
        source.image != BASE_IMAGE
        or training["world_size"] != 4
        or training["target_updates"] != 112
        or training["start_updates"] != 0
        or training.get("resume")
        or training["identity"]["seed"] != 42
        or training["identity"]["buffer_size"] != 262144
        or training["identity"]["dataset_id"] != DATASET
        or source.resources.model_dump() != RESOURCES
    ):
        raise ValueError("Frozen baseline source parameters differ")
    root = get_settings().data_root / "runs" / SOURCE_RUN
    checkpoint = read_native_checkpoint(root, SOURCE_RUN, plan=training)
    if checkpoint["counts"] != RULES["counts"] or not checkpoint["full_state_finite_verified"]:
        raise ValueError("Reference 112-update checkpoint is incomplete or unverified")
    files = {}
    for rank in range(4):
        for name in (f"config-rank-{rank}.json", f"initial-parameters-rank-{rank}.json"):
            relative = "outputs/runtime/" + name
            path = safe_file(root, relative)
            read_json(path)  # bounded JSON, no checkpoint payload hashing in API
            files[relative] = "sha256:" + sha256_file(path)
    files["outputs/receipt.json"] = "sha256:" + sha256_file(safe_file(root, "outputs/receipt.json"))
    image = await EbdClient().image_info(IMAGE)
    if image.get("ok") is not True or not re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("digest", "")):
        raise ValueError("Build the separate approved baseline image before previewing")
    plan = {
        "schema_version": SCHEMA,
        "source_run_id": SOURCE_RUN,
        "source_job_id": SOURCE_JOB,
        "source_spec": source.model_dump(mode="json"),
        "native_train_plan": copy.deepcopy(training),
        "source_manifest_sha256": "sha256:" + sha256_file(root / "outputs/native/checkpoint-manifest.json"),
        "reference_files": files,
        "base_image_digest": BASE_IMAGE,
        "image_digest": image["digest"],
        "comparison_rules": copy.deepcopy(RULES),
    }
    plan["hash"] = plan_hash(plan)
    return plan


async def validate_submission(session, spec):
    supplied = spec.params.get("baseline_plan") or {}
    current = await build_plan(session, supplied.get("source_run_id"))
    if supplied != current or spec.model_dump(mode="json") != spec_for(current, spec.name).model_dump(
        mode="json"
    ):
        raise ValueError("Native baseline spec, frozen source, image or execution limits differ")


def render(spec, run_id):
    from eb.adapters.base import RenderedJob
    from eb.domain.job import JobSpec
    from eb.settings import get_settings
    from eb.ufo_native import plan_hash
    from eb.ufo_native_baseline import execution_training_plan
    from eb.ufo_training import render_training

    plan = spec.params["baseline_plan"]
    if spec.params.get("operation") != OPERATION or spec.model_dump(mode="json") != spec_for(
        plan, spec.name
    ).model_dump(mode="json"):
        raise ValueError("Unsupported native baseline command or resource request")
    # Reuse exactly the already-reviewed native source/data/asset read-only mounts.
    source = JobSpec.model_validate(plan["source_spec"])
    # This new diagnostic Run has a distinct wrapper image. Keep the source
    # spec unchanged, while recording the real image for ebd resource evidence.
    execution = execution_training_plan(plan, run_id)
    source = source.model_copy(
        deep=True, update={"image": spec.image, "params": {**source.params, "native_train_plan": execution}}
    )
    rendered = render_training(source, run_id)
    root = get_settings().data_root / "runs" / run_id
    runtime_plan = {**plan, "run_id": run_id}
    runtime_plan["hash"] = plan_hash(runtime_plan)
    if run_id not in {"run_preview", "run_validate"}:
        path = root / "baseline-plan.json"
        encoded = json.dumps(runtime_plan, sort_keys=True, allow_nan=False)
        if path.is_symlink() or path.exists() and path.read_text() != encoded:
            raise ValueError("Baseline frozen Run plan differs")
        path.write_text(encoded)
    mounts = [item for item in rendered.mounts if item[1] != "/plan.json"]
    mounts.append((str(root / "baseline-plan.json"), "/baseline-plan.json", "ro"))
    reference = get_settings().data_root / "runs" / SOURCE_RUN
    mounts.append((str(reference / "outputs/native"), "/reference/outputs/native", "ro"))
    for relative in plan["reference_files"]:
        mounts.append((str(reference / relative), "/reference/" + relative, "ro"))
    return RenderedJob(
        image=spec.image,
        command=["python", "-m", "eb.runners.ufo_native_baseline", "--plan", "/baseline-plan.json"],
        env=rendered.env,
        mounts=mounts,
        writable_paths=rendered.writable_paths,
        workdir=rendered.workdir,
        outputs=rendered.outputs,
        metrics_glob=rendered.metrics_glob,
        gpus=4,
        shm_gb=8,
    )


def read_receipt(outputs: Path, plan: dict, run_id: str):
    from eb.ufo_native import plan_hash, read_json, safe_file

    receipt = read_json(safe_file(outputs, "baseline-receipt.json"))
    expected = {**plan, "run_id": run_id}
    expected["hash"] = plan_hash(expected)
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("run_id") != run_id
        or receipt.get("operation") != OPERATION
        or receipt.get("source_run_id") != SOURCE_RUN
        or receipt.get("plan_hash") != expected["hash"]
        or receipt.get("image_digest") != plan["image_digest"]
        or receipt.get("base_image_digest") != BASE_IMAGE
        or receipt.get("counts") != RULES["counts"]
        or receipt.get("comparison_rules") != RULES
        or receipt.get("structural_checks_passed") is not True
        or receipt.get("numerical_equivalence_claimed") is not False
        or receipt.get("quality_qualification") != "diagnostic"
    ):
        raise ValueError("Baseline receipt is not bound to its frozen diagnostic plan")
    files, seen = receipt.get("files"), set()
    if not isinstance(files, list) or not 1 <= len(files) <= 200:
        raise ValueError("Baseline file index is unavailable or unbounded")
    for row in files:
        path = safe_file(outputs, row.get("path", ""))
        if (
            row["path"] in seen
            or type(row.get("bytes")) is not int
            or not 0 < row["bytes"] <= 2 * 1024**3
            or path.stat().st_size != row["bytes"]
            or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))
        ):
            raise ValueError("Baseline file index is changed or invalid")
        seen.add(row["path"])
    required = {
        "baseline-report.json",
        "metrics-differences.json",
        "native/checkpoint-manifest.json",
        "native/train_log.txt",
        "analysis/summary.json",
        "metrics-index.json",
    }
    required |= {f"runtime/baseline-initial-rank-{rank}.json" for rank in range(4)}
    if not required <= seen:
        raise ValueError("Baseline diagnostic evidence is incomplete")
    return receipt


def scoped_payload(outputs, receipt, artifact):
    """Interpret legacy raw evidence honestly; never inherit its rank optimizer flags."""
    import math

    from eb.ufo_native import plan_hash, read_json, safe_file, sha256_file
    from eb.ufo_native_baseline import CHECKPOINT_SCHEMA
    from eb.ufo_training_checkpoint import CHECKPOINT_SCHEMA as LEGACY_SCHEMA

    report = read_json(safe_file(outputs, "baseline-report.json"))
    manifest = read_json(safe_file(outputs, "native/checkpoint-manifest.json"))
    ranks = report.get("ranks", [])
    optimizers = report.get("optimizer_internal_steps", {})
    sync = report.get("native_synchronization", {})
    if (
        report.get("counts") != RULES["counts"]
        or len(ranks) != 4
        or {row.get("rank") for row in ranks} != set(range(4))
        or any(
            row.get("initial_hash_exact") is not True
            or row.get("configuration_leaf_values_equal") is not True
            for row in ranks
        )
        or len(optimizers) != 5
        or any(
            row.get("step_minimum") != 112 or row.get("step_maximum") != 112 for row in optimizers.values()
        )
        or sync.get("world_size") != 4
        or sync.get("rank") != 0
        or not isinstance(sync.get("max_abs_diff_from_rank0"), (int, float))
        or not math.isfinite(sync["max_abs_diff_from_rank0"])
        or not 0 <= sync["max_abs_diff_from_rank0"] <= 1e-5
        or any(
            len(sync.get(key, [])) != 4 or not all(math.isfinite(x) for x in sync[key])
            for key in ("rank_param_buffer_sum", "rank_param_buffer_sqsum", "rank_param_buffer_numel")
        )
        or manifest.get("schema_version") not in {CHECKPOINT_SCHEMA, LEGACY_SCHEMA}
    ):
        raise ValueError("Baseline scope lacks actual shared checkpoint/comparison evidence")
    legacy = manifest["schema_version"] == LEGACY_SCHEMA
    observed = manifest.get("rank_results" if legacy else "rank_observations", [])
    if (
        len(observed) != 4
        or {row.get("rank") for row in observed} != set(range(4))
        or any(row.get("replay_loader_verified") is not True for row in observed)
    ):
        raise ValueError("Baseline has not observed every rank Replay loader")
    if not legacy and (
        manifest.get("verification_scope") != VERIFICATION_SCOPE
        or manifest.get("training_restore_supported") is not False
        or any(
            "full_state_finite_verified" in row or "loader_verified" in row
            for row in manifest.get("rank_observations", [])
        )
    ):
        raise ValueError("Baseline manifest must not claim live per-rank optimizer verification")
    for value in (receipt, report):
        if "verification_scope" in value and value["verification_scope"] != VERIFICATION_SCOPE:
            raise ValueError("Baseline verification scope changed")
    bindings = {}
    for name in ("baseline-receipt.json", "baseline-report.json", "native/checkpoint-manifest.json"):
        path = safe_file(outputs, name)
        bindings[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if name != "baseline-receipt.json":
            indexed = next(row for row in receipt["files"] if row["path"] == name)
            if bindings[name] != {key: indexed[key] for key in ("bytes", "sha256")}:
                raise ValueError("Baseline scoped raw file differs from its immutable receipt")
    if (
        plan_hash(receipt) != artifact.sha256
        or plan_hash(read_json(outputs / "baseline-receipt.json")) != artifact.sha256
    ):
        raise ValueError("Baseline scoped receipt differs from the raw Artifact")
    return {
        "schema_version": SCOPED_SCHEMA,
        "run_id": receipt["run_id"],
        "operation": OPERATION,
        "source_run_id": SOURCE_RUN,
        "image_digest": receipt["image_digest"],
        "raw_artifact": {"id": artifact.id, "sha256": artifact.sha256},
        "raw_files": bindings,
        "verification_scope": copy.deepcopy(VERIFICATION_SCOPE),
        "legacy_rank_optimizer_claims_accepted": False,
        "legacy_manifest_overstates_rank_optimizer_verification": legacy,
        "limitations": [
            "Only the shared rank0 model and five saved optimizers were loaded and checked.",
            "Native model synchronization does not verify per-rank optimizer state.",
            "Rank1..3 optimizer readback and all live optimizer states were not observed.",
            "Raw receipt/report/manifest bytes are preserved; legacy rank flags are not accepted.",
            "This diagnostic is not a training restore checkpoint or numerical equivalence claim.",
        ],
    }


async def publish_scoped_verification(session, artifact, outputs, receipt):
    """Append an immutable inline JSON Artifact; never alter the original Artifact."""
    import hashlib

    from sqlalchemy import func, select

    from eb.ids import new_id
    from eb.orchestrator.ufo_native import verify_index
    from eb.store.models import Artifact, LineageEdge
    from eb.ufo_native import plan_hash

    payload = scoped_payload(outputs, receipt, artifact)
    if artifact.storage != str(outputs) or artifact.run_id != receipt["run_id"]:
        raise ValueError("Baseline scoped verification must share its raw Artifact storage and Run")
    # The original receipt is outside its own file index. Verify its bytes via
    # ebd as well; the report/native manifest and all payloads were indexed.
    await verify_index(outputs, [{"path": name, **binding} for name, binding in payload["raw_files"].items()])
    lock = int.from_bytes(
        hashlib.sha256((SCOPED_ARTIFACT_KIND + ":" + receipt["run_id"]).encode()).digest()[:8],
        "big",
        signed=True,
    )
    await session.execute(select(func.pg_advisory_xact_lock(lock)))
    digest = plan_hash(payload)
    scoped = await session.scalar(
        select(Artifact).where(Artifact.run_id == receipt["run_id"], Artifact.kind == SCOPED_ARTIFACT_KIND)
    )
    if scoped is None:
        scoped = Artifact(
            id=new_id("art"),
            kind=SCOPED_ARTIFACT_KIND,
            sha256=digest,
            size_bytes=len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()),
            storage=artifact.storage,
            run_id=receipt["run_id"],
            pinned=True,
            meta={"verification": payload, "storage_kind": "inline_json", "diagnostic_only": True},
        )
        session.add(scoped)
        await session.flush()
        for source_type, source_id, relation in (
            ("artifact", artifact.id, "derived"),
            ("run", receipt["run_id"], "produced"),
        ):
            session.add(
                LineageEdge(
                    src_type=source_type,
                    src_id=source_id,
                    dst_type="artifact",
                    dst_id=scoped.id,
                    relation=relation,
                    run_id=receipt["run_id"],
                )
            )
    elif scoped.sha256 != digest or (scoped.meta or {}).get("verification") != payload:
        raise ValueError("Immutable baseline scoped verification changed")
    return scoped


async def collect(session, job, run, outputs):
    from sqlalchemy import select

    from eb.ids import new_id
    from eb.store.models import Artifact, LineageEdge
    from eb.ufo_native import plan_hash

    plan = job.spec["params"]["baseline_plan"]
    receipt = read_receipt(outputs, plan, run.id)
    from eb.orchestrator.ufo_native import verify_index

    await verify_index(outputs, receipt["files"])
    digest = plan_hash(receipt)
    artifact = await session.scalar(
        select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == ARTIFACT_KIND)
    )
    if artifact is None:
        artifact = Artifact(
            id=new_id("art"),
            kind=ARTIFACT_KIND,
            sha256=digest,
            size_bytes=sum(row["bytes"] for row in receipt["files"]),
            storage=str(outputs),
            run_id=run.id,
            pinned=True,
            meta={
                "receipt": receipt,
                "diagnostic_only": True,
                "verification_scope": copy.deepcopy(VERIFICATION_SCOPE),
                "legacy_rank_optimizer_claims_accepted": False,
            },
        )
        session.add(artifact)
        await session.flush()
        for source_type, source_id, relation in (("run", run.id, "produced"), ("run", SOURCE_RUN, "derived")):
            session.add(
                LineageEdge(
                    src_type=source_type,
                    src_id=source_id,
                    dst_type="artifact",
                    dst_id=artifact.id,
                    relation=relation,
                    run_id=run.id,
                )
            )
    elif artifact.sha256 != digest:
        raise ValueError("Immutable baseline Artifact changed")
    scoped = await publish_scoped_verification(session, artifact, outputs, receipt)
    run.metrics_summary = {
        **(run.metrics_summary or {}),
        "native_baseline": {
            "artifact_id": artifact.id,
            "source_run_id": SOURCE_RUN,
            "actual_optimizer_updates": 112,
            "structural_checks_passed": True,
            "numerical_equivalence_claimed": False,
            "qualification": "diagnostic",
            "receipt": "baseline-receipt.json",
            "scoped_verification_artifact_id": scoped.id,
            "scoped_verification_sha256": scoped.sha256,
            "verification_scope": copy.deepcopy(VERIFICATION_SCOPE),
            "legacy_rank_optimizer_claims_accepted": False,
        },
    }


async def backfill_scoped_verification(session, job, run):
    """Operator domain maintenance for an already completed baseline, without a new Job."""
    from eb.settings import get_settings

    if (
        job.kind != "asset.validate"
        or job.status != "succeeded"
        or run.status != "finished"
        or run.job_id != job.id
        or job.spec.get("params", {}).get("operation") != OPERATION
    ):
        raise ValueError("Only a successful completed native baseline can publish scoped verification")
    await collect(session, job, run, get_settings().data_root / "runs" / run.id / "outputs")
    return run.metrics_summary["native_baseline"]
