"""Explicit recovery of the one verified19984 generation; no failed-Job promotion."""
from copy import deepcopy

from eb.pbfm_training import native_arguments, training_fingerprint

MODE = "training_long_recovery"
IMAGE = "eb/rlinf-pbfm:rpo-depth-recovery-v3"
SOURCE_IMAGE = "sha256:920414f6f14ce06fda68e6f8a59b939782e8ff2559fe98295083128e0959ac0d"
SOURCE_JOB = "job_022c870f9634445380ab"
SOURCE_RUN = "run_a54daed33ff7456eb77f"
SOURCE_COUNTS = {"optimizer_steps": 19984, "local_time": 2578432, "global_time": 10313728}
PARENT_HASH = "sha256:db8bf43117874c759e11a9e71fce44c12c510415b0a7be4a43ea9bb8273b913e"
MANIFEST_HASH = "sha256:d1d9d3773c8fc7738a405ca5eee8b56551bb9539a45c6ccf2e37573848732d62"
HISTORICAL_RECEIPTS = {
    "sha256:c95f1d36d90b0e6e1a154a6bf3955ef4a6ee978b63413c0884c8b9365812eea3":
        "sha256:d3c764c7192b25e67d9a00c91fa4b81d27713281121bc31c4ddd4626eb3a1e4a",
}


def recovery_metadata(source_identity):
    from eb.pbfm_longrun import runtime_compatibility_receipt

    if source_identity["image_digest"] != SOURCE_IMAGE:
        raise ValueError("P-BFM recovery source producer differs")
    return {"schema_version": "eb/pbfm-long-exploration/v1", "protocol": "recovery100000",
            "experiment_target_updates": 100000, "source_identity": deepcopy(source_identity),
            "source_fingerprint": training_fingerprint(source_identity),
            "runtime_receipt_sha256": training_fingerprint(runtime_compatibility_receipt()),
            "checkpoint_interval_global_steps": 5120000, "checkpoint_retention": 2}


def recovery_identity():
    return {"source_job_id": SOURCE_JOB, "source_run_id": SOURCE_RUN, "updates": 19984,
            "local_time": 2578432, "global_time": 10313728,
            "parent_plan_hash": PARENT_HASH, "manifest_sha256": MANIFEST_HASH}


def validate_recovery_plan(plan):
    source = {**plan["identity"], "image_digest": SOURCE_IMAGE}
    expected = recovery_metadata(source)
    receipt = plan["long_exploration"].get("runtime_receipt_sha256")
    if receipt is not None and HISTORICAL_RECEIPTS.get(plan["image_digest"]) == receipt:
        expected["runtime_receipt_sha256"] = receipt
    if (plan["long_exploration"] != expected or plan.get("recovery") != recovery_identity()
            or plan["start_updates"] != 19984 or plan["start_local_time"] != 2578432
            or not 19984 < plan["target_updates"] <= 100000
            or plan["image_digest"] == SOURCE_IMAGE or plan["world_size"] != 4
            or plan["identity"]["num_envs_per_rank"] != 2048
            or plan["identity"]["buffer_size"] != 131072
            or plan["communication_timeout_seconds"] != 900
            or not 60 <= plan["segment_timeout_seconds"] <= 108000):
        raise ValueError("P-BFM recovery identity/producer/counters/budget differs")


async def build_recovery_spec(body, session):
    from sqlalchemy import select

    from eb.domain.job import JobSpec
    from eb.pbfm_evaluation import verify_output_index
    from eb.pbfm_io import read_json, sha256
    from eb.pbfm_runtime_probe import _training_view
    from eb.pbfm_training_checkpoint import canonical_hash, validate_pbfm_checkpoint
    from eb.settings import get_settings
    from eb.store.ebd import EbdClient
    from eb.store.models import Job, Run

    if body.resume_job_id != SOURCE_JOB or body.recovery_update != 19984:
        raise ValueError("P-BFM recovery requires the reviewed failed Job and complete19984 generation")
    job = await session.get(Job, SOURCE_JOB)
    run = await session.scalar(select(Run).where(Run.id == SOURCE_RUN, Run.job_id == SOURCE_JOB))
    if (not job or job.status != "failed" or not run or run.job_id != job.id or run.status != "aborted"):
        raise ValueError("P-BFM recovery parent must remain a failed, stopped Run")
    client = EbdClient()
    state = await client.get_run(SOURCE_RUN)
    if state.get("status") != "exited":
        raise ValueError("P-BFM recovery source runtime is not exited")
    source = JobSpec.model_validate(job.spec)
    training = _training_view(source)
    if source.image != SOURCE_IMAGE or source.params["probe_mode"] != "training_long_exploration":
        raise ValueError("P-BFM recovery parent producer/mode differs")
    root = get_settings().data_root / "runs" / SOURCE_RUN
    parent = read_json(root / "pbfm-train-plan.json", maximum=4 * 1024**2)
    if (parent.get("hash") != PARENT_HASH or canonical_hash(parent) != PARENT_HASH
            or {k:v for k,v in parent.items() if k not in {"hash", "run_id"}}
            != training.params["native_train_plan"] or parent["run_id"] != SOURCE_RUN):
        raise ValueError("P-BFM recovery parent frozen plan differs")
    recovery = root / "outputs/recovery"
    index = read_json(recovery / "index.json", maximum=4 * 1024**2)
    entry = {"path": "generations/u19984/native", "counts": SOURCE_COUNTS,
             "manifest_sha256": MANIFEST_HASH.removeprefix("sha256:")}
    if (index.get("schema_version") != "eb/pbfm-periodic-recovery-index/v1"
            or index.get("run_id") != SOURCE_RUN or index.get("parent_plan_hash") != PARENT_HASH
            or index.get("parent_target_updates") != 100000
            or index.get("latest") != entry or index.get("generations") != [entry]):
        raise ValueError("P-BFM complete recovery index differs")
    native = recovery / entry["path"]
    manifest = read_json(native / "checkpoint-manifest.json", maximum=4 * 1024**2)
    receipt_path = native.parent / "receipt.json"
    receipt = read_json(receipt_path, maximum=4 * 1024**2)
    expected_receipt = {"schema_version": "eb/pbfm-periodic-recovery/v1", "status": "complete",
                        "training_complete": False, "run_id": SOURCE_RUN,
                        "parent_plan_hash": PARENT_HASH, **entry}
    if ("sha256:" + sha256(native / "checkpoint-manifest.json") != MANIFEST_HASH
            or receipt != expected_receipt or manifest["plan_hash"] != PARENT_HASH):
        raise ValueError("P-BFM recovery receipt/manifest differs")
    checked = validate_pbfm_checkpoint(native, manifest, run_id=SOURCE_RUN, plan=parent, resume=True)
    if checked["counts"] != SOURCE_COUNTS:
        raise ValueError("P-BFM recovery actual counters differ")
    for rank in manifest["rank_results"]:
        if (len(rank["optimizer_state_steps"]) != 8 or rank["replay_rows"] != 64
                or any(v != [19984] for v in rank["optimizer_state_steps"].values())):
            raise ValueError("P-BFM recovery native optimizer/Replay proof differs")
    await verify_output_index(native, manifest["files"], {row["path"] for row in manifest["files"]})
    new_image = await client.image_info(IMAGE)
    old_image = await client.image_info(SOURCE_IMAGE)
    if (not new_image.get("ok") or not new_image.get("digest")
            or old_image.get("digest") != SOURCE_IMAGE or not old_image.get("ok")):
        raise ValueError("P-BFM recovery/source producer image unavailable")
    plan = deepcopy(training.params["native_train_plan"])
    identity = plan["identity"]
    if (body.dataset_id != identity["dataset_id"]
            or body.environment_id != identity["environment_profile"]["id"]):
        raise ValueError("P-BFM recovery cannot change frozen Dataset/Profile")
    plan["long_exploration"] = recovery_metadata(identity)
    plan["identity"]["image_digest"] = plan["image_digest"] = new_image["digest"]
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    binding = {"schema_version": "eb/pbfm-native-resume-source/v1", "run_id": SOURCE_RUN,
               "manifest_sha256": MANIFEST_HASH, "receipt_sha256": "sha256:" + sha256(receipt_path),
               "recovery_index_sha256": "sha256:" + sha256(recovery / "index.json"), "artifact": None}
    plan.update(recovery=recovery_identity(), start_updates=19984, start_local_time=2578432,
                target_updates=100000, segment_timeout_seconds=body.timeout_seconds,
                resume={"host_path": str(native), "checkpoint_path": "/resume-native/checkpoint",
                        "step": 19984, "source_run_id": SOURCE_RUN, "source_binding": binding})
    plan["native_args"] = native_arguments(plan)
    params = deepcopy(source.params)
    params.update(native_train_plan=plan, probe_mode=MODE, model_name=body.name,
                  source_diagnostic_job_id=SOURCE_JOB, native_resume_source=binding,
                  resume_step=19984)
    return source.model_copy(deep=True, update={"name": body.name, "params": params,
        "image": new_image["digest"], "inputs": source.inputs.model_copy(update={"resume_from": SOURCE_RUN}),
        "policy": source.policy.model_copy(update={"timeout_h": body.timeout_seconds / 3600})})
