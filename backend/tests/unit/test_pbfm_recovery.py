from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_pbfm_longrun import long_plan

from eb.pbfm_runtime_probe import ProbeIn
from eb.pbfm_training import native_arguments, training_fingerprint, validate_frozen_plan


def recovery_request(**changes):
    return ProbeIn(name="recover100000", dataset_id="ds_pbfm", environment_id="env_train_pbfm",
                   mode="training_long_recovery", gpus=4, num_envs_per_rank=2048,
                   target_updates=100000, resume_job_id="job_022c870f9634445380ab",
                   recovery_update=19984, timeout_seconds=108000, **changes)


def recovery_plan(tmp_path):
    from eb.pbfm_recovery import SOURCE_IMAGE, recovery_metadata

    source = long_plan(tmp_path)
    source["identity"]["image_digest"] = source["image_digest"] = SOURCE_IMAGE
    source["fingerprint"] = training_fingerprint(source["identity"])
    plan = deepcopy(source)
    plan["long_exploration"] = recovery_metadata(source["identity"])
    plan["identity"]["image_digest"] = plan["image_digest"] = "sha256:" + "c" * 64
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["recovery"] = {"source_job_id": "job_022c870f9634445380ab",
        "source_run_id": "run_a54daed33ff7456eb77f", "updates": 19984,
        "local_time": 2578432, "global_time": 10313728,
        "parent_plan_hash": "sha256:db8bf43117874c759e11a9e71fce44c12c510415b0a7be4a43ea9bb8273b913e",
        "manifest_sha256": "sha256:d1d9d3773c8fc7738a405ca5eee8b56551bb9539a45c6ccf2e37573848732d62"}
    plan.update(start_updates=19984, start_local_time=2578432)
    plan["native_args"] = native_arguments(plan)
    return plan


def test_recovery_request_is_explicit_not_historical_or_arbitrary_path():
    assert recovery_request().recovery_update == 19984
    data = recovery_request().model_dump()
    for changes in ({"mode": "training_long_exploration"}, {"recovery_update": 23168},
                    {"gpus": 1}, {"validation_report_id": "ev_fake"}):
        with pytest.raises(ValueError):
            ProbeIn(**{**data, **changes})


def test_recovery_plan_uses_actual_source_identity_and_new_resumed_endpoint(tmp_path):
    from eb.pbfm_longrun import checkpoint_source_plan

    plan = recovery_plan(tmp_path)
    validate_frozen_plan(plan)
    projected = checkpoint_source_plan(plan)
    assert projected["identity"] == plan["long_exploration"]["source_identity"]
    assert projected["image_digest"] != plan["image_digest"]
    assert plan["native_args"][plan["native_args"].index("--num-env-steps")+1] == "51290112"
    for change in ("updates", "manifest_sha256", "source_run_id"):
        altered = deepcopy(plan)
        altered["recovery"][change] = 23168 if change == "updates" else "changed"
        with pytest.raises(ValueError):
            validate_frozen_plan(altered)


@pytest.mark.asyncio
@pytest.mark.parametrize("job_status,run_status,ebd_status", [
    ("running", "running", "running"), ("succeeded", "succeeded", "exited"),
    ("failed", "running", "running"), ("failed", "aborted", "running"),
])
async def test_recovery_builder_rejects_live_or_unreviewed_parent_before_payload_io(
    monkeypatch, job_status, run_status, ebd_status,
):
    from eb.pbfm_recovery import SOURCE_JOB, SOURCE_RUN, build_recovery_spec

    job = SimpleNamespace(id=SOURCE_JOB, status=job_status, spec={})
    run = SimpleNamespace(id=SOURCE_RUN, job_id=SOURCE_JOB, status=run_status)
    session = SimpleNamespace(get=AsyncMock(return_value=job), scalar=AsyncMock(return_value=run))
    client = SimpleNamespace(get_run=AsyncMock(return_value={"status": ebd_status}))
    monkeypatch.setattr("eb.store.ebd.EbdClient", lambda: client)
    with pytest.raises(ValueError, match="failed, stopped|runtime is not exited"):
        await build_recovery_spec(recovery_request(), session)
    if job_status != "failed" or run_status != "aborted":
        client.get_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_builder_cannot_select_another_job_or_unsaved_update():
    from eb.pbfm_recovery import build_recovery_spec

    for changes in ({"resume_job_id": "job_other"}, {"recovery_update": 23168}):
        # Bypass request parsing deliberately: the shared builder still protects its own boundary.
        body = recovery_request().model_copy(update=changes)
        with pytest.raises(ValueError, match="reviewed failed Job and complete19984"):
            await build_recovery_spec(body, None)


def test_frozen_recovery_survives_later_runtime_changes_with_exact_producer_receipt_pair(
    tmp_path, monkeypatch,
):
    from eb import pbfm_longrun

    plan = recovery_plan(tmp_path)
    producer = "sha256:c95f1d36d90b0e6e1a154a6bf3955ef4a6ee978b63413c0884c8b9365812eea3"
    receipt = "sha256:d3c764c7192b25e67d9a00c91fa4b81d27713281121bc31c4ddd4626eb3a1e4a"
    plan["image_digest"] = plan["identity"]["image_digest"] = producer
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["long_exploration"]["runtime_receipt_sha256"] = receipt
    monkeypatch.setattr(pbfm_longrun, "runtime_compatibility_receipt", lambda: {"later_runtime": True})
    validate_frozen_plan(plan)
    for image, frozen_receipt in (("sha256:" + "e"*64, receipt), (producer, "sha256:" + "0"*64),
                                  ("sha256:" + "e"*64, None)):
        changed = deepcopy(plan)
        changed["image_digest"] = changed["identity"]["image_digest"] = image
        changed["fingerprint"] = training_fingerprint(changed["identity"])
        changed["long_exploration"]["runtime_receipt_sha256"] = frozen_receipt
        with pytest.raises(ValueError):
            validate_frozen_plan(changed)
