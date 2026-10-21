from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from eb.api.routes_train import ResumeTrainIn, build_resume_spec, preview_resume, submit_resume
from eb.domain.job import JobSpec
from eb.ufo_native import ENV_ID, TASK_ID, plan_hash
from eb.ufo_training import native_arguments, training_fingerprint


def resume_context(monkeypatch):
    identity = {"world_size": 1, "buffer_size": 262144, "seed": 42}
    plan = {
        "identity": identity,
        "world_size": 1,
        "target_updates": 112,
        "fingerprint": training_fingerprint(identity),
        "start_updates": 0,
        "start_local_time": 0,
        "segment_timeout_seconds": 20011,
    }
    plan["native_args"] = native_arguments(plan)
    source = JobSpec(
        kind="train.rl",
        name="source",
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID,
        image="sha256:" + "a" * 64,
        params={
            "template_id": "rpo_ufo_fb_native",
            "stack": "rlinf",
            "native_train_plan": plan,
            "model_name": "source model",
        },
        resources={"gpus": 1, "cpus": 8, "memory_gb": 32, "shm_gb": 8},
        policy={"timeout_h": 20011 / 3600, "max_retries": 0, "retry_on": []},
    )
    checkpoint = {
        "run_id": "run_source",
        "step": 112,
        "mode": "ufo_native_state",
        "native_path": "/state/runs/run_source/outputs/native",
        "state_coverage": {"model": "restored"},
        "compatibility_fingerprint": plan["fingerprint"],
        "manifest": {"counts": {"optimizer_steps": 112, "local_time": 18432, "global_time": 18432}},
    }
    checkpoint["native_resume_source"] = {
        "schema_version": "eb/ufo-native-resume-source/v1",
        "run_id": "run_source",
        "manifest_sha256": "sha256:" + "b" * 64,
        "receipt_sha256": "sha256:" + "c" * 64,
        "artifact": None,
    }
    source_row = NS(embodiment_id="emb_roboparty_rpo", dataset_ids=["ds_native"], base_model_id=None)
    context = (source_row, NS(id="job_source"), source, checkpoint)
    monkeypatch.setattr("eb.api.routes_train._resume_context", AsyncMock(return_value=context))
    rendered = NS(command=["native-worker"], mounts=[], image=source.image)
    monkeypatch.setattr("eb.adapters.get_adapter", lambda spec: NS(render=lambda *args: rendered))
    body = ResumeTrainIn(name="continued", model_name="new model", checkpoint_step=112, target_updates=512)
    return context, body


@pytest.mark.asyncio
async def test_native_resume_candidate_requires_recorded_complete_state_verification(monkeypatch):
    from eb.api.routes_train import get_resume_candidate

    (row, _, source, checkpoint), _ = resume_context(monkeypatch)
    candidate = await get_resume_candidate("trn_source", NS(), NS())
    assert candidate["available"] is False
    source.params["native_train_plan"]["full_state_finite_required"] = True
    checkpoint.update(full_state_finite_verified=True, bytes=1024)
    row.id = "trn_source"
    candidate = await get_resume_candidate("trn_source", NS(), NS())
    assert candidate["available"] is True and candidate["suggested_target"] == 512


@pytest.mark.asyncio
async def test_native_resume_preview_binds_the_entire_frozen_spec_including_names(monkeypatch):
    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    preview = await preview_resume("trn_source", body, NS(), NS(id="user:kk"))
    expected = build_resume_spec(source, "trn_source", checkpoint, body)
    assert preview["preview_hash"] == plan_hash(expected.model_dump(mode="json"))
    renamed = await preview_resume(
        "trn_source", body.model_copy(update={"name": "renamed"}), NS(), NS(id="user:kk")
    )
    assert renamed["preview_hash"] != preview["preview_hash"]
    renamed_body = body.model_copy(update={"model_name": "another model"})
    model_renamed = await preview_resume("trn_source", renamed_body, NS(), NS(id="user:kk"))
    assert model_renamed["preview_hash"] != preview["preview_hash"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["missing", "name", "model_name", "target", "checkpoint", "source", "budget", "manifest", "receipt"],
)
async def test_native_resume_rejects_missing_or_stale_preview_before_any_write(monkeypatch, change):
    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    preview = await preview_resume("trn_source", body, NS(), NS(id="user:kk"))
    body.preview_hash = preview["preview_hash"]
    if change == "missing":
        body.preview_hash = None
    elif change in {"name", "model_name"}:
        setattr(body, change, "changed name")
    elif change == "target":
        body.target_updates = 1008
    elif change == "checkpoint":
        checkpoint["run_id"] = "run_new_source"
        checkpoint["native_path"] = "/state/runs/run_new_source/outputs/native"
        checkpoint["native_resume_source"]["run_id"] = "run_new_source"
    elif change in {"manifest", "receipt"}:
        checkpoint["native_resume_source"][change + "_sha256"] = "sha256:" + "0" * 64
    elif change == "budget":
        body.timeout_seconds = 3600
    else:
        source.inputs.datasets = [{"id": "ds_other", "path": "/state/other"}]
    submit = AsyncMock()
    monkeypatch.setattr("eb.api.routes_train.orch.submit", submit)
    session = NS(execute=AsyncMock(), scalar=AsyncMock(), commit=AsyncMock())
    with pytest.raises(HTTPException) as expired:
        await submit_resume("trn_source", body, session, NS(id="user:kk"), "resume-key")
    assert expired.value.status_code == 409
    assert "预览" in expired.value.detail
    submit.assert_not_awaited()
    session.execute.assert_not_awaited()
    session.scalar.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 3600])
async def test_matching_native_resume_preview_submits_the_exact_reviewed_spec(monkeypatch, timeout):
    from eb.store.models import TrainJob

    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    body.timeout_seconds = timeout
    reviewed = build_resume_spec(source, "trn_source", checkpoint, body)
    preview = await preview_resume("trn_source", body, NS(), NS(id="user:kk"))
    body.preview_hash = preview["preview_hash"]
    rows = []
    session = NS(
        execute=AsyncMock(),
        scalar=AsyncMock(return_value=None),
        add=rows.append,
        commit=AsyncMock(),
        refresh=AsyncMock(),
    )
    submit = AsyncMock(return_value=NS(id="job_child", status="queued"))
    monkeypatch.setattr("eb.api.routes_train.orch.submit", submit)
    result = await submit_resume("trn_source", body, session, NS(id="user:kk"), "resume-key")
    assert result["resumed_from"] == "trn_source" and result["resume_step"] == 112
    submitted = submit.await_args.args[1]
    assert submitted.model_dump(mode="json") == reviewed.model_dump(mode="json")
    assert any(isinstance(row, TrainJob) for row in rows)
    session.commit.assert_awaited_once()


def test_resume_hash_field_requires_a_sha256_digest():
    from pydantic import ValidationError

    for value in ("unfrozen", "sha256:1234", True, 123):
        with pytest.raises(ValidationError):
            ResumeTrainIn(name="continued", target_updates=512, preview_hash=value)
    assert ResumeTrainIn(name="legacy", max_steps=200).preview_hash is None


@pytest.mark.parametrize("value", [True, False, 59, 21601, 60.0, "60"])
def test_native_resume_timeout_requires_a_strict_bounded_integer(value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResumeTrainIn(name="continued", target_updates=512, timeout_seconds=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 60, 3600, 20011])
async def test_native_resume_timeout_is_frozen_in_plan_policy_and_preview(monkeypatch, timeout):
    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    original = source.model_dump(mode="json")
    unchanged_budget = build_resume_spec(source, "trn_source", checkpoint, body)
    body = ResumeTrainIn(**{**body.model_dump(), "timeout_seconds": timeout})
    resumed = build_resume_spec(source, "trn_source", checkpoint, body)
    expected = 20011 if timeout is None else timeout
    plan = resumed.params["native_train_plan"]
    assert plan["segment_timeout_seconds"] == expected
    assert resumed.policy.timeout_h == expected / 3600
    assert resumed.policy.max_retries == source.policy.max_retries
    for key in ("identity", "fingerprint", "native_args"):
        assert plan[key] == unchanged_budget.params["native_train_plan"][key]
    assert source.model_dump(mode="json") == original
    preview = await preview_resume("trn_source", body, NS(), NS(id="user:kk"))
    assert preview["timeout_seconds"] == expected
    assert preview["policy"] == resumed.policy.model_dump(mode="json")
    assert preview["preview_hash"] == plan_hash(resumed.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_native_resume_cannot_expand_the_source_timeout(monkeypatch):
    (_, _, _, _), body = resume_context(monkeypatch)
    body = ResumeTrainIn(**{**body.model_dump(), "timeout_seconds": 20012})
    with pytest.raises(HTTPException) as rejected:
        await preview_resume("trn_source", body, NS(), NS(id="user:kk"))
    assert rejected.value.status_code == 422
    assert "timeout_seconds" in rejected.value.detail


def test_native_resume_cannot_expand_a_shorter_source_policy(monkeypatch):
    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    source.policy.timeout_h = 1
    body = ResumeTrainIn(**{**body.model_dump(), "timeout_seconds": 3601})
    with pytest.raises(ValueError, match="timeout_seconds"):
        build_resume_spec(source, "trn_source", checkpoint, body)


@pytest.mark.parametrize(
    "template,kind,limit",
    [
        ("libero_pi05_sft", "train.sft", {"max_steps": 200}),
        ("libero_spatial_pi05_ppo", "train.rlft", {"max_epochs": 200}),
    ],
)
def test_legacy_resume_rejects_native_wall_clock_budget(template, kind, limit):
    from eb.checkpoints import resume_fingerprint

    source = JobSpec(
        kind=kind, embodiment_env_id="env_legacy", task_id="tsk_legacy", params={"template_id": template}
    )
    checkpoint = {"compatibility_fingerprint": resume_fingerprint(source)}
    body = ResumeTrainIn(name="continued", timeout_seconds=60, **limit)
    with pytest.raises(ValueError, match="timeout_seconds"):
        build_resume_spec(source, "trn_legacy", checkpoint, body)
