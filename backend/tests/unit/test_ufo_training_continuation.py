from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from test_ufo_resume_preview import resume_context
from test_ufo_training import native_spec

from eb.api.routes_train import ResumeTrainIn, build_resume_spec, get_train


def test_experiment_goal_survives_interval_resume_without_changing_native_identity(monkeypatch):
    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    body = body.model_copy(update={"native_experiment_target_updates": 100000})
    spec = build_resume_spec(source, "trn_parent", checkpoint, body)
    assert spec.params["native_train_plan"]["experiment_target_updates"] == 100000
    assert "experiment_target_updates" not in source.params["native_train_plan"]
    assert spec.params["native_train_plan"]["identity"] == source.params["native_train_plan"]["identity"]
    inherited = build_resume_spec(
        spec, "trn_child", checkpoint, ResumeTrainIn(name="next", target_updates=1008)
    )
    assert inherited.params["native_train_plan"]["experiment_target_updates"] == 100000


@pytest.mark.parametrize("goal", [100, 511, 1000001, True])
def test_invalid_overall_targets_are_rejected(monkeypatch, goal):
    from pydantic import ValidationError

    (_, _, source, checkpoint), body = resume_context(monkeypatch)
    with pytest.raises((ValidationError, ValueError)):
        body = ResumeTrainIn(name="next", target_updates=512, native_experiment_target_updates=goal)
        build_resume_spec(source, "trn_parent", checkpoint, body)


def test_frozen_plan_cannot_claim_a_goal_below_its_interval():
    from eb.ufo_training import validate_train_job

    spec = native_spec()
    spec.params["native_train_plan"]["experiment_target_updates"] = 100
    with pytest.raises(ValueError, match="experiment"):
        validate_train_job(spec)


@pytest.mark.asyncio
async def test_completed_interval_exposes_live_child_without_rewriting_parent(monkeypatch):
    parent=NS(id="trn_parent", template_id="rpo_ufo_fb_native")
    child=NS(id="trn_child", template_id="rpo_ufo_fb_native")
    session=NS(get=AsyncMock(return_value=parent), scalars=AsyncMock(return_value=NS(all=lambda:[child])))
    synced=[]
    async def sync(session,row): synced.append(row.id)
    def output(row):
        return {"id":row.id,"status":"succeeded" if row is parent else "running",
                "params":{"native_train_plan":{"target_updates":20000 if row is parent else 30000,
                          "experiment_target_updates":100000}}}
    monkeypatch.setattr("eb.api.routes_train._sync_run",sync)
    monkeypatch.setattr("eb.api.routes_train._job_out",output)
    result=await get_train(parent.id,session,NS())
    assert result["status"]=="succeeded"
    assert result["continuations"][0]["id"]==child.id
    assert result["continuations"][0]["status"]=="running"
    assert synced==[parent.id,child.id]


@pytest.mark.asyncio
async def test_non_native_detail_does_not_query_continuations(monkeypatch):
    row=NS(template_id="libero_pi05_sft")
    session=NS(get=AsyncMock(return_value=row),scalars=AsyncMock())
    monkeypatch.setattr("eb.api.routes_train._sync_run",AsyncMock())
    monkeypatch.setattr("eb.api.routes_train._job_out",lambda row:{"id":"trn_sft"})
    assert await get_train("trn_sft",session,NS())=={"id":"trn_sft"}
    session.scalars.assert_not_awaited()
