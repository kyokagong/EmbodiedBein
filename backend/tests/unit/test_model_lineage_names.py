from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eb.api.routes import model_lineage
from eb.store.models import Job, ModelVersion, Run, TrainJob, WeightModel


@pytest.mark.asyncio
async def test_lineage_resolves_model_and_training_names_with_detail_links():
    model = WeightModel(id="mdl_1", name="步态模型", path="/models/1")
    train = TrainJob(id="trn_1", name="步态训练", job_id="job_1")
    edge = SimpleNamespace(src_type="train_job", src_id="trn_1", dst_type="model",
                           dst_id="mdl_1", relation="produced_by")
    session = AsyncMock()
    session.scalars = AsyncMock(return_value=SimpleNamespace(all=lambda: [edge]))
    async def get(cls, key):
        return {
        (WeightModel, "mdl_1"): model, (TrainJob, "trn_1"): train,
        }.get((cls, key))
    session.get = get

    result = await model_lineage("mdl_1", session, SimpleNamespace(id="user:kk"))

    assert {(n["name"], n["href"]) for n in result["nodes"]} == {
        ("步态模型", "/models/mdl_1"), ("步态训练", "/train/trn_1")}
    assert result["edges"][0]["relation"] == "produced_by"


@pytest.mark.asyncio
async def test_auto_model_lineage_links_run_to_training_task():
    model = ModelVersion(id="mv_1", family_id="mf_1", name="自动模型")
    train = TrainJob(id="trn_1", name="自动训练", job_id="job_1")
    run = Run(id="run_1", job_id="job_1")
    job = Job(id="job_1", spec={"name": "自动训练"})
    edge = SimpleNamespace(src_type="run", src_id="run_1", dst_type="model_version",
                           dst_id="mv_1", relation="produced")
    session = AsyncMock()
    session.scalars = AsyncMock(return_value=SimpleNamespace(all=lambda: [edge]))
    session.scalar = AsyncMock(return_value=train)
    async def get(cls, key):
        return {(ModelVersion, "mv_1"): model, (Run, "run_1"): run,
                (Job, "job_1"): job}.get((cls, key))
    session.get = get

    result = await model_lineage("mv_1", session, SimpleNamespace(id="user:kk"))

    assert {(n["name"], n["href"]) for n in result["nodes"]} == {
        ("自动模型", "/models/mv_1"), ("自动训练", "/train/trn_1")}
