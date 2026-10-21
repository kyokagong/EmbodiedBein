import copy
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from eb.store.models import Dataset, DatasetVersion, LineageEdge
from eb.ufo_native import ENV_ID
from eb.ufo_training import DATA_OPERATION
from eb.ufo_training_dataset import publish_native_dataset_version


def frozen_dataset(tmp_path):
    run = NS(id="run_data", job_id="job_data", status="finished")
    job = NS(id="job_data", kind="data.convert", status="succeeded", spec={"params": {
        "operation": DATA_OPERATION, "dataset_id": "ds_native"}})
    root = tmp_path / "runs/run_data/outputs/data"
    root.mkdir(parents=True)
    (root / "train.pkl").write_bytes(b"frozen native motion; never deserialize in the API")
    manifest = {"schema_version": "eb/ufo-native-motion/v1", "dataset_id": "ds_native", "split": "train",
                "train_file": "train.pkl", "episodes_count": 3, "fps": 50, "joint_order": ["joint"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    stats = {"format": "ufo_native_motion", "storage_ref": str(root), "episodes_count": 3, "fps": 50,
        "source_refs": [{"kind": "ufo_native_motion", "uri": "readonly/source"}],
        "total_bytes": (root / "train.pkl").stat().st_size,
        "validation": {"status": "validated", "scope": "ufo_native_motion_v1", "training_ready": True,
                       "manifest_sha256": digest, "storage_ref": str(root), "errors": []}}
    row = Dataset(id="ds_native", name="native data", source_kind="ufo_native_motion", status="validated",
        source_uri="readonly/source", embodiment_env_id=ENV_ID, dest_path=str(root),
        format="ufo_native_motion", summary_stats=stats,
        meta={"usage": "train", "validation_job_id": job.id, "validation_run_id": run.id})
    return row, job, run, root


def session_fixture(*responses):
    rows = []
    session = NS(execute=AsyncMock(), scalar=AsyncMock(side_effect=responses), add=rows.append,
                 flush=AsyncMock(), commit=AsyncMock())
    return session, rows


@pytest.mark.asyncio
async def test_publication_freezes_manifest_stats_and_producer_lineage_and_repeats_without_mutation(tmp_path):
    row, job, run, root = frozen_dataset(tmp_path)
    session, rows = session_fixture(None, None, None)
    assert await publish_native_dataset_version(session, row, job, run) is True
    version = next(item for item in rows if isinstance(item, DatasetVersion))
    assert version.version == "native-v1" and version.dataset_id == row.id
    assert version.content_hash == "sha256:" + row.summary_stats["validation"]["manifest_sha256"]
    assert version.compatible_envs == [ENV_ID]
    assert version.stats["manifest"] == json.loads((root / "manifest.json").read_text())
    assert version.stats["storage_ref"] == str(root) and version.stats["producer_run"] == run.id
    edges = [item for item in rows if isinstance(item, LineageEdge)]
    assert {(edge.src_type, edge.src_id, edge.dst_type, edge.dst_id, edge.relation) for edge in edges} == {
        ("run", run.id, "dataset_version", version.id, "produced"),
        ("dataset", row.id, "dataset_version", version.id, "has_version"),
    }
    assert "FOR UPDATE" in str(session.execute.await_args.args[0])
    published_stats = copy.deepcopy(version.stats)
    row.summary_stats["episodes_count"] = 999
    session, rows = session_fixture(version, 1, 2)
    assert await publish_native_dataset_version(session, row, job, run) is False
    assert version.stats == published_stats and version.stats["episodes_count"] == 3
    assert not rows


@pytest.mark.asyncio
async def test_publication_refuses_an_existing_version_digest_conflict(tmp_path):
    row, job, run, _ = frozen_dataset(tmp_path)
    version = DatasetVersion(id="dsv_existing", dataset_id=row.id, version="native-v1",
        content_hash="sha256:" + "f" * 64, compatible_envs=[ENV_ID],
        stats={"storage_ref": row.dest_path, "producer_run": run.id, "unchanged": True})
    session, rows = session_fixture(version)
    previous = copy.deepcopy(version.stats)
    with pytest.raises(ValueError, match="identity changed"):
        await publish_native_dataset_version(session, row, job, run)
    assert not rows and version.stats == previous
    assert version.content_hash == "sha256:" + "f" * 64


@pytest.mark.asyncio
async def test_backfill_refuses_changed_native_manifest_before_database_publication(tmp_path):
    row, job, run, root = frozen_dataset(tmp_path)
    (root / "manifest.json").write_text('{"train_file":"train.pkl","changed":true}')
    session, rows = session_fixture()
    with pytest.raises(ValueError, match="manifest changed"):
        await publish_native_dataset_version(session, row, job, run)
    session.execute.assert_not_awaited()
    assert not rows


@pytest.mark.asyncio
@pytest.mark.parametrize("attribute,status", [
    ("job", "failed"), ("job", "cancelled"), ("job", "infra_failed"), ("job", "running"),
    ("row", "invalid"), ("row", "registered"), ("run", "aborted"), ("run", "running"),
])
async def test_only_successful_validated_native_data_can_publish(tmp_path, attribute, status):
    row, job, run, _ = frozen_dataset(tmp_path)
    setattr({"row": row, "job": job, "run": run}[attribute], "status", status)
    session, rows = session_fixture()
    assert await publish_native_dataset_version(session, row, job, run) is False
    assert not rows
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_cached_validation_backfills_without_reparsing_or_mutating_native_bytes(tmp_path, monkeypatch):
    from eb.api.routes_data import reconcile_dataset

    row, job, run, root = frozen_dataset(tmp_path)
    (root.parent / "dataset-index.json").write_text("do not parse or rerun frozen validation")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    stats = copy.deepcopy(row.summary_stats)
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    session, rows = session_fixture(run, None, None, None)
    session.get = AsyncMock(return_value=job)
    await reconcile_dataset(row, session)
    assert len([item for item in rows if isinstance(item, DatasetVersion)]) == 1
    assert len([item for item in rows if isinstance(item, LineageEdge)]) == 2
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert row.summary_stats == stats and row.status == "validated"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_newly_successful_validation_publishes_through_the_same_reconcile_hook(tmp_path, monkeypatch):
    from eb.api.routes_data import reconcile_dataset

    row, job, run, root = frozen_dataset(tmp_path)
    (root.parent / "dataset-index.json").write_text(json.dumps(row.summary_stats))
    row.status, row.dest_path = "queued", "readonly/source"
    row.meta = {"usage": "train", "validation_job_id": job.id}
    row.summary_stats = {"validation": {"status": "not_validated"}}
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    session, rows = session_fixture(run, None, None, None)
    session.get = AsyncMock(return_value=job)
    await reconcile_dataset(row, session)
    version = next(item for item in rows if isinstance(item, DatasetVersion))
    assert version.stats["producer_run"] == run.id
    assert row.status == "validated" and row.dest_path == str(root)
    session.commit.assert_awaited_once()
