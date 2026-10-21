import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eb.adapters.data_convert import DataConvertAdapter
from eb.api import routes_data
from eb.runners.ufo_training_data import validate_groups, validate_records
from eb.store.models import Dataset
from eb.ufo_native import ENV_ID


def qualification():
    return {"train_groups": ["train"], "validation_groups": ["validation"], "test_groups": ["test"]}


def test_native_source_groups_require_nonempty_disjoint_original_test_groups():
    provenance = {"splits": {"test": {"groups": ["test"]}}}
    assert validate_groups(qualification(), provenance)["train"] == {"train"}
    for value in (["train"], [], ["test", "test"]):
        bad = {**qualification(), "test_groups": value}
        with pytest.raises(ValueError):
            validate_groups(bad, provenance)


def test_native_records_reject_nan_joint_order_and_provenance(monkeypatch):
    import numpy as np

    # This unit isolates the extra platform checks; the actual MotionLib is
    # exercised separately inside the native runtime image on the frozen source.
    monkeypatch.setitem(
        sys.modules, "ufo.motion.formats", SimpleNamespace(validate_ufo_motion_dict=lambda *args: None)
    )
    robot = SimpleNamespace(control_joint_names=["joint"], body_names=["base", "link"])
    record = {
        "root_trans_offset": np.zeros((3, 3)),
        "pose_aa": np.zeros((3, 2, 3)),
        "dof_pos": np.zeros((3, 1)),
        "fps": 50,
        "joint_names": ["joint"],
        "metadata": {
            "motion_key": "motion",
            "source_group": "train",
            "split": "train",
            "source_path": "original.npz",
            "source_sha256": "source_hash",
        },
    }
    provenance = {
        "audit": [{"name": "motion", "group": "train", "source": "original.npz", "sha256": "source_hash"}]
    }
    report = validate_records({"motion__clip000": record}, "train", {"train"}, robot, provenance)
    assert report["episodes_count"] == 1 and report["duration_s"] == 0.04
    bad = copy.deepcopy(record)
    bad["pose_aa"][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_records({"motion__clip000": bad}, "train", {"train"}, robot, provenance)
    for key, value in (("joint_names", ["other"]), ("dof_pos", np.zeros((3, 2)))):
        bad = {**record, key: value}
        with pytest.raises(ValueError):
            validate_records({"motion__clip000": bad}, "train", {"train"}, robot, provenance)
    bad = {**record, "metadata": {**record["metadata"], "source_sha256": "changed"}}
    with pytest.raises(ValueError, match="audit"):
        validate_records({"motion__clip000": bad}, "train", {"train"}, robot, provenance)


def source_fixture(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        ufo_native_root=tmp_path / "ufo", data_root=tmp_path / "state", assets_root=tmp_path / "assets"
    )
    source = settings.ufo_native_root / "runs/rpo_reference_v4"
    for relative in (
        "train_clipped.pkl",
        "validation.pkl",
        "qualification.json",
        "physics_audit/mask_feasibility.json",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture, never deserialized in the API")
    provenance = settings.ufo_native_root / "runs/rpo_reference_v2/import_report.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}")
    monkeypatch.setattr(routes_data, "get_settings", lambda: settings)
    monkeypatch.setattr("eb.settings.get_settings", lambda: settings)
    row = Dataset(
        id="ds_test",
        source_kind="ufo_native_motion",
        source_uri=str(source),
        dest_path=str(source),
        embodiment_env_id=ENV_ID,
        meta={"usage": "train"},
    )
    return settings, source, row


@pytest.mark.asyncio
async def test_native_data_spec_freezes_sources_and_adapter_mounts_only_managed_outputs(
    tmp_path, monkeypatch
):
    settings, source, row = source_fixture(tmp_path, monkeypatch)
    snapshots = []
    for char in ("a", "b"):
        path = settings.data_root / "snapshots" / (char * 64) / "code"
        path.mkdir(parents=True)
        snapshots.append({"path": str(path), "content_hash": "sha256:" + char * 64})
    client = SimpleNamespace(
        image_info=AsyncMock(return_value={"ok": True, "digest": "sha256:" + "c" * 64}),
        snapshot=AsyncMock(side_effect=snapshots),
    )
    monkeypatch.setattr("eb.store.ebd.EbdClient", lambda: client)
    spec = await routes_data.ufo_data_spec(row)
    assert spec.params["src"] == row.source_uri
    assert spec.resources.gpus == 0 and spec.resources.cpus == 4 and spec.resources.memory_gb == 16
    assert (
        spec.policy.max_retries == 0 and spec.policy.timeout_h == 0.5 and spec.runtime == {"network": "none"}
    )
    rendered = DataConvertAdapter().render(spec, "run_test")
    assert rendered.command[:3] == ["python", "-m", "eb.runners.ufo_training_data"]
    assert (str(source), "/motion-source", "ro") in rendered.mounts
    assert all(mode == "ro" or target == "/outputs" for _, target, mode in rendered.mounts)
    assert not any("test.pkl" in target for _, target, _ in rendered.mounts)
    spec.runtime = {"network": "host"}
    with pytest.raises(ValueError):
        DataConvertAdapter().render(spec, "run_test")


def test_native_source_refuses_linked_or_missing_input(tmp_path, monkeypatch):
    _, source, _ = source_fixture(tmp_path, monkeypatch)
    assert routes_data._ufo_motion_source(str(source)) == source
    source.joinpath("train_clipped.pkl").unlink()
    source.joinpath("train_clipped.pkl").symlink_to(source / "validation.pkl")
    with pytest.raises(ValueError):
        routes_data._ufo_motion_source(str(source))
    with pytest.raises(ValueError):
        routes_data._ufo_motion_source(str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_valid", [True, False])
async def test_native_reconcile_publishes_frozen_path_only_after_success(
    tmp_path, monkeypatch, storage_valid
):
    settings, source, row = source_fixture(tmp_path, monkeypatch)
    row.meta = {**row.meta, "validation_job_id": "job_test"}
    row.status = "queued"
    output = settings.data_root / "runs/run_test/outputs"
    output.mkdir(parents=True)
    storage = str(output / "data") if storage_valid else str(source)
    report = {
        "format": "ufo_native_motion",
        "storage_ref": storage,
        "total_bytes": 123,
        "validation": {
            "scope": "ufo_native_motion_v1",
            "training_ready": True,
            "storage_ref": storage,
            "status": "validated",
            "errors": [],
        },
    }
    (output / "dataset-index.json").write_text(json.dumps(report))
    session = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status="succeeded")),
        scalar=AsyncMock(return_value=SimpleNamespace(id="run_test")),
        commit=AsyncMock(),
    )
    await routes_data.reconcile_dataset(row, session)
    assert row.source_uri == str(source)
    assert row.dest_path == str(output / "data") if storage_valid else row.dest_path == str(source)
    assert row.status == ("validated" if storage_valid else "invalid")
