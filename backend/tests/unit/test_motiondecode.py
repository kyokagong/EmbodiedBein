import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eb.motiondecode import HEADER, inspect_motion, safe_member, verify_file


def motion(tmp_path, rows=None):
    path = tmp_path / "sample.csv"
    row = [0, 0, 0.8, 1, 0, 0, 0] + [0] * 29
    path.write_text(
        ",".join(HEADER)
        + "\n"
        + "\n".join(",".join(map(str, values)) for values in (rows if rows is not None else [row, row]))
    )
    return path


def test_motion_index_counts_frames_without_inventing_timestamps(tmp_path):
    path = motion(tmp_path)
    result = inspect_motion(path)
    assert result == {"length": 2, "columns": 36}
    data = path.read_bytes()
    info = {
        "path": path.name,
        "size": len(data),
        "blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }
    assert verify_file(tmp_path, info) == hashlib.sha256(data).hexdigest()
    path.write_bytes(data + b"x")
    with pytest.raises(ValueError, match="size|hash"):
        verify_file(tmp_path, info)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_motion_is_rejected(tmp_path, value):
    row = [value, 0, 1, 1, 0, 0, 0] + [0] * 29
    with pytest.raises(ValueError, match="finite"):
        inspect_motion(motion(tmp_path, [row]))


def test_bad_header_empty_motion_and_bad_quaternion_are_rejected(tmp_path):
    path = motion(tmp_path)
    path.write_text(path.read_text().replace("root_pos_x(m)", "root_pos_x(mm)"))
    with pytest.raises(ValueError, match="header"):
        inspect_motion(path)
    with pytest.raises(ValueError, match="empty"):
        inspect_motion(motion(tmp_path, []))
    with pytest.raises(ValueError, match="quaternion"):
        inspect_motion(motion(tmp_path, [[0] * 36]))


@pytest.mark.parametrize(
    "name", ["../outside", "/etc/passwd", "x/../../outside", "x\\escape", ".cache/huggingface/x"]
)
def test_unsafe_repository_members_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_member(tmp_path, name)


def test_linked_repository_members_are_rejected(tmp_path):
    (tmp_path / "linked").symlink_to(Path("/tmp"), target_is_directory=True)
    with pytest.raises(ValueError):
        safe_member(tmp_path, "linked/x.csv")


def test_offline_acquisition_verifies_all_members_and_remains_raw(tmp_path, monkeypatch):
    from eb.motiondecode import REPO, REVISION, acquire

    monkeypatch.setattr("eb.motiondecode.EXPECTED_FILES", 1)
    monkeypatch.setattr("eb.motiondecode.shutil.disk_usage", lambda _: SimpleNamespace(free=0))

    # The frozen manifest is the input; offline verification must never call HF.
    def no_network(*args, **kwargs):
        raise AssertionError("offline verification contacted HF")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=no_network, hf_hub_download=no_network)
    )
    root = tmp_path / "data"
    source = root / "samples" / "walking"
    source.mkdir(parents=True)
    path = motion(source)
    data = path.read_bytes()
    state = root / ".embodiedbein"
    state.mkdir()
    info = {
        "path": "samples/walking/sample.csv",
        "size": len(data),
        "blob_id": "unused",
        "lfs_sha256": hashlib.sha256(data).hexdigest(),
    }
    (state / "source-manifest.json").write_text(
        json.dumps({"repo_id": REPO, "revision": REVISION, "files": [info]})
    )
    output = tmp_path / "output"
    report = acquire(root, output, local_only=True)
    assert report["validation"]["status"] == "validated"
    assert report["validation"]["training_ready"] is False
    assert report["frames_count"] == 2
    assert report["episodes"][0]["path"] == "samples/walking/sample.csv"
    assert report["motion_schema"]["position_unit"] == "m"
    assert "fps" not in report
    assert report["source_refs"][0]["revision"] == REVISION
    path.write_bytes(data.replace(b"0.8", b"9.8"))
    report = acquire(root, output, local_only=True)
    assert report["validation"]["status"] == "invalid"
    assert report["validation"]["invalid_files"] == 1
    assert report["episodes_count"] == 0
    monkeypatch.setattr("eb.motiondecode.EXPECTED_FILES", 2)
    with pytest.raises(ValueError, match="complete pinned repository"):
        acquire(root, output, local_only=True)


def test_reference_contract_remains_draft_and_blocks_training():
    from eb.domain.contract import load_env
    from eb.domain.job import JobSpec
    from eb.gates.engine import GateContext, g1_submit
    from eb.motiondecode import ENV_ID, JOINTS

    env = load_env(Path("/workspace/contracts") / f"{ENV_ID}.yaml")
    assert env.validate_self().passed
    assert env.joint_order == JOINTS
    convert = JobSpec(kind="data.convert", embodiment_env_id=env.id)
    assert g1_submit(convert, env, GateContext(dataset_versions=[])).passed
    train = JobSpec(kind="train.rl", embodiment_env_id=env.id, task_id="not_published")
    assert not g1_submit(train, env, GateContext(dataset_versions=[])).passed


def test_motiondecode_adapter_freezes_source_and_writes_only_managed_data(tmp_path, monkeypatch):
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.job import JobSpec
    from eb.motiondecode import ENV_ID, REPO, REVISION
    from eb.settings import get_settings

    monkeypatch.setenv("EB_DATASETS_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("EB_DATA_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("EB_ALLOWED_PATH_PREFIXES", str(tmp_path))
    monkeypatch.setenv("EB_HF_ENDPOINT", "https://hf-mirror.com")
    get_settings.cache_clear()
    try:
        spec = JobSpec(
            kind="data.convert",
            embodiment_env_id=ENV_ID,
            params={
                "operation": "acquire_motiondecode",
                "src": str(tmp_path / "datasets/raw"),
                "repo_id": REPO,
                "revision": REVISION,
            },
        )
        rendered = DataConvertAdapter().render(spec, "run_test")
        assert rendered.env["HF_ENDPOINT"] == "https://huggingface.co"
        assert rendered.gpus == 0
        assert rendered.command[:3] == ["python", "-m", "eb.motiondecode"]
        assert rendered.mounts[0][2] == "rw"
        spec.params["revision"] = "0" * 40
        with pytest.raises(ValueError):
            DataConvertAdapter().render(spec, "run_test")
        spec.params.update(revision=REVISION, local_only=True)
        spec.runtime = {"network": "none"}
        assert DataConvertAdapter().render(spec, "run_test").mounts[0][2] == "ro"
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_registration_rejects_using_g1_motion_as_another_robot():
    from fastapi import HTTPException

    from eb.api.routes_data import DatasetIn, create_dataset
    from eb.motiondecode import REPO, REVISION

    with pytest.raises(HTTPException) as error:
        await create_dataset(
            DatasetIn(
                source_kind="huggingface",
                source_uri=REPO,
                revision=REVISION,
                embodiment_env_id="ee_bhl_fixedhand_10dof_v1",
            ),
            None,
            None,
        )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_validation_submits_pinned_cpu_job_and_offline_reindex(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from eb.api.routes_data import validate_dataset
    from eb.motiondecode import ENV_ID, REPO, REVISION
    from eb.store.models import Dataset

    row = Dataset(
        id="ds_test",
        name="MotionDecode",
        source_kind="huggingface",
        source_uri=REPO,
        embodiment_env_id=ENV_ID,
        dest_path=str(tmp_path / "source"),
        status="registered",
        meta={"source_revision": REVISION},
    )
    session = SimpleNamespace(get=AsyncMock(return_value=row), commit=AsyncMock())
    submit = AsyncMock(return_value=SimpleNamespace(id="job_test"))
    monkeypatch.setattr("eb.orchestrator.service.Orchestrator", lambda: SimpleNamespace(submit=submit))
    monkeypatch.setattr(
        "eb.store.ebd.EbdClient",
        lambda: SimpleNamespace(
            image_info=AsyncMock(return_value={"ok": True, "digest": "sha256:" + "a" * 64})
        ),
    )
    await validate_dataset("ds_test", session, SimpleNamespace(id="user:test"), reindex=True)
    spec = submit.await_args.args[1]
    assert spec.params["operation"] == "acquire_motiondecode"
    assert spec.params["repo_id"] == REPO
    assert spec.params["revision"] == REVISION
    assert spec.params["local_only"] is True
    assert spec.image == "sha256:" + "a" * 64
    assert spec.resources.gpus == 0
    assert spec.runtime == {"network": "none"}
    assert row.meta["validation_job_id"] == "job_test"
    assert row.status == "queued"
