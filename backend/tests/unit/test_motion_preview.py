import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import HTTPException

from eb.motion_preview import read_motion_manifest
from eb.runners.motion_preview import build_previews


def motion_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    q = np.arange(69, dtype=np.float32).reshape(3, 23) / 10
    np.savez(root / "walk.npz", q=q, root_pos=np.array([[1, 2, .8]] * 3),
             root_quat=np.array([[1., 0, 0, 0]] * 3), timestamps=np.arange(3) / 50,
             valid=np.array([True, False, True]))
    clip = {"id": "walk", "file": "walk.npz", "frames": 3, "fps": 50,
            "sha256": hashlib.sha256((root / "walk.npz").read_bytes()).hexdigest()}
    manifest = {"schema_version": "eb/rpo-bfm-motion/v1", "output_quaternion": "wxyz",
                "joint_order": [f"joint_{i}" for i in range(23)], "clips": [clip]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return root, hashlib.sha256(path.read_bytes()).hexdigest()


def test_preview_preserves_every_frame_and_does_not_change_source(tmp_path):
    root, digest = motion_source(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    receipt = build_previews(root, tmp_path / "preview", digest)
    payload = json.loads((tmp_path / "preview/walk.json").read_text())
    assert payload["valid"] == [True, False, True]
    assert payload["q"][2][22] == pytest.approx(6.8)
    assert payload["root_quat"][0] == [1., 0, 0, 0]
    assert payload["timestamps"] == [0, .02, .04]
    assert payload["manifest_sha256"] == receipt["manifest_sha256"] == digest
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_preview_rejects_changed_source_and_unsafe_manifest(tmp_path):
    root, digest = motion_source(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        read_motion_manifest(root, "0" * 64)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["clips"][0]["file"] = "../walk.npz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="非法"):
        read_motion_manifest(root)


def test_preview_rejects_corrupted_or_replaced_clip(tmp_path):
    root, digest = motion_source(tmp_path)
    (root / "walk.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        build_previews(root, tmp_path / "preview", digest)


@pytest.mark.parametrize("field", ["timestamps", "root_quat", "q"])
def test_preview_rejects_invalid_time_pose_and_shape(tmp_path, field):
    root, _ = motion_source(tmp_path)
    path = root / "walk.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays[field] = {"timestamps": np.array([0., 0., .04]),
                     "root_quat": np.zeros((3, 4)), "q": np.zeros((3, 22))}[field]
    np.savez(path, **arrays)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["clips"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="非法"):
        build_previews(root, tmp_path / "preview", hashlib.sha256(manifest_path.read_bytes()).hexdigest())


def test_preview_adapter_mounts_frozen_source_readonly(tmp_path, monkeypatch):
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.job import JobSpec
    from eb.motion_preview import PREVIEW_IMAGE, PREVIEW_SCHEMA
    root, digest = motion_source(tmp_path)
    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path,
                               datasets_root=tmp_path, allowed_path_prefixes=[])
    monkeypatch.setattr("eb.settings.get_settings", lambda: settings)
    spec = JobSpec(kind="data.convert", embodiment_env_id="ee_rpo_v2_bfm_23dof_v2", image=PREVIEW_IMAGE,
        params={"operation": "preview_rpo_motion", "dataset_id": "ds_abc", "manifest_sha256": digest,
                "preview_schema": PREVIEW_SCHEMA}, inputs={"datasets": [{"id": "ds_abc", "path": str(root)}]},
        resources={"gpus": 0}, runtime={"network": "none"})
    rendered = DataConvertAdapter().render(spec, "run_abc")
    assert rendered.mounts == [(str(root), "/motion", "ro"),
                               (str(tmp_path / "runs/run_abc/outputs"), "/outputs", "rw")]
    assert rendered.gpus == 0
    from eb.capacity import estimate_job
    estimate = estimate_job(spec, host_vram_gb=0)
    assert estimate.verdict == "fits" and estimate.per_gpu_peak_gb == 0
    spec.runtime = {"network": "bridge"}
    with pytest.raises(ValueError, match="offline"):
        DataConvertAdapter().render(spec, "run_abc")


@pytest.mark.asyncio
async def test_motion_api_rejects_unindexed_dataset():
    from eb.api.routes_motion import motion_dataset
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
        format="rpo_ufo_robot_state", status="validated", summary_stats={})))
    with pytest.raises(HTTPException) as exc:
        await motion_dataset("ds_abc", session)
    assert exc.value.status_code == 409
