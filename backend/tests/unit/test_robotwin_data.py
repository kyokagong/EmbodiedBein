import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from eb.adapters.data_convert import DataConvertAdapter
from eb.api.routes_data import RoboTwinDatasetIn, robotwin_catalog
from eb.domain.job import JobKind, JobSpec
from eb.domain.contract import load_env
from eb.gates.engine import GateContext, g1_submit
from eb.robotwin_worker import safe_extract_zip, validate_robotwin_content
from eb.robotwin import raw_archive_profile


@pytest.mark.asyncio
async def test_catalog_endpoint_exposes_pinned_profiles_and_compatibility():
    result = await robotwin_catalog(SimpleNamespace())
    assert len(result["tasks"]) == 50
    assert len(result["embodiments"]) == 5
    assert result["revision"] == "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042"
    assert next(p for p in result["download_profiles"] if p["id"] == "aloha_lerobot_v21")["requires_probe"]
    assert next(m for m in result["models"] if m["id"] == "gr00t-n1.7")["ready"] is True
    assert next(m for m in result["models"] if m["id"] == "xiaomi-robotics-1")["ready"] is True
    assert next(m for m in result["models"] if m["id"] == "galaxea-g0")["ready"] is False
    assert next(e for e in result["embodiments"] if e["id"] == "piper")["download_ready"] is False


def test_robotwin_request_requires_either_named_profile_or_complete_raw_selection():
    assert RoboTwinDatasetIn(profile_id="aloha_lerobot_v21").profile_id
    with pytest.raises(ValueError):
        RoboTwinDatasetIn(task="adjust_bottle", scenario="clean")


def test_adapter_accepts_only_catalog_derived_robotwin_acquisition(tmp_path, monkeypatch):
    settings = SimpleNamespace(data_root=tmp_path, datasets_root=tmp_path / "datasets",
                               models_root=tmp_path / "models",
                               hf_endpoint="https://huggingface.co",
                               allowed_path_prefixes=str(tmp_path))
    monkeypatch.setattr("eb.settings.get_settings", lambda: settings)
    monkeypatch.setattr("eb.adapters.data_convert.get_settings", lambda: settings, raising=False)
    dest = settings.datasets_root / "ds_abc" / "content"
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        params={"operation":"acquire_robotwin", "dataset_id":"ds_abc", "dest":str(dest),
            "profile_id":"aloha_lerobot_v21"})
    rendered = DataConvertAdapter().render(spec, "run_test")
    assert "eb.robotwin_worker" in rendered.command
    assert "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042" in rendered.command
    assert (str(dest.parent), str(dest.parent), "rw") in rendered.mounts
    bad = spec.model_copy(deep=True)
    bad.params["dest"] = str(tmp_path / "escape")
    with pytest.raises(ValueError):
        DataConvertAdapter().render(bad, "run_test")


def test_draft_robotwin_env_allows_probe_but_not_training():
    env = load_env(Path(__file__).parents[3] / "contracts/ee_robotwin_aloha_joint14_v1.yaml")
    assert env.joint_order == [
        "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6", "fl_joint7",
        "fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6", "fr_joint7",
    ]
    assert all(modality.shape == [240, 320, 3] for modality in env.observation.modalities)
    probe = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id=env.id)
    train = JobSpec(kind=JobKind.TRAIN_SFT, embodiment_env_id=env.id,
                    task_id="tsk_robotwin_multitask")
    assert g1_submit(probe, env, GateContext(dataset_versions=[])).passed
    assert not g1_submit(train, env, GateContext(dataset_versions=[])).passed


def test_safe_extract_rejects_zip_slip(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract_zip(archive, tmp_path / "out")


def test_aloha_lerobot_probe_requires_14d_state_action_and_three_cameras(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    features = {
        "action": {"shape": [14]}, "observation.state": {"shape": [14]},
        "observation.images.cam_high": {"dtype": "video"},
        "observation.images.cam_left_wrist": {"dtype": "video"},
        "observation.images.cam_right_wrist": {"dtype": "video"},
    }
    (meta / "info.json").write_text(json.dumps({"codebase_version":"v2.1", "fps":15, "features":features,
                                                 "total_episodes":50, "total_frames":100}))
    report = validate_robotwin_content(tmp_path, "aloha_lerobot_v21")
    assert report["validation"]["status"] == "validated"
    assert report["validation"]["training_ready"] is True
    features["action"]["shape"] = [7]
    (meta / "info.json").write_text(json.dumps({"codebase_version":"v2.1", "fps":15, "features":features}))
    assert validate_robotwin_content(tmp_path, "aloha_lerobot_v21")["validation"]["status"] == "invalid"


def test_raw_profile_uses_raw_validation_without_named_profile_lookup(tmp_path):
    profile = raw_archive_profile("adjust_bottle", "clean", "piper")
    report = validate_robotwin_content(tmp_path, profile.id, profile)
    assert report["validation"]["scope"] == "robotwin_raw_archive"
    assert report["validation"]["training_ready"] is False
