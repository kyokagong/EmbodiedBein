from pathlib import Path

import pytest

from eb.runners.pbfm_source import CAMERA_CONTRACT, prepare_pbfm_source, rpo_robot_config

SOURCE = Path("/data/kk/P-BFM")


def test_private_source_frozen_patch_and_repeat_rejection(tmp_path):
    if not SOURCE.is_dir():
        pytest.skip("Frozen P-BFM source unavailable")
    original = (SOURCE / "pbfm/robots/training_spec.py").read_bytes()
    dest = tmp_path / "pbfm"
    manifest = prepare_pbfm_source(SOURCE, dest)
    assert (SOURCE / "pbfm/robots/training_spec.py").read_bytes() == original
    assert manifest["schema_version"] == "eb/pbfm-source-adaptation/v1"
    assert manifest["patch_sha256"]
    assert rpo_robot_config(dest).is_file()
    assert "rpo_23dof" in (dest / "pbfm/robots/training_spec.py").read_text()
    assert CAMERA_CONTRACT["simulation_only"] is True
    assert CAMERA_CONTRACT["mount_pos"] != [0.0487988662332928, 0.01, 0.4378029937970051]
    with pytest.raises(ValueError):
        prepare_pbfm_source(SOURCE, dest)
    config = rpo_robot_config(dest).read_text()
    assert "hydra_robot: rpo/rpo_23dof" in config
    assert "action_rescale: true" in config


def test_source_drift_rejected_before_copy(tmp_path):
    bad = tmp_path / "source"
    bad.mkdir()
    (bad / "pyproject.toml").write_text("drift")
    with pytest.raises(ValueError, match="upstream"):
        prepare_pbfm_source(bad, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()
