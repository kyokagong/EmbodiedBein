import json

import pytest
from test_pbfm_training import plan_fixture

from eb.runners.rlinf_pbfm_runtime import build_training_evidence, parse_native_args, verify_source_refs


def test_native_argument_types():
    args = parse_native_args(plan_fixture())
    assert args["num_envs"] == 1024
    assert args["num_env_steps"] == 18432
    assert args["data_path"] == "/training-data/train.pkl"


def test_training_analysis_uses_updates_and_keeps_missing_samples(tmp_path):
    native = tmp_path / "outputs/native"
    native.mkdir(parents=True)
    (native / "train_log.txt").write_text(
        "distributed/optimizer_steps,loss/fb,loss/q\n16,2,\n32,,1\n48,1,nan\n"
    )
    index = build_training_evidence(tmp_path, plan_fixture(), {"optimizer_steps": 112})
    summary = json.loads((tmp_path / "outputs/analysis/summary.json").read_text())
    assert summary["x_axis"] == "distributed/optimizer_steps"
    assert summary["series"]["loss/fb"]["finite_samples"] == 2
    assert summary["series"]["loss/q"]["nonfinite_samples"] == 1
    svg = (tmp_path / "outputs/analysis/train-timeseries.svg").read_text()
    assert "Agent joint updates" in svg and "16" in svg and "48" in svg
    assert len(index["files"]) == 3


def test_training_analysis_refuses_missing_update_axis(tmp_path):
    native = tmp_path / "outputs/native"
    native.mkdir(parents=True)
    (native / "train_log.txt").write_text("timestep,loss\n1000,2\n")
    with pytest.raises(ValueError, match="update"):
        build_training_evidence(tmp_path, plan_fixture(), {"optimizer_steps": 112})


def test_asset_mesh_mutation_rejected_by_frozen_manifest(tmp_path):
    from eb.runners.pbfm_source import file_sha256

    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "foot.stl"
    mesh.write_bytes(b"frozen mesh")
    manifest = tmp_path / "asset-manifest.json"
    manifest.write_text(json.dumps({"entries": [{"path": "foot.stl", "sha256": file_sha256(mesh)}]}))
    plan = {
        "source_refs": {
            "asset": {
                "path": str(assets),
                "manifest_path": str(manifest),
                "snapshot_hash": "sha256:" + file_sha256(manifest),
            }
        }
    }
    verify_source_refs(plan)
    injected = assets / "injected.py"
    injected.write_text("unfrozen payload")
    with pytest.raises(ValueError, match="unindexed"):
        verify_source_refs(plan)
    injected.unlink()
    mesh.write_bytes(b"mutated mesh")
    with pytest.raises(ValueError, match="payload"):
        verify_source_refs(plan)
