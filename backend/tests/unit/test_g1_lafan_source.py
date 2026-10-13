from types import SimpleNamespace

import pytest


def test_g1_acquisition_is_bounded_and_requires_frozen_local_research_use(tmp_path, monkeypatch):
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobKind, JobSpec
    from eb.runners.g1_lafan_source import SOURCES

    assert SOURCES["lafan_29dof_10s-clipped.pkl"]["bytes"] == 205117479
    monkeypatch.setattr("eb.settings.get_settings", lambda:SimpleNamespace(data_root=tmp_path, hf_endpoint="https://huggingface.co"))
    spec = JobSpec(kind=JobKind.ASSET_VALIDATE, embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={"operation":"acquire_g1_lafan_trial", "source_use":"local_noncommercial_no_redistribution"},
        runtime={"network":"bridge"}, resources={"gpus":0, "cpus":2, "memory_gb":4, "shm_gb":1})
    rendered = ToolsAdapter().render(spec, "run_test")
    assert rendered.image == "eb/tools:rpo-g1-source-v1"
    assert rendered.mounts == [(str(tmp_path / "runs/run_test/outputs"), "/outputs", "rw")]
    assert rendered.gpus == 0
    spec.params.pop("source_use")
    with pytest.raises(ValueError):
        ToolsAdapter().render(spec, "run_test")


def test_g1_r2r_converter_binds_source_and_asset_read_only(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.job import JobKind, JobSpec

    source = tmp_path / "state/runs/run_abcdef/outputs/g1-export"
    source.mkdir(parents=True)
    receipt = source / "export-receipt.json"
    receipt.write_text("{}")
    settings = SimpleNamespace(data_root=tmp_path / "state", datasets_root=tmp_path / "datasets", assets_root=Path("/data/kk/assets"))
    monkeypatch.setattr("eb.settings.get_settings", lambda:settings)
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={"operation":"retarget_g1_lafan_trial", "source_run_id":"run_abcdef",
                "source_export_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest(), "dataset_id":"ds_abcdef"},
        resources={"gpus":0, "memory_gb":8, "shm_gb":1}, runtime={"network":"none"})
    rendered = DataConvertAdapter().render(spec, "run_trial")
    assert rendered.image == "eb/tools:rpo-g1-r2r-v7"
    mounts = {b:(a, m) for a,b,m in rendered.mounts}
    assert mounts["/workspace/g1-export"] == (str(source), "ro")
    assert mounts["/workspace/rpo-assets"][1] == "ro"
    assert mounts[str(settings.datasets_root / "ds_abcdef")][1] == "rw"
    spec.runtime["network"] = "bridge"
    with pytest.raises(ValueError):
        DataConvertAdapter().render(spec, "run_trial")
