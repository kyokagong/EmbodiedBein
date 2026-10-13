from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def test_bfm_contract_models_preflattened_history_and_critic_dims():
    from pathlib import Path

    import yaml

    from eb.domain.contract import EmbodimentEnv

    contract = Path(__file__).resolve().parents[3] / "contracts/ee_rpo_v2_bfm_23dof_v1.yaml"
    spec = yaml.safe_load(contract.read_text())
    env = EmbodimentEnv.model_validate(spec)

    assert env.observation.history_length == 4
    assert env.observation.history_preflattened is True
    assert sum(term.dim for term in env.observation.layout) == env.observation.dim == 375
    assert sum(term.dim for term in env.observation.critic_layout) == env.observation.critic_dim == 733
    assert env.self_assertions() == []


def test_bfm_seed_activates_contract_spec_only_with_verified_assets():
    from pathlib import Path

    from eb.bootstrap import _rpo_bfm_env_for_seed
    from eb.domain.contract import load_env

    contract = Path(__file__).resolve().parents[3] / "contracts/ee_rpo_v2_bfm_23dof_v1.yaml"
    env = load_env(contract)

    assert env.status == "draft"
    assert _rpo_bfm_env_for_seed(env, rpo_assets_verified=False).status == "draft"
    active = _rpo_bfm_env_for_seed(env, rpo_assets_verified=True)
    assert active.status == "active"
    assert active.spec_hash() != env.spec_hash()


def test_bfm_recipe_keeps_one_learner_and_requires_motion_view():
    from pathlib import Path

    import yaml

    from eb.domain.rlinf_form import build_overrides, get_template, template_image

    template = get_template("rpo_v2_bfm_zero_pretrain")
    assert template.kind == "ppo" and template.requires_dataset
    assert template.embodiment_env_id == "ee_rpo_v2_bfm_23dof_v1"
    assert template.task_id == "tsk_rpo_bfm_pretrain"
    assert template_image(template) == "eb/rlinf:rpo-v2-bfm-walking-v10"
    config_path = Path(__file__).resolve().parents[2] / "eb/runners/config/rpo_bfm_zero_pretrain.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["actor"]["fsdp_config"]["use_orig_params"] is True
    overrides = build_overrides(template.id, 1, "bfm-smoke", [])
    assert "runner.max_epochs=2" in overrides
    assert "env.train.total_num_envs=4" in overrides
    with pytest.raises(ValueError, match="one learner"):
        build_overrides(template.id, 2, "multi-learner", [])
    with pytest.raises(ValueError, match="batch"):
        build_overrides(template.id, 1, "invalid-batch", ["actor.global_batch_size=16"])


def test_bfm_export_adapter_mounts_checkpoint_config_and_motion_read_only(tmp_path, monkeypatch):
    from eb.adapters import get_adapter
    from eb.domain.job import JobKind, JobSpec

    data_root, models_root, datasets_root = (tmp_path / name for name in ("state", "models", "datasets"))
    checkpoint = models_root / "runs/run_abcdef"
    (checkpoint / "model_state_dict").mkdir(parents=True)
    (checkpoint / "model_state_dict/full_weights.pt").write_bytes(b"weights")
    config = data_root / "runs/run_abcdef/output/resolved-config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("actor: {}\n")
    motion = datasets_root / "ds_motion/content"
    motion.mkdir(parents=True)
    (motion / "manifest.json").write_text("{}")
    settings = SimpleNamespace(
        data_root=data_root, models_root=models_root, datasets_root=datasets_root,
        allowed_path_prefixes=[])
    monkeypatch.setattr("eb.settings.get_settings", lambda:settings)
    monkeypatch.setattr("eb.adapters.isaaclab_rslrl.get_settings", lambda:settings)

    spec = JobSpec(kind=JobKind.EXPORT_POLICY, embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
                   name="bfm-export", image="eb/rlinf:rpo-v2-bfm-zero",
                   params={"source_run_id":"run_abcdef", "asset_sha256":"a" * 64},
                   inputs={"base_model":str(checkpoint), "datasets":[{"id":"ds_motion", "path":str(motion)}]},
                   resources={"memory_gb":16, "shm_gb":8})

    rendered = get_adapter(spec).render(spec, "run_export")

    assert rendered.image == "eb/rlinf:rpo-v2-bfm-zero"
    assert rendered.command[1].endswith("/rpo_bfm_artifact.py")
    assert "--checkpoint" in rendered.command and "--motion-root" in rendered.command
    mounts = {target:(source, mode) for source, target, mode in rendered.mounts}
    assert mounts["/workspace/checkpoint"][1] == "ro"
    assert mounts["/workspace/config/resolved-config.yaml"][1] == "ro"
    assert mounts["/workspace/motion"][1] == "ro"
    assert mounts["/workspace/output"][1] == "rw"
    from eb.rpo_bfm import REFERENCE_IMAGE, REFERENCE_POOL_CONVERTER_REVISION
    spec.image = None
    assert get_adapter(spec).render(spec, "run_export").image == REFERENCE_IMAGE
    (motion / "manifest.json").write_text(__import__("json").dumps({"converter_revision":REFERENCE_POOL_CONVERTER_REVISION}))
    spec.image = "eb/rlinf:rpo-v2-bfm-zero"
    with pytest.raises(ValueError, match="versioned reference image"):
        get_adapter(spec).render(spec, "run_export")


def test_bfm_export_collector_rejects_missing_verified_manifest(tmp_path):
    from eb.orchestrator.collect import validate_rpo_bfm_export

    job = SimpleNamespace(spec={"params":{"operation":"rpo_bfm_export", "source_run_id":"run_abcdef",
                                         "asset_sha256":"a" * 64}})

    with pytest.raises(ValueError, match="BFM export manifest"):
        validate_rpo_bfm_export(tmp_path, job)


@pytest.mark.parametrize("operation,profile", [("convert_rpo_bfm_official", "smoke"), ("convert_rpo_bfm_reference_pool", "reference_pool")])
def test_bfm_official_motion_conversion_is_an_offline_managed_data_job(tmp_path, monkeypatch, operation, profile):
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.job import JobKind, JobSpec

    monkeypatch.setattr("eb.settings.get_settings", lambda: __import__("types").SimpleNamespace(
        datasets_root=tmp_path / "datasets", data_root=tmp_path / "state",
        assets_root=__import__("pathlib").Path("/data/kk/assets")))
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
                   params={"operation":operation, "dataset_id":"ds_abcdef"},
                   runtime={"network":"none"},
                   resources={"gpus":1, "cpus":8, "memory_gb":24, "shm_gb":8})

    rendered = DataConvertAdapter().render(spec, "run_bfm_data")

    assert rendered.image == ("eb/rlinf:rpo-v2-bfm-reference-v5" if profile == "reference_pool" else "eb/rlinf:rpo-v2-bfm-zero")
    assert rendered.command[0].endswith("/.venv/bin/python")
    assert "--source" in rendered.command and "--report" in rendered.command
    assert rendered.command[rendered.command.index("--profile")+1] == profile
    mounts = {target: (source, mode) for source, target, mode in rendered.mounts}
    assert mounts["/workspace/motions"][1] == "ro"
    assert mounts["/workspace/rpo-assets"][1] == "ro"
    assert mounts[str(tmp_path / "datasets/ds_abcdef")][1] == "rw"
    assert rendered.shm_gb == 8 and rendered.shm_gb <= spec.resources.memory_gb
    assert rendered.gpus == 1


@pytest.mark.asyncio
async def test_bfm_dataset_route_requests_shm_within_container_memory(tmp_path, monkeypatch):
    from eb.api.routes_data import OfficialRpoBfmIn, create_official_rpo_bfm_dataset
    from eb.orchestrator.service import Orchestrator

    submitted = {}

    async def capture_submit(self, session, spec, *args, **kwargs):
        submitted["spec"] = spec
        return SimpleNamespace(id="job_bfm_smoke")

    monkeypatch.setattr("eb.api.routes_data.get_settings",
                        lambda:SimpleNamespace(datasets_root=tmp_path))
    monkeypatch.setattr("eb.api.routes_data.disk_preflight",
                        lambda *_args, **_kwargs:SimpleNamespace(ok=True, reason=""))
    monkeypatch.setattr(Orchestrator, "submit", capture_submit)
    session = SimpleNamespace(add=Mock(), commit=AsyncMock(), refresh=AsyncMock())

    result = await create_official_rpo_bfm_dataset(
        OfficialRpoBfmIn(), session, SimpleNamespace(id="usr_smoke"))

    resources = submitted["spec"].resources
    assert result["status"] == "queued"
    assert resources.shm_gb <= resources.memory_gb


def test_bfm_pool_conversion_and_dynamic_probe_are_offline_and_readonly(tmp_path, monkeypatch):
    from pathlib import Path
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.adapters.mujoco import ToolsAdapter
    from eb.domain.job import JobKind, JobSpec
    settings = SimpleNamespace(datasets_root=tmp_path, data_root=tmp_path / "state",
                               assets_root=Path("/data/kk/assets"))
    monkeypatch.setattr("eb.settings.get_settings", lambda:settings)
    spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
                   params={"operation":"convert_rpo_bfm_local_pool", "dataset_id":"ds_abcdef"},
                   runtime={"network":"none"}, resources={"gpus":1, "memory_gb":24, "shm_gb":8})
    rendered = DataConvertAdapter().render(spec, "run_pool")
    assert "--profile" in rendered.command and "local_pool" in rendered.command
    assert all(mode == "ro" for _, target, mode in rendered.mounts if target.startswith("/workspace"))
    dataset = tmp_path / "ds_abcdef/content"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}")
    spec.kind = JobKind.ASSET_VALIDATE
    spec.params["operation"] = "probe_rpo_bfm_motion"
    rendered = ToolsAdapter().render(spec, "run_probe")
    assert rendered.image == "eb/rlinf:rpo-v2-bfm-four-gpu-v6"
    assert rendered.command[1].endswith("/rpo_bfm_motion_probe.py")
    assert (str(dataset), "/workspace/motion", "ro") in rendered.mounts
    spec.runtime["network"] = "bridge"
    with pytest.raises(ValueError):
        ToolsAdapter().render(spec, "run_invalid")


def test_bfm_replay_collection_rejects_unbound_or_forged_quality(tmp_path):
    import json
    from eb.orchestrator.collect import validate_rpo_bfm_motion_probe_receipt
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"schema_version":"eb/rpo-bfm-motion-probe/v1", "status":"succeeded"}))
    with pytest.raises(ValueError, match="BFM"):
        validate_rpo_bfm_motion_probe_receipt(path, "a"*64, "b"*64)
