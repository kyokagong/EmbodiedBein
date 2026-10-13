from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError


def test_reachable_profile_is_closed_and_legacy_default_unchanged():
    from eb.api.routes_data import UfoRobotStateIn
    body = UfoRobotStateIn(name="legacy", parent_dataset_ids=["ds_a"])
    assert body.control_profile is None
    with pytest.raises(ValidationError):
        UfoRobotStateIn(name="bad", parent_dataset_ids=["ds_a"], control_profile="custom")


@pytest.mark.asyncio
async def test_reachable_api_derives_new_version_and_freezes_parent(tmp_path, monkeypatch):
    from eb.api import routes_data
    from eb.rpo_bfm import ENV_V2_ID
    from eb.store.models import Dataset, LineageEdge
    parent = SimpleNamespace(id="ds_aaaa", dest_path=str(tmp_path / "parent"), total_bytes=42,
        license="research", name="parent", status="validated")
    frozen = {"status": "validated", "scope": "rpo_ufo_robot_state_v1", "manifest_sha256": "a" * 64}
    monkeypatch.setattr(routes_data, "reconcile_dataset", AsyncMock())
    monkeypatch.setattr("eb.data_validation.require_ufo_robot_state_dataset", lambda row: frozen)
    monkeypatch.setattr("eb.data_validation.require_bfm_motion_dataset", Mock(side_effect=AssertionError("wrong parent validator")))
    monkeypatch.setattr(routes_data, "validate_allowed_path", lambda path: tmp_path / "parent")
    monkeypatch.setattr(routes_data, "get_settings", lambda: SimpleNamespace(datasets_root=tmp_path / "derived"))
    monkeypatch.setattr(routes_data, "disk_preflight", lambda *a, **kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(routes_data, "new_id", lambda prefix: "ds_bbbb")
    monkeypatch.setattr(routes_data, "_ds_out", lambda row: {"id": row.id})
    submit = AsyncMock(return_value=SimpleNamespace(id="job_new"))
    monkeypatch.setattr("eb.orchestrator.service.Orchestrator.submit", submit)
    session = SimpleNamespace(get=AsyncMock(return_value=parent), add=Mock(), commit=AsyncMock(), refresh=AsyncMock())
    body = routes_data.UfoRobotStateIn(name="reachable", parent_dataset_ids=[parent.id], control_profile="reachable_v2")
    assert await routes_data.create_ufo_robot_state_dataset(body, session, SimpleNamespace(id="user")) == {"id": "ds_bbbb"}
    spec = submit.call_args.args[1]
    assert spec.embodiment_env_id == ENV_V2_ID
    assert spec.params["control_profile"] == "reachable_v2" and spec.resources.gpus == 0
    assert spec.inputs.datasets[0]["validation"] == frozen
    rows = [call.args[0] for call in session.add.call_args_list]
    derived = next(row for row in rows if isinstance(row, Dataset))
    edge = next(row for row in rows if isinstance(row, LineageEdge))
    assert derived.id != parent.id and derived.embodiment_env_id == ENV_V2_ID
    assert derived.meta["parent_dataset_ids"] == [parent.id]
    assert derived.summary_stats["validation"]["status"] == "not_validated"
    assert edge.src_id == parent.id and edge.dst_id == derived.id
    assert edge.relation == "derived_robot_state_control_view"
    assert parent.status == "validated" and parent.dest_path == str(tmp_path / "parent")


def test_robot_state_rejects_cross_interface_before_loading(tmp_path):
    from eb.data_validation import require_ufo_robot_state_dataset
    from eb.rpo_bfm import CODEC_V2_HASH, ENV_ID
    row = SimpleNamespace(status="validated", embodiment_env_id=ENV_ID,
        meta={"usage": "train", "purpose": "behavior_prior"}, dest_path=str(tmp_path),
        summary_stats={"validation": {"status": "validated", "scope": "rpo_ufo_robot_state_v2"}})
    with pytest.raises(ValueError, match="changed after validation"):
        require_ufo_robot_state_dataset(row, expected_codec_sha256=CODEC_V2_HASH)


def test_reachable_adapter_pins_cpu_image_and_readonly_validated_v1_parent(tmp_path, monkeypatch):
    import hashlib
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import UFO_REACHABLE_IMAGE_DIGEST
    from eb.rpo_bfm import CODEC_HASH, ENV_V2_ID
    from eb.settings import get_settings
    source = tmp_path / "parent"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}")
    assets = tmp_path / "assets/roboparty/rpo-v2"
    assets.mkdir(parents=True)
    asset = assets / "asset-manifest.json"
    asset.write_text("{}")
    validator = Mock()
    monkeypatch.setattr("eb.runners.rpo_ufo_motion.validate_robot_state_manifest", validator)
    monkeypatch.setenv("EB_ALLOWED_PATH_PREFIXES", str(tmp_path))
    monkeypatch.setenv("EB_ASSETS_ROOT", str(tmp_path / "assets"))
    get_settings.cache_clear()
    try:
        spec = JobSpec(kind="data.convert", embodiment_env_id=ENV_V2_ID,
            image="eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7",
            params={"operation": "convert_rpo_ufo_robot_state", "dataset_id": "ds_bbbb", "control_profile": "reachable_v2"},
            inputs={"datasets": [{"id": "ds_aaaa", "path": str(source), "validation": {
                "scope": "rpo_ufo_robot_state_v1", "status": "validated",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}}]},
            runtime={"network": "none"}, resources={"gpus": 0, "cpus": 2, "memory_gb": 4, "shm_gb": 1},
            policy={"timeout_h": 1, "max_retries": 0})
        rendered = DataConvertAdapter().render(spec, "run_cpu")
        assert rendered.image == UFO_REACHABLE_IMAGE_DIGEST and rendered.gpus == 0
        assert rendered.command[-2:] == ["--control-profile", "reachable_v2"]
        assert (str(source), "/workspace/parent", "ro") in rendered.mounts
        assert validator.call_args.kwargs["expected_codec_sha256"] == CODEC_HASH
        spec.inputs.datasets[0]["validation"]["scope"] = "rpo_ufo_robot_state_v2"
        with pytest.raises(ValueError, match="parent/asset"):
            DataConvertAdapter().render(spec, "run_reject")
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe_id", ["rpo_bfm_ufo_agent_rlinf_isaac_v1", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7"])
async def test_native_model_promotion_preserves_recipe_interface_and_codec(tmp_path, recipe_id):
    import json
    from eb.domain.job import JobSpec
    from eb.domain.training_recipe import RECIPES
    from eb.orchestrator.collect import promote_rpo_model
    from eb.rpo_bfm import ENV_V2_ID, codec_hash_for_execution, action_control_for_execution
    recipe = RECIPES[recipe_id]
    binding = recipe.binding.model_dump()
    (tmp_path / "actor-normalization.json").write_text(json.dumps({
        "schema_version": "eb/rpo-bfm-normalization/v1", "kind": "frozen_running_stats",
        "observation_dim": 733, "history_length": 4, "latent_dim": 256}))
    spec = JobSpec(kind="train.rl", task_id=recipe.task_id, embodiment_env_id=recipe.embodiment_env_id,
        image=recipe.image, params={"template_id": recipe_id, "execution": binding, "target_learner_updates": 24},
        inputs={"datasets": [{"id": "ds_frozen"}]})
    job = SimpleNamespace(id="job_native", kind="train.rl", spec=spec.model_dump(mode="json"))
    checkpoint = {"path": str(tmp_path), "bytes": 10, "step": 24, "manifest": {
        "execution": binding, "format": recipe.binding.checkpoint_format, "template_id": recipe_id,
        "files": [{"path": "model/actor-normalization.json"}]}}
    session = SimpleNamespace(scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=None),
        flush=AsyncMock(), add=Mock())
    model = await promote_rpo_model(session, job, SimpleNamespace(id="run_native"), checkpoint)
    assert model.embodiment_env_id == recipe.embodiment_env_id
    assert model.io_signature["action_profile"] == (
        "rpo_v2_joint23_position_v2" if recipe.embodiment_env_id == ENV_V2_ID else "rpo_v2_joint23_position_v1")
    assert model.io_signature["obs"] == 375
    assert model.card["codec_sha256"] == codec_hash_for_execution(binding)
    assert model.card["action_control"] == action_control_for_execution(binding)
    assert model.card["execution"] == binding
    assert model.card["training"]["image"] == recipe.image
    assert model.card["motion_prior"]["id"] == "ds_frozen"
