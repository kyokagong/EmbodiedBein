from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from eb.domain.job import JobSpec
from eb.domain.training_recipe import RECIPES, UFO_ISAAC_RECIPE


def test_fixed_agent_probe_renders_without_training_or_source_writes():
    from eb.adapters.mujoco import ToolsAdapter
    spec = JobSpec(kind="asset.validate", embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
        params={"operation": "probe_rpo_bfm_ufo_agent"}, runtime={"network": "none"},
        resources={"gpus": 1}, policy={"timeout_h": .25, "max_retries": 0})
    rendered = ToolsAdapter().render(spec, "run_probe")
    assert "bfm_ufo_agent_smoke.py" in rendered.command[1]
    assert rendered.gpus == 1
    assert all(mode == "rw" and target == "/outputs" for _, target, mode in rendered.mounts)
    spec.runtime["env"] = {"PYTHONPATH": "/arbitrary"}
    with pytest.raises(ValueError):
        ToolsAdapter().render(spec, "run_unsafe")


def test_ufo_raw_conversion_renders_registered_parent_readonly(tmp_path, monkeypatch):
    import hashlib
    from eb.adapters.data_convert import DataConvertAdapter
    from eb.domain.training_recipe import UFO_ISAAC_IMAGE_DIGEST
    from eb.settings import get_settings

    source = tmp_path / "parent"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}")
    assets = tmp_path / "assets/roboparty/rpo-v2"
    assets.mkdir(parents=True)
    asset = assets / "asset-manifest.json"
    asset.write_text("{}")
    monkeypatch.setenv("EB_ALLOWED_PATH_PREFIXES", str(tmp_path))
    monkeypatch.setenv("EB_ASSETS_ROOT", str(tmp_path / "assets"))
    get_settings.cache_clear()
    try:
        spec = JobSpec(kind="data.convert", embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",
            params={"operation": "convert_rpo_ufo_robot_state", "dataset_id": "ds_bbbb"},
            inputs={"datasets": [{"id": "ds_aaaa", "path": str(source), "validation": {
                "scope": "rpo_bfm_motion_v1", "status": "validated",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}}]},
            runtime={"network": "none"}, resources={"gpus": 0, "cpus": 2, "memory_gb": 4, "shm_gb": 1},
            policy={"timeout_h": 1, "max_retries": 0})
        rendered = DataConvertAdapter().render(spec, "run_convert")
        assert rendered.image == UFO_ISAAC_IMAGE_DIGEST and rendered.gpus == 0
        assert (str(source), "/workspace/parent", "ro") in rendered.mounts
        assert "rpo_ufo_motion.py" in rendered.command[1]
        spec.resources.gpus = 1
        with pytest.raises(ValueError):
            DataConvertAdapter().render(spec, "run_invalid")
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_ufo_recipe_preview_submit_identity(monkeypatch):
    from eb.api import routes_train
    from eb.adapters.base import RenderedJob
    from eb.ufo_source import SOURCE_RECEIPT
    recipe = RECIPES[UFO_ISAAC_RECIPE]
    spec = JobSpec(kind="train.rl", task_id=recipe.task_id, embodiment_env_id=recipe.embodiment_env_id,
        image=recipe.image, name="ufo", params={"template_id": UFO_ISAAC_RECIPE, "model_name": "model",
            "execution": recipe.binding.model_dump(), "target_learner_updates": 16,
            "purpose": "qualification", "source_receipt": SOURCE_RECEIPT})
    monkeypatch.setattr(routes_train, "_build_spec", AsyncMock(return_value=spec))
    rendered = RenderedJob(image=recipe.image, command=["fixed-entrypoint"], env={}, writable_paths=[])
    render = Mock(return_value=rendered)
    monkeypatch.setattr("eb.adapters.get_adapter", lambda job: SimpleNamespace(render=render))
    body = routes_train.TrainIn(name="ufo", model_name="model", template_id=UFO_ISAAC_RECIPE,
        embodiment_id="emb_roboparty_rpo", gpus=1, purpose="qualification", target_learner_updates=16)
    preview = await routes_train.preview_train(body, SimpleNamespace(), SimpleNamespace(id="user"))
    assert preview["execution"] == spec.params["execution"]
    assert preview["target_learner_updates"] == 16 and preview["purpose"] == "qualification"
    assert preview["source_receipt"] == SOURCE_RECEIPT
    submit = AsyncMock(return_value=SimpleNamespace(id="job_fixture", status="queued"))
    monkeypatch.setattr(routes_train.orch, "submit", submit)
    session = SimpleNamespace(add=Mock(), commit=AsyncMock(), refresh=AsyncMock())
    await routes_train.submit_train(body, session, SimpleNamespace(id="user"))
    submitted = submit.call_args.args[1]
    assert submitted.params["execution"] == preview["execution"]
    assert submitted.params["source_receipt"] == preview["source_receipt"]
    assert submitted.params["target_learner_updates"] == preview["target_learner_updates"]


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe_id", [UFO_ISAAC_RECIPE, "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_stable_v3", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4"])
async def test_ufo_api_freezes_raw_dataset_and_new_profile(tmp_path, monkeypatch, recipe_id):
    import hashlib
    import json
    from dataclasses import replace
    from eb.api.routes_train import TrainIn, _build_spec
    from eb.domain.environment_profile import profile_hash
    from eb.domain.rlinf_form import get_template
    from eb.environment_catalog import training_definition
    from eb.rpo_bfm import BODY_ORDER, CODEC_HASH, RPO_LAB_JOINT_ORDER
    from eb.settings import get_settings
    from eb.store.models import Dataset, EmbodimentRow, EnvironmentProfile

    assets = tmp_path / "assets/roboparty/rpo-v2"
    assets.mkdir(parents=True)
    asset = assets / "asset-manifest.json"
    asset.write_text('{"status":"verified"}')
    asset_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "eb/rpo-robot-state/v1", "codec_sha256": CODEC_HASH,
        "asset_sha256": asset_hash, "quaternion": "wxyz", "velocity_frame": "world", "velocity_point": "root_link_origin",
        "joint_order": list(RPO_LAB_JOINT_ORDER), "body_order": list(BODY_ORDER),
        "parents": [{"manifest_sha256": "a" * 64}], "source_refs": [{"sha256": "b" * 64, "license": "fixture"}],
        "clips": [{"id": source, "source_group": source, "source": source, "split": "train",
            "fps": 50, "frames": 270, "file": source + ".npz", "sha256": "c" * 64}
            for source in ("official_rpo", "g1")]}))
    ds = SimpleNamespace(id="ds_a", status="validated", dest_path=str(root), source_uri="ds_parent",
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", meta={"usage": "train"}, summary_stats={"validation": {
            "status": "validated", "scope": "rpo_ufo_robot_state_v1", "asset_sha256": asset_hash,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}})
    from eb.domain.training_recipe import four_gpu_recipe, recipe_profile_id
    tpl = get_template(recipe_id)
    profile = SimpleNamespace(id=recipe_profile_id(recipe_id), status="active", usage="train",
        embodiment_env_id=tpl.embodiment_env_id, task_id=tpl.task_id, template_id=tpl.id, suite_id=None,
        definition=training_definition(tpl))
    profile.content_hash = profile_hash(profile)
    rows = {(Dataset, ds.id): ds, (EnvironmentProfile, profile.id): profile,
        (EmbodimentRow, tpl.embodiment_id): SimpleNamespace(root_path=str(assets), source={"kind": "managed_bundle"})}
    session = SimpleNamespace(get=AsyncMock(side_effect=lambda table, key: rows.get((table, key))))
    monkeypatch.setenv("EB_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("EB_ALLOWED_PATH_PREFIXES", str(tmp_path))
    recipe = RECIPES[recipe_id]
    monkeypatch.setitem(RECIPES, recipe_id, replace(recipe, status="dependency_ready"))
    get_settings.cache_clear()
    try:
        body = TrainIn(name="ufo", model_name="model", template_id=tpl.id, embodiment_id=tpl.embodiment_id,
            dataset_ids=[ds.id], environment_id=profile.id, gpus=4 if four_gpu_recipe(recipe_id) else 1, purpose="qualification",
            target_learner_updates=16, seed=7)
        spec = await _build_spec(body, session)
        assert spec.kind.value == "train.rl" and spec.params["execution"] == recipe.binding.model_dump()
        assert spec.params["environment_profile"]["hash"] == profile.content_hash
        assert spec.inputs.datasets[0]["validation"] == ds.summary_stats["validation"]
        assert spec.overrides == ["actor.seed=7"]
        from eb.adapters.mujoco import ToolsAdapter
        probe = JobSpec(kind="asset.validate", embodiment_env_id=tpl.embodiment_env_id,
            params={"operation": "probe_rpo_ufo_environment"}, inputs={"datasets": spec.inputs.datasets},
            runtime={"network": "none"}, resources={"gpus": 1}, policy={"timeout_h": .25, "max_retries": 0})
        rendered = ToolsAdapter().render(probe, "run_probe")
        contact = probe.model_copy(deep=True)
        contact.params = {"operation": "probe_rpo_ufo_initial_contact"}
        contact_rendered = ToolsAdapter().render(contact, "run_contact_probe")
        assert contact_rendered.command[-1].endswith("rpo_ufo_initial_contact.py")
        assert contact_rendered.env["EB_UFO_LEARNING_PROFILE"] == "physical_limits_v2"
        assert contact_rendered.gpus == 1
        assert (str(assets / "rpo_description"),
            "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro") in rendered.mounts
        body.extra_overrides = ["actor.model.replay_capacity=5120000"]
        with pytest.raises(ValueError, match="protects"):
            await _build_spec(body, session)
    finally:
        get_settings.cache_clear()
