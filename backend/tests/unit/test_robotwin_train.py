from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eb.adapters.rlinf import RlinfAdapter
from eb.data_validation import require_validated_dataset
from eb.domain.job import JobKind, JobSpec
from eb.domain.rlinf_form import TEMPLATES, get_template, require_template_ready


def test_six_robotwin_model_recipes_are_visible_but_only_verified_workers_are_ready():
    recipes = {t.model_id: t for t in TEMPLATES if t.id.startswith("robotwin_")}
    assert set(recipes) == {"lingbot-vla-2", "pi05", "gr00t-n1.7", "galaxea-g0", "dm05", "xiaomi-robotics-1"}
    assert recipes["pi05"].ready is True
    assert recipes["lingbot-vla-2"].ready is True
    assert recipes["dm05"].ready is True
    assert recipes["xiaomi-robotics-1"].ready is True
    assert recipes["pi05"].config_name == "robotwin_sft_openpi_pi05"
    assert recipes["gr00t-n1.7"].ready is True
    for model_id, recipe in recipes.items():
        if model_id not in {"pi05", "lingbot-vla-2", "dm05", "gr00t-n1.7", "xiaomi-robotics-1"}:
            assert recipe.ready is False and recipe.blocked_reason
            with pytest.raises(ValueError, match="尚未就绪"):
                require_template_ready(recipe)


def test_dm05_recipe_uses_its_exact_rlinf_worker_and_image():
    from eb.domain.rlinf_form import template_image

    recipe = get_template("robotwin_dm05_sft")
    assert recipe.ready is True
    assert recipe.config_path == "../eb_runner/config"
    assert recipe.config_name == "robotwin_sft_dm05"
    assert recipe.model_framework == "opendm_dm05"
    assert template_image(recipe) == "eb/rlinf:robotwin-dm05"


def test_lingbot_vla2_recipe_uses_exact_worker_and_image():
    from eb.domain.rlinf_form import template_image

    recipe = get_template("robotwin_lingbot_vla2_sft")
    assert recipe.ready is True
    assert recipe.config_path == "../eb_runner/config"
    assert recipe.config_name == "robotwin_sft_lingbot_vla2"
    assert recipe.model_framework == "lingbot_vla2"
    assert template_image(recipe) == "eb/rlinf:robotwin-lingbot-vla2"


def test_gr00t_n17_recipe_uses_exact_worker_and_image():
    from eb.domain.rlinf_form import template_image

    recipe = get_template("robotwin_gr00t_n17_sft")
    assert recipe.ready is True
    assert recipe.config_path == "../eb_runner/config"
    assert recipe.config_name == "robotwin_sft_gr00t_n17"
    assert recipe.model_framework == "gr00t_n1d7"
    assert template_image(recipe) == "eb/rlinf:robotwin-gr00t-n1.7"


def test_xiaomi_xr1_recipe_uses_exact_worker_and_image():
    from eb.domain.rlinf_form import template_image

    recipe = get_template("robotwin_xr1_sft")
    assert recipe.ready is True
    assert recipe.config_path == "../eb_runner/config"
    assert recipe.config_name == "robotwin_sft_xiaomi_xr1"
    assert recipe.model_framework == "xiaomi_xr1"
    assert template_image(recipe) == "eb/rlinf:robotwin-xiaomi-xr1"


def test_lingbot_vla2_keeps_official_joint_mapping_as_strings():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "eb/runners/config/lingbot_vla2_robotwin.yaml").read_text()
    )
    assert all(isinstance(value, str) for value in config["data"]["joints"])
    assert all(isinstance(value, str) for value in config["data"]["norm_type"])


def test_dm05_runtime_can_import_opendm(tmp_path, monkeypatch):
    data = tmp_path / "data"; model = tmp_path / "model"; rlinf = tmp_path / "rlinf"
    data.mkdir(); model.mkdir(); rlinf.mkdir()
    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path / "models", rlinf_repo=rlinf)
    monkeypatch.setattr("eb.adapters.rlinf.get_settings", lambda: settings)
    spec = JobSpec(kind=JobKind.TRAIN_SFT, name="dm05", task_id="tsk_robotwin_multitask",
        embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        params={"template_id":"robotwin_dm05_sft", "stack":"rlinf"},
        inputs={"datasets":[{"id":"ds_rt", "path":str(data)}], "base_model":str(model)},
        resources={"gpus":4, "shm_gb":32})

    rendered = RlinfAdapter().render(spec, "run_dm05")

    assert "/opt/opendm" in rendered.env["PYTHONPATH"].split(":")


def test_dm05_runtime_view_rebases_host_paths(tmp_path):
    import json
    from eb.runners.robotwin_dm05_view import rebase

    source, output = tmp_path / "source", tmp_path / "runtime"
    image = source / "images/episode_000000/cam_high/000000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    row = {key: {"type": "image", "url": "/old/images/episode_000000/cam_high/000000.jpg"}
           for key in ("images_1", "images_2", "images_3")}
    (source / "episode_000000.jsonl").write_text(json.dumps(row) + "\n")
    (source / "index_cache.json").write_text(json.dumps({"data": {"/old/episode_000000.jsonl": 1}}))

    rebase(source, output)

    actual = json.loads((output / "episode_000000.jsonl").read_text())
    assert {actual[key]["url"] for key in row} == {str(image)}


def test_robotwin_dataset_validation_is_native_aloha_and_training_ready(tmp_path):
    row = SimpleNamespace(status="validated", dest_path=str(tmp_path),
        embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        summary_stats={"native_embodiment":"aloha-agilex", "features":{
            "action":{"shape":[14]}, "observation.state":{"shape":[14]}},
            "validation":{"status":"validated", "scope":"robotwin_aloha_lerobot",
                          "training_ready":True, "storage_ref":str(tmp_path)}})
    assert require_validated_dataset(row, "ee_robotwin_aloha_joint14_v1")["training_ready"]
    row.summary_stats["native_embodiment"] = "piper"
    with pytest.raises(ValueError, match="ALOHA"):
        require_validated_dataset(row, "ee_robotwin_aloha_joint14_v1")


def test_pi05_robotwin_renders_managed_rlinf_entry_and_read_only_inputs(tmp_path, monkeypatch):
    data = tmp_path / "data"; model = tmp_path / "model"; rlinf = tmp_path / "rlinf"
    data.mkdir(); model.mkdir(); rlinf.mkdir()
    settings = SimpleNamespace(data_root=tmp_path, models_root=tmp_path / "models", rlinf_repo=rlinf)
    monkeypatch.setattr("eb.adapters.rlinf.get_settings", lambda: settings)
    spec = JobSpec(kind=JobKind.TRAIN_SFT, name="robotwin", task_id="tsk_robotwin_multitask",
        embodiment_env_id="ee_robotwin_aloha_joint14_v1",
        params={"template_id":"robotwin_pi05_sft", "stack":"rlinf"},
        inputs={"datasets":[{"id":"ds_rt", "path":str(data)}], "base_model":str(model)},
        resources={"gpus":4, "shm_gb":32})
    rendered = RlinfAdapter().render(spec, "run_rt")
    assert "/workspace/eb_runner/rlinf_entry.py" in rendered.command
    assert "robotwin_sft_openpi_pi05" in rendered.command
    assert "actor.model.openpi.config_name=pi05_aloha_robotwin" in rendered.command
    assert (str(data), "/workspace/data/ds_rt", "ro") in rendered.mounts
    assert rendered.image == "eb/rlinf:libero-pi05"
