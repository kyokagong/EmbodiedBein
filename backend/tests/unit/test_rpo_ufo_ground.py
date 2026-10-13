import numpy as np
import pytest


def test_ground_coverage_includes_reference_feet_and_actual_origins():
    from eb.runners.rpo_ufo_ground import coverage_receipt

    origins = np.array([[-56.25, -55., 0.], [56.25, 55., 0.]])
    # Includes reference root displacement and the actual collision mesh envelope.
    feet = np.array([[-7.4, -6.5], [8., 8.5]])
    with pytest.raises(ValueError, match='ground coverage'):
        coverage_receipt(origins, feet, (100., 100.))
    receipt = coverage_receipt(origins, feet, (512., 512.))
    assert receipt['minimum_margin_m'] == pytest.approx(191.75)
    assert receipt['num_envs'] == 2
    with pytest.raises(ValueError):
        coverage_receipt(origins * np.nan, feet, (512., 512.))


def test_ground_recipe_derives_contact_without_changing_learning_or_restore_budget():
    from eb.domain.training_recipe import RECIPES, UFO_GROUND_RECIPE, validate_update_target
    from eb.domain.rlinf_form import build_overrides
    from eb.rpo_bfm import CODEC_V2_HASH, codec_hash_for_execution, action_control_for_execution

    binding = RECIPES[UFO_GROUND_RECIPE].binding.model_dump()
    assert codec_hash_for_execution(binding) == CODEC_V2_HASH
    control = action_control_for_execution(binding)
    assert control['ground']['size_m'] == [512., 512.]
    assert control['physics']['physics_dt_s'] == .0025
    values = build_overrides(UFO_GROUND_RECIPE, 4, 'ground', [])
    assert 'env.train.total_num_envs=6144' in values
    assert 'actor.model.batch_size=2048' in values
    assert '+actor.model.contact_profile=contact_stable_v8' in values
    assert '+actor.model.ground_profile=wide_plane_v9' in values
    validate_update_target(524, start=24, recipe_id=UFO_GROUND_RECIPE)
    with pytest.raises(ValueError):
        validate_update_target(525, start=24, recipe_id=UFO_GROUND_RECIPE)


def test_ground_runtime_rejects_mixed_ground_and_contact_profiles(monkeypatch):
    pytest.importorskip('torch')
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from eb.domain.rlinf_form import get_template, build_overrides
    from eb.domain.training_recipe import UFO_GROUND_RECIPE
    from eb.rpo_bfm import CODEC_V2_HASH
    from eb.runners.bfm_ufo_worker import validate_ufo_runtime_config

    monkeypatch.setenv('EMBODIED_PATH', '/workspace/rlinf/examples/embodiment')
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / 'eb/runners/config'), version_base=None):
        cfg = compose(config_name=get_template(UFO_GROUND_RECIPE).config_name,
            overrides=build_overrides(UFO_GROUND_RECIPE, 4, 'fixture', []) + [f'+actor.model.codec_sha256={CODEC_V2_HASH}'])
    validate_ufo_runtime_config(cfg)
    cfg.actor.model.contact_profile = None
    with pytest.raises(ValueError, match='approved'):
        validate_ufo_runtime_config(cfg)
