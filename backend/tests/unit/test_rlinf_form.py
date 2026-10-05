import pytest
from eb.domain.rlinf_form import (TEMPLATES, build_overrides, component_placement,
                                  template_entrypoint, validate_hydra_overrides)


def test_templates_cover_libero_sft_online_rl_and_independent_eval():
    ids = {t.id for t in TEMPLATES}
    assert {'libero_pi05_sft', 'libero_spatial_pi05_ppo', 'libero_spatial_pi05_eval'} <= ids


def test_task_kind_selects_rlinf_native_entrypoint():
    templates = {template.id:template for template in TEMPLATES}
    assert template_entrypoint(templates['libero_pi05_sft']) == 'examples/sft/train_vla_sft.py'
    assert template_entrypoint(templates['libero_spatial_pi05_ppo']) == 'examples/embodiment/train_embodied_agent.py'
    assert template_entrypoint(templates['robotwin_pi05_eval']) == 'evaluations/eval_embodied_agent.py'


def test_templates_cover_robotwin_closed_loop_eval_without_claiming_xr1_ready():
    templates = {t.id: t for t in TEMPLATES}
    assert {
        'robotwin_pi05_eval', 'robotwin_lingbot_vla2_eval',
        'robotwin_gr00t_n17_eval', 'robotwin_dm05_eval', 'robotwin_xr1_eval',
    } <= templates.keys()
    assert all(templates[key].kind == 'eval' for key in templates if key.startswith('robotwin_') and key.endswith('_eval'))
    assert templates['robotwin_pi05_eval'].ready
    assert not templates['robotwin_xr1_eval'].ready
    assert 'controller' in templates['robotwin_xr1_eval'].blocked_reason.lower()


def test_placement_is_hydra_scalar_and_uses_existing_key():
    assert component_placement('berkeley_reach_sft', 4) == r'cluster.component_placement={actor\,\ env\,\ rollout:0-3}'
    assert component_placement('libero_spatial_pi05_ppo', 4) == r'+cluster.component_placement={actor:0-3,env\,\ rollout:0}'
    with pytest.raises(ValueError):
        component_placement('berkeley_reach_sft', 0)


def test_eval_component_placement_has_no_training_actor():
    assert component_placement('robotwin_pi05_eval', 1) == \
        r'cluster.component_placement={env:0,rollout:0}'
    assert component_placement('robotwin_pi05_eval', 4) == \
        r'cluster.component_placement={env:0-3,rollout:0}'
    overrides = build_overrides('robotwin_pi05_eval', 4, 'smoke', [])
    assert 'rollout.model.add_value_head=False' in overrides
    assert 'runner.enable_decoupled_mode=False' in overrides
    assert 'rollout.enable_offload=False' in overrides
    assert 'env.eval.video_cfg.save_video=False' in overrides


def test_protected_overrides_cannot_escape_binding():
    for override in ['runner.logger.log_path=/tmp', '+hydra.searchpath=[file:///tmp]', 'cluster.component_placement.actor=all', 'actor.model.model_path=/tmp', '~runner', 'actor={model:bad}']:
        assert validate_hydra_overrides([override]), override
    assert not validate_hydra_overrides(['actor.optim.lr=1e-5'])
    result = build_overrides('berkeley_reach_sft', 2, 'test name', ['actor.optim.lr=1e-5'])
    assert 'runner.checkpoint.save_dir=/workspace/output/checkpoints' not in result
    assert 'runner.logger.log_path=/workspace/output' in result


def test_libero_sft_explicitly_freezes_vlm_for_action_expert_training():
    overrides = build_overrides('libero_pi05_sft', 4, 'expert-sft', [])
    assert 'actor.model.openpi.train_expert_only=True' in overrides


def test_libero_ppo_uses_periodic_and_final_checkpoint_cadence():
    overrides = build_overrides('libero_spatial_pi05_ppo', 4, 'ppo', [])
    assert 'runner.save_interval=40' in overrides
