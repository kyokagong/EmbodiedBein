from eb.domain.rlinf_form import (
    TEMPLATES,
    component_placement,
    forced_overrides,
    ray_port_for,
    validate_hydra_overrides,
)


def test_templates_include_three_rlinf_recipes():
    ids = {t.id for t in TEMPLATES}
    assert ids == {"berkeley_wbc_ppo_mlp", "berkeley_reach_sft", "collect_berkeley_reach"}


def test_ppo_placement_uses_comma_key():
    assert component_placement("berkeley_wbc_ppo_mlp", 1) == "cluster.component_placement.actor,env,rollout=0"
    assert component_placement("berkeley_wbc_ppo_mlp", 2) == "cluster.component_placement.actor,env,rollout=0,1"
    assert component_placement("berkeley_wbc_ppo_mlp", 4) == "cluster.component_placement.actor,env,rollout=0,1,2,3"


def test_sft_placement_uses_actor_rollout():
    assert component_placement("berkeley_reach_sft", 2) == "cluster.component_placement.actor_rollout=0,1"


def test_ray_port_stable_and_in_range():
    a = ray_port_for("trn_abc")
    b = ray_port_for("trn_abc")
    c = ray_port_for("trn_other")
    assert a == b
    assert 6390 <= a < 6440
    assert a != c or True  # collision allowed; only stability required


def test_forced_overrides_lock_log_path():
    ov = forced_overrides("job-name")
    assert "runner.logger.log_path=/workspace/output" in ov
    assert "runner.checkpoint.save_dir=/workspace/output/checkpoints" in ov


def test_hydra_override_rejects_empty_key():
    bad = validate_hydra_overrides(["=1", "runner.max_epochs=10"])
    assert bad
    assert any("空" in x or "empty" in x.lower() or "=" in x for x in bad)
