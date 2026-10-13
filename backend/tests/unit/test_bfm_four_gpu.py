import pytest


@pytest.mark.parametrize("override,reason", [
    ("runner.max_steps=1", "max_steps"),
    ("env.train.rollout_epoch=2", "rollout_epoch"),
])
def test_bfm_frozen_round_and_update_budget_cannot_be_overridden(override, reason):
    from eb.domain.rlinf_form import build_overrides
    with pytest.raises(ValueError, match=reason):
        build_overrides("rpo_v2_bfm_zero_pretrain", 4, "bad", ["env.train.total_num_envs=24576", override])


def test_bfm_formal_updates_require_frozen_batch_128():
    from eb.domain.rlinf_form import build_overrides
    with pytest.raises(ValueError, match="128"):
        build_overrides("rpo_v2_bfm_zero_pretrain", 4, "bad", ["env.train.total_num_envs=24576", "runner.max_epochs=1000"])
    values = build_overrides("rpo_v2_bfm_zero_pretrain", 4, "formal", [
        "env.train.total_num_envs=24576", "runner.max_epochs=1000", "actor.micro_batch_size=128", "actor.global_batch_size=128",
    ])
    assert "runner.max_epochs=1000" in values


def test_bfm_joint_rollout_fills_replay_batch_from_first_update():
    from eb.domain.rlinf_form import build_overrides
    with pytest.raises(ValueError, match="first.*batch"):
        build_overrides("rpo_v2_bfm_zero_pretrain", 4, "underfilled", [
            "env.train.total_num_envs=6", "runner.max_epochs=1000",
            "actor.micro_batch_size=128", "actor.global_batch_size=128",
        ])


def test_bfm_four_gpu_has_one_learner_and_three_sampler_ranks():
    from eb.domain.rlinf_form import component_placement,build_overrides
    placement=component_placement("rpo_v2_bfm_zero_pretrain",4)
    assert "actor:0" in placement and "rollout:1-3" in placement and "actor:0-3" not in placement
    values=build_overrides("rpo_v2_bfm_zero_pretrain",4,"probe",["env.train.total_num_envs=1536"])
    assert "actor.micro_batch_size=8" in values and "actor.global_batch_size=8" in values
    with pytest.raises(ValueError):
        build_overrides("rpo_v2_bfm_zero_pretrain",4,"bad",["env.train.total_num_envs=1537"])
def test_four_gpu_capacity_uses_separate_measured_roles_and_keeps_replay_round():
    from eb.domain.job import JobSpec,JobKind
    from eb.capacity.estimator import estimate_job
    from eb.domain.rlinf_form import build_overrides
    spec=JobSpec(kind=JobKind.TRAIN_RL,embodiment_env_id="ee_rpo_v2_bfm_23dof_v1",task_id="tsk_rpo_bfm_pretrain",
        params={"template_id":"rpo_v2_bfm_zero_pretrain"},resources={"gpus":4},
        overrides=["env.train.total_num_envs=12288"])
    result=estimate_job(spec)
    assert result.verdict=="fits" and result.headroom_gb>=2.4
    assert result.basis["inputs"]["sampling_gpus"]==3
    with __import__("pytest").raises(ValueError,match="replay"):
        build_overrides("rpo_v2_bfm_zero_pretrain",4,"overflow",["env.train.total_num_envs=30000"])


def test_bfm_eight_updates_per_round_are_scoped_and_counted():
    from eb.domain.rlinf_form import build_overrides
    from eb.rpo_bfm import learner_updates, validate_training_budget
    values = build_overrides("rpo_v2_bfm_zero_pretrain", 4, "balanced", [
        "env.train.total_num_envs=3072", "runner.max_epochs=125",
        "actor.micro_batch_size=128", "actor.global_batch_size=128", "actor.model.updates_per_round=8"])
    assert "actor.model.updates_per_round=8" in values
    assert learner_updates(125, 8) == 1000
    ready = dict(smoke_ready=True, training_ready=True)
    assert validate_training_budget(2000, ready, start_update=1000) == "local_research"
    for target, start in ((1001, 0), (2001, 1000), (100001, 99001), (1000, 1000)):
        with pytest.raises(ValueError): validate_training_budget(target, ready, start_update=start)
    with pytest.raises(ValueError): learner_updates(125, 7)
    with pytest.raises(ValueError):
        build_overrides("rpo_v2_flat_ppo", 4, "invalid", ["actor.model.updates_per_round=8"])


def test_bfm_resume_maintains_recipe_and_extends_real_update_target():
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint
    from eb.domain.job import JobKind, JobSpec
    source = JobSpec(kind=JobKind.TRAIN_RL, embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", task_id="tsk_rpo_bfm_pretrain",
        params=dict(template_id="rpo_v2_bfm_zero_pretrain", bfm_update_budget=1000, bfm_update_target=1000),
        overrides=["runner.max_epochs=125", "actor.model.updates_per_round=8"], resources=dict(gpus=4))
    checkpoint = dict(run_id="run_test", step=125, mode="trainer_state", compatibility_fingerprint=resume_fingerprint(source))
    result = build_resume_spec(source, "trn_test", checkpoint, ResumeTrainIn(name="continued", max_epochs=250))
    assert result.params["bfm_update_target"] == 2000
    assert source.params["bfm_update_target"] == 1000
    assert resume_fingerprint(source) == resume_fingerprint(result)
    result.overrides[-1] = "actor.model.updates_per_round=1"
    assert resume_fingerprint(source) != resume_fingerprint(result)
