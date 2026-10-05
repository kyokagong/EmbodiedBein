from eb.capacity import estimate_job
from eb.domain.job import JobKind, JobSpec


def test_sim2sim_fits_no_gpu():
    spec = JobSpec(kind=JobKind.EVAL_SIM2SIM, embodiment_env_id="ee_bhl_biped_12dof_v1", task_id="tsk_bhl_velocity_biped", resources={"gpus": 0})
    est = estimate_job(spec)
    assert est.verdict == "fits"
    assert est.per_gpu_peak_gb == 0


def test_export_policy_fits_on_4090():
    spec = JobSpec(
        kind=JobKind.EXPORT_POLICY,
        embodiment_env_id="ee_bhl_biped_12dof_v1",
        resources={"gpus": 1},
        params={"num_envs": 16},
    )
    est = estimate_job(spec)
    assert est.verdict == "fits"
    assert est.per_gpu_peak_gb < 24


def test_openvla_full_ft_no_fit_on_24gb():
    spec = JobSpec(
        kind=JobKind.TRAIN_SFT,
        embodiment_env_id="ee_bhl_fixedhand_10dof_v1",
        task_id="tsk_bhl_reach",
        resources={"gpus": 4},
        params={"params_b": 7.5e9, "optimizer": "adamw"},
    )
    est = estimate_job(spec)
    assert est.verdict == "no_fit"


def test_cpu_acquisition_is_not_estimated_as_large_model_training():
    for operation in ('download_model', 'acquire_libero_benchmark'):
        spec = JobSpec(kind=JobKind.DATA_CONVERT, embodiment_env_id='ee_libero_franka_delta_v1',
                       params={'operation': operation}, resources={'gpus':0})
        assert estimate_job(spec).per_gpu_peak_gb == 0


def test_rpo_amp_cpu_motion_conversion_requires_no_gpu_memory():
    for operation in ('convert_rpo_amp_official', 'audit_rpo_amp_official', 'validate_rpo_amp_motion'):
        spec = JobSpec(kind=JobKind.DATA_CONVERT,
                       embodiment_env_id='ee_rpo_v2_amp_locomotion_23dof_v1',
                       params={'operation': operation}, resources={'gpus':0})
        estimate = estimate_job(spec)
        assert estimate.verdict == 'fits'
        assert estimate.per_gpu_peak_gb == 0


def test_libero_expert_sft_accounts_for_frozen_and_trainable_weights():
    spec = JobSpec(kind=JobKind.TRAIN_SFT, embodiment_env_id='ee_libero_franka_delta_v1',
        task_id='tsk_libero_spatial', params={'stack':'rlinf','template_id':'libero_pi05_sft'},resources={'gpus':4})
    estimate = estimate_job(spec)
    assert estimate.basis['inputs']['P_train'] == 430_098_464
    assert estimate.basis['inputs']['P_frozen'] == 3_186_659_056
    assert estimate.confidence == 'analytic'
    assert estimate.per_gpu_peak_gb < 15


def test_libero_large_microbatch_is_not_admitted_as_default_batch():
    spec = JobSpec(kind=JobKind.TRAIN_SFT, embodiment_env_id='ee_libero_franka_delta_v1',
        task_id='tsk_libero_spatial', params={'stack':'rlinf','template_id':'libero_pi05_sft'},
        overrides=['actor.micro_batch_size=128'],resources={'gpus':4})
    assert estimate_job(spec).verdict == 'no_fit'


def test_libero_evaluation_has_no_optimizer_and_no_assumed_weight_sharding():
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id='ee_libero_franka_delta_v1',
        task_id='tsk_libero_spatial',params={'stack':'rlinf','template_id':'libero_spatial_pi05_eval'},resources={'gpus':1})
    one = estimate_job(spec)
    spec.resources.gpus = 4
    assert one.verdict != 'no_fit'
    assert estimate_job(spec).per_gpu_peak_gb == one.per_gpu_peak_gb


def test_robotwin_pi05_evaluation_uses_inference_not_training_memory():
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id='ee_robotwin_aloha_joint14_v1',
        task_id='tsk_robotwin_multitask',
        params={'stack':'rlinf','template_id':'robotwin_pi05_eval'}, resources={'gpus':1})
    estimate = estimate_job(spec)
    assert estimate.verdict != 'no_fit'
    assert estimate.per_gpu_peak_gb < 18
    assert estimate.basis['inputs']['weight_bytes_per_parameter'] == 2
    assert estimate.basis['inputs']['optimizer_bytes'] == 0
    assert estimate.basis['inputs']['policy_replicas'] == 1


def test_rpo_isaac_evaluation_uses_single_env_memory_not_training_optimizer():
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id='ee_rpo_v2_wbc_23dof_v1',
        task_id='tsk_rpo_walk_upperbody_tracking',
        params={'stack':'rlinf', 'template_id':'rpo_v2_wbc_eval', 'record_video':True},
        resources={'gpus':1})
    estimate = estimate_job(spec)
    assert estimate.verdict == 'fits'
    assert estimate.per_gpu_peak_gb < 10
    assert estimate.basis['inputs']['optimizer_bytes'] == 0
    spec.params['record_video'] = False
    assert estimate_job(spec).per_gpu_peak_gb < estimate.per_gpu_peak_gb


def test_rpo_amp_single_gpu_probe_is_estimated_as_small_policy_training():
    spec = JobSpec(kind=JobKind.TRAIN_RL,
                   embodiment_env_id='ee_rpo_v2_amp_locomotion_23dof_v1',
                   task_id='tsk_rpo_amp_stand_walk',
                   params={'stack':'rlinf', 'template_id':'rpo_v2_amp_stand_walk_ppo'},
                   overrides=['env.train.total_num_envs=512'], resources={'gpus':1})
    estimate = estimate_job(spec)
    assert estimate.verdict == 'fits'
    assert estimate.per_gpu_peak_gb < 24


def test_rpo_amp_isaac_evaluation_uses_inference_memory():
    spec = JobSpec(kind=JobKind.EVAL_SIM,
                   embodiment_env_id='ee_rpo_v2_amp_locomotion_23dof_v1',
                   task_id='tsk_rpo_amp_stand_walk',
                   params={'stack':'rlinf', 'template_id':'rpo_v2_amp_eval'},
                   resources={'gpus':1})
    estimate = estimate_job(spec)
    assert estimate.verdict == 'fits'
    assert estimate.per_gpu_peak_gb < 10


def test_rpo_amp_asset_probe_reserves_one_isaac_gpu():
    spec = JobSpec(kind=JobKind.ASSET_VALIDATE,
                   embodiment_env_id='ee_rpo_v2_amp_locomotion_23dof_v1',
                   params={'operation':'probe_rpo_amp_env'}, resources={'gpus':1})
    estimate = estimate_job(spec)
    assert estimate.verdict == 'fits'
    assert 0 < estimate.per_gpu_peak_gb < 10


def test_more_eval_gpus_do_not_multiply_policy_weights():
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id='ee_robotwin_aloha_joint14_v1',
        task_id='tsk_robotwin_multitask', params={'stack':'rlinf',
        'template_id':'robotwin_pi05_eval'}, resources={'gpus':1})
    one = estimate_job(spec)
    spec.resources.gpus = 4
    four = estimate_job(spec)
    assert one.basis['inputs']['policy_replicas'] == 1
    assert four.basis['inputs']['policy_replicas'] == 1


def test_libero_ppo_expert_budget_supports_two_gpu_but_scales_batch():
    spec = JobSpec(kind=JobKind.TRAIN_RLFT, embodiment_env_id='ee_libero_franka_delta_v1',
        task_id='tsk_libero_spatial', params={'stack':'rlinf','template_id':'libero_spatial_pi05_ppo'},resources={'gpus':2})
    assert estimate_job(spec).verdict != 'no_fit'
    spec.overrides = ['actor.micro_batch_size=128']
    assert estimate_job(spec).verdict == 'no_fit'
