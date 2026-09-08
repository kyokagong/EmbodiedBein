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
