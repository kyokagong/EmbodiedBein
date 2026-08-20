from eb.capacity import estimate_job
from eb.domain.job import JobKind, JobSpec


def test_sim2sim_fits_no_gpu():
    spec = JobSpec(kind=JobKind.EVAL_SIM2SIM, contract_id="ec_bhl_biped_12dof_v1", resources={"gpus": 0})
    est = estimate_job(spec)
    assert est.verdict == "fits"
    assert est.per_gpu_peak_gb == 0


def test_openvla_full_ft_no_fit_on_24gb():
    spec = JobSpec(
        kind=JobKind.TRAIN_SFT,
        contract_id="ec_bhl_fixedhand_10dof_v1",
        resources={"gpus": 4},
        params={"params_b": 7.5e9, "optimizer": "adamw"},
    )
    est = estimate_job(spec)
    assert est.verdict == "no_fit"
