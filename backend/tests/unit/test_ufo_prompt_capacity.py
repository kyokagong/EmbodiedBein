from eb.capacity.estimator import estimate_job
from eb.domain.job import JobSpec


def test_native_reward_inference_has_no_gpu_optimizer_capacity_requirement():
    spec = JobSpec(
        kind="data.convert",
        embodiment_env_id="ee_rpo_v2_ufo_native_23dof_v2",
        params={"operation": "ufo_native_prompt"},
        resources={"gpus": 0, "cpus": 4, "memory_gb": 64, "shm_gb": 2},
    )
    estimate = estimate_job(spec, host_vram_gb=0)
    assert estimate.per_gpu_peak_gb == 0
    assert estimate.verdict != "no_fit"
    assert estimate.basis["inputs"]["optimizer_bytes"] == 0
