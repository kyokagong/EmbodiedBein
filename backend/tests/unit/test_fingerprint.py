from eb.domain.fingerprint import fingerprint
from eb.domain.job import JobKind, JobSpec
from eb.scheduler.loop import _free_gpu_indices


def test_fingerprint_pentuple():
    spec = JobSpec(kind=JobKind.TRAIN_RL, embodiment_env_id="ee_bhl_biped_12dof_v1", task_id="tsk_bhl_velocity_biped", params={"seed": 0})
    fp = fingerprint(spec, "sha256:abc", "deadbeef")
    assert set(fp) == {"code_ref", "config_hash", "data_version", "image_digest", "seed"}
    assert fp["seed"] == 0
    assert fp["image_digest"] == "sha256:abc"
    assert fp["code_ref"] == "deadbeef"


def test_four_gpus_busy_fifth_has_no_free():
    doctor = {"gpus": [{"index": i, "run_id": f"run_{i}"} for i in range(4)]}
    assert _free_gpu_indices(doctor) == []
