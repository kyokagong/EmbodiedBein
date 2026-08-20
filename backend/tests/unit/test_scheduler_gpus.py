from eb.scheduler.loop import _free_gpu_indices


def test_empty_doctor_skips_without_fake(monkeypatch):
    monkeypatch.delenv("EB_FAKE_GPUS", raising=False)
    monkeypatch.setattr("eb.settings.get_settings", lambda: type("S", (), {"fake_gpus": ""})())
    assert _free_gpu_indices({"gpus": []}) is None


def test_fake_gpus_env_opens_pool(monkeypatch):
    monkeypatch.setenv("EB_FAKE_GPUS", "0,1,2,3")
    assert _free_gpu_indices({"gpus": []}) == [0, 1, 2, 3]


def test_busy_gpu_excluded_from_free():
    doctor = {
        "gpus": [
            {"index": 0, "run_id": "run_a"},
            {"index": 1, "run_id": None},
            {"index": 2},
        ]
    }
    assert _free_gpu_indices(doctor) == [1, 2]
