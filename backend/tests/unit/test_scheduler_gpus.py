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


def test_gpu_memory_from_external_workloads_limits_admission():
    doctor = {'gpus':[{'index':0,'memory_free_mb':6000},
                      {'index':1,'memory_free_mb':24000},
                      {'index':2,'memory_free_mb':24000,'run_id':'other'}]}
    assert _free_gpu_indices(doctor, min_free_mb=12000) == [1]


def test_memory_admission_keeps_host_reserve_and_running_budgets():
    from eb.scheduler.loop import _memory_budget_mb
    assert _memory_budget_mb({}) == 0
    assert _memory_budget_mb({"memory_available_mb": 80 * 1024}, 0) == 76 * 1024
    assert _memory_budget_mb({"memory_available_mb": 80 * 1024}, 64 * 1024) == 12 * 1024
    assert _memory_budget_mb({"memory_available_mb": 1000}, 0) == 0
