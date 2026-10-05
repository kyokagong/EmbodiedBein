import pytest

from eb.domain.robotwin_eval_runtime import (
    derive_robotwin_runtime,
    list_robotwin_runtime_profiles,
    read_robotwin_runtime,
    resolve_robotwin_runtime,
)


@pytest.mark.parametrize("gpus,parallel,workers,envs,micro", [
    (1, 4, 1, 4, 4),
    (2, 8, 2, 4, 4),
    (4, 16, 4, 4, 4),
])
def test_verified_profiles_use_one_policy(gpus, parallel, workers, envs, micro):
    runtime, resources = derive_robotwin_runtime(gpus, parallel)

    assert runtime.policy_worker_count == 1
    assert runtime.env_worker_count == workers
    assert runtime.envs_per_worker == envs
    assert runtime.parallel_envs == parallel
    assert runtime.inference_micro_batch_size == micro
    assert runtime.max_inflight_batches == 1
    assert runtime.headless and runtime.offscreen_rendering
    assert runtime.save_video is False
    assert resources["memory_gb"] >= resources["shm_gb"]


def test_legacy_runtime_is_readable_but_not_submittable():
    plan = {"model":{"runtime":{"gpu_count":4, "per_gpu_concurrency":4,
                                  "eval_batch_size":16}}}

    assert read_robotwin_runtime(plan, submission=False).parallel_envs == 16
    with pytest.raises(ValueError, match="重新预览"):
        read_robotwin_runtime(plan, submission=True)


def test_four_gpu_profile_has_room_above_measured_ray_peak():
    _, resources = derive_robotwin_runtime(4, 16)

    assert resources["memory_gb"] == 80


def test_runtime_catalog_exposes_three_bounded_product_profiles():
    profiles = list_robotwin_runtime_profiles()

    assert [row["id"] for row in profiles] == [
        "memory_saver_sync", "balanced_async", "throughput_async"]
    assert profiles[0]["status"] == "ready"
    assert profiles[0]["recommended_variant_id"] == "sync_4gpu_16env_80g"
    assert {row["memory_gb"] for row in profiles[0]["variants"]} == {48, 56, 80}
    assert profiles[1]["status"] == profiles[2]["status"] == "blocked"
    assert profiles[1]["blockers"] and profiles[2]["blockers"]


def test_runtime_variant_freezes_container_resources_and_probe():
    runtime, resources = resolve_robotwin_runtime(
        "memory_saver_sync", "sync_4gpu_16env_80g")

    assert runtime.profile_id == "memory_saver_sync"
    assert runtime.resource_variant_id == "sync_4gpu_16env_80g"
    assert runtime.scheduling_mode == "sync"
    assert runtime.probe_receipt_sha256.startswith("sha256:")
    assert resources == {"gpus":4, "cpus":32, "memory_gb":80, "shm_gb":40}


@pytest.mark.parametrize("profile,variant", [
    ("balanced_async", "anything"),
    ("throughput_async", "anything"),
    ("memory_saver_sync", "missing"),
    ("missing", "anything"),
])
def test_unverified_runtime_variant_is_rejected(profile, variant):
    with pytest.raises(ValueError, match="未就绪|不存在|未验证"):
        resolve_robotwin_runtime(profile, variant)


@pytest.mark.parametrize("gpus,parallel", [(1, 8), (2, 4), (4, 20), (3, 12)])
def test_unverified_profiles_are_rejected(gpus, parallel):
    with pytest.raises(ValueError, match="未验证"):
        derive_robotwin_runtime(gpus, parallel)
