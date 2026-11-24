from copy import deepcopy

import pytest

from eb.pbfm_training import native_arguments, segment_endpoint, training_fingerprint
from eb.runners.rlinf_pbfm_runtime import validate_plan


def plan_fixture():
    from eb.pbfm_training import CAMERA_CONTRACT, TERRAIN_CONTRACT

    digest = "sha256:" + "a" * 64
    identity = dict(
        image_digest=digest,
        world_size=1,
        num_envs_per_rank=1024,
        buffer_size=32768,
        seed=42,
        dataset_id="ds_train",
        dataset_manifest="b" * 64,
        contract_hash="c" * 64,
        environment_profile={"id": "ep_train", "purpose": "train"},
        pbfm_snapshot=digest,
        rlinf_snapshot=digest,
        asset_snapshot=digest,
        camera_contract=deepcopy(CAMERA_CONTRACT),
        terrain_contract=deepcopy(TERRAIN_CONTRACT),
    )
    plan = dict(
        schema_version="eb/pbfm-native-train-plan/v1",
        run_id="run_test",
        image_digest=digest,
        world_size=1,
        target_updates=112,
        identity=identity,
        fingerprint=training_fingerprint(identity),
        source_refs={
            name: {
                "path": f"/sources/{name}",
                "snapshot_hash": digest,
                "manifest_path": f"/sources/{name}-manifest.json",
            }
            for name in ("pbfm", "rlinf")
        },
        log_every_steps=1024,
        full_state_finite_required=True,
    )
    plan["native_args"] = native_arguments(plan)
    plan["source_refs"]["asset"] = {
        "path": "/robot-assets",
        "snapshot_hash": digest,
        "manifest_path": "/sources/asset-manifest.json",
    }
    return plan


def test_native_counter_boundaries():
    assert segment_endpoint(112, 1) == 18432
    assert segment_endpoint(512, 4, start_updates=112, start_local_time=18432) == 45056 * 4
    for updates, world in ((100, 1), (112, 3), (128, 0), (100016, 1)):
        with pytest.raises(ValueError):
            segment_endpoint(updates, world)


def test_frozen_plan_and_recipe_semantics():
    plan = plan_fixture()
    assert validate_plan(plan) == plan
    for mutate in (
        lambda p: p["identity"].update(prior_plane_envs=2),
        lambda p: p["native_args"].extend(["--smoke"]),
        lambda p: p["identity"].update(world_size=2),
        lambda p: p["identity"]["camera_contract"].update(mount_body="pelvis"),
        lambda p: p.update(start_updates=112, start_local_time=18432),
    ):
        changed = deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_plan(changed)


@pytest.mark.parametrize(
    ("target", "start_updates", "start_local", "expected"),
    [(112, 0, 0, 106496), (368, 112, 26624, 245760), (512, 368, 61440, 327680)],
)
def test_2048_segment_preserves_native_checker_resume_phase(target, start_updates, start_local, expected):
    assert segment_endpoint(
        target, 4, start_updates=start_updates, start_local_time=start_local, num_envs=2048
    ) == expected


@pytest.mark.parametrize("num_envs", [True, False, 512, 4096, "2048", 2048.0])
def test_segment_rejects_unfrozen_environment_counts(num_envs):
    with pytest.raises(ValueError):
        segment_endpoint(112, 4, num_envs=num_envs)


@pytest.mark.parametrize(
    "counts",
    [{"start_updates": 112, "start_local_time": 26624 + 1024},
     {"start_updates": 112, "start_local_time": 0},
     {"start_updates": 0, "start_local_time": 26624},
     {"start_updates": 113, "start_local_time": 26624}],
)
def test_2048_segment_rejects_unaligned_or_inconsistent_resume_counts(counts):
    with pytest.raises(ValueError):
        segment_endpoint(368, 4, num_envs=2048, **counts)


def plan_2048_fixture():
    plan = plan_fixture()
    plan["world_size"] = 4
    plan["identity"].update(world_size=4, num_envs_per_rank=2048, buffer_size=131072)
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["log_every_steps"] = 4096
    plan["native_args"] = native_arguments(plan)
    return plan


def test_2048_plan_freezes_cli_and_native_log_interval():
    plan = plan_2048_fixture()
    assert validate_plan(plan) == plan
    argv = dict(zip(plan["native_args"][::2], plan["native_args"][1::2]))
    assert argv["--num-envs"] == "2048"
    assert argv["--num-env-steps"] == argv["--checkpoint-every-steps"] == "106496"
    assert argv["--buffer-size"] == "131072"
    plan["log_every_steps"] = 8192
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize("buffer_size", [65536, 71680, 131072 + 1024, True])
def test_2048_replay_requires_integral_slots_beyond_maximum_depth_age(buffer_size):
    plan = plan_2048_fixture()
    plan["identity"]["buffer_size"] = buffer_size
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["native_args"] = native_arguments(plan)
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_2048_replay_accepts_first_capacity_with_a_complete_depth_history():
    plan = plan_2048_fixture()
    plan["identity"]["buffer_size"] = 73728  # 36 slots: one row can reach age 35.
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["native_args"] = native_arguments(plan)
    assert validate_plan(plan) == plan


@pytest.mark.parametrize("drift", ["world", "envs", "fingerprint"])
def test_2048_plan_rejects_topology_and_identity_drift(drift):
    plan = plan_2048_fixture()
    if drift == "world":
        plan["world_size"] = 2
    elif drift == "envs":
        plan["identity"]["num_envs_per_rank"] = 1024
    else:
        plan["identity"]["seed"] += 1
    with pytest.raises(ValueError):
        validate_plan(plan)
