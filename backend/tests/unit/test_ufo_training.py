from copy import deepcopy

import pytest

from eb.domain.job import JobSpec
from eb.ufo_native import ENV_ID, TASK_ID
from eb.ufo_training import segment_endpoint, training_fingerprint, validate_train_job


def test_actual_updates_and_resume_have_distinct_env_step_endpoints():
    assert segment_endpoint(112, 1) == 18432
    assert segment_endpoint(112, 4) == 73728
    # Native update checker starts from the resumed local counter.
    assert segment_endpoint(512, 1, start_updates=112, start_local_time=18432) == 45056
    with pytest.raises(ValueError):
        segment_endpoint(100, 1)
    # Longer approved budgets retain the same native counters and resume image.
    assert segment_endpoint(20000, 4, start_updates=10000, start_local_time=651264) == 5169152


def test_fingerprint_excludes_only_segment_budget_and_paths():
    identity = {"image": "sha256:a", "world_size": 1, "dataset": "sha256:b", "buffer_size": 262144}
    assert training_fingerprint(identity) == training_fingerprint(deepcopy(identity))
    changed = {**identity, "world_size": 2}
    assert training_fingerprint(identity) != training_fingerprint(changed)


def test_native_training_cannot_fall_through_to_deployment_or_ppo():
    spec = JobSpec(
        kind="train.rl",
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID,
        params={"template_id": "rpo_ufo_fb_native", "stack": "rlinf"},
    )
    with pytest.raises(ValueError, match="plan"):
        validate_train_job(spec)


def native_spec():
    from eb.ufo_training import native_arguments

    digest = "sha256:" + "a" * 64
    identity = {
        "image_digest": digest,
        "world_size": 1,
        "buffer_size": 262144,
        "seed": 42,
        "asset_snapshot": digest,
        "ufo_snapshot": digest,
        "rlinf_snapshot": digest,
        "dataset_id": "ds_native",
        "dataset_manifest": "b" * 64,
    }
    plan = {
        "schema_version": "eb/ufo-native-train-plan/v1",
        "identity": identity,
        "fingerprint": training_fingerprint(identity),
        "image_digest": digest,
        "world_size": 1,
        "target_updates": 112,
        "log_every_steps": 1024,
        "source_refs": {
            name: {
                "path": f"/sources/{name}",
                "snapshot_hash": digest,
                "manifest_path": f"/sources/{name}-manifest.json",
            }
            for name in ("ufo", "rlinf")
        },
    }
    plan["native_args"] = native_arguments(plan)
    ref = {"kind": "content_snapshot", "commit": digest, "path": "/frozen"}
    return JobSpec(
        kind="train.rl",
        image=digest,
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID,
        params={
            "template_id": "rpo_ufo_fb_native",
            "stack": "rlinf",
            "native_train_plan": plan,
            "ufo_source_ref": ref,
            "asset_source_ref": ref,
        },
        code_refs={"stack": ref},
        inputs={
            "datasets": [
                {"id": "ds_native", "path": "/data/frozen", "validation": {"manifest_sha256": "b" * 64}}
            ]
        },
        resources={"gpus": 1, "cpus": 8, "memory_gb": 32, "shm_gb": 8},
        runtime={"network": "none", "hostname": "localhost"},
        policy={"timeout_h": 6, "max_retries": 0},
    )


def test_native_job_uses_rlinf_adapter_and_preserves_replicated_capacity():
    from eb.adapters import get_adapter
    from eb.capacity.estimator import estimate_job

    spec = native_spec()
    assert validate_train_job(spec)["target_updates"] == 112
    assert get_adapter(spec).name == "rlinf"
    one = estimate_job(spec)
    four = spec.model_copy(deep=True)
    four.resources.gpus = 4
    assert estimate_job(four).per_gpu_peak_gb == one.per_gpu_peak_gb
    assert one.basis["inputs"]["replicated_per_rank"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_size", 2),
        ("target_updates", 100),
        ("log_every_steps", 81920),
        ("native_args", ["--smoke"]),
        ("image_digest", "eb/ufo-native:rpo-train-v1"),
        ("fingerprint", "sha256:" + "b" * 64),
    ],
)
def test_mutated_native_plans_are_rejected(field, value):
    spec = native_spec()
    spec.params["native_train_plan"][field] = value
    with pytest.raises(ValueError):
        validate_train_job(spec)


def test_native_preview_has_no_write_side_effect_and_no_holdout_mount(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from eb.ufo_training import render_training

    monkeypatch.setattr(
        "eb.settings.get_settings",
        lambda: SimpleNamespace(data_root=tmp_path, assets_root=tmp_path / "assets"),
    )
    rendered = render_training(native_spec(), "run_preview")
    assert not list(tmp_path.iterdir())
    assert rendered.command[2] == "eb.runners.rlinf_ufo_entry"
    assert not any("validation" in src or "holdout" in src for src, _, _ in rendered.mounts)
    assert [dst for _, dst, _ in rendered.mounts if dst.startswith("/training-data/")] == [
        "/training-data/train.pkl",
        "/training-data/manifest.json",
    ]


@pytest.mark.asyncio
async def test_native_preview_hash_is_accepted_by_submit_schema(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from eb.api import routes_train

    body = routes_train.TrainIn(
        name="native112",
        model_name="native112",
        template_id="rpo_ufo_fb_native",
        embodiment_id="emb_roboparty_rpo",
        gpus=1,
    )
    monkeypatch.setattr(routes_train, "_build_spec", AsyncMock(return_value=native_spec()))
    result = await routes_train.preview_train(body, None, SimpleNamespace(id="user:kk"))
    submitted = routes_train.TrainIn.model_validate(
        {**body.model_dump(), "preview_hash": result["preview_hash"]}
    )
    assert submitted.preview_hash == "sha256:" + native_spec().canonical_hash()
