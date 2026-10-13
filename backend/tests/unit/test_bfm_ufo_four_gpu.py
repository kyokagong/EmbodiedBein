import pytest

from eb.domain.job import JobKind, JobSpec
from eb.domain.rlinf_form import build_overrides, get_template
from eb.domain.training_recipe import RECIPES, validate_training_execution

FOUR = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_v1"


@pytest.mark.asyncio
async def test_native_four_evaluation_rejects_unqualified_four_env_batch():
    from types import SimpleNamespace

    from eb.api.routes_eval import RpoEvalIn, _prepare_rpo_flat

    model = SimpleNamespace(card={"schema_version": "eb/rpo-bfm-model/v1",
        "execution": RECIPES[FOUR].binding.model_dump()})
    with pytest.raises(ValueError, match="one environment per GPU"):
        await _prepare_rpo_flat(body=RpoEvalIn(model_version_id="native", backend="isaac",
            profile="validation", eval_gpus=4, envs_per_gpu=4, record_video=True), model=model, session=None)


def test_native_eval_renderer_rejects_unqualified_four_env_batch():
    from eb.adapters import get_adapter
    from eb.rpo_bfm import UFO_VALIDATION_SUITE

    spec = JobSpec(kind=JobKind.EVAL_SIM, task_id="tsk_rpo_bfm_pretrain",
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1", inputs={"base_model": "/native", "datasets": [{}]},
        params={"stack": "rlinf", "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_eval_v1", "rpo_eval_plan": {
            "suite_id": UFO_VALIDATION_SUITE, "profile": "validation", "record_video": True,
            "parallelism": {"gpus": 4, "envs_per_gpu": 4, "protocol": "independent_cases_fixed_terrain_v2"}}})
    with pytest.raises(ValueError, match="one environment per GPU"):
        get_adapter(spec).render(spec, "run_unqualified_batch")
    spec.params.pop("stack")
    with pytest.raises(ValueError, match="requires stack=rlinf"):
        get_adapter(spec)


def test_ufo_four_gpu_has_one_learner_group():
    template = get_template(FOUR)
    overrides = build_overrides(template.id, 4, "qualification", [])
    assert "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}" in overrides
    assert RECIPES[FOUR].binding.topology_id == "one_learner_three_samplers"
    with pytest.raises(ValueError):
        build_overrides(FOUR, 1, "wrong", [])


def test_four_gpu_qualification_keeps_frozen_resources_and_update_budget():
    recipe = RECIPES[FOUR]
    from eb.ufo_source import FOUR_GPU_SOURCE_RECEIPT

    spec = JobSpec(
        kind=JobKind.TRAIN_RL,
        task_id=recipe.task_id,
        embodiment_env_id=recipe.embodiment_env_id,
        image=recipe.image,
        params={
            "template_id": FOUR,
            "execution": recipe.binding.model_dump(),
            "purpose": "qualification",
            "target_learner_updates": 16,
            "source_receipt": FOUR_GPU_SOURCE_RECEIPT,
        },
        resources={"gpus": 4, "cpus": 24, "memory_gb": 64, "shm_gb": 32},
        policy={"timeout_h": 1, "max_retries": 0},
    )
    assert validate_training_execution(spec) == recipe.binding
    spec.resources.gpus = 1
    with pytest.raises(ValueError):
        validate_training_execution(spec)


def test_ufo_sampler_versions_match():
    torch = pytest.importorskip("torch")
    from eb.runners.bfm_ufo_policy import sync_identity, validate_sampler_identities

    packet = {
        "model": {
            "_actor.weight": torch.ones(2, 3),
            "_backward_map.weight": torch.ones(3, 2),
            "_obs_normalizer.running_mean": torch.zeros(2),
        },
        "pool": {"storage": torch.ones(2, 2), "cursor": 1, "full": False, "capacity": 2},
        "version": 16,
        "online_transitions": 11520,
    }
    identity = sync_identity(packet)
    receipts = [{"rank": rank, **identity} for rank in range(3)]
    validate_sampler_identities(identity, receipts, ranks=3)
    bad = {**packet, "model": {**packet["model"], "_backward_map.weight": torch.zeros(3, 2)}}
    receipts[2] = {"rank": 2, **sync_identity(bad)}
    with pytest.raises(ValueError):
        validate_sampler_identities(identity, receipts, ranks=3)
    with pytest.raises(ValueError):
        validate_sampler_identities(identity, receipts[:2], ranks=3)


def test_verified_sampler_sync_is_retained_as_run_evidence(tmp_path):
    pytest.importorskip("torch")
    import json
    from types import SimpleNamespace

    from eb.runners.bfm_ufo_worker import UFOActorWorker
    worker = object.__new__(UFOActorWorker)
    worker.cfg = SimpleNamespace(runner=SimpleNamespace(logger=SimpleNamespace(log_path=str(tmp_path))))
    worker.learner = SimpleNamespace(sampling_rounds=15)
    worker.placement = SimpleNamespace(get_world_size=lambda name: 3)
    worker.sync_expected = {"state_sha256": "a" * 64, "version": 16, "online_transitions": 12288}
    worker.verify_sampler_sync([{**worker.sync_expected, "rank": rank} for rank in range(3)])
    receipt = json.loads((tmp_path / "ufo-sampler-sync.json").read_text())
    assert receipt["sampling_rounds"] == 15 and len(receipt["samplers"]) == 3
    assert receipt["identity"] == worker.sync_expected


def test_eval_partition_has_identical_case_terrain_and_complete_coverage():
    from eb.runners.rpo_bfm_parallel_eval import case_terrain_slot, partition_seeds

    seeds = list(range(61000, 61016))
    single = {seed: case_terrain_slot(seed) for seed in seeds}
    groups = partition_seeds(list(reversed(seeds)), 4)
    parallel = {seed: case_terrain_slot(seed) for group in groups for seed in group}
    assert parallel == single
    assert len(set(single.values())) == len(seeds)
    assert all(0 <= row < 10 and 0 <= column < 20 for row, column in single.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe_id", [FOUR, "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_stable_v3"])
async def test_four_gpu_completed_checkpoint_is_promoted_to_model(tmp_path, monkeypatch, recipe_id):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import eb.checkpoints
    from eb.orchestrator import collect
    monkeypatch.setattr(collect, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(eb.checkpoints, "checkpoint_model", lambda *args, **kwargs: {
        "path": "/model", "bytes": 10, "step": 16, "manifest": {}})
    promote = AsyncMock(return_value=SimpleNamespace(id="mv_four"))
    monkeypatch.setattr(collect, "promote_rpo_model", promote)
    output = tmp_path / "runs/run_four/output"
    output.mkdir(parents=True)
    import json
    identity = {"state_sha256": "a" * 64, "version": 16, "online_transitions": 12288}
    (output / "ufo-sampler-sync.json").write_text(json.dumps({"schema_version": "eb/ufo-sampler-sync/v1",
        "identity": identity, "samplers": [{**identity, "rank": rank} for rank in range(3)]}))
    job = SimpleNamespace(kind="train.rl", spec={"params": {"template_id": recipe_id}},
        embodiment_env_id="ee_rpo_v2_bfm_23dof_v1")
    run = SimpleNamespace(id="run_four", metrics_summary={})
    await collect.claim_outputs(AsyncMock(), job, run)
    assert run.metrics_summary["model_version_id"] == "mv_four"
    promote.assert_awaited_once()
    (output / "ufo-sampler-sync.json").unlink()
    promote.reset_mock()
    with pytest.raises(ValueError, match="sync evidence missing"):
        await collect.claim_outputs(AsyncMock(), job, run)
    promote.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", ["suite_rpo_v2_bfm_isaac_velocity_validation_v2",
    "suite_rpo_ufo_isaac_velocity_validation_v1"])
async def test_native_eval_cases_are_visible_in_report(suite, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import eb.settings
    from eb.api.routes import get_report_episodes
    root = tmp_path / "runs/run_report/output"
    root.mkdir(parents=True)
    (root / "eval-cases.jsonl").write_text(json.dumps({"seed": 61000, "case_id": "rpo-bfm-seed-61000",
        "status": "task_failure"}) + "\n")
    monkeypatch.setattr(eb.settings, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(run_id="run_report",
        suite_id=suite, gate_results={})))
    result = await get_report_episodes("ev_report", session, None)
    assert result["available"] and result["items"][0]["seed"] == 61000
