from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eb.domain.contract import load_env
from eb.domain.rlinf_form import build_overrides, get_template, template_image
from eb.rpo_wbc import RPO_LAB_JOINT_ORDER


def test_flat_recipe_preserves_official_interface_and_online_rl():
    tpl = get_template("rpo_v2_flat_ppo")
    assert tpl.kind == "ppo" and not tpl.requires_dataset and not tpl.requires_base_model
    assert tpl.embodiment_env_id == "ee_rpo_v2_flat_23dof_v1"
    assert template_image(tpl) == "eb/rlinf:rpo-v2-wbc"
    contract = load_env(Path(__file__).resolve().parents[3] / "contracts/ee_rpo_v2_flat_23dof_v1.yaml")
    assert tuple(contract.joint_order) == RPO_LAB_JOINT_ORDER
    assert contract.observation.dim == 780 and contract.observation.critic_dim == 1390
    overrides = build_overrides(tpl.id, 1, "flat-smoke", [])
    assert "runner.max_epochs=500" in overrides
    assert "actor.optim.lr=1e-3" in overrides


def test_flat_four_gpu_batches_and_published_warm_start():
    from eb.rpo_flat import FLAT_ENV, FLAT_TEMPLATE, ROBOLAB_REVISION, flat_warm_start_receipt
    overrides = build_overrides(FLAT_TEMPLATE, 4, "scaled", ["env.train.total_num_envs=8192",
        "actor.global_batch_size=49152", "actor.micro_batch_size=3072"])
    assert "actor.global_batch_size=49152" in overrides
    assert overrides[-4].endswith("0-3}")
    with pytest.raises(ValueError):
        build_overrides(FLAT_TEMPLATE, 4, "bad", ["env.train.total_num_envs=8193"])
    from eb.capacity.estimator import estimate_job
    from eb.domain.job import JobSpec, JobKind
    estimate = estimate_job(JobSpec(kind=JobKind.TRAIN_RL, task_id="tsk_rpo_flat",
        embodiment_env_id=FLAT_ENV, params={"template_id": FLAT_TEMPLATE},
        overrides=["env.train.total_num_envs=8192"], resources={"gpus": 4}))
    assert estimate.verdict != "no_fit"
    card = {"schema_version": "eb/rpo-flat-model/v1", "template_id": FLAT_TEMPLATE,
            "network": {"hidden_dims": [512, 256, 128], "activation": "elu"},
            "normalization": "empirical", "official_source": {"robolab_revision": ROBOLAB_REVISION},
            "run_id": "run_source", "step": 3000, "artifact_sha256": "sha256:package"}
    manifest = {"status": "complete", "run_id": "run_source", "step": 3000,
                "template_id": FLAT_TEMPLATE, "files": [
                    {"path": "model/model_state_dict/full_weights.pt", "bytes": 10, "sha256": "a" * 64},
                    {"path": "model/actor-normalization.json", "bytes": 20, "sha256": "b" * 64}]}
    model = SimpleNamespace(id="mv_source", card=card, family_id="mf_rpo_v2_flat", embodiment_env_id=FLAT_ENV)
    artifact = SimpleNamespace(kind="rlinf_checkpoint", sha256="sha256:package", meta={"manifest": manifest})
    receipt = flat_warm_start_receipt(model, artifact)
    assert receipt["source_step"] == 3000 and receipt["state_coverage"]["optimizer"] == "reset"
    assert receipt["state_coverage"]["normalization"] == "restored"
    manifest["files"].pop()
    with pytest.raises(ValueError, match="归一化"):
        flat_warm_start_receipt(model, artifact)


@pytest.mark.asyncio
async def test_generic_submission_rechecks_published_flat_binding(monkeypatch, tmp_path):
    from eb.orchestrator.service import validate_flat_warm_start_submission
    from eb.domain.job import JobSpec, JobKind
    expected = {"model_id": "mv_source", "source_step": 3000}
    monkeypatch.setattr("eb.rpo_flat.flat_warm_start_receipt", lambda *_: expected)
    model = SimpleNamespace(artifact_id="art_source")
    artifact = SimpleNamespace(storage=str(tmp_path))
    session = SimpleNamespace(get=AsyncMock(side_effect=[model, artifact] * 3))
    spec = JobSpec(kind=JobKind.TRAIN_RL, task_id="tsk_rpo_flat", embodiment_env_id="ee_rpo_v2_flat_23dof_v1",
                   params={"template_id": "rpo_v2_flat_ppo", "warm_start": dict(expected)},
                   inputs={"base_model": str(tmp_path)})
    await validate_flat_warm_start_submission(session, spec)
    spec.params["warm_start"]["source_step"] = 5000
    with pytest.raises(ValueError, match="已发布模型"):
        await validate_flat_warm_start_submission(session, spec)
    spec.params["warm_start"] = dict(expected)
    spec.inputs.base_model = str(tmp_path / "other")
    with pytest.raises(ValueError, match="已发布模型"):
        await validate_flat_warm_start_submission(session, spec)
