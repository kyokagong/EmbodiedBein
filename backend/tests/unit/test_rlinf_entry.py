import json
import os
from types import SimpleNamespace as NS

import pytest

from eb.runners import rlinf_entry
from eb.runners.rlinf_entry import (
    externalize_compressed_eval_plan,
    prepare_eval_model,
    prune_old_checkpoints,
    publish_checkpoint,
    validate_rlinf_image_receipt,
)


def test_compressed_eval_plan_is_externalized_before_child_processes(tmp_path, monkeypatch):
    import base64
    import zlib

    plan = {"schema_version":"eb/robotwin-eval-plan/v1", "cases":[
        {"case_id":f"case-{index}", "instruction":"x" * 200} for index in range(5000)]}
    packed = base64.b64encode(zlib.compress(json.dumps(plan).encode(), 9)).decode()
    monkeypatch.setenv("EB_EVAL_PLAN_ZLIB", packed)

    assert externalize_compressed_eval_plan(tmp_path) == plan
    assert "EB_EVAL_PLAN_ZLIB" not in os.environ
    assert "EB_EVAL_PLAN" not in os.environ
    assert os.environ["EB_EVAL_PLAN_PATH"] == str(tmp_path / "eval-plan.json")
    assert json.loads((tmp_path / "eval-plan.json").read_text()) == plan
    os.environ.pop("EB_EVAL_PLAN_PATH")


def test_robotwin_image_receipt_must_match_frozen_source_and_patch(tmp_path, monkeypatch):
    receipt = tmp_path / ".eb-source-receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": "eb/rlinf-image-source/v1",
        "rlinf_content_hash": "sha256:" + "a" * 64,
        "patch_sha256": "sha256:" + "b" * 64,
    }))
    monkeypatch.setenv("EB_RLINF_CONTENT_HASH", "sha256:" + "a" * 64)
    monkeypatch.setenv("EB_RLINF_PATCH_SHA256", "sha256:" + "b" * 64)

    assert validate_rlinf_image_receipt(receipt)["schema_version"] == "eb/rlinf-image-source/v1"
    monkeypatch.setenv("EB_RLINF_PATCH_SHA256", "sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="patch"):
        validate_rlinf_image_receipt(receipt)


def test_prepare_eval_model_accepts_native_evaluation_config_without_actor(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "omegaconf", NS(OmegaConf=NS(
        merge=lambda left, right: right,
        update=lambda cfg, path, value, **kwargs: setattr(cfg.rollout, "model", value))))
    cfg = NS(rollout=NS(model=NS(model_path="/model", action_dim=14)))
    prepare_eval_model(cfg)
    assert cfg.rollout.model.model_path == "/model"


def test_checkpoint_retention_keeps_only_latest_global_step(tmp_path):
    root = tmp_path / "checkpoints"
    for name in ("global_step_2", "global_step_10"):
        checkpoint = root / name
        metadata = checkpoint / "actor/dcp_checkpoint/.metadata"
        shard = checkpoint / "actor/dcp_checkpoint/__0_0.distcp"
        weights = checkpoint / "actor/model_state_dict/full_weights.pt"
        for path in (metadata, shard, weights):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        monkey = pytest.MonkeyPatch()
        monkey.setenv("EB_RUN_ID", "run_test")
        monkey.setenv("EB_RESUME_FINGERPRINT", "sha256:" + "a" * 64)
        rlinf_entry.write_resume_manifest(checkpoint, "trainer_state")
        monkey.undo()
    for name in ("global_step_12", "global_step_bad", "best_model"):
        (root / name).mkdir(parents=True)
        (root / name / "marker").write_text(name)

    prune_old_checkpoints(root)

    assert {path.name for path in root.iterdir()} == {
        "global_step_10", "global_step_bad", "best_model",
    }


def test_amp_checkpoint_retention_keeps_four_latest_complete_steps(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    monkeypatch.setenv("EB_RUN_ID", "run_amp")
    monkeypatch.setenv("EB_RESUME_FINGERPRINT", "sha256:" + "a" * 64)
    for step in (25, 50, 75, 100, 125):
        checkpoint = root / f"global_step_{step}"
        for name in ("actor/dcp_checkpoint/.metadata", "actor/dcp_checkpoint/__0_0.distcp",
                     "actor/model_state_dict/full_weights.pt"):
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        rlinf_entry.write_resume_manifest(checkpoint, "trainer_state")

    prune_old_checkpoints(root, keep=4)
    assert {path.name for path in root.iterdir()} == {
        "global_step_50", "global_step_75", "global_step_100", "global_step_125",
    }


def test_resume_manifest_is_atomic_and_requires_exact_sft_state(tmp_path, monkeypatch):
    checkpoint = tmp_path / "global_step_42"
    for relative in (
        "actor/dcp_checkpoint/.metadata",
        "actor/dcp_checkpoint/__0_0.distcp",
        "actor/model_state_dict/full_weights.pt",
        "data.pt",
        "rng.pt",
    ):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    monkeypatch.setenv("EB_RUN_ID", "run_source")
    monkeypatch.setenv("EB_RESUME_FINGERPRINT", "sha256:" + "b" * 64)

    manifest = rlinf_entry.write_resume_manifest(checkpoint, "exact")

    saved = json.loads((checkpoint / "resume-manifest.json").read_text())
    assert manifest == saved
    assert saved["schema_version"] == "eb/resume-checkpoint/v1"
    assert saved["run_id"] == "run_source" and saved["step"] == 42
    assert saved["state_coverage"]["dataloader"] == "restored"
    assert {item["path"] for item in saved["files"]} >= {
        "actor/dcp_checkpoint/.metadata", "actor/dcp_checkpoint/__0_0.distcp",
        "actor/model_state_dict/full_weights.pt", "data.pt", "rng.pt",
    }
    assert all(item["bytes"] > 0 for item in saved["files"])
    assert not (checkpoint / "resume-manifest.json.tmp").exists()

    (checkpoint / "data.pt").unlink()
    with pytest.raises(ValueError, match="data.pt"):
        rlinf_entry.write_resume_manifest(checkpoint, "exact")


def test_resume_receipt_must_match_resolved_config_and_marks_loaded(tmp_path, monkeypatch):
    checkpoint = tmp_path / "global_step_42"
    checkpoint.mkdir()
    receipt = {"schema_version":"eb/resume-request/v1", "status":"validated",
               "source_run_id":"run_source", "step":42, "mode":"exact",
               "checkpoint":str(checkpoint), "compatibility_fingerprint":"sha256:" + "a" * 64}
    monkeypatch.setenv("EB_RESUME_RECEIPT", json.dumps(receipt))
    cfg = NS(runner=NS(resume_dir=str(checkpoint)))

    saved = rlinf_entry.prepare_resume_receipt(tmp_path / "output", cfg)
    assert saved["status"] == "validated"
    rlinf_entry.mark_resume_loaded(tmp_path / "output", str(checkpoint))
    assert json.loads((tmp_path / "output/resume-receipt.json").read_text())["status"] == "loaded"

    cfg.runner.resume_dir = str(tmp_path / "other")
    with pytest.raises(ValueError, match="does not match"):
        rlinf_entry.prepare_resume_receipt(tmp_path / "bad-output", cfg)


def test_only_complete_checkpoint_publishes_hashed_parent_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("EB_RUN_ID", "run_test")
    parent = tmp_path / "parent"
    (parent / "physical-intelligence/libero").mkdir(parents=True)
    (parent / "physical-intelligence/libero/norm_stats.json").write_text('{"actions": {}}')
    (parent / "model.safetensors").write_bytes(b"old weights")
    output = tmp_path / "output"
    output.mkdir()
    cfg = NS(runner=NS(logger=NS(experiment_name="exp")), actor=NS(model=NS(model_path=str(parent), openpi=NS(config_name="pi05_libero"))))
    with pytest.raises(RuntimeError, match="without a complete"):
        publish_checkpoint(output, cfg, {})
    assert not (output / "checkpoint-manifest.json").exists()
    for step in [2, 10]:
        checkpoint = output / f"exp/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"weights{step}".encode())
    (output / "model").mkdir()  # Docker creates the empty publication bind mount.
    publish_checkpoint(output, cfg, {"template_id": "sft"})
    manifest = json.loads((output / "checkpoint-manifest.json").read_text())
    assert manifest["step"] == 10
    assert manifest["run_id"] == "run_test"
    assert manifest["status"] == "complete"
    assert len(manifest["files"]) == 2
    assert all(len(file["sha256"]) == 64 for file in manifest["files"])
    assert (output / "model/model_state_dict/full_weights.pt").read_bytes() == b"weights10"
    assert not (output / "model/model.safetensors").exists()


def test_rpo_mlp_checkpoint_publishes_without_parent_model(tmp_path, monkeypatch):
    monkeypatch.setenv("EB_RUN_ID", "run_rpo")
    output = tmp_path / "output"
    weights = output / "rpo/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"rpo weights")
    (output / "model").mkdir()
    cfg = NS(runner=NS(logger=NS(experiment_name="rpo")),
             actor=NS(model=NS(model_path="", model_type="mlp_policy")))

    publish_checkpoint(output, cfg, {"template_id":"rpo_v2_wbc_ppo"})

    manifest = json.loads((output / "checkpoint-manifest.json").read_text())
    assert manifest["config_name"] == "mlp_policy"
    assert (output / "model/model_state_dict/full_weights.pt").read_bytes() == b"rpo weights"
    normalization = json.loads((output / "model/actor-normalization.json").read_text())
    assert normalization["observation_dim"] == 890


def test_amp_checkpoint_requires_discriminator_at_resume_root(tmp_path, monkeypatch):
    monkeypatch.setenv("EB_RUN_ID", "run_amp")
    monkeypatch.setenv("EB_RESUME_FINGERPRINT", "sha256:" + "a" * 64)
    monkeypatch.setenv("EB_INPUT_RECEIPT", json.dumps({"template_id":"rpo_v2_amp_stand_walk_ppo"}))
    output = tmp_path / "output"
    checkpoint = output / "amp-probe/checkpoints/global_step_2"
    for relative in ("actor/dcp_checkpoint/.metadata", "actor/dcp_checkpoint/__0_0.distcp",
                     "actor/model_state_dict/full_weights.pt", "actor/amp/state.pt"):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    cfg = NS(runner=NS(logger=NS(experiment_name="amp-probe")),
             actor=NS(model=NS(model_path="", model_type="mlp_policy")))

    with pytest.raises(ValueError, match="state.pt"):
        rlinf_entry.write_resume_manifest(checkpoint, "trainer_state")
    with pytest.raises(ValueError, match="AMP checkpoint"):
        publish_checkpoint(output, cfg, {"template_id":"rpo_v2_amp_stand_walk_ppo"})

    target = checkpoint / "amp/state.pt"
    target.parent.mkdir()
    (checkpoint / "actor/amp/state.pt").rename(target)
    manifest = rlinf_entry.write_resume_manifest(checkpoint, "trainer_state")
    assert "amp/state.pt" in {item["path"] for item in manifest["files"]}
    publish_checkpoint(output, cfg, {"template_id":"rpo_v2_amp_stand_walk_ppo"})
    assert (output / "model/model_state_dict/full_weights.pt").is_file()


def test_bfm_checkpoint_requires_replay_state_and_publishes_its_normalizer(tmp_path, monkeypatch):
    monkeypatch.setenv("EB_RUN_ID", "run_bfm")
    monkeypatch.setenv("EB_RESUME_FINGERPRINT", "sha256:" + "a" * 64)
    monkeypatch.setenv("EB_INPUT_RECEIPT", json.dumps({"template_id":"rpo_v2_bfm_zero_pretrain"}))
    output = tmp_path / "output"
    checkpoint = output / "bfm/checkpoints/global_step_1"
    for relative in ("actor/dcp_checkpoint/.metadata", "actor/dcp_checkpoint/__0_0.distcp",
                     "actor/model_state_dict/full_weights.pt"):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    cfg = NS(runner=NS(logger=NS(experiment_name="bfm")),
             actor=NS(model=NS(model_path="", model_type="mlp_policy")))

    with pytest.raises(ValueError, match="state.pt"):
        rlinf_entry.write_resume_manifest(checkpoint, "trainer_state")
    (checkpoint / "bfm").mkdir()
    (checkpoint / "bfm/state.pt").write_bytes(b"full BFM trainer state")
    manifest = rlinf_entry.write_resume_manifest(checkpoint, "trainer_state")
    assert "bfm/state.pt" in {item["path"] for item in manifest["files"]}
    progress = checkpoint / "bfm/progress.json"
    progress.write_text(json.dumps(dict(schema_version="eb/bfm-learner-progress/v1", runner_round=1,
        learner_updates=2, updates_per_round=1, z_pool_count=8)))
    with pytest.raises(ValueError, match="learner"):
        publish_checkpoint(output, cfg, {"template_id":"rpo_v2_bfm_zero_pretrain"})
    progress.write_text(json.dumps(dict(schema_version="eb/bfm-learner-progress/v1", runner_round=1,
        learner_updates=1, updates_per_round=1, z_pool_count=8)))
    publish_checkpoint(output, cfg, {"template_id":"rpo_v2_bfm_zero_pretrain"})
    assert json.loads((output / "checkpoint-manifest.json").read_text())["bfm_progress"]["learner_updates"] == 1
    normalization = json.loads((output / "model/actor-normalization.json").read_text())
    assert normalization == {"schema_version":"eb/rpo-bfm-normalization/v1",
                             "kind":"frozen_running_stats", "observation_dim":733,
                             "history_length":4, "latent_dim":256}


@pytest.mark.asyncio
async def test_bfm_checkpoint_is_promoted_to_its_own_unqualified_model_family(tmp_path):
    from eb.orchestrator.collect import promote_rpo_model

    class Session:
        def __init__(self):
            self.rows = []

        async def scalar(self, _query):
            return None

        async def get(self, _model, _key):
            return None

        def add(self, row):
            self.rows.append(row)

        async def flush(self):
            return None

    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "actor-normalization.json").write_text(json.dumps(
        {"schema_version":"eb/rpo-bfm-normalization/v1", "kind":"frozen_running_stats",
         "observation_dim":733, "history_length":4, "latent_dim":256}))
    job = NS(id="job_bfm", kind="train.rl", spec={
        "params":{"template_id":"rpo_v2_bfm_zero_pretrain", "model_name":"bfm-smoke"},
        "inputs":{"datasets":[{"id":"ds_motion"}]}, "image":"eb/rlinf:rpo-v2-bfm-zero",
        "resources":{"gpus":1}, "overrides":[]})
    run = NS(id="run_bfm")
    checkpoint = {"manifest":{"template_id":"rpo_v2_bfm_zero_pretrain",
                               "files":[{"path":"model/actor-normalization.json"}]},
                  "path":str(checkpoint_root), "bytes":123, "step":1}

    session = Session()
    await promote_rpo_model(session, job, run, checkpoint)
    model = next(row for row in session.rows if row.__class__.__name__ == "ModelVersion")
    assert model.family_id == "mf_rpo_v2_bfm_zero"
    assert model.embodiment_env_id == "ee_rpo_v2_bfm_23dof_v1"
    assert model.io_signature == {"observation_profile":"rpo_bfm_actor_history4_z256_v1",
                                  "obs":375, "latent":256, "critic_obs":733,
                                  "action_profile":"rpo_v2_joint23_position_v1", "act":23}
    assert model.card["readiness"] == "simulation_unverified"
    assert model.card["motion_prior"]["id"] == "ds_motion"


def test_rpo_mlp_entry_reaches_upstream_training_without_openpi_assets(tmp_path, monkeypatch):
    import contextlib
    import sys

    class Config(dict):
        __getattr__ = dict.__getitem__

    class ReachedTraining(Exception):
        pass

    cfg = NS(
        actor=NS(model=NS(model_path="", model_type="mlp_policy")),
        runner=Config(task_type="embodied", only_eval=False, resume_dir=None),
    )
    monkeypatch.setitem(sys.modules, "hydra", NS(
        compose=lambda **kwargs: cfg,
        initialize_config_dir=lambda **kwargs: contextlib.nullcontext(),
    ))
    monkeypatch.setitem(sys.modules, "omegaconf", NS(OmegaConf=NS(
        resolve=lambda value: None,
        to_yaml=lambda *args, **kwargs: "resolved",
    )))
    upstream = NS(__wrapped__=lambda value: (_ for _ in ()).throw(ReachedTraining()))
    monkeypatch.setattr(rlinf_entry.runpy, "run_path", lambda *args, **kwargs: {"main": upstream})
    monkeypatch.setenv("EB_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("EB_RUN_ID", "run_rpo")
    monkeypatch.setenv("EB_INPUT_RECEIPT", json.dumps({"template_id": "rpo_v2_wbc_ppo"}))
    monkeypatch.setattr(sys, "argv", ["entry", "--script", "examples/embodiment/train_embodied_agent.py",
                                      "--config-path", str(tmp_path), "--config-name", "rpo"])

    with pytest.raises(ReachedTraining):
        rlinf_entry.main()


def test_eval_dispatch_never_executes_training_or_publishes_model(tmp_path, monkeypatch):
    import contextlib
    import sys

    from eb.runners import rlinf_entry

    class Config(dict):
        __getattr__ = dict.__getitem__

    parent = tmp_path / "parent"
    (parent / "physical-intelligence/libero").mkdir(parents=True)
    (parent / "physical-intelligence/libero/norm_stats.json").write_text("{}")
    (parent / "model_state_dict").mkdir()
    (parent / "model_state_dict/full_weights.pt").write_bytes(b"test")
    cfg = NS(actor=NS(model=NS(model_path=str(parent), openpi=NS(config_name="pi05_libero"))),
             runner=Config(only_eval=True, task_type="embodied_eval"))
    monkeypatch.setitem(sys.modules, "hydra", NS(compose=lambda **kwargs: cfg, initialize_config_dir=lambda **kwargs: contextlib.nullcontext()))
    monkeypatch.setitem(sys.modules, "omegaconf", NS(OmegaConf=NS(resolve=lambda cfg: None, to_yaml=lambda *args, **kwargs: "resolved")))
    monkeypatch.setitem(sys.modules, "libero_eval", NS(evaluate=lambda cfg: {"success_rate": 0.5}))
    monkeypatch.setattr(rlinf_entry, "prepare_eval_model", lambda cfg: None)
    monkeypatch.setattr(rlinf_entry.runpy, "run_path", lambda *args, **kwargs: pytest.fail("training entry executed"))
    monkeypatch.setenv("EB_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("EB_RUN_ID", "eval_test")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    (tmp_path / "big_vision").mkdir()
    (tmp_path / "big_vision/paligemma_tokenizer.model").write_bytes(b"tokenizer")
    monkeypatch.setattr(sys, "argv", ["entry", "--script", "evaluations/eval_embodied_agent.py",
                                      "--config-path", str(tmp_path), "--config-name", "eval"])
    rlinf_entry.main()
    manifest = json.loads((tmp_path / "output/eval-manifest.json").read_text())
    assert manifest["metrics"] == {"success_rate": 0.5}
    assert manifest["status"] == "not_qualified"
    assert not (tmp_path / "output/checkpoint-manifest.json").exists()


def test_robotwin_eval_dispatch_uses_evaluation_runner_and_plan(tmp_path, monkeypatch):
    import contextlib
    import sys

    from eb.runners import rlinf_entry

    class Config(dict):
        __getattr__ = dict.__getitem__

    parent = tmp_path / "parent"
    (parent / "physical-intelligence/robotwin").mkdir(parents=True)
    (parent / "physical-intelligence/robotwin/norm_stats.json").write_text("{}")
    (parent / "model_state_dict").mkdir()
    (parent / "model_state_dict/full_weights.pt").write_bytes(b"test")
    cfg = NS(rollout=NS(model=NS(model_path=str(parent), openpi=NS(config_name="pi05_aloha_robotwin"))),
             runner=Config(only_eval=True, task_type="embodied_eval"),
             env=NS(eval=NS(task_config=NS(task_name="adjust_bottle"))))
    monkeypatch.setitem(sys.modules, "hydra", NS(compose=lambda **kwargs: cfg, initialize_config_dir=lambda **kwargs: contextlib.nullcontext()))
    monkeypatch.setitem(sys.modules, "omegaconf", NS(OmegaConf=NS(resolve=lambda cfg: None, to_yaml=lambda *args, **kwargs: "resolved")))
    monkeypatch.setitem(sys.modules, "robotwin_eval", NS(evaluate=lambda cfg: {"success_once":0.6, "num_trajectories":5}))
    monkeypatch.setattr(rlinf_entry.runpy, "run_path", lambda *args, **kwargs: pytest.fail("training entry executed"))
    monkeypatch.setattr(rlinf_entry, "validate_rlinf_image_receipt", lambda: {})
    monkeypatch.setenv("EB_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("EB_RUN_ID", "eval_robotwin")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    plan = {"schema_version":"eb/robotwin-eval-plan/v1", "hash":"sha256:" + "a" * 64,
            "suite_spec":{"robotwin_revision":"abc"}, "planned":5, "cases":[]}
    monkeypatch.setenv("EB_EVAL_PLAN", json.dumps(plan))
    monkeypatch.setenv("EB_INPUT_RECEIPT", json.dumps({"template_id":"robotwin_pi05_eval"}))
    (tmp_path / "big_vision").mkdir()
    (tmp_path / "big_vision/paligemma_tokenizer.model").write_bytes(b"tokenizer")
    monkeypatch.setattr(sys, "argv", ["entry", "--script", "evaluations/eval_embodied_agent.py",
        "--config-path", str(tmp_path), "--config-name", "eval"])

    rlinf_entry.main()

    manifest = json.loads((tmp_path / "output/eval-manifest.json").read_text())
    assert manifest["metrics"]["success_once"] == 0.6
    assert manifest["plan_hash"] == plan["hash"]
    assert not (tmp_path / "output/checkpoint-manifest.json").exists()


@pytest.mark.asyncio
async def test_successful_process_without_model_package_is_not_complete(tmp_path, monkeypatch):
    from eb.orchestrator.collect import claim_outputs
    monkeypatch.setattr('eb.orchestrator.collect.get_settings', lambda:NS(data_root=tmp_path))
    job = NS(kind='train.sft',spec={'params':{'template_id':'libero_pi05_sft'}})
    with pytest.raises(ValueError,match='checkpoint manifest'):
        await claim_outputs(NS(),job,NS(id='run_missing',metrics_summary={}))
def test_entrypoint_rejects_runner_mode_mismatch():
    from types import SimpleNamespace

    from eb.runners.rlinf_entry import validate_entrypoint_mode

    validate_entrypoint_mode('examples/sft/train_vla_sft.py',
                             SimpleNamespace(runner={'task_type':'sft'}))
    validate_entrypoint_mode('examples/embodiment/train_embodied_agent.py',
                             SimpleNamespace(runner={'task_type':'embodied', 'only_eval':False}))
    validate_entrypoint_mode('evaluations/eval_embodied_agent.py',
                             SimpleNamespace(runner={'task_type':'embodied_eval', 'only_eval':True}))
    with pytest.raises(ValueError, match='entrypoint|入口'):
        validate_entrypoint_mode('examples/embodiment/train_embodied_agent.py',
                                 SimpleNamespace(runner={'task_type':'embodied_eval', 'only_eval':True}))
