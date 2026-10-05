import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from eb.domain.job import JobKind, JobSpec


def make_checkpoint(root, run_id, step, *, mode="exact", fingerprint=None):
    checkpoint = root / "output" / "exp" / "checkpoints" / f"global_step_{step}"
    files = []
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
        files.append({"path": relative, "bytes": path.stat().st_size})
    manifest = {
        "schema_version": "eb/resume-checkpoint/v1",
        "status": "complete",
        "run_id": run_id,
        "step": step,
        "mode": mode,
        "state_coverage": {
            "model": "restored", "optimizer": "restored", "scheduler": "restored",
            "rng_per_rank": "restored", "dataloader": "restored",
            "environment": "not_applicable", "rollout": "not_applicable",
        },
        "compatibility_fingerprint": fingerprint or "sha256:" + "a" * 64,
        "files": files,
    }
    (checkpoint / "resume-manifest.json").write_text(json.dumps(manifest))
    return checkpoint


def source_spec(kind=JobKind.TRAIN_SFT, template="libero_pi05_sft", limit="runner.max_steps=100"):
    return JobSpec(
        kind=kind,
        embodiment_env_id="ee_libero_franka_delta_v1",
        task_id="tsk_libero_spatial",
        name="source",
        image="eb/rlinf:libero-pi05@sha256:test",
        params={"template_id": template, "config_name": "libero", "stack": "rlinf"},
        overrides=[limit, "actor.micro_batch_size=2"],
        inputs={"datasets": [{"id": "ds_1", "path": "/data/kk/data/ds_1"}], "base_model": "/data/kk/models/base"},
        resources={"gpus": 4, "memory_gb": 72, "shm_gb": 32},
        code_refs={
            "embodiment": {"kind": "content_snapshot", "path": "/data/kk/embodiedbein/snapshots/emb", "commit": "sha256:emb"},
            "stack": {"kind": "content_snapshot", "path": "/data/kk/embodiedbein/snapshots/rlinf", "commit": "sha256:rlinf"},
        },
    )


def test_resume_checkpoint_rejects_partial_and_selects_highest_complete(tmp_path):
    from eb.checkpoints import latest_resume_checkpoint, resume_checkpoint

    run_a = tmp_path / "runs/run_a"
    run_b = tmp_path / "runs/run_b"
    make_checkpoint(run_a, "run_a", 40)
    winner = make_checkpoint(run_b, "run_b", 80)
    partial = run_b / "output/exp/checkpoints/global_step_90"
    (partial / "actor/dcp_checkpoint").mkdir(parents=True)
    (partial / "actor/dcp_checkpoint/.metadata").write_text("partial")

    assert resume_checkpoint(run_b, "run_b")["path"] == str(winner.resolve())
    selected = latest_resume_checkpoint([("run_a", run_a), ("run_b", run_b)])
    assert selected["run_id"] == "run_b" and selected["step"] == 80
    assert latest_resume_checkpoint([("run_a", run_a), ("run_b", run_b)], step=40)["run_id"] == "run_a"
    with pytest.raises(ValueError, match="step 50"):
        latest_resume_checkpoint([("run_a", run_a), ("run_b", run_b)], step=50)

    (winner / "actor/dcp_checkpoint/__0_0.distcp").unlink()
    with pytest.raises(ValueError, match="member"):
        resume_checkpoint(run_b, "run_b")


def test_resume_checkpoint_rejects_symlink_member(tmp_path):
    from eb.checkpoints import resume_checkpoint

    run = tmp_path / "run"
    checkpoint = make_checkpoint(run, "run_link", 10)
    member = checkpoint / "data.pt"
    member.unlink()
    member.symlink_to(checkpoint / "rng.pt")
    with pytest.raises(ValueError, match="symlink"):
        resume_checkpoint(run, "run_link")


def test_resume_checkpoint_rejects_non_object_manifest(tmp_path):
    from eb.checkpoints import resume_checkpoint

    checkpoint = tmp_path / "run_bad" / "output" / "exp" / "checkpoints" / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "resume-manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="manifest is invalid"):
        resume_checkpoint(tmp_path / "run_bad", "run_bad")


def test_resume_fingerprint_ignores_only_resume_metadata_and_termination():
    from eb.checkpoints import resume_fingerprint

    source = source_spec()
    resumed = source.model_copy(deep=True)
    resumed.name = "continued"
    resumed.inputs.resume_from = "run_source"
    resumed.params.update({"resume_mode": "exact", "resume_step": 80, "source_train_job_id": "trn_source"})
    resumed.overrides[0] = "runner.max_steps=200"
    assert resume_fingerprint(source) == resume_fingerprint(resumed)

    resumed.resources.gpus = 2
    assert resume_fingerprint(source) != resume_fingerprint(resumed)


def test_build_resume_spec_extends_sft_without_mutating_source():
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint

    source = source_spec()
    checkpoint = {"run_id": "run_source", "step": 80, "mode": "exact",
                  "compatibility_fingerprint": resume_fingerprint(source),
                  "image_digest": "sha256:resolved-image"}
    result = build_resume_spec(source, "trn_source", checkpoint,
                               ResumeTrainIn(name="continued", max_steps=200))
    assert source.name == "source" and source.inputs.resume_from is None
    assert result.name == "continued" and result.inputs.resume_from == "run_source"
    assert result.image == "sha256:resolved-image"
    assert result.params["resume_step"] == 80 and result.params["resume_mode"] == "exact"
    assert "runner.max_steps=200" in result.overrides
    assert resume_fingerprint(result) == checkpoint["compatibility_fingerprint"]

    with pytest.raises(ValueError, match="大于"):
        build_resume_spec(source, "trn_source", checkpoint,
                          ResumeTrainIn(name="short", max_steps=80))


def test_build_resume_spec_extends_ppo_epochs_only():
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint

    source = source_spec(JobKind.TRAIN_RLFT, "libero_spatial_pi05_ppo", "runner.max_epochs=1")
    checkpoint = {"run_id": "run_ppo", "step": 1, "mode": "trainer_state",
                  "compatibility_fingerprint": resume_fingerprint(source)}
    result = build_resume_spec(source, "trn_ppo", checkpoint,
                               ResumeTrainIn(name="ppo continued", max_epochs=2))
    assert "runner.max_epochs=2" in result.overrides
    assert result.inputs.resume_from == "run_ppo"
    with pytest.raises(ValueError, match="max_epochs"):
        build_resume_spec(source, "trn_ppo", checkpoint,
                          ResumeTrainIn(name="bad", max_steps=100))


def test_cancelled_ppo_can_finalize_after_saved_checkpoint():
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint

    source = source_spec(JobKind.TRAIN_RL, "rpo_v2_amp_stand_walk_ppo", "runner.max_epochs=200")
    checkpoint = {"run_id":"run_ppo", "step":50, "mode":"trainer_state",
                  "compatibility_fingerprint":resume_fingerprint(source)}
    result = build_resume_spec(source, "trn_ppo", checkpoint,
                               ResumeTrainIn(name="finalize", max_epochs=51))
    assert "runner.max_epochs=51" in result.overrides


def test_chained_ppo_resume_preserves_original_fingerprint_image():
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.checkpoints import resume_fingerprint

    source = source_spec(JobKind.TRAIN_RL, "rpo_v2_amp_stand_walk_ppo", "runner.max_epochs=100")
    fingerprint = resume_fingerprint(source)
    first = build_resume_spec(source, "trn_first", {
        "run_id": "run_first", "step": 100, "mode": "trainer_state",
        "image_digest": "sha256:" + "b" * 64,
        "compatibility_fingerprint": fingerprint,
    }, ResumeTrainIn(name="continued", max_epochs=200))
    second = build_resume_spec(first, "trn_second", {
        "run_id": "run_second", "step": 200, "mode": "trainer_state",
        "image_digest": "sha256:" + "b" * 64,
        "compatibility_fingerprint": fingerprint,
    }, ResumeTrainIn(name="finalized", max_epochs=201, checkpoint_step=200))
    assert resume_fingerprint(second) == fingerprint
    assert second.params["resume_source_image"] == source.image


def test_scaled_flat_run_can_resume_without_reloading_initial_weights(tmp_path, monkeypatch):
    from eb.api.routes_train import ResumeTrainIn, build_resume_spec
    from eb.adapters.rlinf import RlinfAdapter
    from eb.checkpoints import resume_fingerprint
    from eb.rpo_flat import ROBOLAB_REVISION
    assets = tmp_path / "assets/roboparty/rpo-v2"
    for relative in ("robolab/robolab/tasks/direct/base/base_env.py", "rpo_description/urdf/rpo.urdf",
                     "rpo_description/mjcf/rpo.xml"):
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
    (assets / "asset-manifest.json").write_text(json.dumps({"status": "verified",
        "revisions": {"robolab": ROBOLAB_REVISION}}))
    settings = NS(data_root=tmp_path, models_root=tmp_path / "models", assets_root=tmp_path / "assets",
                  rlinf_repo=tmp_path / "rlinf")
    monkeypatch.setattr("eb.adapters.rlinf.get_settings", lambda: settings)
    source = JobSpec(kind=JobKind.TRAIN_RL, embodiment_env_id="ee_rpo_v2_flat_23dof_v1",
        task_id="tsk_rpo_flat", name="scaled", params={"template_id": "rpo_v2_flat_ppo",
        "warm_start": {"model_id": "mv_source", "source_step": 3000}},
        overrides=["runner.max_epochs=2000", "env.train.total_num_envs=8192",
                   "actor.global_batch_size=49152", "actor.micro_batch_size=3072"],
        inputs={"base_model": str(tmp_path / "original_model")}, resources={"gpus": 4})
    fingerprint = resume_fingerprint(source)
    weight = tmp_path / "original_model/model_state_dict/full_weights.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")
    initial = RlinfAdapter().render(source, "run_initial")
    assert json.loads(initial.env["EB_WARM_START_RECEIPT"]) == source.params["warm_start"]
    assert "runner.ckpt_path=/workspace/models/base/model_state_dict/full_weights.pt" in initial.command
    directory = make_checkpoint(tmp_path / "runs/run_flat", "run_flat", 80, mode="trainer_state", fingerprint=fingerprint)
    manifest = json.loads((directory / "resume-manifest.json").read_text())
    from eb.checkpoints import require_flat_rank_rng
    with pytest.raises(ValueError, match="独立随机状态"):
        require_flat_rank_rng(source, manifest)
    for rank in range(4):
        relative = f"actor/rng/rank_{rank}.pt"
        path = directory / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"rng")
        manifest["files"].append({"path": relative, "bytes": 3})
    (directory / "resume-manifest.json").write_text(json.dumps(manifest))
    result = build_resume_spec(source, "trn_flat", {"run_id": "run_flat", "step": 80,
        "mode": "trainer_state", "compatibility_fingerprint": fingerprint},
        ResumeTrainIn(name="resumed", max_epochs=2000))
    rendered = RlinfAdapter().render(result, "run_resumed")
    assert resume_fingerprint(result) == fingerprint
    assert "runner.resume_dir=/workspace/resume/global_step_80" in rendered.command
    assert not any(item.startswith("runner.ckpt_path=") for item in rendered.command)
    assert "EB_WARM_START_RECEIPT" not in rendered.env
    assert "EB_RESUME_RECEIPT" in rendered.env


@pytest.mark.parametrize("status", ["queued", "preparing", "running", "preempted"])
def test_resume_requires_terminal_source(status):
    from eb.api.routes_train import require_resume_terminal
    with pytest.raises(ValueError, match="终态"):
        require_resume_terminal(NS(status=status))


@pytest.mark.parametrize("status", ["succeeded", "failed", "timeout", "cancelled", "infra_failed"])
def test_resume_accepts_terminal_source(status):
    from eb.api.routes_train import require_resume_terminal
    require_resume_terminal(NS(status=status))


@pytest.mark.parametrize("status", ["created", "running", "restarting", "paused"])
def test_resume_rejects_active_source_container(status):
    from eb.api.routes_train import require_inactive_source_run
    with pytest.raises(ValueError, match="容器"):
        require_inactive_source_run({"status": status})


@pytest.mark.parametrize("status", ["exited", "dead", "unknown"])
def test_resume_accepts_inactive_source_container(status):
    from eb.api.routes_train import require_inactive_source_run
    require_inactive_source_run({"status": status})


@pytest.mark.asyncio
async def test_resume_candidate_hides_host_path_and_reports_sft_target(monkeypatch):
    from eb.api.routes_train import get_resume_candidate

    source = source_spec()
    checkpoint = {"path": "/data/kk/embodiedbein/runs/run_source/output/secret", "run_id": "run_source",
                  "step": 80, "mode": "exact", "bytes": 123,
                  "state_coverage": {"dataloader": "restored"}}
    monkeypatch.setattr("eb.api.routes_train._resume_context",
                        AsyncMock(return_value=(NS(id="trn_source", status="failed"), NS(id="job_source"), source, checkpoint)))
    result = await get_resume_candidate("trn_source", NS(), NS(id="user:kk"))
    assert result["available"] is True
    assert result["source_run_id"] == "run_source" and result["step"] == 80
    assert result["termination"] == "max_steps"
    assert result["suggested_target"] > 80
    assert "path" not in result and "secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_submit_resume_creates_derived_train_job_and_lineage(monkeypatch):
    from eb.api.routes_train import ResumeTrainIn, submit_resume, orch
    from eb.checkpoints import resume_fingerprint
    from eb.store.models import LineageEdge, TrainJob

    source = source_spec()
    checkpoint = {"run_id": "run_source", "step": 80, "mode": "exact",
                  "compatibility_fingerprint": resume_fingerprint(source)}
    source_row = NS(id="trn_source", status="failed", embodiment_id="emb_libero_franka",
                    dataset_ids=["ds_1"], base_model_id="mdl_1")
    monkeypatch.setattr("eb.api.routes_train._resume_context",
                        AsyncMock(return_value=(source_row, NS(id="job_source"), source, checkpoint)))
    rendered = NS(image="eb/rlinf:libero-pi05")
    monkeypatch.setattr("eb.adapters.get_adapter", lambda spec: NS(render=lambda spec, run_id: rendered))
    child_job = NS(id="job_child", status="queued")
    submit = AsyncMock(return_value=child_job)
    monkeypatch.setattr(orch, "submit", submit)

    class Session:
        def __init__(self): self.added = []
        async def scalar(self, query): return None
        def add(self, value): self.added.append(value)
        async def commit(self): pass
        async def refresh(self, value): pass
    session = Session()
    result = await submit_resume("trn_source", ResumeTrainIn(name="continued", max_steps=200),
                                 session, NS(id="user:kk"), "resume-click")
    train = next(value for value in session.added if isinstance(value, TrainJob))
    edge = next(value for value in session.added if isinstance(value, LineageEdge))
    assert result["id"] == train.id and train.job_id == "job_child"
    assert train.params["source_train_job_id"] == "trn_source"
    assert edge.src_id == "trn_source" and edge.dst_id == train.id and edge.relation == "derived_from"
    submitted_spec = submit.await_args.args[1]
    assert submitted_spec.inputs.resume_from == "run_source"
    assert submit.await_args.kwargs["idem_key"].startswith("train-resume:user:kk:")


@pytest.mark.asyncio
async def test_submit_resume_rejects_same_idempotency_key_with_other_request(monkeypatch):
    from eb.api.routes_train import ResumeTrainIn, submit_resume
    from eb.checkpoints import resume_fingerprint

    source = source_spec()
    checkpoint = {"run_id": "run_source", "step": 80, "mode": "exact",
                  "compatibility_fingerprint": resume_fingerprint(source)}
    monkeypatch.setattr("eb.api.routes_train._resume_context",
                        AsyncMock(return_value=(NS(id="trn_source", status="failed"), NS(id="job_source"), source, checkpoint)))
    existing_spec = source.model_copy(deep=True)
    existing_spec.name = "different"
    existing = NS(id="job_existing", spec=existing_spec.model_dump(mode="json"))
    session = NS(scalar=AsyncMock(return_value=existing))
    with pytest.raises(HTTPException) as error:
        await submit_resume("trn_source", ResumeTrainIn(name="continued", max_steps=200),
                            session, NS(id="user:kk"), "same-key")
    assert error.value.status_code == 409
