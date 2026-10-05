import hashlib

from eb.domain.fsm import JobStatus
from eb.pipelines import pipeline_next


def test_empty_pipeline_succeeds():
    assert pipeline_next([], [], None) == ("succeeded", None)


def test_first_step_submits():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, [], None) == ("submit", 0)


def test_waits_while_running():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a"], JobStatus.RUNNING.value) == ("wait", None)


def test_second_step_after_success():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a"], JobStatus.SUCCEEDED.value) == ("submit", 1)


def test_fails_when_previous_failed():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a"], JobStatus.FAILED.value) == ("failed", None)
    assert pipeline_next(steps, ["job_a"], JobStatus.INFRA_FAILED.value) == ("failed", None)


def test_all_steps_done():
    steps = [{"kind": "data.scan"}, {"kind": "data.slice"}]
    assert pipeline_next(steps, ["job_a", "job_b"], JobStatus.SUCCEEDED.value) == ("succeeded", None)


import json
from types import SimpleNamespace

import pytest

from eb.pipelines import advance_pipeline, checkpoint_directory, start_pipeline, step_spec
from eb.store.models import Pipeline, PipelineRun


class MemorySession:
    def __init__(self):
        self.rows = {}
        self.jobs = {}

    def add(self, row):
        self.rows[row.id] = row

    async def scalar(self, query):
        values = list(query.compile().params.values())
        for row in [*self.rows.values(), *self.jobs.values()]:
            if getattr(row, "idem_key", None) in values:
                return row
        return None

    async def get(self, cls, key):
        return self.jobs.get(key) or self.rows.get(key)

    async def commit(self):
        pass

    async def refresh(self, row):
        pass


class FakeOrchestrator:
    def __init__(self, crash=False):
        self.calls = 0
        self.crash = crash

    async def submit(self, session, spec, principal, key):
        self.calls += 1
        job = SimpleNamespace(id=f"job_{self.calls}", idem_key=key, status="queued", error=None)
        session.jobs[job.id] = job
        if self.crash:
            self.crash = False
            raise RuntimeError("process lost after Job commit")
        return job


def scan_step():
    return {"kind": "data.scan", "embodiment_env_id": "ee_test"}


@pytest.mark.asyncio
async def test_repeated_start_freezes_template_and_start_idempotency():
    session = MemorySession()
    template = Pipeline(id="p", name="template", spec={"steps": [scan_step()]})
    one = await start_pipeline(session, template, "user:test", "click-1")
    assert await start_pipeline(session, template, "user:test", "click-1") is one
    two = await start_pipeline(session, template, "user:test", "click-2")
    assert one.id != two.id
    one.spec["steps"][0]["name"] = "changed"
    assert "name" not in template.spec["steps"][0]
    assert "name" not in two.spec["steps"][0]
    assert "status" not in template.spec


@pytest.mark.asyncio
async def test_duplicate_advance_recovers_job_commit_gap_without_resubmitting():
    session, orch = MemorySession(), FakeOrchestrator(crash=True)
    run = PipelineRun(id="pr_test", pipeline_id="p", spec={"steps": [scan_step()]},
                      status="running", job_ids=[], created_by="user:test")
    with pytest.raises(RuntimeError, match="Job commit"):
        await advance_pipeline(session, run, orch)
    assert not run.job_ids
    await advance_pipeline(session, run, orch)
    await advance_pipeline(session, run, orch)
    assert orch.calls == 1
    assert run.job_ids == ["job_1"]
    session.jobs["job_1"].status = "succeeded"
    await advance_pipeline(session, run, orch)
    assert run.status == "succeeded"


def test_typed_binding_rejects_future_step_and_wrong_output():
    for binding in ({"from_step": 1, "output": "model_directory"},
                    {"from_step": 0, "output": "log_file"}):
        with pytest.raises(ValueError, match="model_directory"):
            step_spec({"job": scan_step(), "bindings": {"inputs.base_model": binding}}, 1)


def test_checkpoint_binding_requires_complete_matching_producer_and_contained_weights(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        checkpoint_directory(tmp_path, "run_test")
    output = tmp_path / "output"
    weights = output / "model/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture")
    manifest = output / "checkpoint-manifest.json"
    receipt = {"schema_version": "eb/checkpoint/v1", "status": "complete", "run_id": "other",
               "model_path": "model", "files": [{"path": "model/model_state_dict/full_weights.pt",
                                                "bytes": 7, "sha256": "a" * 64}]}
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="producer"):
        checkpoint_directory(tmp_path, "run_test")
    receipt["run_id"] = "run_test"
    manifest.write_text(json.dumps(receipt))
    assert checkpoint_directory(tmp_path, "run_test") == str(output / "model")
    weights.unlink()
    with pytest.raises(ValueError, match="missing"):
        checkpoint_directory(tmp_path, "run_test")


def test_checkpoint_receipt_rejects_traversal_symlinks_and_changed_sizes(tmp_path):
    from eb.checkpoints import checkpoint_model
    output = tmp_path / "output"
    weights = output / "model/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    receipt = {"schema_version": "eb/checkpoint/v1", "status": "complete", "run_id": "run_safe",
               "model_path": "model", "files": [{"path": "model/model_state_dict/full_weights.pt",
                                                "bytes": 7, "sha256": "a" * 64}]}
    manifest = output / "checkpoint-manifest.json"
    manifest.write_text(json.dumps(receipt))
    assert checkpoint_model(tmp_path, "run_safe")["bytes"] == 7
    with pytest.raises(ValueError, match="digest"):
        checkpoint_model(tmp_path, "run_safe", verify_hashes=True)
    receipt["files"][0]["sha256"] = hashlib.sha256(b"weights").hexdigest()
    manifest.write_text(json.dumps(receipt))
    assert checkpoint_model(tmp_path, "run_safe", verify_hashes=True)["bytes"] == 7
    receipt["model_path"] = "../output/model"
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="escapes"):
        checkpoint_model(tmp_path, "run_safe")
    receipt["model_path"] = "model"
    manifest.write_text(json.dumps(receipt))
    weights.write_bytes(b"changed size")
    with pytest.raises(ValueError, match="size"):
        checkpoint_model(tmp_path, "run_safe")
    weights.unlink()
    weights.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlinks"):
        checkpoint_model(tmp_path, "run_safe")


def test_checkpoint_can_publish_model_under_configured_model_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eb.checkpoints import checkpoint_model
    run_root = tmp_path / "state/runs/run_external"
    output = run_root / "output"
    model = tmp_path / "models/runs/run_external"
    weights = model / "model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    output.mkdir(parents=True)
    (output / "checkpoint-manifest.json").write_text(json.dumps({
        "schema_version":"eb/checkpoint/v1", "status":"complete", "run_id":"run_external",
        "model_path":"model", "files":[{"path":"model/model_state_dict/full_weights.pt",
        "bytes":7, "sha256":"a" * 64}]}))
    monkeypatch.setattr("eb.checkpoints.get_settings", lambda:SimpleNamespace(models_root=tmp_path / "models"))
    assert checkpoint_model(run_root, "run_external")["path"] == str(model)


@pytest.mark.asyncio
async def test_resume_reuses_successful_sft_and_frozen_spec_without_mutating_old_run(monkeypatch):
    from copy import deepcopy
    from unittest.mock import AsyncMock
    import eb.pipelines as pipelines

    session = MemorySession()
    sft = {"kind": "train.sft", "embodiment_env_id": "ee_test", "task_id": "task"}
    ppo = {"job": {"kind": "train.rlft", "embodiment_env_id": "ee_test", "task_id": "task"},
           "bindings": {"inputs.base_model": {"from_step": 0, "output": "model_directory"}}}
    frozen = {"steps": [sft, ppo, scan_step()]}
    previous = PipelineRun(id="pr_old", pipeline_id="p", spec=deepcopy(frozen), status="failed",
                           job_ids=["job_sft", "job_failed"], created_by="user:test", error="snapshot missing")
    template = Pipeline(id="p", name="changed template", spec={"steps": [scan_step()]})
    session.jobs.update(job_sft=SimpleNamespace(id="job_sft", status="succeeded", error=None),
                        job_failed=SimpleNamespace(id="job_failed", status="failed", error="snapshot missing"))
    checkpoint = AsyncMock(return_value="/managed/sft/model")
    monkeypatch.setattr(pipelines, "bound_checkpoint", checkpoint)
    resumed = await start_pipeline(session, template, "user:test", "resume-click", resume_from=previous)
    duplicate = await start_pipeline(session, template, "user:test", "resume-click", resume_from=previous)
    assert duplicate is resumed and len(session.rows) == 1
    assert resumed.job_ids == ["job_sft"] and resumed.spec["resumed_from"] == "pr_old"
    assert resumed.spec["steps"] == frozen["steps"]
    checkpoint.assert_awaited_once_with(session, "job_sft")
    orch = FakeOrchestrator()
    await advance_pipeline(session, resumed, orch)
    await advance_pipeline(session, resumed, orch)
    assert orch.calls == 1 and resumed.job_ids == ["job_sft", "job_1"]
    assert previous.status == "failed" and previous.job_ids == ["job_sft", "job_failed"]
    assert previous.spec == frozen and previous.error == "snapshot missing"
    assert session.jobs["job_sft"].status == "succeeded" and session.jobs["job_failed"].status == "failed"
    session.jobs["job_1"].status = "succeeded"
    await advance_pipeline(session, resumed, orch)
    assert orch.calls == 2 and resumed.job_ids == ["job_sft", "job_1", "job_2"]
    resumed.spec["steps"][0]["name"] = "independent snapshot"
    assert "name" not in previous.spec["steps"][0]


@pytest.mark.asyncio
async def test_resume_rejects_missing_reused_checkpoint_before_creating_run(monkeypatch):
    from unittest.mock import AsyncMock
    import eb.pipelines as pipelines

    session = MemorySession()
    step = {"kind": "train.sft", "embodiment_env_id": "ee_test", "task_id": "task"}
    previous = PipelineRun(id="pr_old", pipeline_id="p", spec={"steps": [step, scan_step()]},
                           status="failed", job_ids=["job_sft"], created_by="user:test")
    template = Pipeline(id="p", name="template", spec={})
    session.jobs["job_sft"] = SimpleNamespace(id="job_sft", status="succeeded")
    monkeypatch.setattr(pipelines, "bound_checkpoint", AsyncMock(side_effect=ValueError("checkpoint missing")))
    with pytest.raises(ValueError, match="checkpoint missing"):
        await start_pipeline(session, template, "user:test", "resume", resume_from=previous)
    assert not session.rows
    previous.status = "running"
    with pytest.raises(ValueError, match="failed run"):
        await start_pipeline(session, template, "user:test", "resume", resume_from=previous)


def test_checkpoint_receipt_rejects_symlinked_output_root(tmp_path):
    from eb.checkpoints import checkpoint_model
    outside = tmp_path / "other-run/output"
    weights = outside / "model/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    receipt = {"schema_version": "eb/checkpoint/v1", "status": "complete", "run_id": "run_alias",
               "model_path": "model", "files": [{"path": "model/model_state_dict/full_weights.pt",
                                                "bytes": 7, "sha256": "a" * 64}]}
    (outside / "checkpoint-manifest.json").write_text(json.dumps(receipt))
    root = tmp_path / "run_alias"
    root.mkdir()
    (root / "output").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint manifest"):
        checkpoint_model(root, "run_alias")


@pytest.mark.asyncio
async def test_resume_adjusts_only_first_remaining_memory_and_preserves_idempotency(monkeypatch):
    from copy import deepcopy
    from unittest.mock import AsyncMock
    import eb.pipelines as pipelines

    session = MemorySession()
    steps = [scan_step(), {"job": {**scan_step(), "resources": {"gpus": 4, "memory_gb": 96, "shm_gb": 32}}},
             {**scan_step(), "resources": {"memory_gb": 64, "shm_gb": 16}}]
    previous = PipelineRun(id="pr_oom", pipeline_id="p", spec={"steps": deepcopy(steps)}, status="failed",
                           job_ids=["job_ok", "job_oom"], created_by="user:test")
    session.jobs.update(job_ok=SimpleNamespace(id="job_ok", status="succeeded"),
                        job_oom=SimpleNamespace(id="job_oom", status="failed"))
    template = Pipeline(id="p", name="template", spec={"steps": []})
    monkeypatch.setattr(pipelines, "bound_checkpoint", AsyncMock())
    resumed = await start_pipeline(session, template, "user:test", "memory-fix", resume_from=previous, memory_gb=72)
    duplicate = await start_pipeline(session, template, "user:test", "memory-fix", resume_from=previous, memory_gb=72)
    assert duplicate is resumed and len(session.rows) == 1
    assert resumed.job_ids == ["job_ok"]
    assert resumed.spec["steps"][1]["job"]["resources"] == {"gpus": 4, "memory_gb": 72, "shm_gb": 32}
    assert resumed.spec["steps"][0] == steps[0] and resumed.spec["steps"][2] == steps[2]
    assert previous.spec == {"steps": steps} and previous.job_ids == ["job_ok", "job_oom"]
    assert previous.status == "failed" and session.jobs["job_oom"].status == "failed"
    assert resumed.spec["resource_adjustment"] == {"step_index": 1, "source_run_id": "pr_oom",
                                                    "memory_gb": {"before": 96, "after": 72}}
    unchanged = await start_pipeline(session, template, "user:test", "no-fix", resume_from=previous)
    assert unchanged.spec["steps"] == steps and "resource_adjustment" not in unchanged.spec
    for invalid in (0, -1, 1025, True, 1.5, 16):
        with pytest.raises(ValueError, match="memory_gb"):
            await start_pipeline(session, template, "user:test", f"bad-{invalid}",
                                 resume_from=previous, memory_gb=invalid)
    assert len(session.rows) == 2
