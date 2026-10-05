import pytest
from types import SimpleNamespace

from eb.domain.fsm import JobStatus, RunStatus
from eb.orchestrator.service import apply_ebd_result


def test_missing_image_is_infra_failed():
    job = SimpleNamespace(status=JobStatus.PREPARING.value, error=None)
    run = SimpleNamespace(
        status=RunStatus.CREATED.value,
        container_id=None,
        abort_reason=None,
        metrics_summary={"Train/mean_reward": 12.4},
    )
    apply_ebd_result(
        job,
        run,
        {
            "container_id": None,
            "status": "image_missing",
            "reason": "image not present: eb/isaac-lab:2.3.2",
        },
    )
    assert job.status == JobStatus.INFRA_FAILED.value
    assert run.status == "aborted"
    assert run.abort_reason == "image_missing"
    assert "mean_reward" not in (run.metrics_summary or {})
    assert job.error == "image not present: eb/isaac-lab:2.3.2"


def test_started_marks_running():
    job = SimpleNamespace(status=JobStatus.PREPARING.value, error=None)
    run = SimpleNamespace(
        status=RunStatus.CREATED.value,
        container_id=None,
        abort_reason=None,
        metrics_summary={},
    )
    apply_ebd_result(job, run, {"container_id": "abc123def", "status": "started"})
    assert job.status == JobStatus.RUNNING.value
    assert run.status == RunStatus.RUNNING.value
    assert run.container_id == "abc123def"


def test_docker_error_is_infra_failed_without_metrics():
    job = SimpleNamespace(status=JobStatus.PREPARING.value, error=None)
    run = SimpleNamespace(
        status=RunStatus.CREATED.value,
        container_id=None,
        abort_reason=None,
        metrics_summary={"Train/mean_reward": 1.0},
    )
    apply_ebd_result(job, run, {"status": "docker_error", "reason": "daemon down"})
    assert job.status == JobStatus.INFRA_FAILED.value
    assert run.status == RunStatus.ABORTED.value
    assert run.abort_reason == "docker_error"
    assert run.metrics_summary == {}
    assert job.error == "daemon down"


@pytest.mark.asyncio
async def test_invalid_output_receipt_terminates_job(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from eb.orchestrator.service import Orchestrator
    monkeypatch.setattr('eb.orchestrator.collect.claim_outputs', AsyncMock(side_effect=ValueError('missing receipt')))
    session = SimpleNamespace(flush=AsyncMock(),commit=AsyncMock())
    job = SimpleNamespace(status='running')
    run = SimpleNamespace(status='running')
    await Orchestrator().collect_and_finish(session, job, run)
    assert job.status == 'failed' and run.status == 'aborted'
    assert run.abort_reason == 'invalid_output'
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_validated_eval_receipt_wins_over_optional_dependency_traceback(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from eb.orchestrator.service import Orchestrator

    async def claim(_session, _job, run):
        run.metrics_summary = {'eval_report_id':'ev_ok', 'validity':'complete'}

    monkeypatch.setattr('eb.orchestrator.collect.claim_outputs', claim)
    monkeypatch.setattr('eb.orchestrator.service.get_settings',
                        lambda: SimpleNamespace(data_root=tmp_path))
    log = tmp_path / 'runs/run_ok/console.log'
    log.parent.mkdir(parents=True)
    log.write_text('optional backend\nTraceback (most recent call last):\nModuleNotFoundError: curobo\n')
    session = SimpleNamespace(flush=AsyncMock(), commit=AsyncMock())
    job = SimpleNamespace(status='running')
    run = SimpleNamespace(id='run_ok', status='running', metrics_summary={})

    await Orchestrator().collect_and_finish(session, job, run)

    assert job.status == 'succeeded' and run.status == 'finished'


def test_mount_validation_rejects_external_writes_and_root_reads():
    from eb.orchestrator.service import validate_mounts
    for mounts in [[('/etc', '/data', 'ro')], [('/data/kk/RLinf', '/code', 'rw')]]:
        with pytest.raises(ValueError):
            validate_mounts(mounts)


def test_mount_validation_allows_each_configured_managed_root(tmp_path, monkeypatch):
    from eb.orchestrator.service import validate_mounts
    from eb.settings import get_settings

    roots = [tmp_path / name for name in ("state", "models", "datasets")]
    for root in roots:
        root.mkdir()
    monkeypatch.setenv("EB_DATA_ROOT", str(roots[0]))
    monkeypatch.setenv("EB_MODELS_ROOT", str(roots[1]))
    monkeypatch.setenv("EB_DATASETS_ROOT", str(roots[2]))
    monkeypatch.setenv("EB_ALLOWED_PATH_PREFIXES", "/data/kk/RLinf")
    get_settings.cache_clear()
    try:
        validate_mounts([(str(root / "output"), f"/workspace/{root.name}", "rw") for root in roots])
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_rlinf_resume_skips_rsl_copy_and_records_run_lineage(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, Mock
    from eb.adapters.base import RenderedJob
    from eb.domain.job import JobSpec
    from eb.orchestrator.service import Orchestrator
    from eb.store.models import LineageEdge

    spec = JobSpec(kind="train.sft", embodiment_env_id="ee_libero_franka_delta_v1",
                   task_id="tsk_libero_spatial", params={"template_id":"libero_pi05_sft","stack":"rlinf"},
                   inputs={"resume_from":"run_source"}, resources={"gpus":1})
    job = SimpleNamespace(id="job_child", spec=spec.model_dump(mode="json"), kind="train.sft",
                          embodiment_env_id=spec.embodiment_env_id, status="queued", attempts=0, version=0)
    session = SimpleNamespace(add=Mock(), commit=AsyncMock(), refresh=AsyncMock())
    ebd = SimpleNamespace(submit_run=AsyncMock(return_value={"status":"started","container_id":"abc"}))
    rendered = RenderedJob("image", ["python"], {}, [], mounts=[])
    monkeypatch.setattr("eb.orchestrator.service.prepare_run_dirs", lambda run_id: tmp_path / run_id)
    monkeypatch.setattr("eb.orchestrator.service.get_adapter", lambda spec: SimpleNamespace(render=lambda *_: rendered))
    copy = Mock(side_effect=AssertionError("RLInf resume must not use RSL copy"))
    monkeypatch.setattr("eb.orchestrator.service.seed_rsl_resume", copy)

    run = await Orchestrator(ebd).admit_and_start(session, job, [0])

    copy.assert_not_called()
    edges = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], LineageEdge)]
    assert len(edges) == 1
    assert edges[0].src_id == "run_source" and edges[0].dst_id == run.id
    assert edges[0].relation == "resumed_from"
