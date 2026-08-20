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
