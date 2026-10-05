from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from eb.scheduler.loop import Scheduler, _classify_container_failure


def test_graphics_buffer_failure_has_actionable_message(tmp_path, monkeypatch):
    log = tmp_path / "runs/run_eval/console.log"
    log.parent.mkdir(parents=True)
    log.write_text("self.left_camera.take_picture()\nRuntimeError: cannot create buffer\n")
    monkeypatch.setattr("eb.scheduler.loop.get_settings",
                        lambda: SimpleNamespace(data_root=tmp_path))
    reason, message = _classify_container_failure("run_eval", "exit_255")
    assert reason == "graphics_buffer_exhausted"
    assert "降低每卡并发环境数" in message


@pytest.mark.asyncio
async def test_timeout_uses_run_clock_and_success_is_collected():
    job = SimpleNamespace(id='job_1', spec={'policy':{'timeout_h':24}}, status='running')
    run = SimpleNamespace(id='run_1', started_at=datetime.now(UTC))
    session = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda:[job])), scalar=AsyncMock(return_value=run), refresh=AsyncMock(), commit=AsyncMock())
    orch = SimpleNamespace(collect_and_finish=AsyncMock())
    scheduler = Scheduler(orch)
    scheduler.ebd = SimpleNamespace(get_run=AsyncMock(return_value={'status':'exited','exit_code':0}))
    await scheduler.reap_running(session)
    orch.collect_and_finish.assert_awaited_once_with(session,job,run)


@pytest.mark.asyncio
async def test_failed_run_records_terminal_time():
    job = SimpleNamespace(id='job_1', spec={}, status='running')
    run = SimpleNamespace(id='run_1', started_at=datetime.now(UTC), finished_at=None)
    session = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda:[job])),
        scalar=AsyncMock(return_value=run),commit=AsyncMock(), refresh=AsyncMock())
    scheduler = Scheduler(SimpleNamespace())
    scheduler.ebd = SimpleNamespace(get_run=AsyncMock(return_value={'status':'exited','exit_code':255}))
    await scheduler.reap_running(session)
    assert run.finished_at is not None
    assert job.status == 'failed'


@pytest.mark.asyncio
async def test_oom_killed_child_cancels_remaining_workers():
    job = SimpleNamespace(id='job_1', spec={}, status='running')
    run = SimpleNamespace(id='run_1', started_at=datetime.now(UTC), finished_at=None)
    session = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda:[job])),
        scalar=AsyncMock(return_value=run), commit=AsyncMock(), refresh=AsyncMock())
    scheduler = Scheduler(SimpleNamespace())
    scheduler.ebd = SimpleNamespace(get_run=AsyncMock(return_value={'status':'running','oom_killed':True}),
        cancel_run=AsyncMock())
    await scheduler.reap_running(session)
    scheduler.ebd.cancel_run.assert_awaited_once_with('run_1')
    assert run.finished_at is not None and run.abort_reason == 'oom'
    assert job.status == 'failed'


@pytest.mark.asyncio
async def test_cancel_failure_and_terminal_jobs_preserve_records(monkeypatch):
    from fastapi import HTTPException
    from eb.api.routes import cancel_job
    monkeypatch.setattr('eb.api.routes._job_out', lambda job: {'status': job.status})
    job = SimpleNamespace(id='job_1', status='running')
    run = SimpleNamespace(id='run_1', status='running')
    session = SimpleNamespace(get=AsyncMock(return_value=job), scalar=AsyncMock(return_value=run), commit=AsyncMock())
    monkeypatch.setattr('eb.api.routes.EbdClient', lambda: SimpleNamespace(cancel_run=AsyncMock(side_effect=OSError('unavailable'))))
    with pytest.raises(HTTPException) as exc:
        await cancel_job('job_1', session, None)
    assert exc.value.status_code == 503
    assert job.status == run.status == 'running'
    session.commit.assert_not_awaited()
    job.status = 'succeeded'
    with pytest.raises(HTTPException) as exc:
        await cancel_job('job_1', session, None)
    assert exc.value.status_code == 409 and job.status == 'succeeded'


@pytest.mark.asyncio
async def test_admission_rechecks_cancelled_state_under_row_lock():
    from unittest.mock import Mock
    from eb.orchestrator.service import Orchestrator
    job = SimpleNamespace(id="job_cancelled", status="queued")
    async def refresh(row, **kwargs):
        assert kwargs == {"with_for_update": True}
        row.status = "cancelled"  # A cancellation committed after the scheduler's queued query.
    session = SimpleNamespace(refresh=AsyncMock(side_effect=refresh), add=Mock(), commit=AsyncMock())
    ebd = SimpleNamespace(submit_run=AsyncMock())
    with pytest.raises(ValueError, match="no longer queued"):
        await Orchestrator(ebd).admit_and_start(session, job, [])
    session.add.assert_not_called()
    ebd.submit_run.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_waits_for_admission_commit_then_cancels_new_run(monkeypatch, tmp_path):
    import asyncio
    from eb.api.routes import cancel_job
    from eb.adapters.base import RenderedJob
    from eb.domain.job import JobSpec
    from eb.orchestrator.service import Orchestrator
    from eb.store.models import Job

    row_lock = asyncio.Lock()
    docker_entered, release_docker = asyncio.Event(), asyncio.Event()
    spec = JobSpec(kind="data.convert", embodiment_env_id="env")
    job = SimpleNamespace(id="job_race", status="queued", attempts=0, version=0,
                          kind="data.convert", embodiment_env_id="env", spec=spec.model_dump(mode="json"))
    runs = []
    async def refresh(row, **kwargs):
        if kwargs.get("with_for_update"):
            await row_lock.acquire()
    async def commit():
        if row_lock.locked():
            row_lock.release()
    async def submit(payload):
        docker_entered.set()
        await release_docker.wait()
        return {"status": "started", "container_id": "container_race"}
    async def locked_get(cls, jid, **kwargs):
        assert cls is Job and kwargs == {"with_for_update": True, "populate_existing": True}
        await row_lock.acquire()
        return job
    admission_session = SimpleNamespace(refresh=AsyncMock(side_effect=refresh), add=runs.append,
                                        commit=AsyncMock(side_effect=commit))
    cancel_session = SimpleNamespace(get=AsyncMock(side_effect=locked_get),
                                    scalar=AsyncMock(side_effect=lambda *_: runs[-1] if runs else None),
                                    commit=AsyncMock(side_effect=commit))
    cancel = AsyncMock()
    monkeypatch.setattr("eb.api.routes.EbdClient", lambda: SimpleNamespace(cancel_run=cancel))
    monkeypatch.setattr("eb.api.routes._job_out", lambda value: {"status": value.status})
    monkeypatch.setattr("eb.orchestrator.service.prepare_run_dirs", lambda _: tmp_path)
    monkeypatch.setattr("eb.orchestrator.service.get_adapter", lambda _: SimpleNamespace(
        render=lambda *_: RenderedJob("test", ["test"], {}, [], mounts=[])))
    admission = asyncio.create_task(Orchestrator(SimpleNamespace(submit_run=submit)).admit_and_start(admission_session, job, []))
    await docker_entered.wait()
    cancellation = asyncio.create_task(cancel_job(job.id, cancel_session, None))
    await asyncio.sleep(0)
    cancel_session.scalar.assert_not_awaited()
    cancel.assert_not_awaited()
    release_docker.set()
    run = await admission
    result = await cancellation
    cancel.assert_awaited_once_with(run.id)
    assert result["status"] == "cancelled" and run.status == "aborted"
    assert run.abort_reason == "cancelled"


@pytest.mark.asyncio
async def test_reap_does_not_overwrite_cancellation_committed_after_running_query():
    job = SimpleNamespace(id="job_cancelled", status="running")
    async def refresh(row, **kwargs):
        assert kwargs == {"with_for_update": True}
        row.status = "cancelled"
    session = SimpleNamespace(
        scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [job])),
        scalar=AsyncMock(), refresh=AsyncMock(side_effect=refresh), commit=AsyncMock())
    scheduler = Scheduler(SimpleNamespace(collect_and_finish=AsyncMock()))
    scheduler.ebd = SimpleNamespace(get_run=AsyncMock(), cancel_run=AsyncMock())
    await scheduler.reap_running(session)
    assert job.status == "cancelled"
    session.scalar.assert_not_awaited()
    scheduler.ebd.get_run.assert_not_awaited()
    scheduler.orch.collect_and_finish.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup_error", [False, True])
async def test_reap_releases_row_lock_when_no_terminal_update(lookup_error):
    job = SimpleNamespace(id="job_active", status="running", spec={})
    run = SimpleNamespace(id="run_active")
    session = SimpleNamespace(
        scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [job])),
        scalar=AsyncMock(return_value=run), refresh=AsyncMock(), commit=AsyncMock())
    scheduler = Scheduler(SimpleNamespace(collect_and_finish=AsyncMock()))
    scheduler.ebd = SimpleNamespace(get_run=AsyncMock(
        return_value={"status": "running"}, side_effect=OSError("unavailable") if lookup_error else None))
    await scheduler.reap_running(session)
    session.refresh.assert_awaited_once_with(job, with_for_update=True)
    session.commit.assert_awaited_once()
    assert job.status == "running"
