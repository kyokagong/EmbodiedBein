"""A failed ebd acknowledgement must leave the Job eligible for cleanup retry."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from eb.scheduler.loop import Scheduler
from eb.store.ebd import EbdClient


def cancellation_client(monkeypatch, payload, status=200):
    def respond(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/runs/run_pending/cancel"
        return httpx.Response(status, json=payload)

    client = EbdClient()
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(client, "_client", lambda **_: httpx.AsyncClient(transport=transport))
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"ok": False, "error": "removal failed"}, {}, [], {"ok": 1}])
async def test_ebd_cancel_rejects_unconfirmed_success_body(monkeypatch, payload):
    client = cancellation_client(monkeypatch, payload)
    with pytest.raises((RuntimeError, ValueError), match="cancel|confirm"):
        await client.cancel_run("run_pending")


@pytest.mark.asyncio
async def test_ebd_cancel_accepts_explicit_confirmation(monkeypatch):
    client = cancellation_client(monkeypatch, {"ok": True, "run_id": "run_pending"})
    assert await client.cancel_run("run_pending") == {"ok": True, "run_id": "run_pending"}


def pending_records(*, expired=False):
    now = datetime.now(UTC)
    job = SimpleNamespace(id="job_pending", status="running", created_at=now - timedelta(days=90),
                          spec={"policy": {"timeout_h": 1}}, error="original diagnostic")
    run = SimpleNamespace(id="run_pending", status="running", finished_at=None, abort_reason=None,
                          started_at=now - timedelta(hours=2 if expired else .25))
    session = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [job])),
                              scalar=AsyncMock(return_value=run), refresh=AsyncMock(), commit=AsyncMock())
    return job, run, session


@pytest.mark.asyncio
@pytest.mark.parametrize("naive", [False, True])
async def test_expired_run_clock_cancels_and_records_timeout(monkeypatch, naive):
    job, run, session = pending_records(expired=True)
    if naive:
        run.started_at = run.started_at.replace(tzinfo=None)
    archive = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.collect.archive_partial_eval", archive)
    scheduler = Scheduler(SimpleNamespace(collect_and_finish=AsyncMock()))
    scheduler.ebd = cancellation_client(monkeypatch, {"ok": True})
    await scheduler.reap_running(session)
    assert job.status == "failed" and run.status == "aborted"
    assert run.abort_reason == "timeout" and run.finished_at is not None
    archive.assert_awaited_once_with(session, job, run)
    scheduler.orch.collect_and_finish.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_old_job_with_fresh_run_does_not_timeout():
    job, run, session = pending_records()
    scheduler = Scheduler(SimpleNamespace(collect_and_finish=AsyncMock()))
    scheduler.ebd = SimpleNamespace(get_run=AsyncMock(return_value={"status": "running"}),
                                    cancel_run=AsyncMock())
    await scheduler.reap_running(session)
    scheduler.ebd.cancel_run.assert_not_awaited()
    assert job.status == run.status == "running" and run.finished_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["timeout", "oom"])
@pytest.mark.parametrize("http_status", [200, 503])
async def test_failed_cleanup_preserves_running_state_and_evidence(
    monkeypatch, caplog, boundary, http_status
):
    job, run, session = pending_records(expired=boundary == "timeout")
    archive = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.collect.archive_partial_eval", archive)
    scheduler = Scheduler(SimpleNamespace(collect_and_finish=AsyncMock()))
    scheduler.ebd = cancellation_client(monkeypatch, {"ok": False, "error": "removal failed"}, http_status)
    monkeypatch.setattr(scheduler.ebd, "get_run", AsyncMock(
        return_value={"status": "running", "oom_killed": True}))
    await scheduler.reap_running(session)
    assert job.status == run.status == "running"
    assert run.finished_at is run.abort_reason is None
    assert job.error == "original diagnostic"
    assert "worker cleanup failed" in caplog.text
    archive.assert_not_awaited()
    scheduler.orch.collect_and_finish.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_unconfirmed_cancel_does_not_commit_terminal_state(monkeypatch):
    from fastapi import HTTPException

    from eb.api.routes import cancel_job

    job, run, session = pending_records()
    session.get = AsyncMock(return_value=job)
    archive = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.collect.archive_partial_eval", archive)
    client = cancellation_client(monkeypatch, {"ok": False, "error": "removal failed"})
    monkeypatch.setattr("eb.api.routes.EbdClient", lambda: client)
    monkeypatch.setattr("eb.api.routes._job_out", lambda row: {"status": row.status})
    with pytest.raises(HTTPException) as failure:
        await cancel_job(job.id, session, None)
    assert failure.value.status_code == 503
    assert job.status == run.status == "running" and run.finished_at is None
    assert job.error == "original diagnostic"
    archive.assert_not_awaited()
    session.commit.assert_not_awaited()
