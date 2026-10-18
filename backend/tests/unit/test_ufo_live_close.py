from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from test_ufo_live import live_plan

from eb.api.routes_ufo_live import router
from eb.store.db import get_session
from eb.store.models import Job

JID = "job_0123456789abcdef0123"
RID = "run_0123456789abcdef0123"
PREFIX = "/api/v1/ufo-native/live/sessions"


@pytest.fixture
def closing_app(monkeypatch):
    job = SimpleNamespace(id=JID, kind="serve.policy", status="running", created_by="user:kk",
                          spec={"params": {"operation": "ufo_native_live", "native_plan": live_plan()}},
                          error=None, priority=0, embodiment_env_id="ee_native")
    run = SimpleNamespace(id=RID, status="running", metrics_summary={}, abort_reason=None)
    records = []
    key = SimpleNamespace(principal="user:kk", actor_type="user")
    pending = {"job": None, "marker": None}
    async def scalar(query):
        if "api_keys" in str(query):
            return key
        if "audit_log" in str(query):
            return pending["marker"]
        if "FROM jobs" in str(query):
            return pending["job"]
        return run
    session = SimpleNamespace(
        get=AsyncMock(side_effect=lambda cls, ident, **kw: job if cls is Job and ident == JID else None),
        scalar=AsyncMock(side_effect=scalar),
        scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [job])),
        add=records.append, commit=AsyncMock(), rollback=AsyncMock(), execute=AsyncMock())
    cancel = AsyncMock()
    monkeypatch.setattr("eb.api.routes.EbdClient", lambda: SimpleNamespace(cancel_run=cancel))
    partial = AsyncMock()
    monkeypatch.setattr("eb.orchestrator.collect.archive_partial_eval", partial)
    async def db():
        yield session
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="close-tests")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_session] = db
    with TestClient(app, headers={"Authorization": "Bearer test"}) as client:
        yield SimpleNamespace(client=client, job=job, run=run, session=session, records=records,
                              cancel=cancel, partial=partial, key=key, pending=pending)


@pytest.mark.parametrize("phase", ["running", "queued", "initializing", "finalizing"])
def test_close_uses_container_cancellation_without_requiring_live_state_or_socket(closing_app, phase):
    s = closing_app
    if phase == "queued":
        s.job.status = "queued"
        s.session.scalar.side_effect = lambda query: s.key if "api_keys" in str(query) else None
    response = s.client.post(f"{PREFIX}/{JID}/close", headers={"Origin": "http://testserver"})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled" and s.job.status == "cancelled"
    if phase != "queued":
        s.cancel.assert_awaited_once_with(RID)
        s.partial.assert_awaited_once()
        assert s.run.status == "aborted" and s.run.abort_reason == "cancelled"
    else:
        s.cancel.assert_not_awaited()
    assert any(call.kwargs.get("with_for_update") for call in s.session.get.await_args_list)
    assert s.session.commit.await_count == 1
    assert s.records[0].action == "ufo_live.close"


def test_close_is_idempotent_after_cancel_or_success_and_preserves_outputs(closing_app):
    s = closing_app
    s.run.metrics_summary = {"files": [{"path": "events.jsonl", "url": "/f"}], "complete": False}
    first = s.client.post(f"{PREFIX}/{JID}/close")
    second = s.client.post(f"{PREFIX}/{JID}/close")
    assert first.status_code == second.status_code == 200
    assert second.json()["files"][0]["path"] == "events.jsonl"
    s.cancel.assert_awaited_once_with(RID)
    s.job.status = "succeeded"
    s.run.metrics_summary["complete"] = True
    assert s.client.post(f"{PREFIX}/{JID}/close").json()["status"] == "succeeded"
    assert s.run.metrics_summary["complete"] is True


@pytest.mark.parametrize("case,expected", [("agent", 403), ("foreign", 404), ("other_operation", 404),
                                          ("cross_origin", 403), ("missing", 404)])
def test_close_never_cancels_another_users_or_unrelated_job(closing_app, case, expected):
    s = closing_app
    headers = {}
    if case == "agent":
        s.key.actor_type = "agent"
    elif case == "foreign":
        s.job.created_by = "user:else"
    elif case == "other_operation":
        s.job.spec["params"]["operation"] = "training"
    elif case == "cross_origin":
        headers["Origin"] = "http://evil.test"
    response = s.client.post(f"{PREFIX}/{'job_missing' if case == 'missing' else JID}/close", headers=headers)
    assert response.status_code == expected
    s.cancel.assert_not_awaited()


def test_failed_removal_is_visible_and_keeps_job_active_for_retry(closing_app):
    s = closing_app
    s.cancel.side_effect = httpx.ConnectError("ebd unavailable")
    response = s.client.post(f"{PREFIX}/{JID}/close")
    assert response.status_code == 503
    assert s.job.status == "running" and s.run.status == "running"
    s.session.commit.assert_not_awaited()


def test_closing_does_not_require_a_valid_model_plan(closing_app):
    s = closing_app
    s.job.spec["params"]["native_plan"] = {}
    assert s.client.post(f"{PREFIX}/{JID}/close").status_code == 200
    s.cancel.assert_awaited_once_with(RID)


def test_partial_evidence_failure_does_not_undo_confirmed_resource_release(closing_app):
    s = closing_app
    s.partial.side_effect = ValueError("invalid live plan or partial evidence")
    response = s.client.post(f"{PREFIX}/{JID}/close")
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert response.json()["metrics_summary"]["complete"] is False
    assert response.json()["metrics_summary"]["evidence_errors"]
    assert response.json()["error"]
    s.cancel.assert_awaited_once_with(RID)
    s.session.commit.assert_awaited_once()


def test_active_sessions_are_bounded_and_sql_filters_owner_operation_and_terminal_states(closing_app):
    s = closing_app
    response = s.client.get(PREFIX)
    assert response.status_code == 200 and response.json()["items"][0]["id"] == JID
    query = s.session.scalars.await_args.args[0]
    params = query.compile().params
    assert "user:kk" in params.values() and "ufo_native_live" in params.values() and 50 in params.values()
    assert "NOT IN" in str(query) and "created_by" in str(query)
    s.key.actor_type = "agent"
    assert s.client.get(PREFIX).status_code == 403


def test_pending_close_before_creation_records_key_and_prevents_later_start(closing_app, monkeypatch):
    s = closing_app
    response = s.client.post(f"{PREFIX}/close-pending", json={"idempotency_key": "pending-key"})
    assert response.status_code == 200 and response.json() == {"closed": True, "job": None}
    assert s.records[0].action == "ufo_live.close_pending" and s.records[0].subject == "pending-key"
    s.pending["marker"] = s.records[0]
    build = AsyncMock()
    monkeypatch.setattr("eb.api.routes_ufo_live.build_plan", build)
    response = s.client.post(PREFIX, headers={"Idempotency-Key": "pending-key"}, json={
        "model_id": "mv_model", "environment_id": "ep_env", "preview_hash": "hash"})
    assert response.status_code == 409
    build.assert_not_awaited()
    # Both paths serialize the same signed 64-bit PostgreSQL advisory transaction lock.
    locks = [tuple(call.args[0].compile().params.values()) for call in s.session.execute.await_args_list]
    assert len(locks) == 2 and locks[0] == locks[1]
    assert -(2**63) <= locks[0][0] < 2**63


def test_preview_remains_read_only_without_requiring_an_idempotency_key(closing_app, monkeypatch):
    s = closing_app
    monkeypatch.setattr("eb.api.routes_ufo_live.build_plan", AsyncMock(return_value=live_plan()))
    response = s.client.post(f"{PREFIX}/preview", json={"model_id": "mv_model", "environment_id": "ep_env"})
    assert response.status_code == 200
    s.session.execute.assert_not_awaited()
    s.session.commit.assert_not_awaited()


def test_pending_close_after_commit_cancels_job_even_if_browser_lost_id(closing_app):
    s = closing_app
    s.pending["job"] = s.job
    response = s.client.post(f"{PREFIX}/close-pending", json={"idempotency_key": "pending-key"})
    assert response.status_code == 200 and response.json()["job"]["status"] == "cancelled"
    s.cancel.assert_awaited_once_with(RID)
    s.pending["marker"] = s.records[0]
    repeated = s.client.post(f"{PREFIX}/close-pending", json={"idempotency_key": "pending-key"})
    assert repeated.status_code == 200
    s.cancel.assert_awaited_once()


@pytest.mark.parametrize("case,expected", [("foreign", 404), ("agent", 403), ("operation", 404),
                                          ("origin", 403), ("invalid_key", 422)])
def test_pending_close_is_scoped_and_cannot_cancel_foreign_submissions(closing_app, case, expected):
    s = closing_app
    s.pending["job"] = s.job
    headers = {}
    key = "pending-key"
    if case == "foreign":
        s.job.created_by = "user:else"
    elif case == "agent":
        s.key.actor_type = "agent"
    elif case == "operation":
        s.job.spec["params"]["operation"] = "training"
    elif case == "origin":
        headers["Origin"] = "http://evil.test"
    else:
        key = "../unbounded"
    response = s.client.post(f"{PREFIX}/close-pending", headers=headers, json={"idempotency_key": key})
    assert response.status_code == expected
    s.cancel.assert_not_awaited()
