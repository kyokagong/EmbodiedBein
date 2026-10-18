import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.websockets import WebSocketDisconnect
from test_ufo_live import live_plan

from eb.api.routes_ufo_live import router
from eb.store.db import get_session

RID = "run_0123456789abcdef0123"
PATH = f"/api/v1/ufo-native/live/runs/{RID}/stream"


@pytest.fixture
def socket_app(monkeypatch):
    records = []
    run = SimpleNamespace(id=RID, job_id="job_live", status="running")
    job = SimpleNamespace(
        id="job_live", created_by="user:kk", status="running",
        spec={"params": {"operation": "ufo_native_live", "native_plan": live_plan()}},
    )
    session = SimpleNamespace(
        get=AsyncMock(side_effect=lambda cls, ident: run if ident == RID else job),
        scalar=AsyncMock(return_value=SimpleNamespace(principal="user:kk", actor_type="user")),
        add=records.append, commit=AsyncMock(), rollback=AsyncMock(),
    )

    async def db():
        yield session

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="websocket-test")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_session] = db
    return app, session, run, job, records


def test_websocket_requires_auth_before_accept(socket_app):
    app, session, *_ = socket_app
    session.scalar.return_value = None
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(PATH):
            pytest.fail("unauthenticated socket accepted")
    assert error.value.code == 4401


@pytest.mark.parametrize("case,code", [("agent", 4403), ("foreign", 4404), ("ended", 4409)])
def test_websocket_requires_owner_user_and_running_job(socket_app, case, code):
    app, session, run, job, _ = socket_app
    if case == "agent":
        session.scalar.return_value = SimpleNamespace(principal="agent:kk", actor_type="agent")
    elif case == "foreign":
        job.created_by = "user:other"
    else:
        run.status = "finished"
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(PATH, headers={"authorization": "Bearer test"}):
            pytest.fail("forbidden socket accepted")
    assert error.value.code == code


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://testserver.evil", "http://testserver:9"])
def test_websocket_rejects_foreign_origin(socket_app, origin):
    app, *_ = socket_app
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(PATH, headers={"authorization": "Bearer test", "origin": origin}):
            pytest.fail("foreign Origin accepted")
    assert error.value.code == 4403


def test_historical_v1_websocket_rejected_before_accept(socket_app):
    from eb.ufo_live import environment_spec, live_suite_spec, suite_id
    from eb.ufo_native import plan_hash

    app, _, _, job, _ = socket_app
    plan = job.spec["params"]["native_plan"]
    plan.update(
        environment_id=environment_spec("mujoco", version=1)["id"],
        environment_hash=environment_spec("mujoco", version=1)["content_hash"],
        suite_id=suite_id("mujoco", version=1), suite_spec=live_suite_spec("mujoco", version=1),
    )
    plan["hash"] = plan_hash(plan)
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(PATH, headers={"authorization": "Bearer test"}):
            pytest.fail("historical stream accepted")
    assert error.value.code == 4409


def test_websocket_forwards_binary_and_audits_only_accepted_controls(socket_app, monkeypatch):
    import eb.api.routes_ufo_live as routes

    app, _, _, _, records = socket_app
    frame = b"\x00\x00\x00\x02{}\xff\xd8opaque-jpeg\xff\xd9"

    class Upstream:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(frame)

        async def send(self, message):
            import json
            value = json.loads(message)
            if value["type"] == "control":
                await self.queue.put(json.dumps({
                    "type": "control_result", "id": value["command"]["id"],
                    "ok": value["command"]["action"] != "reset", "status": 200,
                    "detail": "accepted",
                }))

        async def recv(self):
            return await self.queue.get()

    @asynccontextmanager
    async def connect(path, **kwargs):
        assert path == routes.get_settings().ebd_media_uds
        assert kwargs["uri"] == f"ws://ebd/v1/runs/{RID}/live/stream"
        assert kwargs["max_queue"] == 1
        yield Upstream()

    monkeypatch.setattr(routes, "unix_connect", connect)
    with TestClient(app) as client:
        with client.websocket_connect(
            PATH, headers={"authorization": "Bearer test", "origin": "http://testserver"},
        ) as ws:
            assert ws.receive_bytes() == frame
            ws.send_json({"type": "control", "command": {"id": "pause1", "action": "pause"}})
            assert ws.receive_json()["ok"] is True
            ws.send_json({"type": "control", "command": {"id": "reset1", "action": "reset"}})
            assert ws.receive_json()["ok"] is False
            # Finish server pump cleanup before TestClient cancels its ASGI scope.
            ws.close()
            closing = ws.receive()
            assert closing["type"] == "websocket.close"
            assert closing["code"] == 1000
    assert len(records) == 1
    assert records[0].action == "ufo_live.control"
    assert records[0].after == {"job_id": "job_live", "run_id": RID, "id": "pause1", "action": "pause"}


@pytest.mark.parametrize("value", [
    {"type": "camera", "seq": 1, "azimuth": 0, "elevation": -30, "distance": 2, "path": "/x"},
    {"type": "camera", "seq": True, "azimuth": 0, "elevation": -30, "distance": 2},
    {"type": "camera", "seq": 1, "azimuth": 0, "elevation": -30, "distance": float("nan")},
    {"type": "frame_ack", "seq": -1}, {"type": "heartbeat", "extra": 1},
    {"type": "shell"}, {"type": "control", "command": {"id": "x", "action": "shell"}},
])
def test_client_messages_are_strict_and_bounded(value):
    from eb.api.routes_ufo_live import validate_socket_message
    with pytest.raises(ValueError):
        validate_socket_message(value)


@pytest.mark.parametrize("message", [
    {"type": "heartbeat", "extra": 1}, {"type": "shell"},
    {"type": "camera", "seq": 1, "azimuth": 0, "elevation": -30, "distance": 100},
    "x" * 4097, b"untrusted binary",
])
def test_websocket_cookie_auth_and_invalid_messages_close_without_writes(socket_app, monkeypatch, message):
    import base64
    import json

    from itsdangerous import TimestampSigner

    import eb.api.routes_ufo_live as routes

    app, session, _, _, records = socket_app

    class Upstream:
        async def send(self, value):
            pytest.fail("invalid input was forwarded")

        async def recv(self):
            await asyncio.Future()

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Upstream()

    monkeypatch.setattr(routes, "unix_connect", connect)
    cookie = TimestampSigner("websocket-test").sign(base64.b64encode(json.dumps({"user": "kk"}).encode()))
    with TestClient(app) as client:
        client.cookies.set("session", cookie.decode())
        with client.websocket_connect(PATH, headers={"origin": "http://testserver"}) as ws:
            if isinstance(message, str):
                ws.send_text(message)
            elif isinstance(message, bytes):
                ws.send_bytes(message)
            else:
                ws.send_json(message)
            assert ws.receive()["code"] == 1008
    assert not records
    session.scalar.assert_not_awaited()
