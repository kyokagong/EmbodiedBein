from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_proxy_forwards_range_and_releases_upstream(monkeypatch, method):
    from eb.api.file_stream import stream_storage_file
    closed = []
    class Bytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"abc"
        async def aclose(self):
            closed.append(True)
    def handler(request):
        assert request.method == method
        assert request.headers["range"] == "bytes=1-3"
        assert request.url.params["path"] == "nested/policy.onnx"
        return httpx.Response(206, headers={"content-range": "bytes 1-3/8", "content-length":"3"}, stream=Bytes())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("eb.api.file_stream.media_client", lambda: client)
    result = await stream_storage_file(NS(method=method, headers={"range":"bytes=1-3"}), {"namespace":"assets"}, "nested/policy.onnx")
    assert result.status_code == 206
    if method == "GET":
        iterator = result.body_iterator
        assert await anext(iterator) == b"abc"
        await iterator.aclose()
    else:
        await result.background()
    assert closed and client.is_closed


@pytest.mark.asyncio
async def test_proxy_transport_failure_closes_client(monkeypatch):
    from eb.api.file_stream import stream_storage_file
    from fastapi import HTTPException
    def fail(_):
        raise httpx.ConnectError("unavailable")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    monkeypatch.setattr("eb.api.file_stream.media_client", lambda: client)
    with pytest.raises(HTTPException) as error:
        await stream_storage_file(NS(method="GET", headers={}), {"namespace":"assets"}, "file")
    assert error.value.status_code == 502 and client.is_closed


@pytest.mark.asyncio
async def test_cancel_during_send_finishes_closing_client(monkeypatch):
    import anyio
    from eb.api.file_stream import stream_file
    closed = []
    entered = anyio.Event()
    class Client:
        def build_request(self, *args, **kwargs):
            return None
        async def send(self, *args, **kwargs):
            entered.set()
            await anyio.sleep_forever()
        async def aclose(self):
            await anyio.sleep(0)
            closed.append(True)
    monkeypatch.setattr("eb.api.file_stream.media_client", Client)
    async with anyio.create_task_group() as group:
        group.start_soon(stream_file, NS(method="GET",headers={}),"http://ebd/file")
        await entered.wait()
        group.cancel_scope.cancel()
    assert closed == [True]




@pytest.mark.asyncio
async def test_live_snapshot_releases_read_transaction_before_stream(monkeypatch):
    from eb.api import routes_ufo_live as routes
    session=NS(rollback=AsyncMock())
    monkeypatch.setattr(routes,"owned_run",AsyncMock())
    async def stream(*args,**kwargs):
        session.rollback.assert_awaited_once()
        return "snapshot"
    monkeypatch.setattr(routes,"stream_output",stream)
    assert await routes.snapshot("run_test","state",NS(),session,NS())=="snapshot"
