import json

import httpx
import pytest

from eb.store.ebd import EbdClient


ROOT = {"namespace": "run", "run_id": "run_0123456789abcdef0123", "path": "outputs"}
FILES = [{"path": "model.bin", "bytes": 5, "sha256": "a" * 64}]


def client(monkeypatch, handler):
    value = EbdClient()
    monkeypatch.setattr(value, "_client", lambda **_: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return value


@pytest.mark.asyncio
async def test_verify_artifacts_sends_one_batch_and_explicit_limits(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/v1/artifacts/verify"
        assert json.loads(request.content) == {
            "root": ROOT, "files": FILES * 1, "max_files": 10000, "max_file_bytes": 1024**4,
        }
        return httpx.Response(200, json={"schema_version": "eb/artifact-verification/v1",
                                       "verified": True, "file_count": 1, "total_bytes": 5})

    result = await client(monkeypatch, handle).verify_artifacts(ROOT, FILES, max_files=10000,
                                                              max_file_bytes=1024**4)
    assert result["verified"] is True
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, {}, {"verified": True}, {
    "schema_version": "eb/artifact-verification/v1", "verified": True,
    "file_count": 0, "total_bytes": 5,
}, {"schema_version": "eb/artifact-verification/v1", "verified": True,
    "file_count": True, "total_bytes": 5}])
async def test_verify_artifacts_rejects_malformed_success(monkeypatch, body):
    value = client(monkeypatch, lambda _: httpx.Response(200, json=body))
    with pytest.raises(httpx.RemoteProtocolError):
        await value.verify_artifacts(ROOT, FILES)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 409, 413, 500, 507])
async def test_verify_artifacts_never_swallows_http_failure(monkeypatch, status):
    value = client(monkeypatch, lambda _: httpx.Response(status, json={"error": {"code": "conflict"}}))
    with pytest.raises(httpx.HTTPStatusError):
        await value.verify_artifacts(ROOT, FILES)


@pytest.mark.asyncio
async def test_verify_artifacts_validates_bounds_before_transport(monkeypatch):
    def unexpected(_):
        pytest.fail("invalid request reached ebd")

    value = client(monkeypatch, unexpected)
    for files, limit in [([], 200), (FILES, 0), (FILES, 10001)]:
        with pytest.raises(ValueError):
            await value.verify_artifacts(ROOT, files, max_files=limit)


@pytest.mark.asyncio
async def test_publish_file_validates_cas_response(monkeypatch, tmp_path):
    value = client(monkeypatch, lambda request: httpx.Response(200, json={
        "schema_version": "eb/artifact-publication/v1", "sha256": "a" * 64,
        "size_bytes": 5, "storage": str(tmp_path / "cas/aa" / ("a" * 64)),
    }))
    monkeypatch.setattr(value.settings, "data_root", tmp_path)
    result = await value.publish_file(ROOT, "model.bin", expected_bytes=5, expected_sha256="a" * 64)
    assert result["size_bytes"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["root", "hash", "size", "schema"])
async def test_publish_file_rejects_wrong_identity(monkeypatch, tmp_path, change):
    payload = {"schema_version": "eb/artifact-publication/v1", "sha256": "a" * 64,
               "size_bytes": 5, "storage": str(tmp_path / "cas/aa" / ("a" * 64))}
    payload[{"root": "storage", "hash": "sha256", "size": "size_bytes", "schema": "schema_version"}[change]] = {
        "root": "/etc/passwd", "hash": "b" * 64, "size": 6, "schema": "unknown",
    }[change]
    value = client(monkeypatch, lambda _: httpx.Response(200, json=payload))
    monkeypatch.setattr(value.settings, "data_root", tmp_path)
    with pytest.raises(httpx.RemoteProtocolError):
        await value.publish_file(ROOT, "model.bin", expected_bytes=5, expected_sha256="a" * 64)
