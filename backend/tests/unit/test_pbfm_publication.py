import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eb.orchestrator.ufo_native import verify_index


@pytest.mark.asyncio
async def test_ebd_hash_check_rejects_same_length_checkpoint_mutation(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"state")
    record = {"path": path.name, "bytes": 5, "sha256": hashlib.sha256(b"state").hexdigest()}

    async def scan(name):
        assert name == str(path)
        return {"files": 1, "samples": [{"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}

    client = SimpleNamespace(scan=AsyncMock(side_effect=scan))
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: client)
    await verify_index(tmp_path, [record], max_files=10000, max_file_bytes=1024**4)
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash differs"):
        await verify_index(tmp_path, [record], max_files=10000, max_file_bytes=1024**4)
    assert client.scan.await_count == 2


@pytest.mark.asyncio
async def test_large_replay_uses_explicit_bound_without_relaxing_media_default(tmp_path, monkeypatch):
    path = tmp_path / "buffer.hdf5"
    size = 2 * 1024**3 + 1
    with path.open("wb") as stream:
        stream.truncate(size)  # Sparse metadata fixture; no multi-GiB allocation/hash in a unit test.
    record = {"path": path.name, "bytes": size, "sha256": "b" * 64}
    client = SimpleNamespace(scan=AsyncMock(return_value={"files": 1, "samples": [{"sha256": "b" * 64}]}))
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: client)
    with pytest.raises(ValueError, match="metadata differs"):
        await verify_index(tmp_path, [record])
    client.scan.assert_not_awaited()
    await verify_index(tmp_path, [record], max_files=10000, max_file_bytes=1024**4)
    client.scan.assert_awaited_once_with(str(path))
