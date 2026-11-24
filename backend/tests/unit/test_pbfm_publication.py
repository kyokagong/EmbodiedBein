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

    monkeypatch.setattr("eb.orchestrator.ufo_native.artifact_root", lambda _: {"namespace": "run", "run_id": "run_test", "path": "outputs"})

    async def verify(_, files, **limits):
        assert files == [record]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Native output hash differs")

    client = SimpleNamespace(verify_artifacts=AsyncMock(side_effect=verify))
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: client)
    await verify_index(tmp_path, [record], max_files=10000, max_file_bytes=1024**4)
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash differs"):
        await verify_index(tmp_path, [record], max_files=10000, max_file_bytes=1024**4)
    assert client.verify_artifacts.await_count == 2


@pytest.mark.asyncio
async def test_large_replay_uses_explicit_bound_without_relaxing_media_default(tmp_path, monkeypatch):
    path = tmp_path / "buffer.hdf5"
    size = 2 * 1024**3 + 1
    with path.open("wb") as stream:
        stream.truncate(size)  # Sparse metadata fixture; no multi-GiB allocation/hash in a unit test.
    record = {"path": path.name, "bytes": size, "sha256": "b" * 64}
    client = SimpleNamespace(verify_artifacts=AsyncMock(return_value={"verified": True}))
    root = {"namespace": "run", "run_id": "run_test", "path": "outputs"}
    monkeypatch.setattr("eb.orchestrator.ufo_native.artifact_root", lambda _: root)
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: client)
    with pytest.raises(ValueError, match="metadata differs"):
        await verify_index(tmp_path, [record])
    client.verify_artifacts.assert_not_awaited()
    await verify_index(tmp_path, [record], max_files=10000, max_file_bytes=1024**4)
    client.verify_artifacts.assert_awaited_once_with(root, [record], max_files=10000, max_file_bytes=1024**4)
