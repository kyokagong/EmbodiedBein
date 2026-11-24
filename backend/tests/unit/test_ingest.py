from pathlib import Path

from eb.ingest import infer_contract


def test_infer_biped_and_humanoid():
    cid, io = infer_contract(Path("/data/kk/Berkeley-Humanoid-Lite/checkpoints/policy_biped_25hz_a.onnx"))
    assert cid == "ee_bhl_biped_12dof_v1"
    assert io == {"obs": 45, "act": 12}
    cid, io = infer_contract(Path("logs/rsl_rl/humanoid/x/events"))
    assert cid == "ee_bhl_humanoid_22dof_v1"
    assert io["obs"] == 75


async def test_ingest_enumerates_and_hashes_through_ebd(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, Mock
    from types import SimpleNamespace as NS
    from eb.ingest import ingest_checkpoints
    from eb.store.models import Artifact
    facts = [{"path": "model.pt", "is_dir": False, "is_symlink": False, "bytes": 4, "mtime": 1}]
    monkeypatch.setattr("eb.ingest.storage_inventory", AsyncMock(return_value=facts))
    monkeypatch.setattr("eb.ingest.storage_root", lambda _: {"namespace": "source"})
    digest = AsyncMock(return_value={"bytes": 4, "sha256": "a" * 64})
    monkeypatch.setattr("eb.ingest.EbdClient.hash_storage_file", digest)
    rows = []
    session = NS(scalars=AsyncMock(return_value=NS(all=lambda: [])), get=AsyncMock(return_value=NS(id="mf_legacy")),
                 scalar=AsyncMock(return_value=None), add=rows.append, flush=AsyncMock())
    assert await ingest_checkpoints(session, tmp_path) == 1
    assert digest.await_count == 1
    artifact = next(row for row in rows if isinstance(row, Artifact))
    assert artifact.sha256 == "a" * 64 and artifact.size_bytes == 4
