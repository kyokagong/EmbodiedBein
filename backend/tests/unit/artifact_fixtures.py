"""Adapt existing per-file integrity oracles to the batch transport in unit fixtures."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


def install_artifact_batch(monkeypatch, scan):
    install_storage_inventory(monkeypatch)
    roots = {}

    def reference(root):
        identity = f"run_fixture_{len(roots)}"
        roots[identity] = Path(root)
        return {"namespace": "run", "run_id": identity, "path": ""}

    async def verify(root, files, **limits):
        for item in files:
            value = await scan(str(roots[root["run_id"]] / item["path"]))
            samples = value.get("samples") or []
            if (value.get("files") != 1 or len(samples) != 1
                    or samples[0].get("sha256") != item["sha256"]
                    or samples[0].get("size", item["bytes"]) != item["bytes"]):
                raise ValueError(f"Native output integrity differs (hash differs): {item['path']}")
        return {"verified": True, "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files)}

    batch = AsyncMock(side_effect=verify)
    monkeypatch.setattr("eb.store.ebd.artifact_root", reference)
    monkeypatch.setattr("eb.orchestrator.ufo_native.artifact_root", reference)
    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: SimpleNamespace(verify_artifacts=batch))
    monkeypatch.setattr("eb.store.ebd.EbdClient.verify_artifacts", batch)
    return batch


def install_storage_inventory(monkeypatch):
    async def inventory(root, **_):
        return [{"path": path.relative_to(root).as_posix(), "is_dir": path.is_dir(),
                 "is_symlink": path.is_symlink(), "bytes": path.lstat().st_size,
                 "mtime": int(path.stat().st_mtime)} for path in Path(root).rglob("*")]
    for module in ("eb.orchestrator.collect", "eb.orchestrator.pbfm_training", "eb.ufo_training_prompt",
                   "eb.ufo_native_baseline_service", "eb.pbfm_diagnostic_training", "eb.pbfm_periodic", "eb.pbfm_recovery"):
        monkeypatch.setattr(module + ".storage_inventory", inventory)
