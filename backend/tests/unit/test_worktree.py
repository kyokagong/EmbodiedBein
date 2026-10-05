"""No VCS commands: ebd owns code freezing; Rust tests exercise the bytes."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from eb.store.ebd import EbdClient
from eb.worktree import git_is_dirty, overlay_mounts, prepare_worktree, resolve_commit


class SnapshotTests(unittest.IsolatedAsyncioTestCase):
    def test_stack_snapshot_keeps_adapter_receipt_mount_destination(self):
        from eb.orchestrator.service import _stack_mount_destination

        self.assertEqual(
            _stack_mount_destination([("/snapshot", "/workspace/source-receipts/rlinf", "ro")]),
            "/workspace/source-receipts/rlinf",
        )
        self.assertEqual(_stack_mount_destination([]), "/workspace/rlinf")

    def test_only_git_embodiment_sources_are_snapshotted(self):
        from eb.orchestrator.service import _is_snapshot_source

        self.assertTrue(_is_snapshot_source({"kind": "git_external"}))
        self.assertFalse(_is_snapshot_source({"kind": "managed_bundle"}))

    async def test_submit_forwards_limits_and_keeps_data_mounts(self):
        from eb.adapters.base import RenderedJob
        from eb.domain.job import JobSpec
        from eb.orchestrator.service import Orchestrator

        spec = JobSpec(kind="data.convert", embodiment_env_id="env", resources={"cpus": 6, "memory_gb": 24})
        job = SimpleNamespace(
            id="job_x",
            spec=spec.model_dump(mode="json"),
            kind="data.convert",
            embodiment_env_id="env",
            status="queued",
            attempts=0,
            version=0,
        )
        session = SimpleNamespace(add=Mock(), commit=AsyncMock(), refresh=AsyncMock())
        ebd = SimpleNamespace(submit_run=AsyncMock(return_value={"status": "started", "container_id": "abc"}))
        rendered = RenderedJob("test", ["validate"], {}, [], mounts=[("/data/kk/data/input", "/data", "ro")])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("eb.orchestrator.service.prepare_run_dirs", return_value=Path(tmp)):
                with patch(
                    "eb.orchestrator.service.get_adapter",
                    return_value=SimpleNamespace(render=lambda *_: rendered),
                ):
                    run = await Orchestrator(ebd).admit_and_start(session, job, [])
        payload = ebd.submit_run.call_args.args[0]
        self.assertEqual((payload["cpus"], payload["memory_gb"], payload["network"]), (6, 24, "bridge"))
        self.assertEqual(payload["mounts"], [{"src": "/data/kk/data/input", "dst": "/data", "mode": "ro"}])
        self.assertEqual(run.status, "running")

    async def test_snapshot_uses_ebd_and_never_vcs(self):
        def handle(request):
            self.assertEqual(request.url.path, "/v1/snapshots")
            self.assertEqual(request.read(), b'{"path":"/source","content_hash":null}')
            return httpx.Response(
                200, json={"path": "/owned/snapshots/abc/code", "content_hash": "sha256:abc"}
            )

        client = EbdClient()
        with patch("subprocess.run", side_effect=AssertionError("VCS prohibited")):
            with patch.object(
                client, "_client", return_value=httpx.AsyncClient(transport=httpx.MockTransport(handle))
            ):
                self.assertEqual(
                    await prepare_worktree(Path("/source"), None, "run_1", ebd=client),
                    Path("/owned/snapshots/abc/code"),
                )
            self.assertIsNone(git_is_dirty(Path("/source")))
            with self.assertRaises(ValueError):
                resolve_commit(Path("/source"), "HEAD")

    def test_overlay_rejects_escape_and_never_mutates_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap, run = Path(tmp) / "snap", Path(tmp) / "run"
            snap.mkdir()
            mounts = overlay_mounts(snap, run, ["logs", "outputs"])
            self.assertEqual(list(snap.iterdir()), [])
            self.assertEqual(mounts[0][2], "ro")
            self.assertTrue((run / "logs").is_dir())
            for rel in ["../escape", "/tmp/escape", "."]:
                with self.assertRaises(ValueError):
                    overlay_mounts(snap, run, [rel])
            (run / "link").symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaises(ValueError):
                overlay_mounts(snap, run, ["link/escape"])


if __name__ == "__main__":
    unittest.main()
