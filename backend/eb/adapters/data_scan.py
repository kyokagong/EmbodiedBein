from __future__ import annotations

from eb.adapters.base import Adapter, RenderedJob
from eb.domain.job import JobKind, JobSpec


class DataScanAdapter(Adapter):
    """Containerless: orchestrator calls ebd scan; render is unused for start."""

    kind = JobKind.DATA_SCAN
    name = "data_scan"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        path = spec.params.get("path") or "/data/kk/berkeley_ego_data"
        return RenderedJob(
            image="",
            command=["ebd", "scan", str(path), "--run-id", run_id],
            env={},
            writable_paths=[],
            gpus=0,
            shm_gb=0,
            workdir="/tmp",
        )
