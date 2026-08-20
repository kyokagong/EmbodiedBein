from __future__ import annotations

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec


class MujocoAdapter(Adapter):
    kind = JobKind.EVAL_SIM2SIM
    name = "mujoco"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        return RenderedJob(
            image=_image(spec, "eb/mujoco:latest"),
            command=[
                "python",
                "scripts/play_mujoco.py",
                "--headless",
                "--export-traj",
                "--seeds",
                ",".join(str(s) for s in spec.params.get("seeds", [0])),
            ],
            env={},
            writable_paths=["outputs"],
            gpus=0,
            workdir="/workspace",
        )


class ToolsAdapter(Adapter):
    name = "tools"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        kind = spec.kind.value
        return RenderedJob(
            image=_image(spec, "eb/tools:latest"),
            command=["python", "-m", "eb_tools", kind, "--run-id", run_id],
            env={},
            writable_paths=["outputs"],
            gpus=0,
        )


class NativeAdapter(Adapter):
    name = "native"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        return RenderedJob(
            image="",
            command=["ebd", spec.kind.value, "--run-id", run_id],
            env={},
            writable_paths=[],
            gpus=0,
        )
