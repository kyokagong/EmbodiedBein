from __future__ import annotations

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobSpec


class DataConvertAdapter(Adapter):
    name = "data_convert"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        src = spec.params.get("src") or "/data/kk/berkeley_ego_data"
        return RenderedJob(
            image=_image(spec, "eb/tools:latest"),
            command=["python", "-m", "eb_tools.convert", "--src", str(src), "--run-id", run_id],
            env={},
            writable_paths=["outputs"],
            gpus=0,
            workdir="/workspace",
            outputs=["outputs"],
        )


class DataRetargetAdapter(Adapter):
    name = "data_retarget"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        src = spec.params.get("src") or "/data/kk/berkeley_ego_data"
        return RenderedJob(
            image=_image(spec, "eb/tools:latest"),
            command=[
                "python",
                "-m",
                "eb_tools.retarget",
                "--src",
                str(src),
                "--run-id",
                run_id,
                "--action-dim",
                "10",
                "--fps",
                "20",
            ],
            env={},
            writable_paths=["outputs"],
            gpus=0,
            workdir="/workspace",
            outputs=["outputs"],
        )
