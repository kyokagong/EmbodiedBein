from __future__ import annotations

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec


class RlinfAdapter(Adapter):
    kind = JobKind.TRAIN_SFT
    name = "rlinf"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        cfg_name = spec.params.get("config_name", "maniskill_rlt_stage1_sft_openpi_pi05")
        cfg_path = spec.params.get("config_path", "examples/sft/config/")
        ray_port = (spec.runtime.get("ray") or {}).get("port", 6390)
        script = "examples/sft/train_vla_sft.py"
        if spec.kind == JobKind.TRAIN_RLFT:
            script = "examples/rlft/train.py"
        cmd = [
            "python",
            script,
            "--config-path",
            f"/workspace/rlinf/{cfg_path}",
            "--config-name",
            str(cfg_name),
            "cluster.num_nodes=1",
            f"cluster.ray_port={ray_port}",
            *spec.overrides,
        ]
        env = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
        env.update((spec.runtime.get("env") or {}))
        return RenderedJob(
            image=_image(spec, "eb/rlinf:0.4.0-emb"),
            command=cmd,
            env=env,
            writable_paths=["outputs", "checkpoints"],
            gpus=spec.resources.gpus or 2,
            shm_gb=spec.resources.shm_gb or 32,
            workdir="/workspace/rlinf",
        )
