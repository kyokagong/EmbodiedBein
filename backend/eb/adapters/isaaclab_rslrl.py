from __future__ import annotations

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec

# Image site-packages only has an editable install of berkeley_humanoid_lite.
# The assets/lowlevel packages live under source/ and must be on PYTHONPATH
# so a read-only worktree overlay is importable without writing .egg-info.
_BHL_PYTHONPATH = ":".join(
    [
        "/workspace/isaaclab_extension_template/source/berkeley_humanoid_lite",
        "/workspace/isaaclab_extension_template/source/berkeley_humanoid_lite_assets",
        "/workspace/isaaclab_extension_template/source/berkeley_humanoid_lite_lowlevel",
    ]
)


class IsaacLabRslRlAdapter(Adapter):
    kind = JobKind.TRAIN_RL
    name = "isaaclab_rslrl"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        task = spec.params.get("task", "Velocity-Berkeley-Humanoid-Lite-v0")
        num_envs = spec.params.get("num_envs", 4096)
        max_it = spec.params.get("max_iterations", 6000)
        seed = spec.params.get("seed", 0)
        run_name = spec.name or run_id
        script = {
            JobKind.TRAIN_RL: "scripts/rsl_rl/train.py",
            JobKind.EVAL_SIM: "scripts/rsl_rl/play.py",
            JobKind.EXPORT_POLICY: "scripts/rsl_rl/play.py",
        }.get(spec.kind, "scripts/rsl_rl/train.py")
        cmd = [
            "/isaac-sim/python.sh",
            script,
            "--task",
            str(task),
            "--headless",
            "--device",
            "cuda:0",
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(max_it),
            "--seed",
            str(seed),
            "--run_name",
            run_name,
            "--logger",
            "tensorboard",
            *spec.overrides,
        ]
        if spec.kind == JobKind.EVAL_SIM:
            cmd = [
                "/isaac-sim/python.sh",
                "scripts/rsl_rl/play.py",
                "--task",
                str(task),
                "--headless",
                "--num_envs",
                str(spec.params.get("num_envs", 128)),
                "--seed",
                str(seed),
                "--metrics_output",
                "/workspace/isaaclab_extension_template/outputs/metrics.json",
                *spec.overrides,
            ]
        return RenderedJob(
            image=_image(spec, "eb/isaac-lab:2.3.2"),
            command=cmd,
            env={
                "ACCEPT_EULA": "Y",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PYTHONPATH": _BHL_PYTHONPATH,
            },
            writable_paths=["logs", "configs", "outputs"],
            devices=spec.resources.devices or ["/dev/dri"],
            gpus=spec.resources.gpus or 1,
            shm_gb=spec.resources.shm_gb,
            workdir="/workspace/isaaclab_extension_template",
            outputs=["outputs", "logs"],
            metrics_glob="logs/**/events.out.tfevents.*",
        )
