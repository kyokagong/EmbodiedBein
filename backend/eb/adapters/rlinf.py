from __future__ import annotations

from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec
from eb.domain.rlinf_form import component_placement, forced_overrides, get_template
from eb.settings import get_settings


class RlinfAdapter(Adapter):
    kind = JobKind.TRAIN_SFT
    name = "rlinf"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        settings = get_settings()
        template_id = spec.params.get("template_id") or spec.params.get("config_name")
        tpl = None
        if template_id:
            try:
                tpl = get_template(str(template_id))
            except KeyError:
                tpl = None
        cfg_name = spec.params.get("config_name") or (tpl.config_name if tpl else "berkeley_wbc_ppo_mlp")
        cfg_path = spec.params.get("config_path") or (tpl.config_path if tpl else "examples/embodiment/config")
        script = spec.params.get("script") or (tpl.script if tpl else "examples/embodiment/train_embodied_agent.py")
        if not spec.params.get("script") and spec.kind == JobKind.TRAIN_SFT and not tpl:
            script = "examples/sft/train_vla_sft.py"
            cfg_path = spec.params.get("config_path", "examples/sft/config/")
        ray_port = int((spec.runtime.get("ray") or {}).get("port") or spec.params.get("ray_port") or 6390)
        gpus = spec.resources.gpus or 2
        job_name = spec.name or run_id
        placement = spec.params.get("placement") or (
            component_placement(tpl.id, gpus) if tpl else f"cluster.component_placement.actor_rollout={gpus}"
        )
        overrides = list(spec.overrides or [])
        for item in forced_overrides(job_name):
            if item not in overrides:
                overrides.append(item)
        if placement not in overrides:
            overrides.append(placement)
        cmd = [
            "bash",
            "-lc",
            f"ray start --head --port {ray_port} && python {script} "
            f"--config-path /workspace/rlinf/{cfg_path} --config-name {cfg_name} "
            f"cluster.num_nodes=1 cluster.ray_port={ray_port} "
            + " ".join(overrides),
        ]
        env = {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": "/workspace/berkeley:/workspace/rlinf",
            "RAY_PORT": str(ray_port),
        }
        env.update(spec.runtime.get("env") or {})
        data_root = settings.data_root
        run_root = data_root / "runs" / run_id
        mounts: list[tuple[str, str, str]] = [
            (str(run_root / "output"), "/workspace/output", "rw"),
            (str(run_root / "tb"), "/workspace/tb", "rw"),
            (str(data_root / "cache" / "ov_data"), "/root/.local/share/ov/data", "rw"),
            (str(data_root / "cache" / "isaac_nucleus"), "/root/Documents/Isaac", "rw"),
        ]
        if spec.code_refs and spec.code_refs.embodiment and spec.code_refs.embodiment.path:
            mounts.append((spec.code_refs.embodiment.path, "/workspace/berkeley", "ro"))
        if spec.code_refs and spec.code_refs.stack and spec.code_refs.stack.path:
            mounts.append((spec.code_refs.stack.path, "/workspace/rlinf", "ro"))
        for ds in spec.inputs.datasets:
            host = ds.get("path") or ds.get("dest_path")
            dsid = ds.get("id") or Path(str(host)).name
            if host:
                mounts.append((str(host), f"/workspace/data/{dsid}", "ro"))
        if spec.inputs.base_model:
            mounts.append((spec.inputs.base_model, "/workspace/models/base", "ro"))
        return RenderedJob(
            image=_image(spec, "eb/rlinf:0.4.0-emb"),
            command=cmd,
            env=env,
            writable_paths=["output", "tb"],
            mounts=mounts,
            gpus=spec.resources.gpus or 2,
            shm_gb=spec.resources.shm_gb or 32,
            workdir="/workspace/rlinf",
        )
