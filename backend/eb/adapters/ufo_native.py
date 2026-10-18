from __future__ import annotations

import json

from eb.adapters.base import Adapter, RenderedJob
from eb.settings import get_settings
from eb.ufo_native import validate_job


class UfoNativeAdapter(Adapter):
    name = "ufo_native"

    def render(self, spec, run_id):
        plan = validate_job(spec)
        settings = get_settings()
        root = settings.data_root / "runs" / run_id
        mounts = [(str(root / "outputs"), "/outputs", "rw")]
        if plan["operation"] == "ufo_native_import":
            mounts.append((plan["source"]["path"], "/source", "ro"))
            mounts.extend(
                (item["source"], f"/source-extra/{i}", "ro")
                for i, item in enumerate(plan["source"]["extras"])
            )
        else:
            if spec.inputs.base_model != plan["input_path"]:
                raise ValueError("Native input path differs from frozen model")
            mounts.append((plan["input_path"], "/input", "ro"))
            if plan["operation"] in ("ufo_native_eval", "ufo_native_live"):
                mounts.append((plan["native_source_path"], "/native-source", "ro"))
            mounts.extend(
                (item["path"], f"/reports/{i}", "ro") for i, item in enumerate(plan.get("reports", []))
            )
        # The immutable plan is materialized only for a real Run, never at preview.
        if run_id not in ("run_preview", "run_validate"):
            if any(part.is_symlink() for part in (root, *root.parents)):
                raise ValueError("Native Run path must not contain symlinks")
            root.mkdir(parents=True, exist_ok=True)
            path = root / "native-plan.json"
            if path.is_symlink():
                raise ValueError("Native Run plan must not be a symlink")
            encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
            if path.exists() and path.read_text() != encoded:
                raise ValueError("Native Run plan already differs")
            if not path.exists():
                path.write_text(encoded)
            if plan["operation"] == "ufo_native_live":
                control = root / "live-control"
                if control.is_symlink():
                    raise ValueError("Live control path must not be a symlink")
                control.mkdir(exist_ok=True)
        if plan["operation"] == "ufo_native_live":
            mounts.append((str(root / "live-control"), "/control", "ro"))
        mounts.append((str(root / "native-plan.json"), "/plan.json", "ro"))
        return RenderedJob(
            image=spec.image,
            command=["python", "-m", "eb.runners.ufo_native_package", "--plan", "/plan.json"],
            env={
                **(
                    {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void"}
                    if not spec.resources.gpus
                    else {}
                ),
                "EB_RUN_ID": run_id,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "MUJOCO_GL": "egl" if spec.resources.gpus else "osmesa",
                "PYOPENGL_PLATFORM": "egl" if spec.resources.gpus else "osmesa",
                "MPLCONFIGDIR": "/tmp/mpl",
                "WARP_CACHE_PATH": "/outputs/warp-cache",
                "OMP_NUM_THREADS": "4",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
            },
            mounts=mounts,
            writable_paths=["outputs"],
            workdir="/opt/eb-native",
            outputs=["outputs"],
            metrics_glob="",
            gpus=spec.resources.gpus,
            shm_gb=2,
        )
