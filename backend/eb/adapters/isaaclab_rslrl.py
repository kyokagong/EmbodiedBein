from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec
from eb.settings import get_settings

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
        if spec.embodiment_env_id == "ee_rpo_v2_bfm_23dof_v1" and spec.kind == JobKind.EXPORT_POLICY:
            from eb.paths import validate_allowed_path

            source_run_id = str(spec.params.get("source_run_id", ""))
            asset_sha256 = str(spec.params.get("asset_sha256", ""))
            if not re.fullmatch(r"run_[0-9a-f]+", source_run_id):
                raise ValueError("BFM export requires a pinned source Run")
            if not re.fullmatch(r"[0-9a-f]{64}", asset_sha256):
                raise ValueError("BFM export requires a pinned RPO asset hash")
            from eb.rpo_bfm import REFERENCE_IMAGE,OFFICIAL_REFERENCE_IMAGE, REFERENCE_POOL_CONVERTER_REVISION, UNCHECKED_REFERENCE_REVISION,G1_CONVERTER_REVISION,G1_IMAGE
            native = spec.params.get("model_execution")
            if native:
                from eb.domain.training_recipe import ExecutionBinding, RECIPES, UFO_ISAAC_IMAGE_DIGEST, UFO_SLEW_RECIPE, UFO_STABLE_RECIPE, UFO_VELOCITY_RECIPE, UFO_SCALE_RECIPE, recipe_eval_image_digest
                binding = ExecutionBinding.model_validate(native)
                recipe = RECIPES.get(binding.recipe_id)
                if recipe is None or binding != recipe.binding:
                    raise ValueError("Unknown native BFM export binding")
                native_image = recipe_eval_image_digest(binding.recipe_id) if binding.recipe_id in {UFO_STABLE_RECIPE, UFO_SLEW_RECIPE, UFO_VELOCITY_RECIPE, UFO_SCALE_RECIPE} else UFO_ISAAC_IMAGE_DIGEST
                if spec.image not in (None, recipe.image, native_image):
                    raise ValueError("Native BFM export requires its pinned agent image")
            if not native and spec.image and spec.image not in {REFERENCE_IMAGE,OFFICIAL_REFERENCE_IMAGE, "eb/rlinf:rpo-v2-bfm-zero"}:
                raise ValueError("RPO BFM export requires the dedicated RLinf image")
            checkpoint = validate_allowed_path(spec.inputs.base_model or "")
            if native:
                from eb.runners.bfm_ufo_checkpoint import validate_ufo_checkpoint
                metadata = checkpoint / "native-checkpoint.json"
                if metadata.is_symlink() or not metadata.is_file():
                    raise ValueError("Native model package receipt missing")
                source = checkpoint / json.loads(metadata.read_text())["path"]
                if not source.resolve().is_relative_to(checkpoint):
                    raise ValueError("Native checkpoint escapes model package")
                native_manifest = validate_ufo_checkpoint(source, run_id=source_run_id, verify_hashes=False)
                if native_manifest["execution"] != native:
                    raise ValueError("Native checkpoint and model binding differ")
            if (checkpoint.name != source_run_id or not checkpoint.is_dir()
                    or (checkpoint / "model_state_dict/full_weights.pt").is_symlink()
                    or (not native and not (checkpoint / "model_state_dict/full_weights.pt").is_file())):
                raise ValueError("BFM export checkpoint does not match its source Run")
            datasets = spec.inputs.datasets
            if len(datasets) != 1:
                raise ValueError("BFM export requires its single pinned motion view")
            motion = validate_allowed_path(str(datasets[0].get("path", "")))
            if (motion.is_symlink() or not motion.is_dir()
                    or (motion / "manifest.json").is_symlink()
                    or not (motion / "manifest.json").is_file()):
                raise ValueError("BFM export motion view is unavailable")
            revision=json.loads((motion / "manifest.json").read_text()).get("converter_revision")
            if revision==G1_CONVERTER_REVISION and spec.image not in (None,G1_IMAGE):
                raise ValueError("G1 BFM view needs the calibrated G1 codec image")
            if (revision
                    in {REFERENCE_POOL_CONVERTER_REVISION, UNCHECKED_REFERENCE_REVISION}
                    and spec.image == "eb/rlinf:rpo-v2-bfm-zero"):
                raise ValueError("BFM reference motion needs the versioned reference image")
            settings = get_settings()
            config = settings.data_root / "runs" / source_run_id / "output/resolved-config.yaml"
            if config.is_symlink() or not config.is_file():
                raise ValueError("BFM export source Run config is unavailable")
            output = settings.data_root / "runs" / run_id / "output"
            return RenderedJob(
                image=native_image if native else spec.image or REFERENCE_IMAGE,
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_bfm_artifact.py", "export",
                         "--checkpoint", "/workspace/checkpoint",
                         "--resolved-config", "/workspace/config/resolved-config.yaml",
                         "--motion-root", "/workspace/motion",
                         "--asset-sha256", asset_sha256,
                         "--output", "/workspace/output/exported"],
                env={"PYTHONUNBUFFERED":"1"}, writable_paths=["output"],
                mounts=[(str(checkpoint), "/workspace/checkpoint", "ro"),
                        (str(config), "/workspace/config/resolved-config.yaml", "ro"),
                        (str(motion), "/workspace/motion", "ro"),
                        (str(output), "/workspace/output", "rw")],
                gpus=spec.resources.gpus or 1, shm_gb=spec.resources.shm_gb,
                workdir="/workspace", outputs=["output"], metrics_glob="",
            )
        if spec.embodiment_env_id == "ee_rpo_v2_wbc_23dof_v1" and spec.kind == JobKind.EXPORT_POLICY:
            source = Path(spec.inputs.base_model or "")
            checkpoint = source / "model_state_dict/full_weights.pt" if source.is_dir() else source
            if not checkpoint.is_file():
                raise ValueError("RPO 导出必须绑定 RLinf full_weights.pt 或其模型目录")
            settings = get_settings()
            return RenderedJob(
                image=_image(spec, "eb/rlinf:rpo-v2-wbc"),
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_wbc_artifact.py", "export",
                         "--checkpoint", "/workspace/model/full_weights.pt",
                         "--output", "/workspace/output/model"],
                env={"PYTHONUNBUFFERED":"1"}, writable_paths=["output"],
                mounts=[(str(checkpoint), "/workspace/model/full_weights.pt", "ro"),
                        (str(settings.data_root / "runs" / run_id / "output"), "/workspace/output", "rw"),
                        (str(settings.models_root / "runs" / run_id), "/workspace/output/model", "rw")],
                gpus=spec.resources.gpus, shm_gb=spec.resources.shm_gb,
                workdir="/workspace", outputs=["output"], metrics_glob="",
            )
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
        if spec.kind == JobKind.EXPORT_POLICY:
            # Committed BHL play.py has no --export_only/--seed; it writes ONNX then loops.
            play = [
                "/isaac-sim/python.sh",
                "scripts/rsl_rl/play.py",
                "--task",
                str(task),
                "--headless",
                "--device",
                "cuda:0",
                "--num_envs",
                str(spec.params.get("num_envs", 16)),
                *spec.overrides,
            ]
            quoted = " ".join(shlex.quote(p) for p in play)
            cmd = [
                "bash",
                "-lc",
                (
                    f"{quoted} & pid=$!; "
                    "for i in $(seq 1 360); do "
                    "  if find logs outputs -name policy.onnx 2>/dev/null | grep -q .; then "
                    "    sleep 2; kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true; exit 0; "
                    "  fi; "
                    "  if ! kill -0 $pid 2>/dev/null; then wait $pid; exit $?; fi; "
                    "  sleep 2; "
                    "done; "
                    "kill $pid 2>/dev/null || true; exit 1"
                ),
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
