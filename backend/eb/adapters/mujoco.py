from __future__ import annotations

from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec
from eb.domain.environment_profile import rpo_eval_arguments
from eb.settings import get_settings


class MujocoAdapter(Adapter):
    kind = JobKind.EVAL_SIM2SIM
    name = "mujoco"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        if spec.embodiment_env_id in {"ee_rpo_v2_wbc_23dof_v1", "ee_rpo_v2_amp_locomotion_23dof_v1"}:
            settings = get_settings()
            amp = spec.embodiment_env_id == "ee_rpo_v2_amp_locomotion_23dof_v1"
            if spec.params.get("rpo_checkpoint_eval"):
                checkpoint = Path(spec.inputs.base_model or "")
                if checkpoint.is_dir():
                    checkpoint /= "model_state_dict/full_weights.pt"
                assets = settings.assets_root / "roboparty/rpo-v2"
                mjcf = assets / "rpo_description/mjcf/rpo.xml"
                if not checkpoint.is_file() or not mjcf.is_file():
                    raise ValueError("RPO 评估缺少受管 checkpoint 或 MJCF")
                seeds = spec.params.get("seeds")
                if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 12
                        or any(type(seed) is not int or not 0 <= seed <= 1_000_000 for seed in seeds)
                        or len(seeds) != len(set(seeds))):
                    raise ValueError("RPO 评估 seeds 必须为 1–12 个不重复的非负整数")
                plan = spec.params.get("rpo_eval_plan") or {}
                if plan.get("implicit_seed_defaults") is True and plan.get("seed_override") is False:
                    expected = list(range(3 if plan.get("profile") == "smoke" else 12))
                    if seeds != expected:
                        raise ValueError("RPO 评估计划的默认 seed 与规模不一致")
                seed_args = (["--profile", plan["profile"]]
                             if (plan.get("implicit_seed_defaults") is True
                                 and plan.get("seed_override") is False
                                 and plan.get("profile") in ("smoke", "standard"))
                             else ["--seeds", ",".join(map(str, seeds))])
                return RenderedJob(
                    image=_image(spec, "eb/rlinf:rpo-v2-wbc"),
                    command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                             "/workspace/eb_runner/rpo_wbc_artifact.py", "eval-checkpoint-mujoco",
                             *(["--recipe", "amp"] if amp else []),
                             "--checkpoint", "/workspace/model/full_weights.pt",
                             "--mjcf", "/workspace/rpo/mjcf/rpo.xml",
                             "--output", "/workspace/output", *seed_args,
                             *(["--criterion", plan["criterion"]] if plan.get("criterion") in ("stability", "stand_walk") else []),
                             *rpo_eval_arguments(plan),
                             *(["--record-video"] if spec.params.get("record_video") else [])],
                    env={"MUJOCO_GL":"egl", "PYTHONUNBUFFERED":"1"}, writable_paths=["output"],
                    mounts=[(str(checkpoint), "/workspace/model/full_weights.pt", "ro"),
                            (str(assets / "rpo_description"), "/workspace/rpo", "ro"),
                            (str(settings.data_root / "runs" / run_id / "output"), "/workspace/output", "rw")],
                    gpus=1, shm_gb=spec.resources.shm_gb, workdir="/workspace",
                    outputs=["output"], metrics_glob="",
                )
            policy = Path(spec.inputs.base_model or "")
            if policy.is_dir():
                policy = policy / "policy.onnx"
            assets = settings.assets_root / "roboparty/rpo-v2"
            mjcf = assets / "rpo_description/mjcf/rpo.xml"
            if not policy.is_file() or not mjcf.is_file():
                raise ValueError("RPO MuJoCo 评估必须绑定实际 policy.onnx 和受管 MJCF")
            seeds = spec.params.get("seeds", list(range(12)))
            if not seeds or any(type(seed) is not int for seed in seeds):
                raise ValueError("RPO MuJoCo seeds 必须是非空整数列表")
            return RenderedJob(
                image=_image(spec, "eb/mujoco:latest"),
                command=["python", "/workspace/eb_runner/rpo_wbc_artifact.py", "eval-mujoco",
                         "--policy", "/workspace/model/policy.onnx", "--mjcf", "/workspace/rpo/mjcf/rpo.xml",
                         "--output", "/workspace/output", "--seeds", ",".join(map(str, seeds))],
                env={"MUJOCO_GL":"egl", "PYTHONUNBUFFERED":"1"}, writable_paths=["output"],
                mounts=[(str(policy), "/workspace/model/policy.onnx", "ro"),
                        (str(assets / "rpo_description"), "/workspace/rpo", "ro"),
                        (str(settings.data_root / "runs" / run_id / "output"), "/workspace/output", "rw")],
                gpus=0, workdir="/workspace", outputs=["output"], metrics_glob="",
            )
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
        if spec.params.get("operation") in {"probe_rpo_amp_env", "probe_rpo_amp_motion"}:
            import json
            import re
            from eb.settings import get_settings

            if (spec.kind != JobKind.ASSET_VALIDATE
                    or spec.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1"
                    or spec.resources.gpus != 1 or spec.runtime.get("network") != "none"
                    or spec.image not in (None, "eb/rlinf:rpo-v2-wbc")):
                raise ValueError("RPO AMP probe requires one isolated Isaac GPU and the pinned image")
            settings = get_settings()
            assets = settings.assets_root / "roboparty/rpo-v2"
            manifest = assets / "asset-manifest.json"
            if (assets.is_symlink() or manifest.is_symlink() or not manifest.is_file()
                    or json.loads(manifest.read_text()).get("status") != "verified"
                    or not (assets / "rpo_description").is_dir()):
                raise ValueError("RPO AMP probe requires verified managed assets")
            output = settings.data_root / "runs" / run_id / "outputs"
            motion = spec.params.get("operation") == "probe_rpo_amp_motion"
            initial_pose = spec.params.get("initial_pose")
            if initial_pose not in (None, "official_a1_stand_v1"):
                raise ValueError("RPO AMP probe initial pose is unsupported")
            dataset_id = str(spec.params.get("dataset_id", ""))
            dataset = settings.datasets_root / dataset_id / "content" if motion else None
            if motion and (not re.fullmatch(r"ds_[0-9a-f]+", dataset_id)
                           or dataset.is_symlink() or not (dataset / "manifest.json").is_file()):
                raise ValueError("RPO AMP motion probe needs a managed motion DatasetVersion")
            return RenderedJob(
                image="eb/rlinf:rpo-v2-wbc",
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_amp_probe.py", "--output", "/outputs/probe.json",
                         *(["--motion-dir", "/workspace/motion", "--asset-manifest",
                            "/workspace/rpo-assets/asset-manifest.json"] if motion else [])],
                env={"RPO_ASSET_ROOT":"/workspace/rpo-assets", "PYTHONUNBUFFERED":"1",
                     "PYTHONPATH":"/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf",
                     **({"EB_RPO_AMP_INITIAL_POSE":initial_pose} if initial_pose else {})},
                workdir="/workspace", gpus=1, shm_gb=spec.resources.shm_gb,
                mounts=[(str(assets), "/workspace/rpo-assets", "ro"),
                        (str(assets / "rpo_description"),
                         "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"),
                        *([(str(dataset), "/workspace/motion", "ro")] if motion else []),
                        (str(output), "/outputs", "rw")], writable_paths=[],
            )
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
