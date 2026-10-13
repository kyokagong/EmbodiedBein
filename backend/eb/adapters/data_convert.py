from __future__ import annotations

import json
import os
from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobKind, JobSpec


def _download_env(settings, endpoint: str | None = None) -> dict[str, str]:
    env = {"HF_ENDPOINT": endpoint or settings.hf_endpoint, "HF_HUB_DISABLE_XET": "1"}
    env.update({key: os.environ[key] for key in ("http_proxy", "https_proxy", "no_proxy") if os.environ.get(key)})
    return env


class DataConvertAdapter(Adapter):
    name = "data_convert"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        if spec.params.get("operation") == "preview_rpo_motion":
            from eb.motion_preview import PREVIEW_IMAGE, PREVIEW_SCHEMA, read_motion_manifest
            from eb.paths import validate_allowed_path
            from eb.settings import get_settings

            if (spec.kind != JobKind.DATA_CONVERT or spec.image not in (None, PREVIEW_IMAGE)
                    or spec.resources.gpus != 0 or spec.runtime != {"network": "none"}
                    or len(spec.inputs.datasets) != 1 or spec.overrides
                    or spec.inputs.base_model or spec.inputs.resume_from
                    or spec.params.get("preview_schema") != PREVIEW_SCHEMA):
                raise ValueError("Motion preview requires an offline CPU dataset Job")
            binding = spec.inputs.datasets[0]
            if binding.get("id") != spec.params.get("dataset_id"):
                raise ValueError("Motion preview dataset binding differs")
            source = validate_allowed_path(binding["path"])
            read_motion_manifest(source, spec.params["manifest_sha256"])
            output = get_settings().data_root / "runs" / run_id / "outputs"
            return RenderedJob(image=PREVIEW_IMAGE,
                command=["python", "-m", "eb.runners.motion_preview", "--source", "/motion",
                         "--output", "/outputs/motion-preview", "--manifest-sha256", spec.params["manifest_sha256"]],
                mounts=[(str(source), "/motion", "ro"), (str(output), "/outputs", "rw")],
                env={"PYTHONDONTWRITEBYTECODE": "1"}, workdir="/workspace", gpus=0, shm_gb=1,
                writable_paths=[])
        if spec.params.get("operation")=="retarget_g1_lafan_pool":
            import hashlib,re
            from eb.settings import get_settings
            settings=get_settings()
            did=str(spec.params.get("dataset_id","")); sid=str(spec.params.get("source_run_id",""))
            image="eb/tools:rpo-g1-pool-v1"
            if (spec.resources.gpus!=0 or spec.runtime.get("network")!="none"
                    or spec.embodiment_env_id!="ee_rpo_v2_bfm_23dof_v1" or spec.image not in (None,image)
                    or not re.fullmatch(r"ds_[0-9a-f]+",did) or not re.fullmatch(r"run_[0-9a-f]+",sid)):
                raise ValueError("G1 pool requires fixed offline managed CPU resources")
            source=settings.data_root/"runs"/sid/"outputs/g1"
            receipt=source/"source-manifest.json"
            if (source.is_symlink() or receipt.is_symlink() or not receipt.is_file()
                    or hashlib.sha256(receipt.read_bytes()).hexdigest()!=spec.params.get("source_manifest_sha256")):
                raise ValueError("G1 pool source receipt hash mismatch")
            assets=settings.assets_root/"roboparty/rpo-v2"
            asset=assets/"asset-manifest.json"
            if json.loads(asset.read_text()).get("status")!="verified":
                raise ValueError("G1 pool asset is not frozen/verified")
            destination=settings.datasets_root/did/"content"
            output=settings.data_root/"runs"/run_id/"outputs"
            return RenderedJob(image=image,command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                "/workspace/eb_runner/g1_lafan_pool.py","--source","/workspace/source","--assets","/workspace/rpo-assets",
                "--output",str(destination),"--report","/outputs/dataset-index.json","--asset-sha256",hashlib.sha256(asset.read_bytes()).hexdigest()],
                mounts=[(str(source),"/workspace/source","ro"),(str(assets),"/workspace/rpo-assets","ro"),
                        (str(destination.parent),str(destination.parent),"rw"),(str(output),"/outputs","rw")],
                env={"PYTHONUNBUFFERED":"1"},workdir="/workspace",gpus=0,shm_gb=1,writable_paths=[])
        if spec.params.get("operation") == "retarget_g1_lafan_trial":
            import hashlib
            import re
            from eb.settings import get_settings
            settings = get_settings()
            did, source_run = str(spec.params.get("dataset_id", "")), str(spec.params.get("source_run_id", ""))
            if (spec.runtime.get("network") != "none" or spec.resources.gpus != 0
                    or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.image not in (None, "eb/tools:rpo-g1-r2r-v7")
                    or not re.fullmatch(r"ds_[0-9a-f]+", did) or not re.fullmatch(r"run_[0-9a-f]+", source_run)):
                raise ValueError("G1 R2R needs fixed offline CPU DatasetVersion resources")
            source = settings.data_root / "runs" / source_run / "outputs/g1-export"
            receipt = source / "export-receipt.json"
            assets = settings.assets_root / "roboparty/rpo-v2"
            asset = assets / "asset-manifest.json"
            if (source.is_symlink() or receipt.is_symlink() or not receipt.is_file()
                    or receipt.stat().st_size > 1024*1024
                    or hashlib.sha256(receipt.read_bytes()).hexdigest() != spec.params.get("source_export_sha256")
                    or asset.is_symlink() or not asset.is_file() or json.loads(asset.read_text()).get("status") != "verified"):
                raise ValueError("G1 R2R needs hash-bound export and verified RPO assets")
            destination = settings.datasets_root / did / "content"
            output = settings.data_root / "runs" / run_id / "outputs"
            return RenderedJob(image="eb/tools:rpo-g1-r2r-v7",
                command=["/workspace/isaaclab_extension_template/.venv/bin/python", "/workspace/eb_runner/rpo_g1_r2r_trial.py",
                    "--source", "/workspace/g1-export", "--assets", "/workspace/rpo-assets",
                    "--output", str(destination), "--report", "/outputs/dataset-index.json",
                    "--asset-sha256", hashlib.sha256(asset.read_bytes()).hexdigest()],
                env={"PYTHONUNBUFFERED":"1"}, workdir="/workspace", gpus=0, shm_gb=1, writable_paths=[],
                mounts=[(str(source), "/workspace/g1-export", "ro"), (str(assets), "/workspace/rpo-assets", "ro"),
                    (str(destination.parent), str(destination.parent), "rw"), (str(output), "/outputs", "rw")])
        if spec.params.get("operation") == "convert_rpo_ufo_robot_state":
            import re
            import hashlib
            from eb.settings import get_settings
            from eb.paths import validate_allowed_path
            from eb.domain.training_recipe import UFO_ISAAC_IMAGE_DIGEST
            from eb.rpo_bfm import CODEC_HASH, ENV_ID, ENV_V2_ID
            profile = spec.params.get("control_profile")
            if profile not in (None, "reachable_v2"):
                raise ValueError("Unsupported RobotState control profile")
            reachable = profile == "reachable_v2"
            if reachable:
                from eb.domain.training_recipe import UFO_REACHABLE_IMAGE_DIGEST
            image = "eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7" if reachable else "eb/rlinf:rpo-bfm-ufo-agent-v1"
            settings = get_settings()
            dataset_id = str(spec.params.get("dataset_id", ""))
            if (spec.kind != JobKind.DATA_CONVERT or spec.resources.gpus != 0
                    or spec.runtime != {"network": "none"} or len(spec.inputs.datasets) != 1
                    or not re.fullmatch(r"ds_[0-9a-f]+", dataset_id)
                    or spec.embodiment_env_id != (ENV_V2_ID if reachable else ENV_ID)
                    or spec.image not in (None, image)
                    or spec.overrides or spec.inputs.base_model or spec.inputs.resume_from
                    or spec.resources.model_dump() != {"gpus": 0, "gpu_mode": "exclusive",
                        "cpus": 2, "memory_gb": 4, "shm_gb": 1, "devices": []}
                    or not 0 < spec.policy.timeout_h <= 1 or spec.policy.max_retries != 0):
                raise ValueError("RobotState conversion requires fixed CPU/offline parent DatasetVersion")
            parent = spec.inputs.datasets[0]
            if not re.fullmatch(r"ds_[0-9a-f]+", str(parent.get("id", ""))) or parent["id"] == dataset_id:
                raise ValueError("RobotState requires a distinct registered parent DatasetVersion")
            source = validate_allowed_path(parent["path"])
            manifest = source / "manifest.json"
            validation = parent.get("validation") or {}
            asset = settings.assets_root / "roboparty/rpo-v2/asset-manifest.json"
            if (source.is_symlink() or manifest.is_symlink() or not manifest.is_file()
                    or validation.get("scope") != ("rpo_ufo_robot_state_v1" if reachable else "rpo_bfm_motion_v1")
                    or validation.get("status") != "validated"
                    or hashlib.sha256(manifest.read_bytes()).hexdigest() != validation.get("manifest_sha256")
                    or asset.is_symlink() or not asset.is_file()
                    or hashlib.sha256(asset.read_bytes()).hexdigest() != validation.get("asset_sha256")):
                raise ValueError("RobotState parent/asset hash differs from validated version")
            if reachable:
                from eb.runners.rpo_ufo_motion import validate_robot_state_manifest
                validate_robot_state_manifest(json.loads(manifest.read_text()),
                    asset_sha256=validation["asset_sha256"], expected_codec_sha256=CODEC_HASH)
            destination = settings.datasets_root / dataset_id / "content"
            output = settings.data_root / "runs" / run_id / "outputs"
            return RenderedJob(image=UFO_REACHABLE_IMAGE_DIGEST if reachable else UFO_ISAAC_IMAGE_DIGEST,
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                    "/workspace/eb_runner/rpo_ufo_motion.py", "--source-manifest", "/workspace/parent/manifest.json",
                    "--asset-manifest", "/workspace/assets/asset-manifest.json", "--output", str(destination),
                    "--report", "/outputs/dataset-index.json", *(["--control-profile", "reachable_v2"] if reachable else [])], env={"PYTHONDONTWRITEBYTECODE": "1"},
                mounts=[(str(source), "/workspace/parent", "ro"), (str(asset.parent), "/workspace/assets", "ro"),
                    (str(destination.parent), str(destination.parent), "rw"), (str(output), "/outputs", "rw")],
                workdir="/workspace", gpus=0, shm_gb=spec.resources.shm_gb, writable_paths=[])
        if spec.params.get("operation") in {"convert_rpo_bfm_official", "convert_rpo_bfm_local_pool", "convert_rpo_bfm_reference_pool", "convert_rpo_bfm_g1","convert_rpo_bfm_mixed_pool"}:
            import re
            from eb.settings import get_settings

            settings = get_settings()
            profile = {"convert_rpo_bfm_official":"smoke", "convert_rpo_bfm_local_pool":"local_pool", "convert_rpo_bfm_reference_pool":"reference_pool", "convert_rpo_bfm_g1":"g1_reference","convert_rpo_bfm_mixed_pool":"mixed_pool"}[spec.params["operation"]]
            dataset_id = str(spec.params.get("dataset_id", ""))
            if (spec.runtime.get("network") != "none"
                    or spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                    or spec.resources.gpus != 1
                    or not re.fullmatch(r"ds_[0-9a-f]+", dataset_id)):
                raise ValueError("BFM official motion conversion requires a valid offline DatasetVersion")
            source = Path("/data/kk/roboparty/roboparty_train/robolab/data/motions/rpo_lab")
            asset_root = settings.assets_root / "roboparty/rpo-v2"
            if profile != "smoke":
                source = asset_root / "robolab/data/motions/rpo_lab"
            extra_mounts,extra_args=[],[]
            if profile in {"g1_reference","mixed_pool"}:
                import hashlib
                source_id=str(spec.params.get("source_dataset_id",""))
                if not re.fullmatch(r"ds_[0-9a-f]+",source_id) or source_id==dataset_id:
                    raise ValueError("G1 codec needs a distinct managed source DatasetVersion")
                source=settings.datasets_root/source_id/"content"
                receipt=source/("pool-receipt.json" if profile=="mixed_pool" else "retarget-receipt.json")
                if (receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size>1024*1024
                        or hashlib.sha256(receipt.read_bytes()).hexdigest()!=spec.params.get("source_retarget_sha256")
                        or profile=="g1_reference" and json.loads(receipt.read_text()).get("calibration_verified") is not True):
                    raise ValueError("G1 codec requires hash-bound calibrated source")
                if profile=="mixed_pool":
                    official_id=str(spec.params.get("official_dataset_id",""))
                    if not re.fullmatch(r"ds_[0-9a-f]+",official_id) or official_id in {dataset_id,source_id}:
                        raise ValueError("Mixed codec requires distinct official source DatasetVersion")
                    official=settings.datasets_root/official_id/"content"
                    official_manifest=official/"manifest.json"
                    if (official.is_symlink() or official_manifest.is_symlink() or not official_manifest.is_file()
                            or hashlib.sha256(official_manifest.read_bytes()).hexdigest()!=spec.params.get("official_manifest_sha256")):
                        raise ValueError("Mixed codec official source manifest changed")
                    extra_mounts=[(str(official),"/workspace/official-view","ro")]
                    extra_args=["--official-view","/workspace/official-view"]
            asset = asset_root / "asset-manifest.json"
            if (source.is_symlink() or not source.is_dir() or asset.is_symlink() or not asset.is_file()
                    or json.loads(asset.read_text()).get("status") != "verified"):
                raise ValueError("BFM conversion needs the pinned local motion source and verified RPO assets")
            destination = settings.datasets_root / dataset_id / "content"
            output = settings.data_root / "runs" / run_id / "outputs"
            from eb.rpo_bfm import OFFICIAL_REFERENCE_IMAGE,G1_IMAGE
            default_image = G1_IMAGE if profile in {"g1_reference","mixed_pool"} else OFFICIAL_REFERENCE_IMAGE if profile == "reference_pool" else "eb/rlinf:rpo-v2-bfm-zero"
            if profile in {"g1_reference","mixed_pool"} and spec.image not in (None,G1_IMAGE):
                raise ValueError("G1 codec needs the versioned G1 BFM image")
            return RenderedJob(image=_image(spec, default_image),
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_bfm_motion.py", "--source", "/workspace/motions",
                         "--output", str(destination), "--report", "/outputs/dataset-index.json",
                         "--asset-manifest", "/workspace/rpo-assets/asset-manifest.json",
                         "--profile", profile,*extra_args],
                env={"PYTHONPATH":"/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_tasks:/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf",
                     "PYTHONUNBUFFERED":"1", "PYTHONDONTWRITEBYTECODE":"1"},
                writable_paths=[], workdir="/workspace/rlinf", gpus=1, shm_gb=8,
                mounts=[(str(source), "/workspace/motions", "ro"),
                        (str(asset_root), "/workspace/rpo-assets", "ro"),
                        (str(asset_root / "rpo_description"),
                         "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"),
                        (str(destination.parent), str(destination.parent), "rw"),
                        (str(output), "/outputs", "rw"),*extra_mounts])
        if spec.params.get("operation") == "audit_rpo_amp_official":
            from eb.settings import get_settings

            if (spec.runtime.get("network") != "none"
                    or spec.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1"):
                raise ValueError("official AMP audit requires the RPO interface and no network")
            source = Path("/data/kk/roboparty/roboparty_train/robolab/data/motions/rpo_lab")
            if not source.is_dir() or source.is_symlink():
                raise ValueError("official RPO motion source is missing")
            output = get_settings().data_root / "runs" / run_id / "outputs"
            return RenderedJob(image=_image(spec, "eb/tools:latest"),
                command=["python", "-m", "eb.rpo_amp_convert", "--source", "/workspace/motions",
                         "--audit", "--output", "/outputs/motion-audit.json"],
                env={}, writable_paths=[], workdir="/",
                mounts=[(str(source), "/workspace/motions", "ro"),
                        (str(output), "/outputs", "rw")])
        if spec.params.get("operation") == "convert_rpo_amp_official":
            import re
            from eb.settings import get_settings

            settings = get_settings()
            if spec.runtime.get("network") != "none":
                raise ValueError("official AMP pickle conversion must run without network")
            dataset_id = str(spec.params.get("dataset_id", ""))
            if (not re.fullmatch(r"ds_[0-9a-f]+", dataset_id)
                    or spec.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1"):
                raise ValueError("official AMP conversion has an invalid dataset identity")
            source = Path("/data/kk/roboparty/roboparty_train/robolab/data/motions/rpo_lab")
            destination = settings.datasets_root / dataset_id / "content"
            asset = settings.assets_root / "roboparty/rpo-v2/asset-manifest.json"
            if not source.is_dir() or source.is_symlink() or not asset.is_file() or asset.is_symlink():
                raise ValueError("official RPO motion source or verified asset receipt is missing")
            output = settings.data_root / "runs" / run_id / "outputs"
            return RenderedJob(image=_image(spec, "eb/tools:latest"),
                command=["python", "-m", "eb.rpo_amp_convert", "--source", "/workspace/motions",
                         "--destination", str(destination), "--asset-manifest", "/workspace/asset-manifest.json",
                         "--output", "/outputs/dataset-index.json"],
                env={}, writable_paths=[], workdir="/",
                mounts=[(str(source), "/workspace/motions", "ro"),
                        (str(asset), "/workspace/asset-manifest.json", "ro"),
                        (str(destination.parent), str(destination.parent), "rw"),
                        (str(output), "/outputs", "rw")])
        if spec.params.get("operation") == "acquire_robotwin":
            import re

            from eb.paths import validate_allowed_path
            from eb.robotwin import get_download_profile, raw_archive_profile
            from eb.settings import get_settings

            settings = get_settings()
            dataset_id = str(spec.params.get("dataset_id", ""))
            if not re.fullmatch(r"ds_[A-Za-z0-9]+", dataset_id):
                raise ValueError("invalid RoboTwin dataset ID")
            destination = validate_allowed_path(str(spec.params.get("dest", "")))
            if destination != settings.datasets_root / dataset_id / "content":
                raise ValueError("RoboTwin download must use its managed dataset directory")
            profile_id = str(spec.params.get("profile_id") or "")
            if profile_id.startswith("raw:"):
                profile = raw_archive_profile(str(spec.params.get("task")), str(spec.params.get("scenario")),
                                              str(spec.params.get("embodiment")))
                if profile.id != profile_id:
                    raise ValueError("RoboTwin raw profile does not match the frozen selection")
            else:
                profile = get_download_profile(profile_id)
            if spec.embodiment_env_id != profile.embodiment_env_id:
                raise ValueError("RoboTwin dataset and embodiment interface do not match")
            output = settings.data_root / "runs" / run_id / "outputs"
            command = ["python", "-m", "eb.robotwin_worker", "--dest", str(destination),
                       "--output", "/outputs/dataset-index.json", "--revision", profile.revision]
            if profile_id.startswith("raw:"):
                command += ["--task", str(profile.task), "--scenario", profile.scenario,
                            "--embodiment", profile.native_embodiment]
            else:
                command += ["--profile-id", profile.id]
            return RenderedJob(image=_image(spec, "eb/tools:latest"), command=command,
                env={**_download_env(settings), "EB_RUN_ID":run_id},
                writable_paths=[], workdir="/", mounts=[(str(destination.parent), str(destination.parent), "rw"),
                    (str(output), "/outputs", "rw")])
        if spec.params.get("operation") == "acquire_libero_benchmark":
            from eb.settings import get_settings
            from eb.paths import validate_allowed_path

            settings = get_settings()
            dataset_id = str(spec.params.get("dataset_id", ""))
            destination = validate_allowed_path(str(spec.params.get("dest", "")))
            if (not dataset_id.startswith("ds_") or not dataset_id[3:].isalnum()
                    or destination != settings.datasets_root / dataset_id / "bundle"
                    or spec.params.get("repo_id") != "RLinf/LIBERO-assets"
                    or spec.params.get("revision") != "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"):
                raise ValueError("LIBERO benchmark acquisition must use the managed pinned source")
            parent = destination.parent
            output = settings.data_root / "runs" / run_id / "outputs"
            return RenderedJob(image="eb/rlinf:libero-pi05",
                command=["python", "/workspace/eb_runner/libero_asset_worker.py",
                         "--dest", str(destination), "--output", "/outputs/dataset-index.json"],
                env=_download_env(settings),
                writable_paths=[], workdir="/workspace/rlinf", gpus=0,
                mounts=[(str(parent), str(parent), "rw"), (str(output), "/outputs", "rw")])
        if spec.params.get("operation") == "download_model":
            import re

            from eb.settings import get_settings
            from eb.model_sources import MODEL_SOURCES

            repo, revision = spec.params.get("repo_id", ""), spec.params.get("revision", "")
            model_id = spec.params.get("model_id", "")
            source_id = str(spec.params.get("source_id", ""))
            if (not re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
                    or not re.fullmatch(r"[0-9a-f]{40}", revision)
                    or not re.fullmatch(r"mdl_[0-9a-f]+", model_id)
                    or source_id not in MODEL_SOURCES
                    or MODEL_SOURCES[source_id]["repo_id"] != repo):
                raise ValueError("invalid public model revision")
            output = get_settings().data_root / "runs" / run_id / "outputs"
            model = get_settings().models_root / model_id
            return RenderedJob(
                image=_image(spec, "eb/tools:latest"),
                command=["python", "-m", "eb.model_worker", "--repo-id", repo,
                         "--revision", revision, "--output", "/outputs", "--model-dir", str(model),
                         "--source-id", source_id],
                env=_download_env(get_settings(), "https://huggingface.co"),
                writable_paths=[], workdir="/",
                mounts=[(str(output), "/outputs", "rw"), (str(model), str(model), "rw")],
            )
        if spec.params.get("operation") in {"validate_lerobot_v3", "validate_rpo_amp_motion", "validate_rpo_bfm_motion"}:
            from eb.paths import validate_allowed_path
            from eb.settings import get_settings
            source = validate_allowed_path(str(spec.params["src"]))
            amp = spec.params["operation"] == "validate_rpo_amp_motion"
            bfm = spec.params["operation"] == "validate_rpo_bfm_motion"
            if amp and spec.runtime.get("network") != "none":
                raise ValueError("AMP motion validation must run without network")
            if bfm and spec.runtime.get("network") != "none":
                raise ValueError("BFM motion validation must run without network")
            if amp and spec.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1":
                raise ValueError("AMP motion validation requires the RPO AMP interface")
            if bfm and spec.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1":
                raise ValueError("BFM motion validation requires the RPO BFM interface")
            if spec.params.get("repo_id") and not source.is_relative_to(get_settings().datasets_root):
                raise ValueError("HF output must be under managed datasets root")
            src = str(source)
            output = get_settings().data_root / "runs" / run_id / "outputs"
            command = ["python", "-m", "eb.data_worker", "--src", src,
                       "--output", "/outputs/dataset-index.json"]
            mounts = [(src, src, "rw" if spec.params.get("repo_id") and not spec.params.get("local_only") else "ro"),
                      (str(output), "/outputs", "rw")]
            if amp:
                asset = get_settings().assets_root / "roboparty/rpo-v2/asset-manifest.json"
                if asset.is_symlink() or not asset.is_file():
                    raise ValueError("AMP motion validation needs a verified RPO asset receipt")
                command += ["--amp-asset-manifest", "/workspace/rpo-asset-manifest.json"]
                mounts.append((str(asset), "/workspace/rpo-asset-manifest.json", "ro"))
            if bfm:
                asset_root = get_settings().assets_root / "roboparty/rpo-v2"
                asset = asset_root / "asset-manifest.json"
                if asset.is_symlink() or not asset.is_file():
                    raise ValueError("BFM motion validation needs a verified RPO asset receipt")
                command += ["--bfm-asset-manifest", "/workspace/rpo-asset-manifest.json"]
                mounts.append((str(asset), "/workspace/rpo-asset-manifest.json", "ro"))
            if spec.params.get("repo_id"):
                command += ["--repo-id", spec.params["repo_id"], "--revision", spec.params["revision"]]
                if spec.params.get("local_only"):
                    command += ["--local-only"]
            return RenderedJob(image=_image(spec, "eb/tools:latest"), command=command, env=_download_env(get_settings()),
                               writable_paths=[], workdir="/", mounts=mounts)
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
