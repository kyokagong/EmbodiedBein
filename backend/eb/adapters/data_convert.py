from __future__ import annotations

import os
from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.domain.job import JobSpec


def _download_env(settings, endpoint: str | None = None) -> dict[str, str]:
    env = {"HF_ENDPOINT": endpoint or settings.hf_endpoint, "HF_HUB_DISABLE_XET": "1"}
    env.update({key: os.environ[key] for key in ("http_proxy", "https_proxy", "no_proxy") if os.environ.get(key)})
    return env


class DataConvertAdapter(Adapter):
    name = "data_convert"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
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
        if spec.params.get("operation") in {"validate_lerobot_v3", "validate_rpo_amp_motion"}:
            from eb.paths import validate_allowed_path
            from eb.settings import get_settings
            source = validate_allowed_path(str(spec.params["src"]))
            amp = spec.params["operation"] == "validate_rpo_amp_motion"
            if amp and spec.runtime.get("network") != "none":
                raise ValueError("AMP motion validation must run without network")
            if amp and spec.embodiment_env_id != "ee_rpo_v2_amp_locomotion_23dof_v1":
                raise ValueError("AMP motion validation requires the RPO AMP interface")
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
