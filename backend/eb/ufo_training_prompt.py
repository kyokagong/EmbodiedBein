"""Frozen metadata for deriving task prompts from a managed UFO training run."""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from eb.ufo_native import ENV_ID, TASKS, plan_hash, read_json, safe_file, sha256_file

OPERATION = "ufo_native_prompt"
PLAN_SCHEMA = "eb/ufo-native-prompt-plan/v1"
PROMPT_SCHEMA = "eb/ufo-native-prompt-derivation/v1"
TRAINED_SCHEMA = "eb/rpo-ufo-native-trained-model/v1"
INFERENCE_FILES = (
    "config.json", "config.yaml", "effective_config.json", "checkpoint/train_status.json",
    "checkpoint/model/config.json", "checkpoint/model/init_kwargs.json", "checkpoint/model/model.safetensors",
)


async def trained_source(session, model_id):
    from eb.settings import get_settings
    from eb.store.models import Artifact, Job, ModelVersion, Run
    from eb.ufo_training_checkpoint import canonical_hash, read_native_checkpoint

    model = await session.get(ModelVersion, model_id)
    if (not model or model.embodiment_env_id != ENV_ID
            or model.card.get("schema_version") != TRAINED_SCHEMA
            or model.card.get("training_provenance") != "platform"):
        raise ValueError("Only a managed native trained model can derive new task prompts")
    artifact = await session.get(Artifact, model.artifact_id)
    resume = await session.get(Artifact, model.card.get("resume_artifact_id"))
    run = await session.get(Run, model.card.get("run_id"))
    job = await session.get(Job, run.job_id) if run else None
    if (not artifact or artifact.kind != "ufo_native_trained_model" or not resume
            or resume.kind != "ufo_native_resume" or not run or not job or job.kind != "train.rl"):
        raise ValueError("Managed native model training lineage or Replay artifact is missing")
    root = get_settings().data_root / "runs" / run.id
    checkpoint = read_native_checkpoint(root, run.id,
        plan=(job.spec.get("params") or {}).get("native_train_plan"))
    if (resume.storage != checkpoint["native_path"] or canonical_hash(checkpoint["manifest"]) != resume.sha256
            or artifact.storage != str(Path(checkpoint["path"]) / "model")
            or canonical_hash(artifact.meta.get("manifest") or {}) != artifact.sha256
            or model.card.get("model_sha256") != next(item["sha256"] for item in checkpoint["files"]
                if item["path"] == "checkpoint/model/model.safetensors")):
        raise ValueError("Managed model, training checkpoint and Replay identities differ")
    return model, artifact, resume, job, checkpoint


def source_path(value: str, job, checkpoint: dict) -> Path:
    """Map paths saved inside the training container back to its frozen inputs."""
    from eb.settings import get_settings

    params = job.spec.get("params") or {}
    native = Path(checkpoint["native_path"])
    run_root = native.parents[1]
    mappings = {
        "/sources/ufo": Path(params["ufo_source_ref"]["path"]),
        "/run/eb-output": run_root,
        str(get_settings().assets_root / "roboparty/rpo-v2"): Path(params["asset_source_ref"]["path"]),
    }
    path = Path(value)
    for prefix, root in mappings.items():
        if path.is_relative_to(prefix):
            relative = path.relative_to(prefix).as_posix()
            return safe_file(root, relative)
    raise ValueError("Native prompt auxiliary input is outside the frozen training sources")


async def prompt_plan(session, model_id, *, seed=71000, num_samples=16384):
    from eb.api.routes_ufo_native import scanned_file
    from eb.store.ebd import EbdClient

    model, artifact, resume, job, checkpoint = await trained_source(session, model_id)
    native = Path(checkpoint["native_path"])
    params = job.spec["params"]
    env = read_json(safe_file(native, "config.json"))["env"]
    xml = source_path(env["mjcf_path"], job, checkpoint)
    robot = source_path(env["robot_config_path"], job, checkpoint)
    tree = ET.parse(xml)
    if (tree.findall(".//include")
            or any("file" in node.attrib and node.tag != "mesh" for node in tree.iter())):
        raise ValueError("Native prompt XML contains an unregistered external asset")
    compiler = tree.getroot().find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    raw_xml = Path(env["mjcf_path"])
    meshroot = Path(meshdir) if Path(meshdir).is_absolute() else raw_xml.parent / meshdir
    motion = safe_file(Path(job.spec["inputs"]["datasets"][0]["path"]), "train.pkl")
    sources = [(xml, "assets/rpo.xml"), (robot, "assets/robot.yaml"), (motion, "sim-data/motion.pkl"),
               (safe_file(Path(params["ufo_source_ref"]["path"]), "LICENSE"), "licenses/UFO.txt"),
               # Native RewardInference checks the agent config before loading model files.
               (safe_file(native, "checkpoint/config.json"), "checkpoint/config.json")]
    for mesh in tree.findall(".//asset/mesh"):
        filename = mesh.get("file", "")
        sources.append((source_path(str(meshroot / filename), job, checkpoint), "assets/meshes/" + filename))
    if len(sources) > 180 or len({destination for _, destination in sources}) != len(sources):
        raise ValueError("Native prompt assets exceed the bounded unique input index")
    extras = await asyncio.gather(*(scanned_file(path) for path, _ in sources))
    for item, (_, destination) in zip(extras, sources):
        item["path"] = destination
    records = {item["path"]: item for item in checkpoint["files"]}
    files = [records[name] for name in INFERENCE_FILES]
    if sum(item["bytes"] for item in files + extras) > 2 * 1024**3:
        raise ValueError("Native inference package inputs exceed the 2 GiB budget")
    image = await EbdClient().image_info(checkpoint["image_digest"])
    if not image.get("ok") or image.get("digest") != checkpoint["image_digest"]:
        raise ValueError("Prompt derivation requires the frozen native training task image")
    plan = {
        "schema_version": PLAN_SCHEMA, "operation": OPERATION, "name": model.name,
        "model_id": model.id, "input_path": str(native), "source_run_id": checkpoint["run_id"],
        "source_artifact_sha256": artifact.sha256, "source_resume_artifact_sha256": resume.sha256,
        "source_model_sha256": model.card["model_sha256"],
        "native_manifest_sha256": sha256_file(safe_file(native, "checkpoint-manifest.json")),
        "source_refs": checkpoint["source_refs"],
        "source_fingerprint": checkpoint["compatibility_fingerprint"],
        "ufo_source_ref": params["ufo_source_ref"], "rlinf_source_ref": job.spec["code_refs"]["stack"],
        "image_digest": image["digest"], "files": files, "extras": extras, "codec": model.card["codec"],
        "buffer_rank": 0, "replay_id": checkpoint["replay_id"],
        "snapshot": checkpoint["replay"][0]["snapshot"],
        "seed": seed, "num_samples": num_samples, "n_inferences": 1, "cpu_workers": 4,
        "tasks": [task["id"] for task in TASKS],
    }
    plan["hash"] = plan_hash(plan)
    validate_prompt_plan(plan)
    return plan


def validate_prompt_plan(plan):
    if (plan.get("schema_version") != PLAN_SCHEMA or plan.get("operation") != OPERATION
            or plan.get("hash") != plan_hash(plan) or plan.get("buffer_rank") != 0
            or type(plan.get("seed")) is not int or not 0 <= plan["seed"] <= 1000000
            or type(plan.get("num_samples")) is not int or not 256 <= plan["num_samples"] <= 262144
            or plan["num_samples"] % 256 or plan.get("n_inferences") != 1 or plan.get("cpu_workers") != 4
            or plan.get("tasks") != [task["id"] for task in TASKS]
            or (plan.get("snapshot") or {}).get("id") != plan.get("replay_id")
            or (plan.get("snapshot") or {}).get("transition_format") != 2):
        raise ValueError("Native prompt budget, tasks, seed or Replay binding differs")
    for key in ("image_digest", "source_artifact_sha256", "source_resume_artifact_sha256",
                "source_fingerprint"):
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(plan.get(key, ""))):
            raise ValueError("Native prompt source/image identity is invalid")
    for key in ("source_model_sha256", "native_manifest_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(plan.get(key, ""))):
            raise ValueError("Native prompt model/manifest digest is invalid")
    return plan


def validate_prompt_job(spec):
    plan = validate_prompt_plan(spec.params.get("native_plan") or {})
    if (set(spec.params) != {"operation", "native_plan"} or spec.params["operation"] != OPERATION
            or spec.kind != "data.convert" or spec.embodiment_env_id != ENV_ID or spec.task_id
            or spec.image != plan["image_digest"] or spec.inputs.base_model != plan["input_path"]
            or spec.inputs.datasets or spec.inputs.resume_from or spec.overrides or spec.code_refs
            and (spec.code_refs.stack or spec.code_refs.embodiment)
            or spec.runtime != {"network": "none"} or spec.resources.gpus != 0
            or spec.resources.cpus != 4 or spec.resources.memory_gb != 64 or spec.resources.shm_gb != 2
            or spec.resources.devices or spec.policy.timeout_h != 1 or spec.policy.max_retries != 0
            or spec.fanout or spec.outputs):
        raise ValueError("Native prompt job differs from frozen offline resources and model")
    return plan


def render_prompt(spec, run_id):
    from eb.adapters.base import RenderedJob
    from eb.settings import get_settings

    plan = validate_prompt_job(spec)
    root = get_settings().data_root / "runs" / run_id
    if run_id not in ("run_preview", "run_validate"):
        if any(part.is_symlink() for part in (root, *root.parents)):
            raise ValueError("Native prompt Run path contains a symlink")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "native-plan.json"
        encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
        if path.is_symlink() or path.exists() and path.read_text() != encoded:
            raise ValueError("Native prompt frozen Run plan differs")
        path.write_text(encoded)
    mounts = [(str(root / "outputs"), "/outputs", "rw"),
              (str(root / "native-plan.json"), "/plan.json", "ro"),
              (plan["input_path"], "/training-source", "ro")]
    for name, source in (("ufo", plan["ufo_source_ref"]), ("rlinf", plan["rlinf_source_ref"])):
        path = Path(source["path"])
        mounts.extend(((str(path), f"/sources/{name}", "ro"),
                       (str(path.parent / "manifest.json"), f"/sources/{name}-manifest.json", "ro")))
    mounts.extend((item["source"], f"/source-extra/{index}", "ro")
                  for index, item in enumerate(plan["extras"]))
    return RenderedJob(image=spec.image,
        command=["python", "-m", "eb.runners.ufo_training_prompt", "--plan", "/plan.json"],
        env={"EB_RUN_ID": run_id, "PYTHONPATH": "/sources/ufo:/sources/rlinf:/opt/eb-native",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "osmesa",
             "PYOPENGL_PLATFORM": "osmesa", "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "TORCHINDUCTOR_CACHE_DIR": "/outputs/cache/torchinductor",
             "TRITON_CACHE_DIR": "/outputs/cache/triton", "XDG_CACHE_HOME": "/outputs/cache/xdg",
             "UFO_CACHE_DIR": "/outputs/cache", "TMPDIR": "/outputs/tmp", "WANDB_MODE": "disabled"},
        mounts=mounts, writable_paths=["outputs"], outputs=["outputs"], workdir="/outputs", metrics_glob="",
        gpus=0, shm_gb=2)


async def collect_prompt(session, job, run, outputs, plan, value):
    from sqlalchemy import select

    from eb.ids import new_id
    from eb.orchestrator.ufo_native import verify_index
    from eb.store.models import Artifact, LineageEdge, ModelVersion
    from eb.ufo_native import MODEL_SCHEMA

    root, prompt_root = outputs / "model", outputs / "prompts"
    manifest = read_json(safe_file(root, "manifest.json"))
    prompt_manifest = read_json(safe_file(prompt_root, "manifest.json"))
    derivation = prompt_manifest.get("derivation") or {}
    expected = {
        "schema_version": PROMPT_SCHEMA, "operation": OPERATION, "source_model_id": plan["model_id"],
        "source_model_sha256": plan["source_model_sha256"],
        "source_artifact_sha256": plan["source_artifact_sha256"],
        "source_resume_artifact_sha256": plan["source_resume_artifact_sha256"],
        "source_run_id": plan["source_run_id"], "snapshot": plan["snapshot"], "buffer_rank": 0,
        "seed": plan["seed"], "num_samples_per_inference": plan["num_samples"], "n_inferences": 1,
        "cpu_workers": 4, "process_executor": True, "tasks": plan["tasks"],
        "source_refs": plan["source_refs"], "image_digest": plan["image_digest"],
    }
    if (value.get("manifest") != manifest or value.get("prompt_manifest") != prompt_manifest
            or value.get("package") != "model" or manifest.get("schema_version") != "eb/ufo-native-import/v1"
            or manifest.get("loader_verified") is not True or manifest.get("codec") != plan["codec"]
            or manifest.get("source_model_id") != plan["model_id"]
            or manifest.get("training_provenance") != "platform"
            or manifest.get("prompt_derivation") != "native_reward_inference"
            or manifest.get("prompt_binding") != derivation
            or manifest.get("prompt_artifact_sha256") != plan_hash(prompt_manifest)
            or manifest.get("source_refs") != plan["source_refs"]
            or manifest.get("code_sources") != {"ufo": plan["ufo_source_ref"],
                                               "rlinf": plan["rlinf_source_ref"]}
            or any(derivation.get(key) != expected_value for key, expected_value in expected.items())
            or sum(item["bytes"] for item in manifest.get("files", [])) > 2 * 1024**3):
        raise ValueError("Derived prompts do not bind the frozen model, Replay, seed and sample budget")
    await verify_index(root, manifest["files"])
    await verify_index(prompt_root, prompt_manifest["files"])
    prompts = read_json(safe_file(root, "prompts.json"))
    prompt_records = {item["path"]: item for item in prompt_manifest["files"]}
    model_records = {item["path"]: item for item in manifest["files"]}
    if (set(prompt_records) != {"prompts.json", "derivation.json", "reward_locomotion.pkl"}
            or prompts.get("source_model_sha256") != plan["source_model_sha256"]
            or prompts.get("buffer_snapshot_id") != plan["replay_id"]
            or prompts.get("seed") != plan["seed"]
            or model_records.get("checkpoint/model/model.safetensors", {}).get("sha256")
            != plan["source_model_sha256"]
            or prompt_records["prompts.json"]["sha256"] != derivation.get("prompts_sha256")
            or model_records.get("prompts.json", {}).get("sha256") != derivation.get("prompts_sha256")
            or prompt_records["reward_locomotion.pkl"]["sha256"]
            != derivation.get("raw_native_prompt_sha256")):
        raise ValueError("Derived native prompt file identities differ")
    import math

    tasks = prompts.get("tasks") or {}
    if (set(tasks) != set(plan["tasks"]) or any(not isinstance(z, list) or len(z) != 256
            or any(type(number) not in (int, float) or not math.isfinite(number) for number in z)
            or sum(number * number for number in z) < 1e-12 for z in tasks.values())):
        raise ValueError("Derived native prompt vectors are invalid")
    artifacts = []
    for kind, payload, storage in (("ufo_native_prompt", prompt_manifest, prompt_root),
                                   ("ufo_native_model", manifest, root)):
        digest = plan_hash(payload)
        artifact = await session.scalar(select(Artifact).where(
            Artifact.sha256 == digest, Artifact.kind == kind))
        if artifact is None:
            artifact = Artifact(id=new_id("art"), kind=kind, sha256=digest, storage=str(storage),
                size_bytes=sum(item["bytes"] for item in payload["files"]), run_id=run.id, pinned=True,
                meta={"manifest": payload, "source_model_id": plan["model_id"], "simulation_only": True})
            session.add(artifact)
            await session.flush()
            session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="artifact",
                                    dst_id=artifact.id, relation="produced", run_id=run.id))
        artifacts.append(artifact)
    prompt_artifact, model_artifact = artifacts
    model = await session.scalar(select(ModelVersion).where(ModelVersion.artifact_id == model_artifact.id))
    if model is None:
        source = await session.get(ModelVersion, plan["model_id"])
        if source is None or source.card.get("schema_version") != TRAINED_SCHEMA:
            raise ValueError("Derived native prompt source model lineage is missing")
        model = ModelVersion(id=new_id("mv"), family_id=source.family_id, artifact_id=model_artifact.id,
            name=plan["name"] + " · native task prompts", embodiment_env_id=ENV_ID, provenance="full",
            io_signature={"obs": 375, "latent": 256, "actor_obs": 631, "act": 23},
            tags=["rpo-v2", "ufo-native", "simulation-only", "safetensors", "native-prompts"],
            card={"schema_version": MODEL_SCHEMA, "package_path": str(root), "run_id": run.id,
                  "source_model_id": source.id, "training_provenance": "platform",
                  "training_run_id": plan["source_run_id"], "prompt_derivation": "native_reward_inference",
                  "prompt_binding": derivation, "prompt_artifact_id": prompt_artifact.id,
                  "readiness": "prompt_derived", "qualification": "diagnostic", "simulation_only": True,
                  "hardware_verified": False, "codec": plan["codec"]})
        session.add(model)
        session.add(LineageEdge(src_type="model_version", src_id=source.id, dst_type="model_version",
                                dst_id=model.id, relation="prompt_derived_from", run_id=run.id))
        session.add(LineageEdge(src_type="artifact", src_id=prompt_artifact.id, dst_type="model_version",
                                dst_id=model.id, relation="conditioned_by", run_id=run.id))
    run.metrics_summary = {"model_version_id": model.id, "artifact_id": model_artifact.id,
        "prompt_artifact_id": prompt_artifact.id, "native_operation": OPERATION,
        "qualification": "diagnostic", "prompt_derivation": "native_reward_inference",
        "source_model_id": plan["model_id"], "seed": plan["seed"], "num_samples": plan["num_samples"]}
