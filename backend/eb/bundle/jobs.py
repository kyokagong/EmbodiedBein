"""Freeze generic Bundle Job inputs and register checked Rust publications."""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from eb.domain.contract import EmbodimentEnv
from eb.gates.engine import g3_model, g4_export
from eb.ids import new_id, sha256_text
from eb.settings import get_settings
from eb.store.ebd import EbdClient, storage_root
from eb.store.models import Artifact, Bundle, LineageEdge, ModelVersion
from eb.bundle.worker import MEMBERS


def fingerprint(plan: dict) -> str:
    return sha256_text(json.dumps({key:value for key,value in plan.items() if key != "hash"}, sort_keys=True, separators=(",", ":")))


async def freeze_bundle(session, spec, env: EmbodimentEnv):
    if env.id.startswith(("ee_rpo_v2_ufo_native_", "ee_rpo_v2_pbfm_")) or "native" in env.id or spec.resources.gpus:
        raise ValueError("generic bundle requires a CPU ONNX model; native simulation bundles use their dedicated flow")
    model = await session.get(ModelVersion, spec.inputs.base_model) if spec.inputs.base_model else None
    artifact = await session.get(Artifact, model.artifact_id) if model else None
    if (not model or model.embodiment_env_id != env.id or not artifact or artifact.kind != "onnx"
            or (model.card or {}).get("simulation_only") is True or not 0 < artifact.size_bytes <= 2 * 1024**3):
        raise ValueError("bundle input must reference a registered compatible ONNX ModelVersion")
    path = Path(artifact.storage)
    identity = await EbdClient().hash_storage_file(storage_root(path.parent), path.name)
    if identity["sha256"] != artifact.sha256 or identity["bytes"] != artifact.size_bytes:
        raise ValueError("bundle registered model identity differs")
    deploy = spec.params.get("deploy") or {"num_observations":env.observation.dim,
        "num_actions":env.action.dim,"action_scale":env.action.scale}
    if not g4_export(deploy, env).passed:
        raise ValueError("bundle G4 deployment configuration differs")
    manifest = {"embodiment_env_id":env.id,"io_signature":{"obs":env.observation.dim,"act":env.action.dim},
        "gates_passed":["G3", "G4", "numerical_check"],
        "g3":g3_model(env.observation.dim,env.action.dim,env).model_dump(),
        "g4":g4_export(deploy, env).model_dump(),
        "safety":{"action_clip":(env.action.scale if env.action.scale is not None else .25)*1.5,
                  "max_cmd_vel":[1.0,.5,.8],"watchdog_ms":100,"fallback":"freeze_or_damp"},
        "provenance":{"model_version_id":model.id,"artifact_id":artifact.id,"sha256":artifact.sha256}}
    plan = {"schema_version":"eb/bundle-plan/v1","model_version_id":model.id,"artifact_id":artifact.id,
        "onnx_path":str(path),"onnx_sha256":artifact.sha256,"onnx_bytes":artifact.size_bytes,
        "obs":env.observation.dim,"act":env.action.dim,"deploy":deploy,"manifest":manifest}
    if spec.image not in (None, "eb/tools:latest"):
        raise ValueError("bundle checks require the platform tools image")
    image = await EbdClient().image_info("eb/tools:latest")
    import re
    if image.get("ok") is not True or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(image.get("digest", ""))):
        raise ValueError("bundle tools image is unavailable")
    plan["image_digest"] = image["digest"]
    spec.image = image["digest"]
    plan["hash"] = fingerprint(plan)
    spec.params = {"bundle_plan":plan}
    spec.runtime = {"network":"none"}
    return spec


async def collect_bundle(session, job, run):
    plan = job.spec["params"]["bundle_plan"]
    reference = {"namespace":"run","run_id":run.id,"path":"outputs"}
    receipt = json.loads(await EbdClient().read_storage_text(reference, "bundle-checks.json", max_bytes=1024**2))
    if (not isinstance(receipt,dict) or not isinstance(receipt.get("numerical_check"),dict)
            or receipt.get("schema_version") != "eb/bundle-checks/v1" or receipt.get("status") != "complete"
            or receipt.get("plan_hash") != plan["hash"] or fingerprint(plan) != plan["hash"]
            or receipt.get("numerical_check", {}).get("passed") is not True
            or receipt.get("io_signature") != {"obs":plan["obs"],"act":plan["act"]}):
        raise ValueError("bundle model checks are incomplete or differ from the frozen plan")
    manifest = receipt.get("manifest")
    if (not isinstance(manifest, dict) or {k:v for k,v in manifest.items() if k not in {"numerical_check"}} != plan["manifest"]
            or manifest.get("numerical_check") != receipt["numerical_check"]):
        raise ValueError("bundle manifest differs from checked model")
    members = receipt.get("members")
    if (not isinstance(members,list) or len(members)!=len(MEMBERS)
            or any(not isinstance(item,dict) for item in members)
            or {item.get("path") for item in members}!=MEMBERS):
        raise ValueError("bundle members are missing or unbounded")
    policy = next(item for item in members if item["path"] == "policy.onnx")
    if policy.get("sha256") != plan["onnx_sha256"] or policy.get("bytes") != plan["onnx_bytes"]:
        raise ValueError("bundle policy differs from the frozen input")
    # Small generated metadata must bind the archived manifest to the receipt too.
    import yaml
    staging = {**reference,"path":"outputs/bundle-staging"}
    for name, expected in (("manifest.yaml", manifest), ("deploy.yaml", plan["deploy"]),
                           ("policy_meta.yaml",plan["deploy"]), ("gate_report.json",{"manifest":manifest})):
        text = await EbdClient().read_storage_text(staging, name)
        record = next(item for item in members if item["path"] == name)
        if (sha256_text(text) != record.get("sha256") or len(text.encode()) != record.get("bytes")
                or yaml.safe_load(text) != expected):
            raise ValueError("bundle staged metadata differs")
    existing = await session.scalar(select(Bundle).where(Bundle.manifest["run_id"].astext == run.id))
    if existing:
        run.metrics_summary = {"bundle_id":existing.id}
        return
    published = await EbdClient().pack_bundle(staging, members)
    artifact = Artifact(id=new_id("art"), run_id=run.id, kind="bundle", sha256=published["sha256"],
                        size_bytes=published["size_bytes"], storage=published["storage"], meta={"members":members})
    bundle = Bundle(id=new_id("bnd"), model_version_id=plan["model_version_id"], embodiment_env_id=job.embodiment_env_id,
        stage="S0", signed=False, manifest={**manifest,"schema_version":"eb/generic-bundle/v1","run_id":run.id,
        "cas":published["storage"],"can_sign":False,"download_url":f"/media/{published['sha256'][:2]}/{published['sha256']}"})
    session.add(artifact); session.add(bundle)
    session.add(LineageEdge(src_type="model_version",src_id=plan["model_version_id"],dst_type="bundle",dst_id=bundle.id,relation="packaged"))
    session.add(LineageEdge(src_type="run",src_id=run.id,dst_type="artifact",dst_id=artifact.id,relation="produced"))
    run.metrics_summary = {"bundle_id":bundle.id,"artifact_id":artifact.id}


async def validate_bundle_plan(spec):
    plan = spec.params.get("bundle_plan")
    if (not isinstance(plan, dict) or fingerprint(plan) != plan.get("hash")
            or spec.image != plan.get("image_digest") or spec.runtime != {"network":"none"} or spec.resources.gpus != 0
            or spec.embodiment_env_id != plan["manifest"]["embodiment_env_id"]):
        raise ValueError("bundle frozen plan or runtime differs")
    path = Path(plan["onnx_path"])
    identity = await EbdClient().hash_storage_file(storage_root(path.parent), path.name)
    if identity["sha256"] != plan["onnx_sha256"] or identity["bytes"] != plan["onnx_bytes"]:
        raise ValueError("bundle frozen input changed before admission")
