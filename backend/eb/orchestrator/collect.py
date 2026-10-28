"""Claim run outputs into CAS / models using measured ONNX I/O."""

from __future__ import annotations

import json
import math
from pathlib import Path
import math
import hashlib
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.contract import EmbodimentEnv
from eb.domain.fsm import JobStatus, RunStatus
from eb.eval.parse_play import parse_play
from eb.gates.engine import g3_model
from eb.ids import new_id
from eb.onnx_io import read_signature, sha256_file
from eb.orchestrator.pbfm_training import collect_pbfm_training, read_pbfm_training_evidence
from eb.settings import get_settings
from eb.store.models import (
    Artifact,
    EmbodimentEnvRow,
    EvalReport,
    Job,
    LineageEdge,
    ModelFamily,
    ModelVersion,
    Run,
)


def robotwin_model_package(job: Job, run: Run, checkpoint: dict) -> dict:
    """Describe one immutable trained package without trusting a UI supplied path."""
    template_id = (job.spec.get("params") or {}).get("template_id", "")
    if not template_id.startswith("robotwin_") or not template_id.endswith("_sft"):
        raise ValueError("训练产物不是 RoboTwin SFT 模型")
    model_key = template_id.removeprefix("robotwin_").removesuffix("_sft")
    package_files = {item.get("path") for item in checkpoint.get("manifest", {}).get("files", [])}
    if (model_key == "pi05"
            and "model/physical-intelligence/robotwin/norm_stats.json" not in package_files):
        raise ValueError("RoboTwin π0.5 model package lacks normalization assets")
    eval_template = f"robotwin_{model_key}_eval"
    from eb.domain.rlinf_form import get_template
    template = get_template(eval_template)
    canonical = json.dumps(checkpoint["manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    blocker_code = template.blocked_reason.split(":", 1)[0] if template.blocked_reason else None
    return {
        "family_id": "mf_robotwin_" + re.sub(r"[^a-z0-9]+", "_", template.model_id or model_key),
        "family_name": template.model_id or model_key,
        "artifact": {"sha256": digest, "kind": "model_package", "size_bytes": checkpoint["bytes"],
                     "storage": checkpoint["path"], "run_id": run.id,
                     "meta": {"manifest": checkpoint["manifest"], "step": checkpoint["step"]}},
        "model": {"name": (job.spec.get("params") or {}).get("model_name") or job.spec.get("name") or f"{model_key}-{run.id}",
                  "embodiment_env_id": job.spec["embodiment_env_id"],
                  "io_signature": {"observation_profile":"robotwin_aloha_3cam_joint14_v1",
                                   "action_profile":"aloha_joint14_position_v1"},
                  "tags": ["robotwin", model_key, "sft", "model_package"], "provenance":"full",
                  "card": {"schema_version":"eb/robotwin-model-package/v1", "train_job_id":job.id,
                           "run_id":run.id, "step":checkpoint["step"], "template_id":template_id,
                           "eval_template_id":eval_template, "base_model":(job.spec.get("inputs") or {}).get("base_model"),
                           "readiness":"adapter_ready" if template.ready else "blocked",
                           "blocker_code":blocker_code, "blocker":template.blocked_reason,
                           "artifact_sha256":digest, "package_path":checkpoint["path"]}}}


async def promote_robotwin_model(session: AsyncSession, job: Job, run: Run, checkpoint: dict) -> ModelVersion:
    package = robotwin_model_package(job, run, checkpoint)
    artifact = await session.scalar(select(Artifact).where(Artifact.sha256 == package["artifact"]["sha256"]))
    if artifact:
        existing = await session.scalar(select(ModelVersion).where(ModelVersion.artifact_id == artifact.id))
        if existing:
            return existing
    else:
        artifact = Artifact(id=new_id("art"), **package["artifact"], pinned=True)
        session.add(artifact)
        await session.flush()
    family = await session.get(ModelFamily, package["family_id"])
    if not family:
        family = ModelFamily(id=package["family_id"], name=package["family_name"], family="vla")
        session.add(family)
        await session.flush()
    model = ModelVersion(id=new_id("mv"), family_id=family.id, artifact_id=artifact.id, **package["model"])
    session.add(model)
    session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="model_version",
                            dst_id=model.id, relation="produced", run_id=run.id))
    return model


async def promote_rpo_model(session: AsyncSession, job: Job, run: Run, checkpoint: dict) -> ModelVersion:
    template_id = (job.spec.get("params") or {}).get("template_id")
    amp = template_id == "rpo_v2_amp_stand_walk_ppo"
    flat = template_id == "rpo_v2_flat_ppo"
    if flat:
        from eb.rpo_flat import ROBOLAB_REVISION
        files = {item["path"] for item in checkpoint["manifest"]["files"]}
        normalization = json.loads((Path(checkpoint["path"]) / "actor-normalization.json").read_text())
        if (checkpoint["manifest"].get("template_id") != template_id
                or "model/actor-normalization.json" not in files
                or normalization.get("kind") != "empirical" or normalization.get("running_stats_in_weights") is not True
                or (checkpoint["manifest"].get("official_source") or {}).get("robolab_revision") != ROBOLAB_REVISION):
            raise ValueError("RPO-Flat model package lacks empirical normalization or official provenance")
    if amp:
        files = {item["path"] for item in checkpoint["manifest"]["files"]}
        if (checkpoint["manifest"].get("template_id") != template_id
                or "model/actor-normalization.json" not in files
                or len((job.spec.get("inputs") or {}).get("datasets") or []) != 1):
            raise ValueError("AMP model package lacks normalization or motion provenance")
    canonical = json.dumps(checkpoint["manifest"], sort_keys=True, separators=(",", ":"))
    digest = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    artifact = await session.scalar(select(Artifact).where(Artifact.sha256 == digest))
    if artifact:
        existing = await session.scalar(select(ModelVersion).where(ModelVersion.artifact_id == artifact.id))
        if existing:
            return existing
    else:
        artifact = Artifact(id=new_id("art"), sha256=digest, kind="rlinf_checkpoint",
                            size_bytes=checkpoint["bytes"], storage=checkpoint["path"], run_id=run.id,
                            meta={"manifest":checkpoint["manifest"], "step":checkpoint["step"]}, pinned=True)
        session.add(artifact)
        await session.flush()
    family_id = "mf_rpo_v2_flat" if flat else "mf_rpo_v2_amp_locomotion" if amp else "mf_rpo_v2_wbc"
    family = await session.get(ModelFamily, family_id)
    if not family:
        family = ModelFamily(id=family_id, name="RPO-Flat" if flat else "RPO V2 AMP Locomotion" if amp else "RPO V2 WBC", family="mlp")
        session.add(family)
        await session.flush()
    model = ModelVersion(
        id=new_id("mv"), family_id=family.id, artifact_id=artifact.id,
        name=(job.spec.get("params") or {}).get("model_name") or job.spec.get("name") or f"rpo-wbc-{run.id}",
        embodiment_env_id="ee_rpo_v2_flat_23dof_v1" if flat else "ee_rpo_v2_amp_locomotion_23dof_v1" if amp else "ee_rpo_v2_wbc_23dof_v1",
        io_signature={"observation_profile":"rpo_v2_flat_actor_history10_v1" if flat else "rpo_v2_amp_actor_history10_v1" if amp else "rpo_v2_wbc_actor_history10_v1", "obs":780 if amp or flat else 890,
                      "action_profile":"rpo_v2_flat_joint23_position_v1" if flat else "rpo_v2_joint23_position_v1", "act":23},
        tags=["roboparty", "rpo-v2", "flat", "rlinf", "checkpoint"] if flat else ["roboparty", "rpo-v2", "amp", "locomotion", "rlinf", "checkpoint"] if amp else ["roboparty", "rpo-v2", "wbc", "rlinf", "checkpoint"], provenance="full",
        card={"schema_version":"eb/rpo-flat-model/v1" if flat else "eb/rpo-amp-model/v1" if amp else "eb/rpo-wbc-model/v1", "train_job_id":job.id,
              "run_id":run.id, "step":checkpoint["step"],
              "template_id":template_id,
              **({"official_source":checkpoint["manifest"].get("official_source"),
                  "network":{"hidden_dims":[512,256,128], "activation":"elu"},
                  "normalization":"empirical"} if flat else {}),
              "environment_profile":(job.spec.get("params") or {}).get("environment_profile"),
              **({"warm_start": job.spec["params"]["warm_start"],
                  "cumulative_updates": job.spec["params"]["warm_start"]["source_cumulative_updates"] + checkpoint["step"]}
                 if flat and (job.spec.get("params") or {}).get("warm_start") else {}),
              **({"initial_pose":(job.spec.get("params") or {}).get("rpo_amp_initial_pose")} if amp else {}),
              **({"initial_logstd":(job.spec.get("params") or {}).get("rpo_amp_initial_logstd")} if amp else {}),
              **({"velocity_tracking_std":(job.spec.get("params") or {}).get("rpo_amp_velocity_tracking_std")} if amp else {}),
              **({"velocity_reward_weight":(job.spec.get("params") or {}).get("rpo_amp_velocity_reward_weight")} if amp else {}),
              **({"actuator_delay_steps":(job.spec.get("params") or {}).get("rpo_amp_actuator_delay_steps")} if amp else {}),
              **({"motion_prior": (job.spec.get("inputs") or {}).get("datasets")[0]} if amp else {}),
              "readiness":"simulation_unverified", "hardware_verified":False,
              "artifact_sha256":digest, "checkpoint_path":checkpoint["path"],
              "training":{"kind":job.kind, "image":job.spec.get("image"),
                          "resources":job.spec.get("resources") or {},
                          "overrides":job.spec.get("overrides") or []}},
    )
    session.add(model)
    session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="model_version",
                            dst_id=model.id, relation="produced", run_id=run.id))
    return model


async def read_native_training_evidence(root: Path, run_id: str, *, target_updates: int) -> dict:
    """Expose only indexed, verified evidence from a completed native segment."""
    from urllib.parse import quote

    from eb.orchestrator.ufo_native import verify_index
    from eb.ufo_native import finite_tree, read_json, safe_file

    outputs = root / "outputs"
    index_path = safe_file(outputs, "metrics-index.json")
    index = read_json(index_path)
    files = index.get("files")
    if (index.get("schema_version") != "eb/ufo-native-metrics-index/v1"
            or index.get("run_id") != run_id
            or index.get("decision") != "wait_for_independent_validation"
            or not isinstance(files, list) or not 1 <= len(files) <= 200
            or any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
                   or not re.fullmatch(r"analysis/[\w.-]+\.(?:png|csv|json)|native/(?:train_log\.txt|ufo_tracking_eval\.csv)",
                                       item["path"]) for item in files)):
        raise ValueError("Native training evidence index identity or file scope differs")
    names = {item["path"] for item in files}
    if (not {"analysis/summary.json", "analysis/train-log-snapshot.csv", "native/train_log.txt",
             "analysis/resources.png", "analysis/resource-snapshot.csv"}.issubset(names)
            or not any(re.fullmatch(r"analysis/training-\d+\.png", name) for name in names)):
        raise ValueError("Native training evidence lacks training or resource plots/raw snapshots")
    await verify_index(outputs, files)
    summary = read_json(safe_file(outputs, "analysis/summary.json"))
    if (summary.get("schema_version") != "eb/ufo-native-training-analysis/v1"
            or summary.get("run_id") != run_id or summary.get("target_updates") != target_updates
            or summary.get("decision") != index["decision"] or not finite_tree(summary)
            or (summary.get("resources") or {}).get("status") != "snapshot"):
        raise ValueError("Native training analysis summary identity or resource evidence differs")
    entries = [{"path": "metrics-index.json", "bytes": index_path.stat().st_size,
                "sha256": sha256_file(index_path)}, *files]
    return {"available": True, "run_id": run_id, "decision": index["decision"],
            "files": [{**item, "url": f"/api/v1/ufo-native/runs/{run_id}/files/{quote(item['path'], safe='/')}"}
                      for item in entries]}


async def collect_ufo_native_training(session: AsyncSession, job: Job, run: Run, root: Path) -> None:
    """Register native training state separately from the inference network files."""
    from eb.ufo_native import validate_config
    from eb.ufo_training_checkpoint import canonical_hash, read_native_checkpoint

    params = job.spec.get("params") or {}
    plan = params.get("native_train_plan") or {}
    checkpoint = read_native_checkpoint(root, run.id, plan=plan)
    manifest = checkpoint["manifest"]
    counts = checkpoint["counts"]
    if (manifest["plan_hash"] != canonical_hash({**plan, "run_id": run.id})
            or type(plan.get("target_updates")) is not int or counts["optimizer_steps"] < plan["target_updates"]):
        raise ValueError("Native training receipt differs from its frozen Run plan or target updates")
    evidence = await read_native_training_evidence(root, run.id, target_updates=plan["target_updates"])
    native = Path(checkpoint["native_path"])
    codec = validate_config(
        json.loads((native / "config.json").read_text()),
        json.loads((native / "effective_config.json").read_text()),
        json.loads((native / "checkpoint/model/config.json").read_text()),
        json.loads((native / "checkpoint/model/init_kwargs.json").read_text()),
    )
    model_files = [{**item, "path": item["path"].removeprefix("checkpoint/model/")}
                   for item in manifest["files"] if item["path"].startswith("checkpoint/model/")]
    inference_manifest = {
        "schema_version": "eb/ufo-native-trained-network/v1", "run_id": run.id,
        "training_provenance": "platform", "source_refs": manifest["source_refs"],
        "image_digest": manifest["image_digest"], "counts": counts, "codec": codec,
        "loader_verified": True, "files": model_files,
    }
    artifacts = []
    for kind, value, storage in (
        ("ufo_native_resume", manifest, native),
        ("ufo_native_trained_model", inference_manifest, native / "checkpoint/model"),
    ):
        digest = canonical_hash(value)
        artifact = await session.scalar(select(Artifact).where(Artifact.sha256 == digest, Artifact.kind == kind))
        if artifact is None:
            artifact = Artifact(
                id=new_id("art"), sha256=digest, kind=kind, run_id=run.id, pinned=True,
                storage=str(storage), size_bytes=sum(item["bytes"] for item in value["files"]),
                meta={"manifest": value, "step": checkpoint["step"], "counts": counts,
                      "simulation_only": True, "qualification": "incomplete"},
            )
            session.add(artifact)
            await session.flush()
            session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="artifact",
                                    dst_id=artifact.id, relation="produced", run_id=run.id))
        artifacts.append(artifact)
    resume_artifact, inference_artifact = artifacts
    model = await session.scalar(select(ModelVersion).where(ModelVersion.artifact_id == inference_artifact.id))
    if model is None:
        family_id = "mf_rpo_ufo_native"
        if await session.get(ModelFamily, family_id) is None:
            session.add(ModelFamily(id=family_id, name="RPO native UFO FB-CPR", family="mlp"))
            await session.flush()
        model = ModelVersion(
            id=new_id("mv"), family_id=family_id, artifact_id=inference_artifact.id,
            name=params.get("model_name") or job.spec.get("name") or f"ufo-native-{run.id}",
            embodiment_env_id=job.spec["embodiment_env_id"], provenance="full",
            io_signature={"obs": 375, "latent": 256, "actor_obs": 631, "act": 23},
            tags=["rpo-v2", "ufo-native", "rlinf", "safetensors", "simulation-only"],
            card={"schema_version": "eb/rpo-ufo-native-trained-model/v1", "training_provenance": "platform",
                  "train_job_id": job.id, "run_id": run.id, "template_id": params["template_id"],
                  "package_path": str(native / "checkpoint/model"), "checkpoint_path": checkpoint["path"],
                  "resume_artifact_id": resume_artifact.id, "resume_mode": checkpoint["mode"],
                  "full_state_finite_verified": checkpoint.get("full_state_finite_verified", False),
                  "resume_limitations": checkpoint["limitations"], "counts": counts,
                  "state_coverage": checkpoint["state_coverage"], "source_refs": manifest["source_refs"],
                  "image_digest": manifest["image_digest"], "codec": codec,
                  "compatibility_fingerprint": manifest["compatibility_fingerprint"],
                  "model_sha256": next(item["sha256"] for item in model_files if item["path"] == "model.safetensors"),
                  "reward_inference_replay": checkpoint["replay"][0], "prompt_derivation": "required",
                  "datasets": (job.spec.get("inputs") or {}).get("datasets") or [],
                  "readiness": "prompt_required", "qualification": "incomplete", "simulation_only": True,
                  "hardware_verified": False},
        )
        session.add(model)
        session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="model_version",
                                dst_id=model.id, relation="produced", run_id=run.id))
        for dataset in (job.spec.get("inputs") or {}).get("datasets") or []:
            if dataset.get("id"):
                session.add(LineageEdge(src_type="dataset", src_id=dataset["id"], dst_type="model_version",
                                        dst_id=model.id, relation="trained_on", run_id=run.id))
    run.metrics_summary = {
        **(run.metrics_summary or {}), "model_version_id": model.id,
        "resume_artifact_id": resume_artifact.id, "artifact_id": inference_artifact.id,
        "native_operation": "ufo_native_train", "resume_mode": checkpoint["mode"],
        "full_state_finite_verified": checkpoint.get("full_state_finite_verified", False),
        "qualification": "incomplete", **counts,
        "segment_optimizer_steps": counts["optimizer_steps"] - plan.get("start_updates", 0),
        "segment_local_env_steps": counts["local_time"] - plan.get("start_local_time", 0),
        "checkpoint": {key: checkpoint[key] for key in ("path", "bytes", "step", "mode")},
        "training_evidence": evidence,
    }


def validate_amp_case_evidence(rows: list, seeds: list, receipt: dict, output: Path,
                               record_video: bool) -> None:
    from eb.runners.rpo_wbc_artifact import case_success

    metrics = receipt.get("metrics") or {}
    if (len(rows) != len(seeds) or len(rows) != (receipt.get("cases") or {}).get("planned")
            or any(not isinstance(row, dict) or type(row.get("seed")) is not int for row in rows)
            or sorted(row["seed"] for row in rows) != sorted(seeds)
            or len({row.get("case_id") for row in rows}) != len(rows)):
        raise ValueError("AMP case set differs from the frozen evaluation plan")
    for row in rows:
        seed = row["seed"]
        scenario = "stand" if seed % 2 == 0 else "walk"
        if (row.get("case_id") != f"rpo-seed-{seed}"
                or row.get("task_id") != "tsk_rpo_amp_stand_walk"
                or row.get("scenario") != scenario
                or row.get("completed") is not True
                or type(row.get("episode_len")) is not int
                or not 1 <= row["episode_len"] <= 1000
                or any(type(row.get(key)) is not bool for key in ("success", "fall", "nan", "hard_limit"))
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                       for key in ("distance_m", "velocity_rmse"))):
            raise ValueError("AMP case evidence is malformed")
        expected = case_success("stand_walk", row["episode_len"], row["fall"], row["nan"],
                                row["hard_limit"], None, row["velocity_rmse"],
                                scenario=scenario, distance_m=row["distance_m"],
                                settings=receipt.get("environment_settings") or {})
        if row["success"] is not expected or row.get("status") != ("success" if expected else "task_failure"):
            raise ValueError("AMP case success conflicts with measured trajectory")
        if record_video:
            relative = f"seed_{seed}/0.mp4"
            video = output / "video/eval" / relative
            if (row.get("video_rel") != relative or video.is_symlink()
                    or not video.is_file() or video.stat().st_size == 0):
                raise ValueError("AMP requested video evidence is missing")
    if (metrics.get("n_trials") != len(rows)
            or metrics.get("successes") != sum(row["success"] for row in rows)
            or metrics.get("fall_count") != sum(row["fall"] for row in rows)
            or metrics.get("nan_count") != sum(row["nan"] for row in rows)
            or metrics.get("hard_limit_count") != sum(row["hard_limit"] for row in rows)
            or type(metrics.get("success_rate")) not in (int, float)
            or not math.isclose(metrics["success_rate"], metrics["successes"] / len(rows), abs_tol=1e-8)
            or type(metrics.get("velocity_rmse")) not in (int, float)
            or not math.isclose(metrics["velocity_rmse"],
                                sum(row["velocity_rmse"] for row in rows) / len(rows), abs_tol=1e-8)):
        raise ValueError("AMP evaluation summary differs from case evidence")


def validate_rpo_amp_probe_receipt(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("RPO AMP probe receipt is missing or invalid")
    receipt = json.loads(path.read_text())
    expected = {"schema_version":"eb/rpo-amp-isaac-probe/v1", "status":"passed",
                "num_envs":2, "actor_dim":780, "critic_dim":1390,
                "action_dim":23, "amp_frame_dim":74, "amp_window_frames":3,
                "partial_reset":True}
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("RPO AMP probe receipt did not pass the frozen numeric contract")
    return receipt


def validate_rpo_amp_motion_probe_receipt(path: Path, manifest_sha256: str) -> dict:
    receipt = validate_rpo_amp_probe_receipt(path)
    motion = receipt.get("motion") or {}
    if (motion.get("status") != "passed"
            or motion.get("dataset_manifest_sha256") != manifest_sha256
            or type(motion.get("clips_checked")) is not int or motion["clips_checked"] < 2
            or type(motion.get("max_key_body_error_m")) not in (int, float)
            or not math.isfinite(motion["max_key_body_error_m"])
            or motion["max_key_body_error_m"] > 0.03
            or motion.get("dynamic_replay_verified") is not False):
        raise ValueError("RPO AMP motion replay did not pass the frozen kinematic contract")
    return receipt


async def claim_outputs(session: AsyncSession, job: Job, run: Run) -> None:
    root = get_settings().data_root / "runs" / run.id
    outputs = root / "outputs"
    cas_root = get_settings().data_root / "cas"
    cas_root.mkdir(parents=True, exist_ok=True)

    if (job.kind == "train.rl"
            and (job.spec.get("params") or {}).get("template_id") == "rpo_ufo_fb_native"):
        await collect_ufo_native_training(session, job, run, root)
        return
    if (job.kind == "train.rl"
            and (job.spec.get("params") or {}).get("template_id") == "rpo_pbfm_fb_depth_native"):
        await collect_pbfm_training(session, job, run, root)
        return

    operation = (job.spec.get("params") or {}).get("operation")
    if operation == "probe_pbfm_runtime":
        from eb.pbfm_runtime_probe import collect as collect_pbfm_probe
        await collect_pbfm_probe(session, job, run, outputs)
        return
    if operation in {"prepare_pbfm_rpo_motion", "export_pbfm_policy", "evaluate_pbfm_terrain"}:
        from eb.pbfm_evaluation import collect_pbfm
        await collect_pbfm(session, job, run, outputs)
        return
    from eb.ufo_failure_probe import OPERATION as FAILURE_PROBE_OPERATION
    if operation == FAILURE_PROBE_OPERATION:
        from eb.ufo_failure_probe import collect as collect_failure_probe
        await collect_failure_probe(session, job, run, outputs)
        return
    from eb.ufo_native_baseline import OPERATION as BASELINE_OPERATION
    if operation == BASELINE_OPERATION:
        from eb.ufo_native_baseline_service import collect as collect_baseline
        await collect_baseline(session, job, run, outputs)
        return
    from eb.ufo_grounded_reference import OPERATION as GROUNDED_OPERATION
    if operation == GROUNDED_OPERATION:
        from eb.ufo_grounded_reference import collect_candidate
        await collect_candidate(session, job, run, outputs)
        return
    from eb.ufo_native import OPERATIONS as UFO_NATIVE_OPERATIONS
    if operation in UFO_NATIVE_OPERATIONS:
        from eb.orchestrator.ufo_native import collect_native
        await collect_native(session, job, run, outputs)
        return
    if job.kind == "asset.validate" and operation in {"probe_rpo_amp_env", "probe_rpo_amp_motion"}:
        if operation == "probe_rpo_amp_motion":
            dataset_id = str((job.spec.get("params") or {}).get("dataset_id", ""))
            manifest = get_settings().datasets_root / dataset_id / "content/manifest.json"
            if not re.fullmatch(r"ds_[0-9a-f]+", dataset_id) or manifest.is_symlink() or not manifest.is_file():
                raise ValueError("RPO AMP motion probe dataset is unavailable")
            receipt = validate_rpo_amp_motion_probe_receipt(
                outputs / "probe.json", hashlib.sha256(manifest.read_bytes()).hexdigest())
        else:
            receipt = validate_rpo_amp_probe_receipt(outputs / "probe.json")
        run.metrics_summary = {**(run.metrics_summary or {}), "probe":receipt}
        return

    train_template = (job.spec.get("params") or {}).get("template_id", "")
    if job.kind in {"train.rl", "train.sft", "train.rlft"} and train_template.startswith(("libero_", "robotwin_", "rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_")):
        from eb.checkpoints import checkpoint_model
        checkpoint = checkpoint_model(root, run.id, verify_hashes=train_template.startswith(("robotwin_", "rpo_v2_amp_", "rpo_v2_flat_")))
        if train_template.startswith("libero_") and "model/physical-intelligence/libero/norm_stats.json" not in {item["path"] for item in checkpoint["manifest"]["files"]}:
            raise ValueError("LIBERO checkpoint manifest lacks normalization assets")
        model = (await promote_robotwin_model(session, job, run, checkpoint)
                 if train_template.startswith("robotwin_") else
                 await promote_rpo_model(session, job, run, checkpoint)
                 if train_template in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_amp_stand_walk_ppo", "rpo_v2_flat_ppo"} else None)
        run.metrics_summary = {**(run.metrics_summary or {}),
            "checkpoint":{key:checkpoint[key] for key in ("path", "bytes", "step")},
            **({"model_version_id": model.id} if model else {})}
        return

    eval_manifest = root / "output" / "eval-manifest.json"
    params = job.spec.get("params") or {}
    template_id = params.get("template_id", "")
    embodiment_env_id = getattr(job, "embodiment_env_id", None) or job.spec.get("embodiment_env_id")
    if job.kind == "eval.sim" and template_id == "rpo_v2_flat_eval":
        from eb.rpo_flat import validate_flat_eval
        plan = params.get("rpo_eval_plan") or {}
        evidence = validate_flat_eval(root / "output", plan, run.id)
        existing = await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id))
        if existing:
            run.metrics_summary = {**existing.metrics, "eval_report_id": existing.id,
                                   "qualification": existing.gate_results.get("qualification")}
            return
        report = EvalReport(id=new_id("ev"), suite_id=plan["suite_id"],
            model_version_id=params["model_version_id"], run_id=run.id, metrics=evidence["metrics"],
            gate_results={key:evidence[key] for key in ("qualification", "validity", "receipt")})
        session.add(report)
        session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="eval_report",
                                dst_id=report.id, relation="produced", run_id=run.id))
        run.metrics_summary = {**report.metrics, "eval_report_id":report.id,
                              "qualification":evidence["qualification"]}
        return
    if job.kind in {"eval.sim", "eval.sim2sim"} and embodiment_env_id in {
            "ee_rpo_v2_wbc_23dof_v1", "ee_rpo_v2_amp_locomotion_23dof_v1"}:
        amp = embodiment_env_id == "ee_rpo_v2_amp_locomotion_23dof_v1"
        if not eval_manifest.is_file() or eval_manifest.is_symlink() or eval_manifest.stat().st_size > 1024 * 1024:
            raise ValueError("RPO 评估缺少有界结果凭证")
        receipt = json.loads(eval_manifest.read_text())
        from eb.rpo_wbc import validate_eval_receipt
        suite_id = params.get("suite_id") or ""
        criterion = ("stability" if suite_id.endswith("_stability_v1") else
                     "stand_walk" if suite_id.endswith(("_stand_walk_v1", "_stand_walk_v2")) else "tracking")
        errors = validate_eval_receipt(receipt, criterion=criterion, amp=amp)
        eval_plan = params.get("rpo_eval_plan") or {}
        if amp:
            evidence = root / "output" / "eval-cases.jsonl"
            onnx_file = root / "output" / "exported" / "policy.onnx"
            if (receipt.get("backend") != eval_plan.get("backend")
                    or eval_plan.get("schema_version") != "eb/rpo-amp-eval-plan/v1"
                    or not evidence.is_file() or evidence.is_symlink() or evidence.stat().st_size > 1024 * 1024
                    or not onnx_file.is_file() or onnx_file.is_symlink()
                    or sha256_file(onnx_file) != (receipt.get("onnx") or {}).get("sha256")):
                raise ValueError("AMP independent evaluation evidence is incomplete")
            try:
                rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                        for line in evidence.read_text().splitlines() if line]
            except (ValueError, TypeError) as exc:
                raise ValueError("AMP case evidence contains invalid JSON") from exc
            if receipt.get("record_video") is not eval_plan.get("record_video"):
                raise ValueError("AMP video mode differs from the frozen evaluation plan")
            validate_amp_case_evidence(rows, eval_plan.get("seeds") or [], receipt,
                                       root / "output", bool(eval_plan.get("record_video")))
        for key, value in (eval_plan.get("environment_settings") or {}).items():
            if (receipt.get("environment_settings") or {}).get(key) != value:
                errors.append(f"environment_settings.{key}")
        diagnostic = (amp or eval_plan.get("diagnostic_environment") or eval_plan.get("seed_override")
                      or eval_plan.get("profile") != "standard")
        qualification = "fail" if errors else "not_requested" if diagnostic else "pass"
        existing = await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id))
        if existing:
            run.metrics_summary = {**existing.metrics, "eval_report_id":existing.id,
                                   "qualification":existing.gate_results.get("qualification")}
            return
        report = EvalReport(
            id=new_id("ev"), suite_id=params.get("suite_id") or "suite_rpo_v2_wbc_sim2sim_v1",
            model_version_id=params.get("model_version_id"), run_id=run.id,
            metrics=receipt.get("metrics") or {},
            gate_results={"qualification":qualification, "validity":"complete",
                          "errors":errors, "receipt":receipt,
                          **({"naturalness_status":"not_measured", "suite_scope":"diagnostic"} if amp else {}),
                          "success_definition":(f"diagnostic_environment:{eval_plan.get('environment_settings')}"
                                                if eval_plan.get("diagnostic_environment") else
                                                "survive_1000_steps_without_fall_or_nan" if criterion == "stability"
                                                else "stand_within_0.5m_or_walk_forward_4m_upright_1000_steps"
                                                if criterion == "stand_walk" else
                                                "velocity_and_upper_tracking_without_fall_or_limit_violation")},
        )
        session.add(report)
        session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="eval_report",
                                dst_id=report.id, relation="produced", run_id=run.id))
        run.metrics_summary = {**report.metrics, "eval_report_id":report.id, "qualification":qualification}
        return
    if job.kind == "eval.sim" and template_id.startswith(("libero_", "robotwin_")):
        if not eval_manifest.is_file() or eval_manifest.is_symlink() or eval_manifest.stat().st_size > 1024 * 1024:
            raise ValueError("评估缺少有界结果凭证")
        receipt = json.loads(eval_manifest.read_text())
        metrics = libero_eval_metrics(receipt, run.id)
        plan = params.get("eval_plan")
        if plan:
            if receipt.get("plan_hash") != plan.get("hash"):
                raise ValueError("evaluation plan hash changed")
            rows = []
            for path in sorted((root / "output").glob("eval-episodes-rank*.jsonl")):
                if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                    raise ValueError("episode evidence is unsafe or too large")
                rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
            if plan.get("schema_version") == "eb/robotwin-eval-plan/v1":
                policy_path = root / "output/policy-receipt.json"
                if (policy_path.is_symlink() or not policy_path.is_file()
                        or policy_path.stat().st_size > 64 * 1024):
                    raise ValueError("RoboTwin 缺少模型加载凭证")
                policy = json.loads(policy_path.read_text())
                policy_sha = "sha256:" + hashlib.sha256(policy_path.read_bytes()).hexdigest()
                model_input = plan.get("model") or {}
                if (policy.get("schema_version") != "eb/robotwin-policy-receipt/v1"
                        or policy.get("status") != "loaded"
                        or policy.get("model_version_id") != model_input.get("model_version_id")
                        or policy.get("artifact_sha256") != model_input.get("artifact_sha256")
                        or policy.get("robotwin_content_hash") != (model_input.get("runtime") or {}).get("robotwin_content_hash")
                        or policy.get("evidence_patch_sha256") != (model_input.get("runtime") or {}).get("robotwin_evidence_patch_sha256")
                        or policy.get("patched_base_task_sha256") != (model_input.get("runtime") or {}).get("patched_base_task_sha256")
                        or policy.get("policy_worker_count") != 1
                        or policy.get("model_load_count") != 1
                        or policy.get("headless") is not True
                        or policy.get("offscreen_rendering") is not True
                        or policy.get("save_video") is not False
                        or type(policy.get("inference_micro_batch_size")) is not int
                        or policy.get("inference_micro_batch_size") <= 0
                        or any(row.get("policy_receipt_sha256") != policy_sha for row in rows)):
                    raise ValueError("RoboTwin 模型加载凭证与逐 case 证据不一致")
            metrics.update(validate_eval_episodes(plan, rows, metrics, run.id))
        existing = await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id)) if plan else None
        if existing:
            run.metrics_summary = {**existing.metrics, "eval_report_id": existing.id,
                                   "qualification": existing.gate_results.get("qualification")}
            return
        qualification = "not_requested" if plan else "not_qualified"
        model_version_id = ((plan.get("model") or {}).get("model_version_id") if plan else None)
        report = EvalReport(id=new_id("ev"), suite_id=params.get("suite_id") or "suite_" + job.spec["task_id"],
            model_version_id=model_version_id, run_id=run.id, metrics=metrics,
            gate_results={"qualification":qualification, "validity":"complete" if plan else "legacy_unverified",
                          "reason":"尚未执行发布验收", "receipt":receipt,
                          "plan_hash":plan.get("hash") if plan else None,
                          "protocol":{"runner":"rlinf", "image":job.spec.get("image"),
                                      "suite_spec":plan.get("suite_spec"),
                                      "profile":plan.get("profile"),
                                      "overrides":job.spec.get("overrides"),
                                      "inference":params.get("eval_source_inference")} if plan else None,
                          "success_definition":"at_least_once_in_completed_trajectory",
                          "model_input":(job.spec.get("inputs") or {}).get("base_model"),
                          "checkpoint":{"run_id":(job.spec.get("params") or {}).get("eval_checkpoint_run_id"),
                                        "step":(job.spec.get("params") or {}).get("eval_checkpoint_step")}})
        session.add(report)
        if plan:
            session.add(LineageEdge(src_type="job", src_id=job.id, dst_type="eval_report",
                                    dst_id=report.id, relation="produced", run_id=run.id))
            if plan.get("schema_version") == "eb/robotwin-eval-plan/v1" and model_version_id:
                model = await session.get(ModelVersion, model_version_id)
                if model:
                    profile = plan.get("profile")
                    readiness = "smoke_passed" if profile == "smoke" else "validated"
                    model.card = {**(model.card or {}), "readiness":readiness,
                                  "closed_loop_evidence":{"report_id":report.id,
                                                          "run_id":run.id,
                                                          "profile":profile,
                                                          "plan_hash":plan["hash"]}}
        run.metrics_summary = {**metrics, "eval_report_id":report.id, "qualification":qualification}
        return

    metrics_path = outputs / "metrics.json"
    if job.kind == "eval.sim" and metrics_path.exists():
        parsed = parse_play(metrics_path)
        report = EvalReport(
            id=new_id("ev"),
            suite_id=(job.spec.get("params") or {}).get("suite_id") or "suite_biped_locomotion_v2",
            model_version_id=(job.spec.get("inputs") or {}).get("base_model"),
            run_id=run.id,
            metrics={
                "success_rate": parsed["success_rate"],
                "per_seed": parsed["per_seed"],
                "mean_reward": parsed["mean_reward"],
            },
            gate_results={"g3": "pending"},
        )
        session.add(report)
        run.metrics_summary = {k: v for k, v in report.metrics.items() if k != "per_seed"}

    search_roots = [p for p in (outputs, root / "output", root / "logs",
                                get_settings().models_root / "runs" / run.id) if p.exists()]
    if not search_roots:
        return
    onnx_files: list[Path] = []
    for base in search_roots:
        onnx_files.extend(base.rglob("exported/policy.onnx"))
        onnx_files.extend(p for p in base.rglob("*.onnx") if "exported" not in p.parts or p.name != "policy.onnx")
    # unique
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in onnx_files:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    ckpt_files: list[Path] = []
    for base in search_roots:
        ckpt_files.extend(base.rglob("model_*.pt"))
    ckpt_seen: set[str] = set()
    ckpt_uniq: list[Path] = []
    for p in ckpt_files:
        k = str(p)
        if k not in ckpt_seen:
            ckpt_seen.add(k)
            ckpt_uniq.append(p)
    ckpt_files = ckpt_uniq
    if not uniq and not ckpt_files:
        return

    fam = await session.get(ModelFamily, "mf_run")
    if fam is None:
        fam = ModelFamily(id="mf_run", name="run policies", family="mlp")
        session.add(fam)
        await session.flush()
    ctr_row = await session.get(EmbodimentEnvRow, job.embodiment_env_id)
    ctr = EmbodimentEnv.model_validate(ctr_row.spec) if ctr_row else None

    for path in uniq:
        try:
            digest = sha256_file(path)
            sig = read_signature(path)
        except Exception:
            job.status = JobStatus.FAILED.value
            job.error = f"onnx unreadable: {path.name}"
            run.status = RunStatus.ABORTED.value
            run.abort_reason = "g3_unreadable"
            return
        if ctr:
            gate = g3_model(sig.get("obs"), sig.get("act"), ctr)
            if not gate.passed:
                job.status = JobStatus.FAILED.value
                job.error = "; ".join(v.hint or str(v.path) for v in gate.violations)
                run.status = RunStatus.ABORTED.value
                run.abort_reason = "g3_failed"
                run.metrics_summary = {"g3": gate.model_dump()}
                return
        dest_dir = cas_root / digest[:2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / digest
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        existing = await session.scalar(select(Artifact).where(Artifact.sha256 == digest))
        if existing:
            art = existing
        else:
            art = Artifact(
                id=new_id("art"),
                sha256=digest,
                kind="onnx",
                size_bytes=path.stat().st_size,
                storage=str(dest),
                run_id=run.id,
                meta={"name": path.name, **sig},
            )
            session.add(art)
            await session.flush()
        mv = ModelVersion(
            id=new_id("mv"),
            family_id=fam.id,
            name=(job.spec.get("params") or {}).get("model_name") or path.stem,
            embodiment_env_id=job.embodiment_env_id,
            io_signature={"obs": sig.get("obs"), "act": sig.get("act")},
            provenance="full",
            artifact_id=art.id,
            tags=["policy", "run"],
            card={"source_path": str(path), "measured": True},
        )
        session.add(mv)
        await session.flush()
        session.add(
            LineageEdge(
                src_type="run",
                src_id=run.id,
                dst_type="model",
                dst_id=mv.id,
                relation="produced",
                run_id=run.id,
            )
        )
    for path in ckpt_files:
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        dest_dir = cas_root / digest[:2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / digest
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        session.add(
            Artifact(
                id=new_id("art"),
                sha256=digest,
                kind="checkpoint",
                size_bytes=path.stat().st_size,
                storage=str(dest),
                run_id=run.id,
                meta={"name": path.name, "folded": True},
            )
        )


def libero_eval_metrics(receipt: dict, run_id: str) -> dict:
    """Normalize only RLinf's measured completed-trajectory LIBERO statistics."""
    if (not isinstance(receipt, dict) or receipt.get("run_id") != run_id
            or receipt.get("status") != "not_qualified"):
        raise ValueError("LIBERO evaluation receipt does not match the run or qualification state")
    raw = receipt.get("metrics")
    if (not isinstance(raw, dict) or not raw
            or any(type(value) not in (int, float) or (type(value) is float and not math.isfinite(value))
                   for value in raw.values())):
        raise ValueError("LIBERO evaluation needs nonempty finite scalar metrics")
    count = raw.get("num_trajectories")
    success = raw.get("success_once")
    if type(count) is not int or count <= 0:
        raise ValueError("LIBERO evaluation needs a positive measured trajectory count")
    if success is None or not 0 <= success <= 1:
        raise ValueError("LIBERO evaluation needs success_once in [0, 1]")
    # Upstream concatenates newly-done trajectory records before averaging success_once.
    # Keep the untouched raw receipt in gate_results; this alias is not success-at-end.
    return {**raw, "success_rate": success, "n_trials": count}


def validate_libero_episodes(plan: dict, episodes: list[dict], metrics: dict, run_id: str | None = None) -> dict:
    """Reject a green aggregate unless every planned case has one measured outcome."""
    expected = {case["case_id"]: case for case in plan.get("cases", [])}
    if (len(expected) != plan.get("planned") or len(episodes) != len(expected)
            or metrics.get("n_trials") != len(expected)):
        raise ValueError("LIBERO planned and measured case counts differ")
    seen, successes, per_task = set(), 0, {}
    for row in episodes:
        if not isinstance(row, dict) or row.get("case_id") not in expected or row["case_id"] in seen:
            raise ValueError("LIBERO episode is missing, duplicated or outside the plan")
        case = expected[row["case_id"]]
        if (type(row.get("success")) is not bool
                or (run_id is not None and (row.get("run_id") != run_id or row.get("plan_hash") != plan["hash"]))
                or row.get("status") != ("success" if row["success"] else "task_failure")
                or row.get("task_id") != case["task_id"]
                or row.get("init_state_id") != case["init_state_id"]):
            raise ValueError("LIBERO episode outcome conflicts with its planned case")
        seen.add(row["case_id"])
        successes += row["success"]
        task = per_task.setdefault(str(case["task_id"]), {"successes": 0, "n_trials": 0})
        task["successes"] += row["success"]
        task["n_trials"] += 1
    if abs(successes / len(expected) - metrics["success_rate"]) > 1e-5:
        raise ValueError("LIBERO aggregate success conflicts with episode evidence")
    return {"planned": len(expected), "completed": len(episodes), "successes": successes,
            "per_task": per_task, "profile": plan["profile"], "validity": "complete"}


def validate_eval_episodes(plan: dict, episodes: list[dict], metrics: dict,
                           run_id: str | None = None) -> dict:
    """Validate complete per-case evidence for a supported frozen benchmark plan."""
    if plan.get("schema_version") == "eb/libero-eval-plan/v1":
        return validate_libero_episodes(plan, episodes, metrics, run_id)
    if plan.get("schema_version") != "eb/robotwin-eval-plan/v1":
        raise ValueError("未知评估计划协议")
    expected = {case["case_id"]: case for case in plan.get("cases", [])}
    if (len(expected) != plan.get("planned") or len(episodes) != len(expected)
            or metrics.get("n_trials") != len(expected)):
        raise ValueError("RoboTwin 计划与实测 case 数不一致")
    seen: set[str] = set()
    successes = task_failures = timeouts = 0
    per_task: dict[str, dict] = {}
    valid_statuses = {"success", "task_failure", "timeout"}
    for row in episodes:
        case = expected.get(row.get("case_id")) if isinstance(row, dict) else None
        status = row.get("status") if isinstance(row, dict) else None
        if case is None or row["case_id"] in seen or status not in valid_statuses:
            raise ValueError("RoboTwin episode 缺失、重复、未知或状态非法")
        if (run_id is not None and (row.get("run_id") != run_id or row.get("plan_hash") != plan["hash"])):
            raise ValueError("RoboTwin episode 与计划或 Run 不一致")
        validate_robotwin_episode(plan, row, case, run_id)
        seen.add(row["case_id"])
        successes += status == "success"
        task_failures += status == "task_failure"
        timeouts += status == "timeout"
        task = per_task.setdefault(case["task_id"], {"successes": 0, "n_trials": 0})
        task["successes"] += status == "success"
        task["n_trials"] += 1
    success_rate = successes / len(expected)
    if abs(success_rate - metrics.get("success_rate", -1)) > 1e-5:
        raise ValueError("RoboTwin aggregate success 与 episode 证据冲突")
    return {"planned": len(expected), "completed": len(episodes), "valid": len(episodes),
            "invalid": 0, "infra_failed": 0, "cancelled": 0, "not_started": 0,
            "successes": successes, "task_failures": task_failures, "timeouts": timeouts,
            "success_rate": success_rate, "per_task": per_task,
            "profile": plan["profile"], "validity": "complete"}


def validate_robotwin_episode(plan: dict, row: dict, case: dict, run_id: str | None) -> None:
    status = row.get("status")
    if (row.get("task_id") != case["task_id"] or row.get("scenario") != case["scenario"]
            or row.get("seed_planned") != case["seed"] or row.get("seed_actual") != case["seed"]
            or type(row.get("success_once")) is not bool
            or row["success_once"] != (status == "success")
            or (run_id is not None and (row.get("run_id") != run_id or row.get("plan_hash") != plan["hash"]))):
        raise ValueError("RoboTwin episode outcome 与计划冲突")
    for field in ("env_steps", "sim_steps", "chunk_requested", "chunk_executed"):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(f"RoboTwin episode {field} 非法")
    if row["chunk_executed"] > row["chunk_requested"]:
        raise ValueError("RoboTwin episode chunk 与计划冲突")
    task_config = next((item.get("task_config_sha256")
        for item in (plan.get("suite_spec") or {}).get("tasks", [])
        if item.get("task_id") == case["task_id"]), None)
    scene = {"task_id":case["task_id"], "scenario":case["scenario"],
             "seed":case["seed"], "instruction":row.get("instruction"),
             "task_config_sha256":task_config}
    scene_hash = "sha256:" + hashlib.sha256(json.dumps(
        scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if (row.get("task_config_sha256") != task_config or row.get("scene_hash") != scene_hash
            or not isinstance(row.get("instruction"), str) or not row["instruction"].strip()
            or not isinstance(row.get("media"), list)
            or not isinstance(row.get("sim_seconds"), (int, float))
            or not math.isfinite(row["sim_seconds"]) or row["sim_seconds"] < 0
            or row.get("action_index") != max(0, row["env_steps"] - 1)
            or row.get("termination_reason") != status
            or row.get("model_version_id") != (plan.get("model") or {}).get("model_version_id")):
        raise ValueError("RoboTwin episode 场景、指令或精确执行证据非法")


async def archive_partial_eval(session: AsyncSession, job: Job, run: Run, *, invalid: bool = False) -> None:
    """Keep measured cases when an eval stops before its aggregate receipt is complete."""
    if ((getattr(job, 'spec', None) or {}).get('params') or {}).get('operation') == 'evaluate_pbfm_terrain':
        from eb.orchestrator.pbfm_partial import archive_partial_pbfm
        await archive_partial_pbfm(session, job, run, invalid=invalid)
        return
    if ((getattr(job, 'spec', None) or {}).get('params') or {}).get('operation') == 'ufo_native_live':
        from eb.orchestrator.ufo_live import archive_partial_live
        await archive_partial_live(session, job, run, invalid=invalid)
        return
    if ((getattr(job, 'spec', None) or {}).get('params') or {}).get('operation') == 'ufo_native_eval':
        from eb.orchestrator.ufo_native_partial import archive_partial_native
        await archive_partial_native(session, job, run, invalid=invalid)
        return
    if getattr(job, "kind", None) != "eval.sim":
        return
    params = (getattr(job, "spec", None) or {}).get("params") or {}
    if params.get("template_id") == "rpo_v2_flat_eval":
        output = get_settings().data_root / "runs" / run.id / "output"
        if (output / "eval-manifest.json").is_file():
            await claim_outputs(session, job, run)
        return
    plan = ((getattr(job, "spec", None) or {}).get("params") or {}).get("eval_plan")
    if not isinstance(plan, dict):
        return
    if await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id)):
        return
    expected = {case["case_id"]: case for case in plan["cases"]}
    robotwin = plan.get("schema_version") == "eb/robotwin-eval-plan/v1"
    output = get_settings().data_root / "runs" / run.id / "output"
    found: dict[str, dict] = {}
    evidence_error = None
    try:
        if output.is_symlink():
            raise ValueError("评估输出目录为符号链接")
        for path in sorted(output.glob("eval-episodes-rank*.jsonl")):
            if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                raise ValueError("逐案例证据文件不可信")
            for line in path.read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                case = expected.get(row.get("case_id")) if isinstance(row, dict) else None
                success = row.get("success_once") if robotwin else row.get("success")
                status_ok = (row.get("status") in {"success", "task_failure", "timeout"}
                             if robotwin else row.get("status") == ("success" if success else "task_failure"))
                case_ok = (row.get("seed_planned") == case.get("seed") == row.get("seed_actual")
                           if robotwin else row.get("init_state_id") == case.get("init_state_id"))
                if (case is None or row["case_id"] in found or type(success) is not bool
                        or row.get("run_id") != run.id or row.get("plan_hash") != plan["hash"]
                        or not status_ok or success != (row.get("status") == "success")
                        or row.get("task_id") != case["task_id"] or not case_ok):
                    raise ValueError("逐案例证据重复或与计划不一致")
                if robotwin:
                    validate_robotwin_episode(plan, row, case, run.id)
                found[row["case_id"]] = row
    except (ValueError, OSError, TypeError, KeyError) as exc:
        evidence_error = str(exc)[:300]
        found.clear()
    successes = sum(row.get("success_once", row.get("success", False)) for row in found.values())
    timeouts = sum(row.get("status") == "timeout" for row in found.values())
    task_failures = sum(row.get("status") == "task_failure" for row in found.values())
    per_task: dict[str, dict] = {}
    for row in found.values():
        task = per_task.setdefault(str(row["task_id"]), {"successes": 0, "n_trials": 0})
        task["successes"] += row.get("success_once", row.get("success", False))
        task["n_trials"] += 1
    metrics = {"planned": len(expected), "completed": len(found), "n_trials": len(found),
               "not_started": len(expected) - len(found), "successes": successes,
               "task_failures": task_failures, "timeouts": timeouts,
               "success_rate": successes / len(found) if found else None,
               "per_task": per_task, "profile": plan["profile"]}
    validity = "invalid" if invalid or evidence_error else "incomplete"
    report = EvalReport(id=new_id("ev"), suite_id=plan["suite_id"],
                        model_version_id=((plan.get("model") or {}).get("model_version_id")),
                        run_id=run.id, metrics=metrics,
                        gate_results={"validity": validity, "qualification": "not_requested",
                                      "reason": job.error or run.abort_reason or "评估未完成",
                                      "evidence_error": evidence_error, "plan_hash": plan["hash"],
                                      "protocol":{"runner":"rlinf", "image":job.spec.get("image"),
                                                  "suite_spec":plan.get("suite_spec"),
                                                  "profile":plan.get("profile"),
                                                  "overrides":job.spec.get("overrides"),
                                                  "inference":(job.spec.get("params") or {}).get("eval_source_inference")},
                                      "checkpoint": plan.get("checkpoint"),
                                      "model": plan.get("model")})
    session.add(report)
    session.add(LineageEdge(src_type="job", src_id=job.id, dst_type="eval_report",
                            dst_id=report.id, relation="produced", run_id=run.id))
    run.metrics_summary = {**metrics, "eval_report_id": report.id, "validity": validity}
