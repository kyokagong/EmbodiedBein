"""Claim run outputs into CAS / models using measured ONNX I/O."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.domain.contract import EmbodimentEnv
from eb.domain.fsm import JobStatus, RunStatus
from eb.eval.parse_play import parse_play
from eb.gates.engine import g3_model
from eb.ids import new_id
from eb.onnx_io import read_signature, sha256_file
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
    from eb.domain.training_recipe import execution_binding, RECIPES
    from eb.domain.job import JobSpec
    native_binding = (execution_binding(JobSpec.model_validate(job.spec))
        if (job.spec.get("params") or {}).get("execution") or template_id in RECIPES else None)
    if native_binding and (checkpoint["manifest"].get("execution") != native_binding.model_dump()
            or checkpoint["manifest"].get("format") != native_binding.checkpoint_format
            or checkpoint["step"] != job.spec["params"]["target_learner_updates"]):
        raise ValueError("Native model package differs from frozen recipe/target")
    amp = template_id == "rpo_v2_amp_stand_walk_ppo"
    flat = template_id == "rpo_v2_flat_ppo"
    bfm = template_id == "rpo_v2_bfm_zero_pretrain" or native_binding is not None
    from eb.rpo_bfm import ENV_ID, ENV_V2_ID, codec_hash_for_execution, action_control_for_execution, auxiliary_profile_for_execution
    bfm_env_id = RECIPES[template_id].embodiment_env_id if native_binding else ENV_ID
    action_profile = "rpo_v2_joint23_position_v2" if bfm and bfm_env_id == ENV_V2_ID else "rpo_v2_joint23_position_v1"
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
    if bfm:
        files = {item["path"] for item in checkpoint["manifest"]["files"]}
        normalization = json.loads((Path(checkpoint["path"]) / "actor-normalization.json").read_text())
        if (checkpoint["manifest"].get("template_id") != template_id
                or "model/actor-normalization.json" not in files
                or normalization != {"schema_version":"eb/rpo-bfm-normalization/v1",
                                    "kind":"frozen_running_stats", "observation_dim":733,
                                    "history_length":4, "latent_dim":256}
                or len((job.spec.get("inputs") or {}).get("datasets") or []) != 1):
            raise ValueError("BFM model package lacks frozen normalization or motion provenance")
        if (job.spec.get("params") or {}).get("bfm_update_target") is not None:
            progress = checkpoint["manifest"].get("bfm_progress") or {}
            if (progress.get("learner_updates") != job.spec["params"]["bfm_update_target"]
                    or "model/bfm-progress.json" not in files
                    or json.loads((Path(checkpoint["path"]) / "bfm-progress.json").read_text()) != progress):
                raise ValueError("BFM completed learner updates differ from frozen target")
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
    family_id = "mf_rpo_v2_bfm_zero" if bfm else "mf_rpo_v2_flat" if flat else "mf_rpo_v2_amp_locomotion" if amp else "mf_rpo_v2_wbc"
    family = await session.get(ModelFamily, family_id)
    if not family:
        family = ModelFamily(id=family_id, name="RPO V2 BFM-Zero" if bfm else "RPO-Flat" if flat else "RPO V2 AMP Locomotion" if amp else "RPO V2 WBC", family="mlp")
        session.add(family)
        await session.flush()
    model = ModelVersion(
        id=new_id("mv"), family_id=family.id, artifact_id=artifact.id,
        name=(job.spec.get("params") or {}).get("model_name") or job.spec.get("name") or f"rpo-bfm-{run.id}" if bfm else (job.spec.get("params") or {}).get("model_name") or job.spec.get("name") or f"rpo-wbc-{run.id}",
        embodiment_env_id=bfm_env_id if bfm else "ee_rpo_v2_flat_23dof_v1" if flat else "ee_rpo_v2_amp_locomotion_23dof_v1" if amp else "ee_rpo_v2_wbc_23dof_v1",
        io_signature={"observation_profile":"rpo_bfm_actor_history4_z256_v1" if bfm else "rpo_v2_flat_actor_history10_v1" if flat else "rpo_v2_amp_actor_history10_v1" if amp else "rpo_v2_wbc_actor_history10_v1", "obs":375 if bfm else 780 if amp or flat else 890,
                      **({"latent":256, "critic_obs":733} if bfm else {}),
                      "action_profile":action_profile, "act":23},
        tags=["roboparty", "rpo-v2", "bfm-zero", "rlinf", "checkpoint"] if bfm else ["roboparty", "rpo-v2", "flat", "rlinf", "checkpoint"] if flat else ["roboparty", "rpo-v2", "amp", "locomotion", "rlinf", "checkpoint"] if amp else ["roboparty", "rpo-v2", "wbc", "rlinf", "checkpoint"], provenance="full",
        card={"schema_version":"eb/rpo-bfm-model/v1" if bfm else "eb/rpo-flat-model/v1" if flat else "eb/rpo-amp-model/v1" if amp else "eb/rpo-wbc-model/v1", "train_job_id":job.id,
              "run_id":run.id, "step":checkpoint["step"],
              "template_id":template_id,
              **({"official_source":checkpoint["manifest"].get("official_source"),
                  "network":{"hidden_dims":[512,256,128], "activation":"elu"},
                  "normalization":"empirical"} if flat else {}),
              "environment_profile":(job.spec.get("params") or {}).get("environment_profile"),
              **({"execution": native_binding.model_dump(), "checkpoint_format": native_binding.checkpoint_format,
                  "progress_unit": native_binding.progress_unit,
                  "codec_sha256": codec_hash_for_execution(native_binding.model_dump()),
                  "action_control": action_control_for_execution(native_binding.model_dump()),
                  "auxiliary_profile": auxiliary_profile_for_execution(native_binding.model_dump())} if native_binding else {}),
              **({"warm_start": job.spec["params"]["warm_start"],
                  "cumulative_updates": job.spec["params"]["warm_start"]["source_cumulative_updates"] + checkpoint["step"]}
                 if flat and (job.spec.get("params") or {}).get("warm_start") else {}),
              **({"initial_pose":(job.spec.get("params") or {}).get("rpo_amp_initial_pose")} if amp else {}),
              **({"initial_logstd":(job.spec.get("params") or {}).get("rpo_amp_initial_logstd")} if amp else {}),
              **({"velocity_tracking_std":(job.spec.get("params") or {}).get("rpo_amp_velocity_tracking_std")} if amp else {}),
              **({"velocity_reward_weight":(job.spec.get("params") or {}).get("rpo_amp_velocity_reward_weight")} if amp else {}),
              **({"actuator_delay_steps":(job.spec.get("params") or {}).get("rpo_amp_actuator_delay_steps")} if amp else {}),
              **({"motion_prior": (job.spec.get("inputs") or {}).get("datasets")[0]} if amp else {}),
              **({"motion_prior": (job.spec.get("inputs") or {}).get("datasets")[0],
                  "learner_progress":checkpoint["manifest"].get("bfm_progress"),
                  "normalization":"frozen_running_stats",
                  "network":{"latent_dim":256, "history_length":4, "components":["actor", "forward_map", "backward_map", "discriminator", "critic", "aux_critic"]}} if bfm else {}),
              "readiness":"simulation_unverified", "hardware_verified":False,
              "artifact_sha256":digest, "checkpoint_path":checkpoint["path"],
              "training":{"kind":job.kind, "image":job.spec.get("image"),
                          "resources":job.spec.get("resources") or {},
                          "overrides":job.spec.get("overrides") or []}},
    )
    if template_id == "rpo_v2_bfm_zero_tracking_teacher":
        normalization = json.loads((Path(checkpoint["path"]) / "actor-normalization.json").read_text())
        if (checkpoint["manifest"].get("template_id") != template_id
                or normalization.get("schema_version") != "eb/rpo-tracking-normalization/v1"
                or normalization.get("observation_dim") != 547 or normalization.get("critic_dim") != 905
                or checkpoint["manifest"].get("tracking_reference") != job.spec["params"].get("tracking_reference")):
            raise ValueError("Tracking teacher package lacks its frozen codec/reference")
        family_id = "mf_rpo_v2_tracking_teacher"
        if not await session.get(ModelFamily, family_id):
            session.add(ModelFamily(id=family_id, name="RPO RLinf tracking teacher", family="mlp"))
            await session.flush()
        model.family_id = family_id
        model.embodiment_env_id = "ee_rpo_v2_tracking_23dof_v1"
        model.io_signature = {"observation_profile":"rpo_tracking_actor547_v1", "obs":547,
                              "critic_obs":905, "act":23, "action_profile":"rpo_reference_residual23_v1"}
        model.tags = ["roboparty", "rlinf", "tracking-teacher", "checkpoint"]
        model.card.update({"schema_version":"eb/rpo-tracking-model/v1",
                           "tracking_reference":job.spec["params"]["tracking_reference"],
                           "normalization":"empirical", "network":{"hidden_dims":[512,256,128], "activation":"elu"}})
    session.add(model)
    session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="model_version",
                            dst_id=model.id, relation="produced", run_id=run.id))
    return model


def validate_rpo_bfm_export(root: Path, job: Job) -> dict:
    params = job.spec.get("params") or {}
    source_run_id = str(params.get("source_run_id", ""))
    asset_sha256 = str(params.get("asset_sha256", ""))
    if (params.get("operation") != "rpo_bfm_export"
            or not re.fullmatch(r"run_[0-9a-f]+", source_run_id)
            or not re.fullmatch(r"[0-9a-f]{64}", asset_sha256)):
        raise ValueError("BFM export manifest has no pinned source Run or asset")
    output = Path(root) / "output/exported"

    def read_receipt(name: str) -> dict:
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValueError("BFM export manifest or prompt manifest is missing or invalid")
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("BFM export manifest JSON is invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("BFM export manifest JSON is invalid")
        return value

    receipt = read_receipt("export-manifest.json")
    prompt = read_receipt("prompt-manifest.json")
    datasets = (job.spec.get("inputs") or {}).get("datasets") or []
    if len(datasets) != 1:
        raise ValueError("BFM export manifest requires one pinned motion dataset")
    motion_manifest = Path(datasets[0].get("path", "")) / "manifest.json"
    checkpoint = Path(get_settings().models_root) / "runs" / source_run_id
    weights = checkpoint / "model_state_dict/full_weights.pt"
    config = Path(get_settings().data_root) / "runs" / source_run_id / "output/resolved-config.yaml"
    if params.get("model_execution"):
        from eb.checkpoints import checkpoint_model
        from eb.runners.bfm_ufo_checkpoint import digest, validate_ufo_checkpoint

        package = checkpoint_model(config.parent.parent, source_run_id)
        if package["manifest"].get("format") not in {"ufo_agent_state_v1", "ufo_agent_state_v2"}:
            raise ValueError("Native BFM export requires a native model package")
        relative = Path(package["manifest"]["native_checkpoint_path"]).relative_to("model")
        native = validate_ufo_checkpoint(checkpoint / relative, run_id=source_run_id, verify_hashes=False)
        if (native["execution"] != params["model_execution"]
                or receipt.get("execution") != native["execution"]
                or receipt.get("checkpoint_format") != native["format"]
                or receipt.get("learner_joint_updates") != native["step"]
                or receipt.get("native_manifest_sha256") != digest(checkpoint / relative / "resume-manifest.json")):
            raise ValueError("Native BFM export differs from its checkpoint binding/step")
        from eb.rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
        if receipt.get("action_control") != action_control_for_execution(native["execution"]):
            raise ValueError("Native BFM export action control differs from its checkpoint")
        if receipt.get("auxiliary_profile") != auxiliary_profile_for_execution(native["execution"]):
            raise ValueError("Native BFM export auxiliary profile differs from its checkpoint")
        weights = checkpoint / relative / "model/model.safetensors"
    for path in (motion_manifest, weights, config):
        if path.is_symlink() or not path.is_file():
            raise ValueError("BFM export source, checkpoint or motion manifest is missing")
    from eb.rpo_bfm import codec_hash_for_execution
    expected_codec_sha256 = codec_hash_for_execution(params.get("model_execution"))
    onnx_receipt = receipt.get("onnx") or {}
    prompt_receipt = receipt.get("prompts") or {}
    golden_receipt = receipt.get("golden") or {}
    command_receipt = receipt.get("command_latent") or {}
    onnx_path = output / "policy.onnx"
    command_path = output / "command-to-latent.onnx"
    prompts_path = output / "prompts.npz"
    golden_path = output / "golden-traces.npz"
    prompt_items = prompt.get("prompts")
    if (receipt.get("schema_version") != "eb/rpo-bfm-export/v1" or receipt.get("status") != "verified"
            or receipt.get("readiness") != "diagnostic_smoke"
            or receipt.get("model_weights_sha256") != sha256_file(weights)
            or receipt.get("resolved_config_sha256") != sha256_file(config)
            or receipt.get("codec_sha256") != expected_codec_sha256
            or receipt.get("asset_sha256") != asset_sha256
            or receipt.get("motion_manifest_sha256") != sha256_file(motion_manifest)
            or onnx_receipt.get("path") != onnx_path.name or onnx_path.is_symlink()
            or not onnx_path.is_file() or onnx_path.stat().st_size > 512 * 1024 * 1024
            or onnx_receipt.get("sha256") != sha256_file(onnx_path)
            or onnx_receipt.get("obs") != 375 or onnx_receipt.get("latent") != 256
            or onnx_receipt.get("act") != 23 or onnx_receipt.get("opset") != 17
            or onnx_receipt.get("inputs") != ["observation", "z"]
            or onnx_receipt.get("batch_sizes") != [1, 4]
            or onnx_receipt.get("normalization") != "embedded"
            or type(onnx_receipt.get("max_joint_target_error_rad")) not in (float, int)
            or not math.isfinite(onnx_receipt["max_joint_target_error_rad"])
            or onnx_receipt["max_joint_target_error_rad"] > 1e-4):
        raise ValueError("BFM export manifest differs from its pinned model/codec/motion or actor contract")
    command_frontend = prompt.get("command_frontend") or {}
    from eb.rpo_bfm import PROMPT_COMMAND
    if (command_receipt.get("path") != command_path.name or command_path.is_symlink()
            or not command_path.is_file() or command_path.stat().st_size > 64 * 1024 * 1024
            or command_receipt.get("sha256") != sha256_file(command_path)
            or command_receipt.get("command_dim") != 3 or command_receipt.get("latent_dim") != 256
            or command_receipt.get("opset") != 17 or command_receipt.get("reference_split") != "train"
            or type(command_receipt.get("reference_count")) is not int
            or not 1 <= command_receipt["reference_count"] <= 4096
            or command_receipt.get("reference_limit") != 4096
            or command_receipt.get("batch_sizes") != [1, 6]
            or type(command_receipt.get("max_abs_error")) not in (float, int)
            or not math.isfinite(command_receipt["max_abs_error"])
            or command_receipt["max_abs_error"] > 1e-4
            or command_frontend.get("path") != command_path.name
            or command_frontend.get("sha256") != command_receipt.get("sha256")
            or command_frontend.get("inputs") != ["command"]
            or command_frontend.get("outputs") != ["z"]
            or command_frontend.get("reference_split") != "train"
            or command_frontend.get("reference_count") != command_receipt.get("reference_count")
            or command_frontend.get("reference_limit") != 4096
            or command_frontend.get("minimum") != PROMPT_COMMAND["minimum"]
            or command_frontend.get("maximum") != PROMPT_COMMAND["maximum"]
            or command_frontend.get("formula") != "rpo-bfm-reward-inference/v1"):
        raise ValueError("BFM command-to-latent artifact differs from its frozen reference or command contract")
    if (prompt.get("schema_version") != "eb/rpo-bfm-prompts/v1"
            or prompt.get("status") != "diagnostic_smoke"
            or prompt.get("model_weights_sha256") != receipt["model_weights_sha256"]
            or prompt.get("codec_sha256") != expected_codec_sha256
            or prompt.get("asset_sha256") != asset_sha256
            or prompt.get("motion_manifest_sha256") != receipt["motion_manifest_sha256"]
            or prompt_receipt.get("path") != prompts_path.name or prompts_path.is_symlink()
            or not prompts_path.is_file() or prompt_receipt.get("sha256") != sha256_file(prompts_path)
            or prompt_receipt.get("manifest") != "prompt-manifest.json"
            or prompt_receipt.get("manifest_sha256") != sha256_file(output / "prompt-manifest.json")
            or not isinstance(prompt_items, dict) or set(prompt_items) != {
                "goal_train_reference", "tracking_train_sequence", "reward_stand", "reward_walk_0p3mps"}
            or any(not isinstance(item, dict) or type(item.get("z_norm")) not in (int, float)
                   or not math.isclose(item["z_norm"], 16., abs_tol=1e-4)
                   for item in (prompt_items or {}).values())):
        raise ValueError("BFM export prompt manifest differs from its frozen source or latent contract")
    if (golden_receipt.get("path") != golden_path.name or golden_path.is_symlink()
            or not golden_path.is_file() or golden_receipt.get("sha256") != sha256_file(golden_path)
            or golden_receipt.get("n_inputs") != 4 or golden_receipt.get("batch_sizes") != [1, 4]
            or golden_receipt.get("n_command_inputs") != 6
            or golden_receipt.get("command_batch_sizes") != [1, 6]):
        raise ValueError("BFM export golden trace manifest is incomplete")
    import onnx
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    command_graph = onnx.load(str(command_path), load_external_data=False).graph
    def last_dimension(value):
        dims = value.type.tensor_type.shape.dim
        return dims[-1].dim_value if dims else None

    if (len(graph.input) != 2
            or {item.name:last_dimension(item) for item in graph.input} != {"observation":375, "z":256}
            or len(graph.output) != 1 or graph.output[0].name != "action"
            or last_dimension(graph.output[0]) != 23):
        raise ValueError("BFM export ONNX measured inputs or action dimension are invalid")
    if (len(command_graph.input) != 1 or command_graph.input[0].name != "command"
            or last_dimension(command_graph.input[0]) != 3
            or len(command_graph.output) != 1 or command_graph.output[0].name != "z"
            or last_dimension(command_graph.output[0]) != 256):
        raise ValueError("BFM command-to-latent ONNX measured dimensions are invalid")
    return receipt


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


def validate_rpo_bfm_motion_probe_receipt(path, manifest_sha, asset_sha, teacher_sha=None):
    from eb.rpo_bfm import CODEC_HASH
    from eb.runners.rpo_bfm_motion_probe import RECIPE, tracking_pass

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8*1024*1024:
        raise ValueError("BFM motion replay receipt is missing or oversized")
    receipt = json.loads(path.read_text())
    recipe = {**RECIPE, "id":"rpo-tracking-teacher-validation-v1", "seeds":[142,143,144]} if teacher_sha else RECIPE
    if (receipt.get("schema_version") != "eb/rpo-bfm-motion-probe/v1" or receipt.get("status") != "succeeded"
            or receipt.get("manifest_sha256") != manifest_sha or receipt.get("asset_sha256") != asset_sha
            or receipt.get("codec_sha256") != CODEC_HASH or receipt.get("recipe") != recipe
            or receipt.get("training_ready") is not False):
        raise ValueError("BFM motion replay receipt has an invalid input/recipe binding")
    cases = receipt.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > 10000:
        raise ValueError("BFM motion replay has no bounded physical cases")
    try:
        passed = [tracking_pass(item) for item in cases]
        consistent = all(item.get("passed") == value for item, value in zip(cases, passed))
        seed_groups = {}
        for item in cases:
            key = (item["clip"], item["begin"], item["end"])
            seed_groups.setdefault(key, []).append(item["seed"])
        consistent &= all(sorted(seeds) == recipe["seeds"] for seeds in seed_groups.values())
    except (KeyError, TypeError, ValueError):
        raise ValueError("BFM motion replay case metrics are invalid") from None
    if (not consistent or receipt.get("total") != len(cases) or receipt.get("passed") != sum(passed)
            or receipt.get("intervals_checked") != len(seed_groups)
            or receipt.get("dynamic_replay_verified") != all(passed)):
        raise ValueError("BFM motion replay quality disagrees with case evidence")
    trajectory = receipt.get("trajectory") or {}
    file = path.parent / "replay-trajectory.npz"
    if (trajectory.get("file") != file.name or file.is_symlink() or not file.is_file()
            or file.stat().st_size > 64*1024*1024
            or hashlib.sha256(file.read_bytes()).hexdigest() != trajectory.get("sha256")):
        raise ValueError("BFM motion replay trajectory hash mismatch")
    if teacher_sha:
        if (receipt.get("teacher_weights_sha256") != teacher_sha or len(receipt.get("videos", [])) != 12
                or receipt.get("startup_randomization") != "fresh_process_per_seed"
                or receipt.get("startup_seeds") != [142,143,144]):
            raise ValueError("Tracking teacher evaluation lacks bound weights/videos")
        startups = receipt.get("startups", [])
        if [item.get("seed") for item in startups] != [142,143,144]:
            raise ValueError("Tracking teacher evaluation lacks sampled physics witnesses")
        for item in startups:
            file = path.parent / item["file"]
            if (item["file"] != f"seed_{item['seed']}/startup.json" or file.is_symlink()
                    or file.parent.is_symlink() or not file.is_file() or file.stat().st_size > 1024*1024
                    or hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]
                    or json.loads(file.read_text()).get("seed") != item["seed"]):
                raise ValueError("Tracking teacher sampled physics witness changed")
        for video in receipt["videos"]:
            file = path.parent / video["file"]
            if (not re.fullmatch(r"seed_(142|143|144)/[A-Za-z0-9_-]+\.mp4", video["file"])
                    or file.is_symlink() or file.parent.is_symlink() or not file.is_file()
                    or hashlib.sha256(file.read_bytes()).hexdigest() != video["sha256"]):
                raise ValueError("Tracking teacher video evidence changed")
    return receipt


async def claim_outputs(session: AsyncSession, job: Job, run: Run) -> None:
    root = get_settings().data_root / "runs" / run.id
    outputs = root / "outputs"
    cas_root = get_settings().data_root / "cas"
    cas_root.mkdir(parents=True, exist_ok=True)

    operation = (job.spec.get("params") or {}).get("operation")
    if job.kind == 'asset.validate' and operation in {'probe_rpo_ufo_tracking_contact','probe_rpo_ufo_compliance_v11','probe_rpo_ufo_mjlab_v12','probe_rpo_ufo_mjlab_viability_v13','probe_rpo_ufo_learning_diagnostics_v1','probe_rpo_ufo_learning_diagnostics_v2','probe_rpo_ufo_learning_value_v3','probe_rpo_ufo_learning_diagnostics_v4','probe_rpo_ufo_dynamics_diagnostics_v1','probe_rpo_ufo_dynamics_finalize_v2','probe_rpo_ufo_contact_factor_v1','probe_rpo_ufo_integration_factor_v1','probe_rpo_ufo_action_replay_v1','probe_rpo_ufo_solver_cone_v1','probe_rpo_ufo_control_damping_v1','probe_rpo_ufo_friction_impedance_v1','probe_rpo_ufo_ground_stiffness_v1','probe_rpo_ufo_contact_margin_v1'}:
        dynamics=operation in {'probe_rpo_ufo_dynamics_diagnostics_v1','probe_rpo_ufo_dynamics_finalize_v2','probe_rpo_ufo_contact_factor_v1','probe_rpo_ufo_integration_factor_v1','probe_rpo_ufo_action_replay_v1','probe_rpo_ufo_solver_cone_v1','probe_rpo_ufo_control_damping_v1','probe_rpo_ufo_friction_impedance_v1','probe_rpo_ufo_ground_stiffness_v1','probe_rpo_ufo_contact_margin_v1'}
        value_only=operation=='probe_rpo_ufo_learning_value_v3'
        learning=operation in {'probe_rpo_ufo_learning_diagnostics_v1','probe_rpo_ufo_learning_diagnostics_v2','probe_rpo_ufo_learning_diagnostics_v4'} or value_only
        compliance=operation=='probe_rpo_ufo_compliance_v11'
        native=operation in {'probe_rpo_ufo_mjlab_v12','probe_rpo_ufo_mjlab_viability_v13'}
        if dynamics:
            from eb.runners.rpo_ufo_dynamics_diagnostics import validate_receipt
            if operation=='probe_rpo_ufo_contact_factor_v1':
                from eb.runners.rpo_ufo_contact_factor import validate_receipt
            if operation=='probe_rpo_ufo_integration_factor_v1':
                from eb.runners.rpo_ufo_integration_factor import validate_receipt
            if operation=='probe_rpo_ufo_action_replay_v1':
                from eb.runners.rpo_ufo_action_replay import validate_receipt
            if operation=='probe_rpo_ufo_solver_cone_v1':
                from eb.runners.rpo_ufo_solver_cone import validate_receipt
            if operation=='probe_rpo_ufo_control_damping_v1':
                from eb.runners.rpo_ufo_control_damping import validate_receipt
            if operation=='probe_rpo_ufo_friction_impedance_v1':
                from eb.runners.rpo_ufo_friction_impedance import validate_receipt
            if operation=='probe_rpo_ufo_ground_stiffness_v1':
                from eb.runners.rpo_ufo_ground_stiffness import validate_receipt
            if operation=='probe_rpo_ufo_contact_margin_v1':
                from eb.runners.rpo_ufo_contact_margin import validate_receipt
        elif value_only:
            from eb.runners.rpo_ufo_learning_diagnostics import validate_value_receipt as validate_receipt
        elif learning:
            from eb.runners.rpo_ufo_learning_diagnostics import validate_receipt
        elif native:
            from eb.runners.rpo_ufo_mjlab_probe import validate_receipt
        elif compliance:
            from eb.runners.rpo_ufo_compliance_probe import validate_receipt
        else:
            from eb.runners.rpo_ufo_tracking_contact_probe import validate_receipt
        path=outputs/('dynamics-diagnostics.json' if dynamics else 'learning-value.json' if value_only else 'learning-diagnostics.json' if learning else 'mjlab-probe.json' if native else 'contact-compliance-probe.json' if compliance else 'tracking-contact-probe.json')
        if path.is_symlink() or not path.is_file() or path.stat().st_size>1024**2:
            raise ValueError('Tracking/contact diagnostic receipt missing or unbounded')
        receipt=json.loads(path.read_text())
        validate_receipt(receipt,job.spec['inputs']['datasets'][0]['validation'],outputs)
        if operation=='probe_rpo_ufo_contact_factor_v1' and receipt.get('contact_factor')!=job.spec['params']['candidate']:
            raise ValueError('Contact factor receipt differs from requested candidate')
        if operation in {'probe_rpo_ufo_integration_factor_v1','probe_rpo_ufo_action_replay_v1','probe_rpo_ufo_solver_cone_v1','probe_rpo_ufo_control_damping_v1','probe_rpo_ufo_friction_impedance_v1','probe_rpo_ufo_ground_stiffness_v1','probe_rpo_ufo_contact_margin_v1'}:
            params=job.spec['params']
            if (receipt.get('integration_factor')!=params['candidate']
                    or receipt.get('trace_source_run_id')!=params['trace_source_run_id']
                    or receipt.get('trace_source_receipt_sha256')!=params['trace_source_receipt_sha256']):
                raise ValueError('Integration factor or trace source differs from requested identity')
            if operation in {'probe_rpo_ufo_action_replay_v1','probe_rpo_ufo_solver_cone_v1','probe_rpo_ufo_control_damping_v1','probe_rpo_ufo_friction_impedance_v1','probe_rpo_ufo_ground_stiffness_v1','probe_rpo_ufo_contact_margin_v1'} and receipt.get('action_program_stage')!=params['stage']:
                raise ValueError('Action program stage differs from requested identity')
        if operation=='probe_rpo_ufo_dynamics_finalize_v2':
            source=json.loads((outputs/'rsi-source-audit.json').read_text())
            if (receipt.get('source_run_id')!=job.spec['params']['source_run_id']
                    or receipt.get('source_receipt_sha256')!=job.spec['params']['source_receipt_sha256']
                    or source.get('source_receipt_sha256')!=receipt['source_receipt_sha256']
                    or source.get('environment_integration_steps')!=0 or source.get('optimizer_updates')!=0
                    or source.get('dataset_changed') is not False or len(source.get('records',[]))!=3):
                raise ValueError('RSI source audit identity differs')
        run.metrics_summary={**(run.metrics_summary or {}),
            ('dynamics_diagnostics' if dynamics else 'learning_value_diagnostics' if value_only else 'learning_diagnostics' if learning else 'mjlab_physics_probe' if native else 'contact_compliance_probe' if compliance else 'tracking_contact_probe'):receipt}
        return
    if job.kind == "asset.validate" and operation == "probe_rpo_ufo_contact_v8":
        from eb.runners.rpo_ufo_contact_probe import validate_probe
        output = get_settings().data_root / "runs" / run.id / "outputs"
        path = output / "contact-probe.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError("Contact v8 receipt invalid")
        receipt = json.loads(path.read_text())
        validate_probe(receipt, job.spec["inputs"]["datasets"][0]["validation"], output)
        run.metrics_summary = {**(run.metrics_summary or {}), "contact_probe": receipt}
        return
    if job.kind == "asset.validate" and operation == "probe_rpo_ufo_initial_contact":
        from eb.runners.rpo_ufo_initial_contact import validate_receipt
        path = outputs / "initial-contact.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValueError("RSI contact receipt unavailable or unbounded")
        receipt = json.loads(path.read_text())
        validation = job.spec["inputs"]["datasets"][0]["validation"]
        validate_receipt(receipt, validation, outputs)
        run.metrics_summary = {**(run.metrics_summary or {}), "initial_contact": receipt}
        return
    if job.kind == "asset.validate" and operation == "probe_rpo_ufo_tracking":
        from eb.runners.bfm_ufo_tracking import SUITE, validate_tracking_receipt
        path = outputs / "tracking.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
            raise ValueError("Held-out tracking receipt missing or unbounded")
        receipt = json.loads(path.read_text())
        validate_tracking_receipt(receipt, (job.spec.get("params") or {}).get("tracking_plan"), outputs)
        import os
        media = []
        for row in receipt["cases"]:
            digest = row["video_integrity"]["sha256"]
            destination = cas_root / digest[:2] / digest
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(outputs / row["video"], destination)
            artifact = await session.scalar(select(Artifact).where(Artifact.run_id == run.id, Artifact.sha256 == digest))
            if artifact is None:
                artifact = Artifact(id=new_id("art"), kind="video", sha256=digest,
                    size_bytes=destination.stat().st_size, storage=str(destination), run_id=run.id,
                    meta={"role": "heldout_tracking", "clip_id": row["clip_id"], "plan_hash": receipt["plan"]["hash"]})
                session.add(artifact)
                session.add(LineageEdge(src_type="run", src_id=run.id, dst_type="artifact", dst_id=artifact.id,
                    relation="produced", run_id=run.id))
            media.append({"artifact_id": artifact.id, "url": f"/media/{digest[:2]}/{digest}",
                **{key: row[key] for key in ("clip_id", "case_id", "steps", "reason", "q_rmse_rad")}})
        run.metrics_summary = {**(run.metrics_summary or {}), "tracking": receipt, "tracking_media": media, "suite_id": SUITE["id"]}
        return
    if job.kind == "asset.validate" and operation == "probe_rpo_ufo_environment":
        from eb.runners.bfm_ufo_agent import AUX_NAMES
        path = outputs / "probe.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValueError("UFO environment probe receipt missing or unbounded")
        receipt = json.loads(path.read_text())
        dataset = (job.spec.get("inputs") or {}).get("datasets", [None])[0]
        validation = (dataset or {}).get("validation") or {}
        if (receipt.get("schema_version") != "eb/rpo-ufo-environment-probe/v1" or receipt.get("status") != "passed"
                or receipt.get("backend") != "isaaclab" or receipt.get("gym_id") != "RPO-UFO-Isaac-RSI-v1"
                or receipt.get("envs") != 2 or receipt.get("aux_names") != list(AUX_NAMES)
                or receipt.get("checks") != ["observation_idempotent", "reference_partial_reset", "timeout_terminal", "raw_aux_eight"]
                or receipt.get("asset_sha256") != validation.get("asset_sha256")
                or receipt.get("manifest_sha256") != validation.get("manifest_sha256")):
            raise ValueError("UFO environment probe evidence differs from frozen input")
        run.metrics_summary = {**(run.metrics_summary or {}), "probe": receipt}
        return
    if job.kind == "asset.validate" and operation == "probe_rpo_bfm_ufo_agent":
        from eb.runners.bfm_ufo_agent import validate_source_receipt, OPTIMIZERS, TARGETS
        from eb.ufo_source import SOURCE_RECEIPT
        import math
        path = outputs / "probe.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValueError("UFO agent probe receipt missing or unbounded")
        receipt = json.loads(path.read_text())
        validate_source_receipt(receipt.get("source") or {})
        if (receipt.get("status") != "passed" or receipt.get("device") != "cuda"
                or receipt.get("joint_updates") != 1
                or receipt.get("source") != SOURCE_RECEIPT
                or receipt.get("configuration") != {"hidden_dim": 1024, "hidden_layers": 3, "batch_size": 128}
                or receipt.get("targets") != list(TARGETS) or receipt.get("z_pool_count") != 128
                or receipt.get("optimizer_steps") != {name: 1 for name in OPTIMIZERS}
                or not math.isfinite(receipt.get("parameter_delta_l1", float("nan")))
                or receipt.get("parameter_delta_l1", 0) <= 0
                or receipt.get("atol") != 1e-6 or receipt.get("rtol") != 1e-5):
            raise ValueError("UFO agent native differential/update evidence invalid")
        run.metrics_summary = {**(run.metrics_summary or {}), "probe": receipt}
        return
    if job.kind == "asset.validate" and operation in {"probe_rpo_bfm_motion", "probe_100style_retarget", "probe_100style_tracking_teacher"}:
        params = job.spec.get("params") or {}
        if operation == "probe_rpo_bfm_motion":
            source_id = str(params.get("dataset_id", ""))
            manifest = get_settings().datasets_root / source_id / "content/manifest.json"
            valid_id = re.fullmatch(r"ds_[0-9a-f]+", source_id)
        else:
            source_id = str(params.get("source_run_id", ""))
            manifest = get_settings().data_root / "runs" / source_id / "outputs/retargeted/retarget-receipt.json"
            valid_id = re.fullmatch(r"run_[0-9a-f]+", source_id)
        asset = get_settings().assets_root / "roboparty/rpo-v2/asset-manifest.json"
        if not valid_id or manifest.is_symlink() or not manifest.is_file():
            raise ValueError("BFM motion replay dataset is unavailable")
        receipt = validate_rpo_bfm_motion_probe_receipt(outputs / "probe.json",
            hashlib.sha256(manifest.read_bytes()).hexdigest(), hashlib.sha256(asset.read_bytes()).hexdigest(),
            params.get("teacher_weights_sha256") if operation == "probe_100style_tracking_teacher" else None)
        run.metrics_summary = {**(run.metrics_summary or {}), "probe":receipt}
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
    if job.kind in {"train.rl", "train.sft", "train.rlft"} and train_template.startswith(("libero_", "robotwin_", "rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_")):
        from eb.checkpoints import checkpoint_model
        from eb.domain.training_recipe import RECIPES, four_gpu_recipe
        checkpoint = checkpoint_model(root, run.id, verify_hashes=train_template.startswith(("robotwin_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_")))
        if train_template.startswith("libero_") and "model/physical-intelligence/libero/norm_stats.json" not in {item["path"] for item in checkpoint["manifest"]["files"]}:
            raise ValueError("LIBERO checkpoint manifest lacks normalization assets")
        sync = None
        if train_template in RECIPES and four_gpu_recipe(train_template):
            from eb.runners.bfm_ufo_checkpoint import validate_sampler_identities
            sync_path = root / "output/ufo-sampler-sync.json"
            if sync_path.is_symlink() or not sync_path.is_file() or sync_path.stat().st_size > 64 * 1024:
                raise ValueError("Four GPU sampler sync evidence missing")
            sync = json.loads(sync_path.read_text())
            validate_sampler_identities(sync["identity"], sync["samplers"], ranks=3)
            if sync.get("schema_version") != "eb/ufo-sampler-sync/v1" or not 0 <= sync["identity"]["version"] <= checkpoint["step"]:
                raise ValueError("Four GPU sync version differs from learner counter")
        model = (await promote_robotwin_model(session, job, run, checkpoint)
                 if train_template.startswith("robotwin_") else
                 await promote_rpo_model(session, job, run, checkpoint)
                 if train_template in RECIPES or train_template in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_amp_stand_walk_ppo", "rpo_v2_flat_ppo", "rpo_v2_bfm_zero_pretrain", "rpo_v2_bfm_zero_tracking_teacher"} else None)
        run.metrics_summary = {**(run.metrics_summary or {}),
            "checkpoint":{key:checkpoint[key] for key in ("path", "bytes", "step")},
            **({"model_version_id": model.id} if model else {})}
        if sync is not None:
            run.metrics_summary["sampler_sync"] = sync
        return

    params = job.spec.get("params") or {}
    bfm_export_receipt = None
    bfm_command_artifact = None
    source_model = None
    if job.kind == "export.policy" and params.get("operation") == "rpo_bfm_export":
        source_run_id = params.get("source_run_id")
        source_model = await session.get(ModelVersion, params.get("source_model_version_id"))
        source_artifact = await session.get(Artifact, source_model.artifact_id) if source_model else None
        source_path = Path(get_settings().models_root) / "runs" / str(source_run_id)
        if (not source_model or source_model.family_id != "mf_rpo_v2_bfm_zero"
                or source_model.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"
                or (source_model.card or {}).get("run_id") != source_run_id
                or not source_artifact or Path(source_artifact.storage).resolve() != source_path.resolve()
                or job.embodiment_env_id != "ee_rpo_v2_bfm_23dof_v1"):
            raise ValueError("BFM export source model version does not match its checkpoint")
        bfm_export_receipt = validate_rpo_bfm_export(root, job)
        command_path = root / "output/exported/command-to-latent.onnx"
        command_digest = sha256_file(command_path)
        command_dest_dir = cas_root / command_digest[:2]
        command_dest_dir.mkdir(parents=True, exist_ok=True)
        command_dest = command_dest_dir / command_digest
        if not command_dest.exists():
            command_dest.write_bytes(command_path.read_bytes())
        bfm_command_artifact = await session.scalar(select(Artifact).where(Artifact.sha256 == command_digest))
        if bfm_command_artifact is None:
            bfm_command_artifact = Artifact(
                id=new_id("art"), sha256=command_digest, kind="onnx",
                size_bytes=command_path.stat().st_size, storage=str(command_dest), run_id=run.id,
                meta={"name":command_path.name, "role":"bfm_command_to_latent",
                      "command_dim":3, "latent_dim":256,
                      "export_manifest":bfm_export_receipt["command_latent"]},
            )
            session.add(bfm_command_artifact)
            await session.flush()
        run.metrics_summary = {**(run.metrics_summary or {}),
                               "bfm_export":{"onnx":bfm_export_receipt["onnx"],
                                             "command_latent":bfm_export_receipt["command_latent"],
                                             "prompts":bfm_export_receipt["prompts"],
                                             "readiness":"diagnostic_smoke"}}

    eval_manifest = root / "output" / "eval-manifest.json"
    template_id = params.get("template_id", "")
    embodiment_env_id = getattr(job, "embodiment_env_id", None) or job.spec.get("embodiment_env_id")
    if job.kind == "eval.sim" and template_id in {"rpo_v2_flat_eval","rpo_v2_bfm_zero_eval", "rpo_bfm_ufo_agent_rlinf_isaac_eval_v1"}:
        from eb.rpo_flat import validate_flat_eval
        plan = params.get("rpo_eval_plan") or {}
        evidence = validate_flat_eval(root / "output", plan, run.id,bfm=template_id!="rpo_v2_flat_eval")
        existing = await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id))
        if existing:
            run.metrics_summary = {**existing.metrics, "eval_report_id": existing.id,
                                   "qualification": existing.gate_results.get("qualification")}
            return
        report = EvalReport(id=new_id("ev"), suite_id=plan["suite_id"],
            model_version_id=params["model_version_id"], run_id=run.id, metrics=evidence["metrics"],
            gate_results={**{key:evidence[key] for key in ("qualification", "validity", "receipt")},
                **({"execution": params["model_execution"], "source_step": plan["checkpoint_step"],
                    "progress_unit": "learner_joint_update"} if params.get("model_execution") else {})})
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
    if bfm_export_receipt:
        uniq = [path for path in uniq if path == root / "output/exported/policy.onnx"]
        if not uniq:
            raise ValueError("BFM export actor ONNX is missing")

    family_id = "mf_rpo_v2_bfm_zero" if bfm_export_receipt else "mf_run"
    fam = await session.get(ModelFamily, family_id)
    if fam is None:
        fam = ModelFamily(id=family_id,
                          name="RPO V2 BFM-Zero" if bfm_export_receipt else "run policies", family="mlp")
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
        artifact_meta = {"name":path.name, **sig}
        if bfm_export_receipt:
            artifact_meta["bfm_export_manifest"] = bfm_export_receipt
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
                meta=artifact_meta,
            )
            session.add(art)
            await session.flush()
        io_signature = ({"observation_profile":"rpo_bfm_actor_history4_z256_v1",
                         "obs":375, "latent":256, "critic_obs":733,
                         "action_profile":"rpo_v2_joint23_position_v2" if job.embodiment_env_id == "ee_rpo_v2_bfm_23dof_v2" else "rpo_v2_joint23_position_v1", "act":23}
                        if bfm_export_receipt else {"obs":sig.get("obs"), "act":sig.get("act")})
        mv = ModelVersion(
            id=new_id("mv"),
            family_id=fam.id,
            name=(job.spec.get("params") or {}).get("model_name") or path.stem,
            embodiment_env_id=job.embodiment_env_id,
            io_signature=io_signature,
            provenance="full",
            artifact_id=art.id,
            tags=(["policy", "run", "rpo-v2", "bfm-zero", "onnx"]
                  if bfm_export_receipt else ["policy", "run"]),
            card=({"schema_version":"eb/rpo-bfm-exported-model/v1", "source_path":str(path),
                   "measured":True, "readiness":"diagnostic_smoke",
                   "source_model_version_id":source_model.id,
                   "command_latent_artifact_id":bfm_command_artifact.id if bfm_command_artifact else None,
                   "export_manifest":bfm_export_receipt}
                  if bfm_export_receipt else {"source_path":str(path), "measured":True}),
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
        if bfm_export_receipt:
            session.add(LineageEdge(src_type="model_version", src_id=source_model.id,
                                    dst_type="model_version", dst_id=mv.id,
                                    relation="exported_from", run_id=run.id))
            run.metrics_summary = {**(run.metrics_summary or {}),
                                   "exported_model_version_id":mv.id}
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
