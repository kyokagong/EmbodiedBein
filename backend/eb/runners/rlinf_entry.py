"""Container-only RLinf entry; compose inputs and publish complete model artifacts."""
import argparse
import base64
import hashlib
import json
import os
import runpy
import shutil
import sys
import zlib
from datetime import UTC, datetime
from pathlib import Path

SFT_ENTRYPOINT = "examples/sft/train_vla_sft.py"
RL_ENTRYPOINT = "examples/embodiment/train_embodied_agent.py"
EVAL_ENTRYPOINT = "evaluations/eval_embodied_agent.py"
RLINF_PATCH_SHA256 = "sha256:f1a4b0a266bf843db06d34cd34518e2bd6dceeada8f4ae982bb72a4c41441255"


def validate_rlinf_image_receipt(path=Path("/workspace/rlinf/.eb-source-receipt.json")):
    expected = {
        "rlinf_content_hash": os.environ.get("EB_RLINF_CONTENT_HASH"),
        "patch_sha256": os.environ.get("EB_RLINF_PATCH_SHA256"),
    }
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("RLInf patched image receipt is missing or invalid") from exc
    if receipt.get("schema_version") != "eb/rlinf-image-source/v1":
        raise ValueError("RLInf patched image receipt schema is invalid")
    for field, value in expected.items():
        if not value or receipt.get(field) != value:
            raise ValueError(f"RLInf image {field} does not match frozen evaluation input")
    return receipt


def validate_entrypoint_mode(entrypoint, cfg):
    expected = {
        SFT_ENTRYPOINT:("sft", False),
        RL_ENTRYPOINT:("embodied", False),
        EVAL_ENTRYPOINT:("embodied_eval", True),
    }
    if entrypoint not in expected:
        raise ValueError("Unapproved RLInf entrypoint")
    task_type, only_eval = expected[entrypoint]
    runner = cfg.runner
    if runner.get("task_type") != task_type or bool(runner.get("only_eval", False)) != only_eval:
        raise ValueError("RLInf 入口与 runner task_type/only_eval 不一致")


def load_eval_plan_env():
    raw = os.environ.get("EB_EVAL_PLAN")
    packed = os.environ.get("EB_EVAL_PLAN_ZLIB")
    path = os.environ.get("EB_EVAL_PLAN_PATH")
    if sum(bool(value) for value in (raw, packed, path)) > 1:
        raise ValueError("Evaluation plan environment is ambiguous")
    if path:
        try:
            raw = Path(path).read_text()
        except OSError as exc:
            raise ValueError("Evaluation plan file is missing or unreadable") from exc
    if packed:
        try:
            raw = zlib.decompress(base64.b64decode(packed, validate=True)).decode()
        except (ValueError, zlib.error, UnicodeDecodeError) as exc:
            raise ValueError("Compressed evaluation plan is invalid") from exc
    return json.loads(raw) if raw else None


def externalize_compressed_eval_plan(output):
    """Keep a full evaluation plan out of subprocess environments."""
    if "EB_EVAL_PLAN_ZLIB" not in os.environ:
        return load_eval_plan_env()
    plan = load_eval_plan_env()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "eval-plan.json"
    atomic_json(path, plan)
    os.environ.pop("EB_EVAL_PLAN_ZLIB")
    os.environ["EB_EVAL_PLAN_PATH"] = str(path)
    return plan


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_eval_model(cfg):
    from omegaconf import OmegaConf
    if getattr(cfg, "actor", None) is None:
        return
    model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    OmegaConf.update(cfg, "rollout.model", model, merge=False, force_add=True)


def _complete_resume_manifest(checkpoint):
    manifest = checkpoint / "resume-manifest.json"
    if checkpoint.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text())
        step = int(checkpoint.name.removeprefix("global_step_"))
    except (ValueError, OSError, json.JSONDecodeError):
        return None
    if (value.get("schema_version") != "eb/resume-checkpoint/v1"
            or value.get("status") != "complete" or value.get("step") != step):
        return None
    return value


def write_resume_manifest(checkpoint, mode):
    checkpoint = Path(checkpoint)
    try:
        step = int(checkpoint.name.removeprefix("global_step_"))
    except ValueError as exc:
        raise ValueError("checkpoint directory must be global_step_N") from exc
    if step < 0 or checkpoint.is_symlink():
        raise ValueError("checkpoint directory is invalid")
    required = [
        checkpoint / "actor/dcp_checkpoint/.metadata",
        checkpoint / "actor/model_state_dict/full_weights.pt",
    ]
    shards = list((checkpoint / "actor/dcp_checkpoint").glob("*.distcp"))
    if not shards:
        raise ValueError("checkpoint has no DCP shards")
    if mode == "exact":
        required += [checkpoint / "data.pt", checkpoint / "rng.pt"]
    elif mode != "trainer_state":
        raise ValueError("unsupported resume mode")
    if json.loads(os.environ.get("EB_INPUT_RECEIPT", "{}")).get("template_id") == "rpo_v2_amp_stand_walk_ppo":
        required.append(checkpoint / "amp/state.pt")
    for path in [*required, *shards]:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"checkpoint member is incomplete: {path.name}")
    fingerprint = os.environ.get("EB_RESUME_FINGERPRINT", "")
    if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
        raise ValueError("EB_RESUME_FINGERPRINT is missing or invalid")
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if path.name in {"resume-manifest.json", "resume-manifest.json.tmp"}:
            continue
        if path.is_symlink():
            raise ValueError("checkpoint members must not be symlinks")
        if path.is_file():
            files.append({"path": path.relative_to(checkpoint).as_posix(), "bytes": path.stat().st_size})
    exact = mode == "exact"
    value = {
        "schema_version": "eb/resume-checkpoint/v1",
        "status": "complete",
        "run_id": os.environ["EB_RUN_ID"],
        "step": step,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "state_coverage": {
            "model": "restored", "optimizer": "restored", "scheduler": "restored",
            "rng_per_rank": "restored", "dataloader": "restored" if exact else "not_applicable",
            "environment": "not_applicable" if exact else "not_available",
            "rollout": "not_applicable" if exact else "not_available",
        },
        "compatibility_fingerprint": fingerprint,
        "files": files,
    }
    atomic_json(checkpoint / "resume-manifest.json", value)
    return value


def prepare_resume_receipt(output, cfg):
    raw = os.environ.get("EB_RESUME_RECEIPT")
    configured = cfg.runner.get("resume_dir") if hasattr(cfg.runner, "get") else getattr(cfg.runner, "resume_dir", None)
    if not raw:
        if configured:
            raise ValueError("runner.resume_dir requires a platform resume receipt")
        return None
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("EB_RESUME_RECEIPT is invalid") from exc
    checkpoint = Path(str(receipt.get("checkpoint", "")))
    if (receipt.get("schema_version") != "eb/resume-request/v1"
            or receipt.get("status") != "validated" or not checkpoint.is_dir()
            or checkpoint.resolve() != Path(str(configured)).resolve()):
        raise ValueError("resume receipt checkpoint does not match resolved config")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "resume-receipt.json", receipt)
    return receipt


def mark_resume_loaded(output, load_path):
    path = Path(output) / "resume-receipt.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("validated resume receipt is missing")
    receipt = json.loads(path.read_text())
    if Path(str(receipt.get("checkpoint", ""))).resolve() != Path(load_path).resolve():
        raise ValueError("loaded checkpoint does not match resume receipt")
    receipt["status"] = "loaded"
    atomic_json(path, receipt)


def prepare_warm_start_receipt(output, cfg):
    raw = os.environ.get("EB_WARM_START_RECEIPT")
    if not raw:
        return
    import torch
    receipt = json.loads(raw)
    weight = Path(str(cfg.runner.ckpt_path))
    normalization = weight.parent.parent / "actor-normalization.json"
    if (receipt.get("schema_version") != "eb/warm-start/v1" or receipt.get("status") != "validated"
            or cfg.runner.get("resume_dir") or weight.is_symlink() or normalization.is_symlink()
            or weight.stat().st_size != receipt.get("weight_bytes")):
        raise ValueError("invalid Flat warm start binding")
    for path, key in ((weight, "weight_sha256"), (normalization, "normalization_sha256")):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt.get(key):
                raise ValueError("published Flat warm start package changed")
    state = torch.load(weight, map_location="cpu", weights_only=True)
    for prefix, dimension in (("actor_norm", 780), ("critic_norm", 1390)):
        for field in ("_mean", "_std", "_var"):
            tensor = state[f"{prefix}.{field}"]
            if tensor.numel() != dimension or not torch.isfinite(tensor).all():
                raise ValueError("invalid warm start normalization")
    count = float(state["actor_norm.count"])
    if not 0 < count < float("inf") or count != float(state["critic_norm.count"]):
        raise ValueError("invalid warm start normalization counts")
    receipt["normalization_count"] = count
    atomic_json(Path(output) / "warm-start-receipt.json", receipt)


def prune_old_checkpoints(root, *, keep=1):
    if type(keep) is not int or keep < 1:
        raise ValueError("checkpoint retention must be positive")
    candidates = []
    for directory in root.glob("global_step_*"):
        suffix = directory.name.removeprefix("global_step_")
        if suffix.isdigit() and directory.is_dir() and not directory.is_symlink():
            candidates.append((int(suffix), directory))
    complete = [item for item in candidates if _complete_resume_manifest(item[1])]
    if not complete:
        return
    retained = {directory for _, directory in sorted(complete, reverse=True)[:keep]}
    for _, directory in candidates:
        if directory not in retained:
            shutil.rmtree(directory)


def publish_checkpoint(output, cfg, receipt):
    root = output / str(cfg.runner.logger.experiment_name) / "checkpoints"
    candidates = []
    for directory in root.glob("global_step_*"):
        suffix = directory.name.removeprefix("global_step_")
        weights = directory / "actor/model_state_dict/full_weights.pt"
        if suffix.isdigit() and weights.is_file():
            candidates.append((int(suffix), weights))
    if not candidates:
        raise RuntimeError("Training exited without a complete full_weights.pt checkpoint")
    step, weights = max(candidates)
    amp = receipt.get("template_id") == "rpo_v2_amp_stand_walk_ppo"
    flat = receipt.get("template_id") == "rpo_v2_flat_ppo"
    if amp and not (weights.parents[2] / "amp/state.pt").is_file():
        raise ValueError("AMP checkpoint lacks discriminator and sampler state")
    if any(path.is_symlink() for path in [weights, *weights.parents] if path != output.parent):
        raise ValueError("Checkpoint source must not contain symlinks")
    parent = Path(str(cfg.actor.model.model_path))
    destination = output / "model"
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError("Refusing to replace an existing model publication")
    destination.mkdir(exist_ok=True)
    staging = destination / ".publishing"
    if staging.exists():
        raise FileExistsError("Refusing to replace an existing model publication")
    staging.mkdir()
    try:
        model_type = str(getattr(cfg.actor.model, "model_type", "openpi"))
        # Preserve parent metadata for VLA models. The RPO MLP has no parent model.
        if model_type != "mlp_policy":
            for source in parent.rglob("*"):
                if source.is_symlink():
                    raise ValueError("Parent model assets must not contain symlinks")
                if source.is_file() and source.suffix not in {".pt", ".safetensors", ".bin"}:
                    target = staging / source.relative_to(parent)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        if model_type == "lingbot_vla2":
            shutil.copyfile("/opt/lingbot-vla2/assets/norm_stats/robotwin.json",
                            staging / "norm_stats.json")
        elif model_type == "xiaomi_xr1":
            shutil.copyfile(output / "xr1-normalization.json", staging / "norm_stats.json")
        required_stats = None if model_type == "mlp_policy" else \
            "statistics.json" if model_type == "gr00t_n1d7" else "norm_stats.json"
        if required_stats and not list(staging.rglob(required_stats)):
            raise ValueError(f"Model publication requires parent {required_stats}")
        if model_type == "mlp_policy":
            if flat:
                import torch
                state = torch.load(weights, map_location="cpu", weights_only=True)
                for prefix, dimension in (("actor_norm", 780), ("critic_norm", 1390)):
                    for field in ("_mean", "_var", "_std"):
                        tensor = state.get(f"{prefix}.{field}")
                        if tensor is None or tensor.numel() != dimension or not torch.isfinite(tensor).all():
                            raise ValueError("RPO-Flat checkpoint lacks finite empirical normalization")
                    if float(state[f"{prefix}.count"]) <= 0:
                        raise ValueError("RPO-Flat checkpoint normalization was never fitted")
            atomic_json(staging / "actor-normalization.json", {
                "schema_version":"eb/rpo-flat-normalization/v1" if flat else "eb/rpo-amp-normalization/v1" if amp else "eb/rpo-wbc-normalization/v1",
                "kind":"empirical" if flat else "fixed_env_scaling",
                "observation_dim":780 if amp or flat else 890, "history_length":10, "clip":100.0,
                **({"running_stats_in_weights":True, "critic_dim":1390} if flat else {}),
            })
        target = staging / "model_state_dict/full_weights.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weights, target)
        files = []
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                with file.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files.append({"path": "model/" + str(file.relative_to(staging)), "bytes": file.stat().st_size, "sha256": digest})
        for child in staging.iterdir():
            child.rename(destination / child.name)
        staging.rmdir()
        config_name = str(cfg.actor.model.openpi.config_name) if model_type == "openpi" else model_type
        atomic_json(output / "checkpoint-manifest.json", {
            "schema_version": "eb/checkpoint/v1", "run_id": os.environ["EB_RUN_ID"],
            "template_id": receipt.get("template_id"), "status": "complete",
            "model_path": "model", "step": step, "files": files,
            "parent_model": str(parent), "config_name": config_name,
            "source_checkpoint": str(weights.relative_to(output)),
            **({"official_source":receipt.get("official_source")} if flat else {}),
        })
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    output = Path(os.environ.get("EB_OUTPUT_DIR", "/workspace/output"))
    eval_plan = externalize_compressed_eval_plan(output)
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--cfg", choices=["job"])
    arguments, overrides = parser.parse_known_args()
    script = Path(arguments.script).resolve()
    allowed = {SFT_ENTRYPOINT, RL_ENTRYPOINT, EVAL_ENTRYPOINT}
    if arguments.script not in allowed:
        raise ValueError("Unapproved RLinf entry script")
    receipt = json.loads(os.environ.get("EB_INPUT_RECEIPT", "{}"))
    template_id = receipt.get("template_id")
    if template_id == "robotwin_pi05_eval":
        validate_rlinf_image_receipt()
    if template_id in {"robotwin_dm05_sft", "robotwin_lingbot_vla2_sft", "robotwin_gr00t_n17_sft", "robotwin_xr1_sft"}:
        from rlinf.config import EMBODIED_MODEL, SupportedModel
        model_name = {"robotwin_dm05_sft": "opendm_dm05",
                      "robotwin_lingbot_vla2_sft": "lingbot_vla2",
                      "robotwin_gr00t_n17_sft": "gr00t_n1d7",
                      "robotwin_xr1_sft": "xiaomi_xr1"}[template_id]
        model_type = SupportedModel.models.get(model_name) or SupportedModel.register(model_name)
        setattr(SupportedModel, model_name.upper(), model_type)
        EMBODIED_MODEL.add(model_type)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base="1.1", config_dir=str(Path(arguments.config_path).resolve())):
        cfg = compose(config_name=arguments.config_name, overrides=overrides)
        validate_entrypoint_mode(arguments.script, cfg)
        if (arguments.script == SFT_ENTRYPOINT
                and template_id not in {"robotwin_dm05_sft", "robotwin_lingbot_vla2_sft", "robotwin_gr00t_n17_sft", "robotwin_xr1_sft"}):
            from libero_fsdp import configure_sft_wrapping
            configure_sft_wrapping(cfg)
        elif arguments.script == EVAL_ENTRYPOINT:
            prepare_eval_model(cfg)
        OmegaConf.resolve(cfg)
    output.mkdir(parents=True, exist_ok=True)
    prepare_resume_receipt(output, cfg)
    prepare_warm_start_receipt(output, cfg)
    atomic_json(output / "input-receipt.json", receipt)
    (output / "resolved-config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    if arguments.cfg:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    model_cfg = cfg.rollout.model if cfg.runner.get("only_eval", False) and hasattr(cfg, "rollout") else cfg.actor.model
    model_type = str(getattr(model_cfg, "model_type", "openpi"))
    parent = Path(str(model_cfg.model_path))
    if template_id == "robotwin_dm05_sft":
        required = [parent / "config.json", parent / "model.safetensors",
                    parent / "processor_config.json", parent / "norm_stats.json"]
        missing = [str(path.name) for path in required if not path.is_file()]
        if missing:
            raise ValueError("DM0.5 parent is incomplete: " + ", ".join(missing))
        data = Path(str(cfg.data.train_data_paths))
        if not (data / "index_cache.json").is_file() or not (data / "norm_stats.json").is_file():
            raise ValueError("DM0.5 requires a frozen OpenDM JSONL/video view")
    elif template_id == "robotwin_lingbot_vla2_sft":
        required = [parent / "model.safetensors.index.json", parent / "depth/model.pt",
                    parent / "dino_video/teacher_step_10000.pth"]
        if missing := [str(path.relative_to(parent)) for path in required if not path.is_file()]:
            raise ValueError("LingBot-VLA-2 parent is incomplete: " + ", ".join(missing))
        tokenizer = Path(str(cfg.actor.model.tokenizer_path))
        if not (tokenizer / "config.json").is_file() or not (tokenizer / "tokenizer.json").is_file():
            raise ValueError("LingBot-VLA-2 requires the frozen Qwen3-VL tokenizer/config assets")
        data = Path(str(cfg.data.train_data_paths))
        if not (data / "meta/info.json").is_file():
            raise ValueError("LingBot-VLA-2 requires the validated RoboTwin ALOHA LeRobot view")
    elif template_id == "robotwin_gr00t_n17_sft":
        required = [parent / "config.json", parent / "model.safetensors.index.json",
                    parent / "processor_config.json", parent / "statistics.json",
                    parent / "embodiment_id.json"]
        if missing := [path.name for path in required if not path.is_file()]:
            raise ValueError("GR00T-N1.7 parent is incomplete: " + ", ".join(missing))
        cosmos = Path(str(cfg.actor.model.cosmos_path))
        if not (cosmos / "config.json").is_file() or not list(cosmos.glob("*.safetensors")):
            raise ValueError("GR00T-N1.7 requires the local Cosmos-Reason2-2B backbone")
        data = Path(str(cfg.data.train_data_paths))
        if not (data / "meta/info.json").is_file():
            raise ValueError("GR00T-N1.7 requires the validated RoboTwin ALOHA LeRobot view")
    elif template_id == "robotwin_xr1_sft":
        if not (parent / "model_states.pt").is_file():
            raise ValueError("Xiaomi-Robotics-1 parent is incomplete: model_states.pt")
        processor = Path(str(cfg.actor.model.processor_path))
        if not (processor / "config.json").is_file() or not (processor / "tokenizer.json").is_file():
            raise ValueError("Xiaomi-Robotics-1 requires the frozen Qwen3-VL processor")
        data = Path(str(cfg.data.train_data_paths))
        if not (data / "meta/info.json").is_file() or not (data / "meta/episodes.jsonl").is_file():
            raise ValueError("Xiaomi-Robotics-1 requires the validated RoboTwin ALOHA LeRobot view")
    elif model_type == "mlp_policy":
        pass
    else:
        tokenizer = Path(os.environ.get("OPENPI_DATA_HOME", "/opt/eb-openpi-cache")) / "big_vision/paligemma_tokenizer.model"
        if not tokenizer.is_file():
            raise FileNotFoundError("Job image is missing the offline PaliGemma tokenizer asset")
        openpi_name = str(model_cfg.openpi.config_name)
        if openpi_name not in {"pi05_libero", "pi05_aloha_robotwin"}:
            raise ValueError("This runner supports only the validated pi05_libero and pi05_aloha_robotwin profiles")
        asset_id = "physical-intelligence/libero" if openpi_name == "pi05_libero" else "physical-intelligence/robotwin"
        if not (parent / asset_id / "norm_stats.json").is_file():
            raise ValueError(f"OpenPI parent requires {asset_id}/norm_stats.json")
        full_weights = parent / "model_state_dict/full_weights.pt"
        if not full_weights.is_file():
            from safetensors import safe_open
            shards = sorted(parent.glob("*.safetensors"))
            if not shards:
                raise ValueError("OpenPI parent has no weights")
            valid = False
            for shard in shards:
                with safe_open(str(shard), framework="pt", device="cpu") as weights:
                    valid = valid or any(key.startswith("paligemma_with_expert.") for key in weights.keys())
            if not valid:
                raise ValueError("Base weights are not an OpenPI model; convert LeRobot weights explicitly")
    if arguments.script == EVAL_ENTRYPOINT:
        plan = eval_plan
        if plan:
            if plan.get("schema_version") == "eb/libero-eval-plan/v1":
                from libero_asset_worker import verify_bundle
                verify_bundle(Path("/workspace/libero-benchmark"), os.environ["EB_LIBERO_BENCHMARK_SHA256"])
                atomic_json(output / "eval-plan.json", plan)
            elif plan.get("schema_version") != "eb/robotwin-eval-plan/v1":
                raise ValueError("Unsupported evaluation plan protocol")
        if plan and plan.get("schema_version") == "eb/robotwin-eval-plan/v1":
            from robotwin_eval import evaluate
        else:
            from libero_eval import evaluate
        metrics = evaluate(cfg)
        atomic_json(output / "eval-manifest.json", {"run_id": os.environ["EB_RUN_ID"], "status": "not_qualified", "metrics": metrics,
            "plan_hash": plan["hash"] if plan else None,
            "reason": "Per-task qualification requires separate validation", "input_receipt": "input-receipt.json"})
        return
    sys.argv = [str(script)]
    namespace = runpy.run_path(str(script), run_name="eb_upstream_entry")
    if template_id == "robotwin_dm05_sft":
        from dm05_sft import Dm05SftWorker
        namespace["main"].__wrapped__.__globals__["FSDPVlaSftWorker"] = Dm05SftWorker
    elif template_id == "robotwin_lingbot_vla2_sft":
        from lingbot_vla2_sft import LingbotVla2SftWorker
        namespace["main"].__wrapped__.__globals__["FSDPVlaSftWorker"] = LingbotVla2SftWorker
    elif template_id == "robotwin_gr00t_n17_sft":
        from gr00t_n17_sft import Gr00tN17SftWorker
        namespace["main"].__wrapped__.__globals__["FSDPVlaSftWorker"] = Gr00tN17SftWorker
    elif template_id == "robotwin_xr1_sft":
        from xiaomi_xr1_sft import XiaomiXr1SftWorker
        namespace["main"].__wrapped__.__globals__["FSDPVlaSftWorker"] = XiaomiXr1SftWorker
    elif arguments.script == SFT_ENTRYPOINT and openpi_name == "pi05_libero":
        from libero_sft import LiberoSftWorker
        namespace["main"].__wrapped__.__globals__["FSDPVlaSftWorker"] = LiberoSftWorker
    elif arguments.script == RL_ENTRYPOINT and template_id.startswith("libero_"):
        from libero_ppo import LiberoPpoActor, LiberoRolloutWorker
        from libero_env import LiberoEnvWorker
        import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
        actor_module.EmbodiedFSDPActor = LiberoPpoActor
        namespace["main"].__wrapped__.__globals__["EnvWorker"] = LiberoEnvWorker
        namespace["main"].__wrapped__.__globals__["MultiStepRolloutWorker"] = LiberoRolloutWorker
    namespace["main"].__wrapped__(cfg)
    publish_checkpoint(output, cfg, receipt)


if __name__ == "__main__":
    main()
