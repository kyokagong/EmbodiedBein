"""Bounded checkpoint receipt checks; payload hashing happens in the producer task."""
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from eb.domain.job import JobSpec
from eb.settings import get_settings


def resume_fingerprint(spec: JobSpec) -> str:
    payload = spec.model_dump(mode="json", exclude={"name", "labels", "experiment"})
    payload["inputs"]["resume_from"] = None
    source_image = payload["params"].pop("resume_source_image", None)
    if source_image:
        payload["image"] = source_image
    for key in ("resume_mode", "resume_step", "source_train_job_id", "model_name"):
        payload["params"].pop(key, None)
    if payload["params"].get("template_id") == "rpo_v2_bfm_zero_pretrain":
        payload["params"].pop("bfm_update_target", None)
    if payload["params"].get("execution"):
        from eb.domain.training_recipe import execution_binding
        execution_binding(spec)
        for key in ("target_learner_updates", "purpose"):
            payload["params"].pop(key, None)
        if payload["params"]["execution"]["recipe_id"] in {"rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13", "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9"}:
            payload["policy"].pop("timeout_h", None)
    payload["overrides"] = [item for item in payload["overrides"]
                            if item.split("=", 1)[0] not in {"runner.max_steps", "runner.max_epochs"}]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def require_flat_rank_rng(spec: JobSpec, checkpoint: dict) -> None:
    if spec.params.get("template_id") == "rpo_v2_flat_ppo" and spec.resources.gpus > 1:
        files = {item["path"] for item in checkpoint["files"]}
        if any(f"actor/rng/rank_{rank}.pt" not in files for rank in range(spec.resources.gpus)):
            raise ValueError("Flat 多卡 checkpoint 缺少各 rank 独立随机状态")


def _resume_candidate(checkpoint: Path, run_id: str) -> dict:
    manifest = checkpoint / "resume-manifest.json"
    if checkpoint.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("checkpoint has no complete resume manifest")
    if manifest.stat().st_size > 64 * 1024:
        raise ValueError("resume manifest is too large")
    try:
        receipt = json.loads(manifest.read_text())
        directory_step = int(checkpoint.name.removeprefix("global_step_"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("resume manifest is invalid") from exc
    if not isinstance(receipt, dict):
        raise ValueError("resume manifest is invalid")
    if receipt.get("format") in {"ufo_agent_state_v1", "ufo_agent_state_v2"}:
        from eb.domain.training_recipe import ExecutionBinding, RECIPES
        from eb.runners.bfm_ufo_checkpoint import validate_ufo_checkpoint
        binding = ExecutionBinding.model_validate(receipt.get("execution"))
        if binding.recipe_id not in RECIPES or binding != RECIPES[binding.recipe_id].binding:
            raise ValueError("Unknown native resume execution binding")
        return validate_ufo_checkpoint(checkpoint, run_id=run_id, verify_hashes=False)
    if receipt.get("format") is not None:
        raise ValueError("Unknown checkpoint format")
    fingerprint = receipt.get("compatibility_fingerprint")
    if (receipt.get("schema_version") != "eb/resume-checkpoint/v1"
            or receipt.get("status") != "complete" or receipt.get("run_id") != run_id
            or receipt.get("step") != directory_step or directory_step < 0
            or receipt.get("mode") not in {"exact", "trainer_state"}
            or not isinstance(fingerprint, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)
            or not isinstance(receipt.get("files"), list) or not receipt["files"]
            or len(receipt["files"]) > 10000):
        raise ValueError("resume manifest completion or producer mismatch")
    coverage = receipt.get("state_coverage")
    required_states = {"model", "optimizer", "scheduler", "rng_per_rank", "dataloader", "environment", "rollout"}
    if (not isinstance(coverage, dict) or set(coverage) != required_states
            or any(value not in {"restored", "not_applicable", "not_available"} for value in coverage.values())):
        raise ValueError("resume state coverage is invalid")
    total, seen = 0, set()
    for item in receipt["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("resume checkpoint member is invalid")
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"] or not relative.parts:
            raise ValueError("resume checkpoint member escapes source")
        path = checkpoint
        for component in relative.parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("resume checkpoint member must not be a symlink")
        if (not path.resolve().is_relative_to(checkpoint.resolve()) or not path.is_file()
                or path in seen or type(item.get("bytes")) is not int
                or item["bytes"] <= 0 or path.stat().st_size != item["bytes"]):
            raise ValueError("resume checkpoint member is missing or changed")
        seen.add(path)
        total += item["bytes"]
    required = {
        checkpoint / "actor/dcp_checkpoint/.metadata",
        checkpoint / "actor/model_state_dict/full_weights.pt",
    }
    if not any(path.parent == checkpoint / "actor/dcp_checkpoint" and path.suffix == ".distcp" for path in seen):
        raise ValueError("resume checkpoint has no DCP shard")
    if receipt["mode"] == "exact":
        required |= {checkpoint / "data.pt", checkpoint / "rng.pt"}
    if not required.issubset(seen):
        raise ValueError("resume checkpoint required member is missing")
    return {**receipt, "path": str(checkpoint.resolve()), "bytes": total}


def resume_checkpoint(run_root: Path, run_id: str, *, step: int | None = None) -> dict:
    run_root = Path(run_root)
    output = run_root / "output"
    if run_root.is_symlink() or output.is_symlink() or not output.is_dir():
        raise ValueError("source Run output is unavailable")
    published = output / "checkpoint-manifest.json"
    if published.is_file() and not published.is_symlink() and published.stat().st_size <= 64 * 1024:
        receipt = json.loads(published.read_text())
        if isinstance(receipt, dict) and receipt.get("format") in {"ufo_agent_state_v1", "ufo_agent_state_v2"}:
            model = checkpoint_model(run_root, run_id)
            relative = PurePosixPath(model["manifest"]["native_checkpoint_path"]).relative_to("model")
            candidate = _resume_candidate(Path(model["path"]) / relative, run_id)
            if step is None or candidate["step"] == step:
                return candidate
    candidates, errors = [], []
    paths = [(output, path) for path in output.glob("*/checkpoints/global_step_*")]
    configured_models = getattr(get_settings(), "models_root", None)
    if configured_models:
        model_root = Path(configured_models) / "runs" / run_id
        if (model_root.is_symlink()
                or not model_root.resolve().is_relative_to(Path(configured_models).resolve())):
            raise ValueError("native checkpoint escapes source model storage")
        paths.extend((model_root, path) for path in (model_root / ".checkpoint-source").glob("global_step_*"))
    for boundary, checkpoint in paths:
        if not checkpoint.is_dir() or not checkpoint.name.removeprefix("global_step_").isdigit():
            continue
        if step is not None and int(checkpoint.name.removeprefix("global_step_")) != step:
            continue
        try:
            if not checkpoint.resolve().is_relative_to(boundary.resolve()):
                raise ValueError("checkpoint escapes source Run output")
            candidates.append(_resume_candidate(checkpoint, run_id))
        except ValueError as exc:
            if (checkpoint / "resume-manifest.json").exists():
                errors.append(exc)
    if candidates:
        return max(candidates, key=lambda item: item["step"])
    if errors:
        raise errors[-1]
    raise ValueError(f"source Run has no complete resumable checkpoint{f' at step {step}' if step is not None else ''}")


def latest_resume_checkpoint(runs: Iterable[tuple[str, Path]], *, step: int | None = None) -> dict:
    candidates = []
    for run_id, root in runs:
        try:
            candidates.append(resume_checkpoint(root, run_id, step=step))
        except (ValueError, OSError):
            continue
    if not candidates:
        raise ValueError(f"training job has no complete resumable checkpoint{f' at step {step}' if step is not None else ''}")
    return max(candidates, key=lambda item: item["step"])


def checkpoint_weights(run_root: Path, step: int) -> dict:
    """Resolve a legacy RLInf full-weight file without claiming resumability."""
    output = Path(run_root) / "output"
    if type(step) is not int or step < 0 or Path(run_root).is_symlink() or output.is_symlink() or not output.is_dir():
        raise ValueError("checkpoint source is invalid")
    for checkpoint in output.glob(f"*/checkpoints/global_step_{step}"):
        weight = checkpoint / "actor/model_state_dict/full_weights.pt"
        path = output
        for component in weight.relative_to(output).parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("checkpoint contains a symlink")
        if not weight.resolve().is_relative_to(output.resolve()):
            raise ValueError("checkpoint escapes source Run")
        if weight.is_file() and weight.stat().st_size > 0:
            return {"path": str(weight.resolve()), "bytes": weight.stat().st_size, "step": step}
    raise ValueError("checkpoint has no nonempty full_weights.pt")


def checkpoint_inference(run_root: Path, template_id: str, source_overrides: list[str]) -> dict[str, int]:
    """Read the producer's actual inference knobs, with a legacy SFT fallback."""
    from eb.domain.rlinf_form import get_template
    resolved = Path(run_root) / "output" / "resolved-config.yaml"
    if resolved.is_symlink():
        raise ValueError("源训练推理配置不可信")
    if resolved.exists():
        if resolved.stat().st_size > 1024 * 1024:
            raise ValueError("源训练推理配置不可信")
        import yaml
        try:
            source_config = yaml.safe_load(resolved.read_text()) or {}
            actor_model = ((source_config.get("actor") or {}).get("model") or {})
            values = {key: actor_model.get(key) for key in ("num_action_chunks", "num_steps")}
        except (yaml.YAMLError, AttributeError, TypeError) as exc:
            raise ValueError("源训练推理配置无法解析") from exc
    elif get_template(template_id).kind == "sft":
        defaults = dict(item.split("=", 1) for item in get_template(template_id).default_overrides)
        effective = {**defaults, **dict(item.split("=", 1) for item in source_overrides)}
        try:
            values = {key: int(effective[f"actor.model.{key}"]) for key in ("num_action_chunks", "num_steps")}
        except (KeyError, ValueError) as exc:
            raise ValueError("旧 SFT 训练缺少可验证的推理配置") from exc
    else:
        raise ValueError("源 PPO 训练缺少 resolved-config.yaml，无法验证推理配置")
    if any(type(value) is not int or not 1 <= value <= 100 for value in values.values()):
        raise ValueError("源训练推理配置超出支持范围")
    return {f"actor.model.{key}": value for key, value in values.items()}


def checkpoint_model(run_root: Path, run_id: str, *, verify_hashes: bool = False) -> dict:
    output = run_root / "output"
    manifest = output / "checkpoint-manifest.json"
    if (run_root.is_symlink() or output.is_symlink() or manifest.is_symlink()
            or not manifest.is_file() or manifest.stat().st_size > 64 * 1024):
        raise ValueError("producer has no bounded completed checkpoint manifest")
    receipt = json.loads(manifest.read_text())
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != "eb/checkpoint/v1" or receipt.get("status") != "complete"
            or receipt.get("run_id") != run_id or not isinstance(receipt.get("files"), list)
            or not receipt["files"]):
        raise ValueError("checkpoint manifest completion or producer mismatch")
    configured_models = getattr(get_settings(), "models_root", None)
    external_model = Path(configured_models) / "runs" / run_id if configured_models else None
    model_root = external_model if external_model and external_model.is_dir() else output / "model"

    def member(relative) -> Path:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("checkpoint member path is invalid")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or relative == ".":
            raise ValueError("checkpoint member path escapes outputs")
        if parts.parts[0] == "model":
            path, components, boundary = model_root, parts.parts[1:], model_root
        else:
            path, components, boundary = output, parts.parts, output
        for component in components:
            path = path / component
            if path.is_symlink():
                raise ValueError("checkpoint members must not be symlinks")
        if not path.resolve().is_relative_to(boundary.resolve()):
            raise ValueError("checkpoint member escapes outputs")
        return path

    model = member(receipt.get("model_path"))
    if not model.is_dir():
        raise ValueError("completed checkpoint model directory is missing")
    total, seen = 0, set()
    for item in receipt["files"]:
        if not isinstance(item, dict):
            raise ValueError("invalid checkpoint file record")
        file = member(item.get("path"))
        if not file.is_relative_to(model) or not file.is_file() or file in seen:
            raise ValueError("checkpoint member missing, duplicate or outside model directory")
        if (type(item.get("bytes")) is not int or item["bytes"] < 0
                or file.stat().st_size != item["bytes"]
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("checkpoint file size or digest receipt is invalid")
        if verify_hashes:
            with file.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError("checkpoint file digest changed")
        seen.add(file)
        total += item["bytes"]
    if receipt.get("format") in {"ufo_agent_state_v1", "ufo_agent_state_v2"}:
        from eb.domain.training_recipe import ExecutionBinding, RECIPES
        from eb.runners.bfm_ufo_checkpoint import validate_ufo_checkpoint
        binding = ExecutionBinding.model_validate(receipt.get("execution"))
        recipe = RECIPES.get(binding.recipe_id)
        if not recipe or binding != recipe.binding or receipt.get("template_id") != binding.recipe_id:
            raise ValueError("Unknown native model execution binding")
        native = validate_ufo_checkpoint(member(receipt.get("native_checkpoint_path")), run_id=run_id,
            verify_hashes=verify_hashes)
        if native["step"] != receipt.get("step") or native["execution"] != receipt["execution"]:
            raise ValueError("Native model and trainer manifest disagree")
        weights = member(receipt.get("weights_path"))
    elif receipt.get("format") is not None:
        raise ValueError("Unknown model checkpoint format")
    else:
        weights = model / "model_state_dict" / "full_weights.pt"
    if weights not in seen or weights.stat().st_size == 0:
        raise ValueError("completed checkpoint has no weights")
    return {"path": str(model), "bytes": total, "step": receipt.get("step"), "manifest": receipt}
