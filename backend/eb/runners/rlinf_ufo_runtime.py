"""Python 3.10 task-only validation and native UFO configuration helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

OUTPUT_ROOT = Path("/run/eb-output")
NATIVE_ROOT = OUTPUT_ROOT / "outputs/native"
VALUE_FLAGS = {
    "--robot-config",
    "--data-path",
    "--work-dir",
    "--num-envs",
    "--num-env-steps",
    "--buffer-size",
    "--seed",
    "--checkpoint-every-steps",
    "--buffer-every-steps",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2))
    temporary.replace(path)


def canonical_plan_hash(plan: dict) -> str:
    value = {key: value for key, value in plan.items() if key != "hash"}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def validate_plan(plan: dict) -> dict:
    if plan.get("schema_version") != "eb/ufo-native-train-plan/v1":
        raise ValueError("Unsupported UFO training plan")
    if not re.fullmatch(r"run_[a-zA-Z0-9_-]+", str(plan.get("run_id", ""))):
        raise ValueError("Invalid run_id")
    for key in ("fingerprint", "image_digest"):
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(plan.get(key, ""))):
            raise ValueError(f"Missing frozen {key}")
    for key, maximum in (("world_size", 4), ("target_updates", 1_000_000)):
        if type(plan.get(key)) is not int or not 1 <= plan[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    if plan["target_updates"] < 112 or plan["target_updates"] % 16:
        raise ValueError("target_updates must be 112..1000000 in native batches of 16")
    if plan.get("hash") and plan["hash"] != canonical_plan_hash(plan):
        raise ValueError("Frozen plan hash mismatch")
    argv = plan.get("native_args")
    if not isinstance(argv, list) or not argv or len(argv) > 64 or not all(isinstance(x, str) for x in argv):
        raise ValueError("native_args must be a bounded argument list")
    flags = set()
    active = None
    values = 0
    for token in argv:
        if token.startswith("--"):
            if active is not None and values == 0:
                raise ValueError(f"Missing value for {active}")
            if token not in VALUE_FLAGS or token in flags:
                raise ValueError(f"Forbidden or duplicate native argument: {token}")
            flags.add(token)
            active, values = token, 0
        else:
            if active is None or (values and active != "--data-path") or "\x00" in token:
                raise ValueError("Invalid native argument value")
            values += 1
    if not values or not {"--robot-config", "--data-path", "--work-dir", "--num-env-steps"} <= flags:
        raise ValueError("Required frozen native arguments are missing")
    if argv[argv.index("--work-dir") + 1] != str(NATIVE_ROOT):
        raise ValueError("Native work directory must belong to this Run")
    for key, default, maximum in (
        ("communication_timeout_seconds", 300, 1800),
        ("segment_timeout_seconds", 21600, 21600),
    ):
        value = plan.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"Invalid {key}")
    return plan


def verify_source_refs(plan: dict) -> dict:
    identities = {}
    for name in ("ufo", "rlinf"):
        ref = plan["source_refs"][name]
        root = Path(ref["path"])
        if str(root) != f"/sources/{name}":
            raise ValueError(f"Unexpected {name} source root")
        manifest_path = Path(ref["manifest_path"])
        if str(manifest_path) != f"/sources/{name}-manifest.json":
            raise ValueError("Unexpected source manifest path")
        if file_hash(manifest_path) != ref["snapshot_hash"]:
            raise ValueError(f"{name} source manifest identity mismatch")
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["entries"]:
            relative = Path(entry["path"])
            path = root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError("Source manifest path escape")
            if path.is_symlink():
                raise ValueError("Source manifest symlink")
            if entry["sha256"] is None:
                if not path.is_dir():
                    raise ValueError("Source directory missing")
            elif file_hash(path) != "sha256:" + entry["sha256"].removeprefix("sha256:"):
                raise ValueError(f"{name} source file mismatch: {relative}")
        identities[name] = dict(ref)
    return identities


def verify_training_data(plan: dict, root: Path = Path("/training-data")) -> dict:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not 0 < manifest_path.stat().st_size <= 4 * 1024 * 1024:
        raise ValueError("Frozen training manifest is missing or unbounded")
    identity = plan["identity"]
    if file_hash(manifest_path).removeprefix("sha256:") != identity["dataset_manifest"]:
        raise ValueError("Frozen training manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != "eb/ufo-native-motion/v1"
        or manifest.get("split") != "train"
        or manifest.get("train_file") != "train.pkl"
        or manifest.get("dataset_id") != identity["dataset_id"]
        or manifest.get("native_loader", {}).get("all_motions_loaded") is not True
        or {path.name for path in root.iterdir()} != {"train.pkl", "manifest.json"}
    ):
        raise ValueError("Training mount must contain only its frozen native training split")
    payload = root / "train.pkl"
    if payload.is_symlink() or file_hash(payload).removeprefix("sha256:") != manifest["file_sha256"]:
        raise ValueError("Frozen training pickle digest mismatch")
    argv = plan["native_args"]
    data_index = argv.index("--data-path") + 1
    if argv[data_index] != "/training-data/train.pkl" or (
        data_index + 1 < len(argv) and not argv[data_index + 1].startswith("--")
    ):
        raise ValueError("Only the frozen training split may enter native training")
    robot = Path(argv[argv.index("--robot-config") + 1])
    if file_hash(robot).removeprefix("sha256:") != manifest["robot_config_sha256"]:
        raise ValueError("Training dataset robot configuration differs from frozen source")
    import yaml

    xml = Path(yaml.safe_load(robot.read_text())["xml_path"])
    if not xml.is_absolute():
        xml = Path(plan["source_refs"]["ufo"]["path"]) / xml
    if file_hash(xml).removeprefix("sha256:") != manifest["robot_xml_sha256"]:
        raise ValueError("Training dataset robot XML differs from frozen source")
    return manifest


def configure_environment(plan: dict) -> None:
    runtime = OUTPUT_ROOT / "runtime"
    ray_root = runtime / "ray"
    if os.environ.get("RAY_ADDRESS"):
        raise ValueError("External RAY_ADDRESS is forbidden")
    if os.environ.get("RAY_TMPDIR") not in (None, "", str(ray_root)):
        raise ValueError("Inherited/shared Ray temporary directory is forbidden")
    if os.environ.get("RLINF_EXT_MODULE"):
        raise ValueError("Unfrozen RLinf extension is forbidden")
    # Ray appends /ray to RAY_TMPDIR; it must start without a discovery file.
    if (ray_root / "ray/ray_current_cluster").exists():
        raise ValueError("Run already has a Ray discovery file")
    cache = OUTPUT_ROOT / "outputs/cache"
    env = {
        "RAY_TMPDIR": str(ray_root),
        "UFO_CACHE_DIR": str(cache),
        "TMPDIR": str(runtime / "tmp"),
        "TEMP": str(runtime / "tmp"),
        "TMP": str(runtime / "tmp"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "WARP_CACHE_PATH": str(cache / "warp"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "XDG_CONFIG_HOME": str(cache / "config"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "WANDB_MODE": "disabled",
        "WANDB_DIR": str(cache / "wandb"),
        "RLINF_CODE_WORKING_DIR": "0",
        "RLINF_CATCH_FAILURE": "0",
        # RLinf interprets this setting in minutes; native NCCL uses seconds.
        "RLINF_TIMEOUT": str((plan.get("communication_timeout_seconds", 300) + 59) // 60),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "MUJOCO_GL": "egl",
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(512 * 1024 * 1024),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for value in env.values():
        if value.startswith(str(OUTPUT_ROOT)):
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    for name in ("rlinf", "ufo"):
        root = plan["source_refs"][name]["path"]
        if name in sys.modules:
            raise RuntimeError(f"{name} imported before frozen source selection")
        sys.path.insert(0, root)
    os.environ["PYTHONPATH"] = ":".join(
        [plan["source_refs"]["ufo"]["path"], plan["source_refs"]["rlinf"]["path"], "/opt/eb-native"]
    )
    os.chdir(OUTPUT_ROOT)


@contextmanager
def scoped_argv(argv: list[str]):
    previous = sys.argv
    sys.argv = ["ufo.train", *argv]
    try:
        yield
    finally:
        sys.argv = previous


def parse_native_args(plan: dict) -> dict:
    from ufo.train import parse_args

    argv = list(plan["native_args"])
    if plan.get("log_every_steps") is not None:
        import yaml

        interval = plan["log_every_steps"]
        if type(interval) is not int or interval < 1:
            raise ValueError("Invalid log_every_steps")
        index = argv.index("--robot-config") + 1
        source = Path(argv[index])
        robot = yaml.safe_load(source.read_text())
        # Relocating YAML must retain the original robot XML resolution.
        from ufo.tools.robot_config import resolve_path

        xml_path = resolve_path(robot["xml_path"], base_dir=source.parent)
        if not xml_path.is_file():
            xml_path = Path(plan["source_refs"]["ufo"]["path"]) / robot["xml_path"]
        if not xml_path.is_file():
            raise ValueError("Frozen robot XML is missing")
        robot["xml_path"] = str(xml_path.resolve())
        robot.setdefault("training", {}).setdefault("train_overrides", {})["log_every_steps"] = interval
        derived = OUTPUT_ROOT / "inputs/robot.yaml"
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_text(yaml.safe_dump(robot, sort_keys=False))
        write_json(
            OUTPUT_ROOT / "outputs/robot-config-receipt.json",
            {
                "source_path": str(source),
                "source_sha256": file_hash(source),
                "derived_path": str(derived),
                "derived_sha256": file_hash(derived),
                "xml_path": str(xml_path),
                "xml_sha256": file_hash(xml_path),
                "overrides": {"training.train_overrides.log_every_steps": interval},
            },
        )
        argv[index] = str(derived)
    with scoped_argv(argv):
        args = parse_args()
    paths = [args.data_path] if isinstance(args.data_path, str) else args.data_path
    for value in [*paths, args.robot_config]:
        if not Path(value).is_file():
            raise ValueError(f"Frozen native input missing: {value}")
    if args.agent != "fb" or args.use_wandb or args.gpu_ids != "single":
        raise ValueError("Unsupported native launch mode")
    for key in ("num_envs", "num_env_steps", "buffer_size", "checkpoint_every_steps", "buffer_every_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"Invalid native {key}")
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def build_native_config(args: dict, rank: int, world_size: int):
    from ufo.train import build_mjlab_config

    # Keep this mapping identical to native run_train; no algorithm defaults here.
    names = (
        "num_envs",
        "num_env_steps",
        "wandb_run_name",
        "checkpoint_every_steps",
        "buffer_every_steps",
        "data_path",
        "data_mix_weights",
        "update_z_every_step",
        "buffer_size",
        "lr_scale",
        "clip_grad_norm",
        "num_agent_updates",
        "robot_config",
        "network_width",
        "shared_critic",
        "agent",
    )
    values = {key: args[key] for key in names}
    values.update(
        {
            key: bool(args[key])
            for key in (
                "disable_eval_prioritization",
                "smoke",
                "disable_dr",
                "disable_obs_noise",
                "cartwheel_aux_safe",
            )
        }
    )
    return build_mjlab_config(
        device="cuda:0",
        work_dir=args["work_dir"],
        seed=args["seed"] + rank,
        use_wandb=False,
        distributed_rank=rank,
        distributed_world_size=world_size,
        **values,
    )


def prepare_segment(plan: dict, args: dict) -> None:
    native = NATIVE_ROOT
    if (native / "checkpoint").exists():
        raise ValueError("New Run work directory already contains a checkpoint")
    native.mkdir(parents=True, exist_ok=True)
    start = buffer_start = 0
    if plan.get("resume"):
        source = Path(plan["resume"]["checkpoint_path"]).resolve(strict=True)
        if source.is_relative_to(OUTPUT_ROOT):
            raise ValueError("Resume source must be a separate read-only input")
        if (source / ".saving").exists():
            raise ValueError("Incomplete resume checkpoint")
        from eb.ufo_training_checkpoint import validate_native_checkpoint

        manifest = json.loads((source.parent / "checkpoint-manifest.json").read_text())
        fresh = plan["resume"].get("replay_policy") == "fresh"
        source_plan = plan["resume"].get("source_plan") if fresh else plan
        if fresh:
            from eb.ufo_training_checkpoint import validate_reconfigured_resume
            validate_reconfigured_resume(plan, source_plan)
        validate_native_checkpoint(
            source.parent, manifest, run_id=manifest["run_id"], plan=source_plan, verify_hashes=True
        )
        status = json.loads((source / "train_status.json").read_text())
        if status["world_size"] != plan["world_size"]:
            raise ValueError("Resume world_size mismatch")
        start = status["global_time"]
        buffer_start = status["buffer_global_time"]
        if not status.get("replay_id"):
            raise ValueError("Resume requires complete all-rank Replay")
        for rank in range(plan["world_size"]):
            shard = source / "buffers" / f"train_rank_{rank}"
            marker = json.loads((shard / "snapshot.json").read_text())
            if any(
                marker.get(key) != value
                for key, value in {
                    "id": status["replay_id"],
                    "global_time": start,
                    "world_size": plan["world_size"],
                    "transition_format": 2,
                }.items()
            ):
                raise ValueError("Resume Replay snapshot mismatch")
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Resume symlinks forbidden")
        # Missing buffers deliberately selects UFO's native fresh-collection branch.
        shutil.copytree(source, native / "checkpoint", ignore=(
            lambda root, names: ["buffers"] if Path(root) == source and "buffers" in names else []
        ) if fresh else None)
        write_json(native / "initialization.json", {
            "source_run_id": manifest["run_id"], "source_counts": {
                key: status[key] for key in ("local_time", "global_time", "optimizer_steps")
            }, "model": "restored", "optimizer": "restored",
            "replay_per_rank": "reinitialized" if fresh else "restored",
            "source_fingerprint": manifest.get("compatibility_fingerprint"),
            "target_fingerprint": plan["fingerprint"],
        })
    endpoint = args["num_env_steps"]
    increment = args["num_envs"] * plan["world_size"]
    if endpoint <= start or endpoint % increment:
        raise ValueError("Endpoint must advance and align to global parallel environments")
    if (
        start + args["checkpoint_every_steps"] < endpoint
        or buffer_start + args["buffer_every_steps"] < endpoint
    ):
        raise ValueError("SAVE_AND_STOP would stop at an earlier saving boundary")
    (native / "SAVE_AND_STOP").touch()


def verify_full_native_state_finite(
    agent, *, rank: int, local_time: int, global_time: int, optimizer_steps: int
) -> None:
    """Use UFO's complete model and nested optimizer checks before publication."""
    from ufo.training.logging import _assert_finite, _assert_model_finite

    counts = {
        "rank": rank,
        "local_time": local_time,
        "global_time": global_time,
        "optimizer_steps": optimizer_steps,
    }
    _assert_model_finite(agent._model, **counts)
    _assert_finite(agent.optimizer_dict, label="agent.optimizer_dict", **counts)


def parameter_snapshot(model) -> tuple[dict, dict]:
    """Read model parameters without changing native training or gradient state."""
    import torch

    digest, samples, records = hashlib.sha256(), {}, []
    prefixes = ("_actor.", "_forward_map.", "_backward_map.")
    for name, value in sorted(model.named_parameters()):
        if not value.requires_grad or not name.startswith(prefixes):
            continue
        cpu = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(cpu).all()):
            raise ValueError("Native trainable parameters are nonfinite")
        digest.update(json.dumps([name, list(cpu.shape), str(cpu.dtype)]).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        samples[name] = cpu.reshape(-1)[:64].float().clone()
        records.append({"name": name, "shape": list(cpu.shape), "sample_elements": samples[name].numel()})
    if not records or any(not any(row["name"].startswith(prefix) for row in records) for prefix in prefixes):
        raise ValueError("Native actor/forward/backward trainable parameters are missing")
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "parameters": records,
        "sample_definition": "First 64 flattened elements per trainable actor/forward/backward parameter",
    }, samples


def parameter_change(before: tuple[dict, dict], after: tuple[dict, dict]) -> dict:
    if before[0]["parameters"] != after[0]["parameters"]:
        raise ValueError("Native parameter layout changed during training")
    squared_norm, maximum = 0.0, 0.0
    for name, value in before[1].items():
        delta = (after[1][name].double() - value.double()).abs()
        squared_norm += delta.square().sum().item()
        maximum = max(maximum, delta.max().item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or not math.isfinite(maximum):
        raise ValueError("Native parameter update evidence is nonfinite")
    if before[0]["sha256"] == after[0]["sha256"] or norm <= 0:
        raise ValueError("Native training did not change sampled trainable weights")
    return {
        "before_sha256": before[0]["sha256"],
        "after_sha256": after[0]["sha256"],
        "sample_delta_l2": norm,
        "sample_delta_max_abs": maximum,
        "parameters": after[0]["parameters"],
        "sample_definition": after[0]["sample_definition"],
    }
