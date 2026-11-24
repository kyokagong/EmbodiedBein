"""Python 3.10 task helpers; P-BFM retains ownership of its numerical loop."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

from eb.pbfm_training import native_arguments, training_fingerprint, validate_frozen_plan
from eb.runners.pbfm_source import CAMERA_CONTRACT, file_sha256, prepare_pbfm_source

OUTPUT_ROOT = Path("/run/eb-output")
STAGING_ROOT = OUTPUT_ROOT / "runtime/native-staging"
NATIVE_ROOT = OUTPUT_ROOT / "outputs/native"
PRIVATE_SOURCE = OUTPUT_ROOT / "runtime/pbfm"
OPTIMIZERS = {
    "actor_optimizer",
    "backward_optimizer",
    "forward_optimizer",
    "critic_optimizer",
    "discriminator_optimizer",
    "aux_critic_optimizer",
    "velocity_critic_optimizer",
    "heading_critic_optimizer",
}


def resolved_environment_config(config) -> dict:
    """Resolve native robot metadata with its standard length interpolation."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("len"):
        OmegaConf.register_new_resolver("len", len)
    return OmegaConf.to_container(config, resolve=True)


def write_json(path: Path, value: dict) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Task output symlink is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def canonical_plan_hash(plan: dict) -> str:
    return training_fingerprint({key: value for key, value in plan.items() if key != "hash"})


def validate_plan(plan: dict) -> dict:
    validate_frozen_plan(plan)
    if plan.get("long_exploration"):
        from eb.pbfm_longrun import runtime_compatibility_receipt

        receipt = json.loads(Path("/opt/pbfm-long-runtime.json").read_text())
        if (receipt != runtime_compatibility_receipt()
                or training_fingerprint(receipt) != plan["long_exploration"]["runtime_receipt_sha256"]):
            raise ValueError("P-BFM derived runtime receipt differs")
    if not re.fullmatch(r"run_[a-zA-Z0-9_-]+", str(plan.get("run_id", ""))):
        raise ValueError("Invalid P-BFM run_id")
    if plan.get("hash") and plan["hash"] != canonical_plan_hash(plan):
        raise ValueError("P-BFM frozen plan hash differs")
    return plan


def verify_source_refs(plan: dict) -> dict:
    for name, ref in plan["source_refs"].items():
        root, manifest_path = Path(ref["path"]), Path(ref["manifest_path"])
        if (
            manifest_path.is_symlink()
            or not 0 < manifest_path.stat().st_size <= 16 * 1024**2
            or "sha256:" + file_sha256(manifest_path) != ref["snapshot_hash"]
        ):
            raise ValueError(name + " snapshot manifest identity differs")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest.get("entries"), list) or not 0 < len(manifest["entries"]) <= 100000:
            raise ValueError("Frozen source manifest is empty or unbounded")
        seen = set()
        for entry in manifest["entries"]:
            relative = Path(entry["path"])
            path = root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() in seen
                or any(p.is_symlink() for p in (path, *path.parents))
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError("Frozen source member path escape or symlink")
            seen.add(relative.as_posix())
            if entry["sha256"] is None:
                if not path.is_dir():
                    raise ValueError("Frozen source directory is missing")
            elif file_sha256(path) != entry["sha256"].removeprefix("sha256:"):
                raise ValueError("Frozen source payload differs: " + str(relative))
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError("Frozen source contains a symlink or special file")
            actual.add(path.relative_to(root).as_posix())
        if actual != seen:
            raise ValueError("Frozen source contains an unindexed or missing payload")
    return plan["source_refs"]


def configure_environment(plan: dict) -> None:
    runtime, cache = OUTPUT_ROOT / "runtime", OUTPUT_ROOT / "runtime/cache"
    ray_root = runtime / "ray"
    if os.environ.get("RAY_ADDRESS") or os.environ.get("RLINF_EXT_MODULE"):
        raise ValueError("External Ray or RLinf extension is forbidden")
    if (
        os.environ.get("RAY_TMPDIR") not in (None, "", str(ray_root))
        or (ray_root / "ray/ray_current_cluster").exists()
    ):
        raise ValueError("Shared or reused Ray discovery directory is forbidden")
    env = {
        "RAY_TMPDIR": str(ray_root),
        "UFO_CACHE_DIR": str(cache / "expert"),
        "TMPDIR": str(runtime / "tmp"),
        "WARP_CACHE_PATH": str(cache / "warp"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "XDG_CONFIG_HOME": str(cache / "config"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "WANDB_DIR": str(cache / "wandb"),
        "WANDB_MODE": "disabled",
        "MUJOCO_GL": "egl",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RLINF_CODE_WORKING_DIR": "0",
        "RLINF_CATCH_FAILURE": "0",
        "RLINF_TIMEOUT": str((plan.get("communication_timeout_seconds", 300) + 59) // 60),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(512 * 1024**2),
    }
    for value in env.values():
        if value.startswith(str(OUTPUT_ROOT)):
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    receipt = prepare_pbfm_source(Path(plan["source_refs"]["pbfm"]["path"]), PRIVATE_SOURCE)
    write_json(OUTPUT_ROOT / "outputs/source-adaptation.json", receipt)
    for name, root in (("pbfm", str(PRIVATE_SOURCE)), ("rlinf", plan["source_refs"]["rlinf"]["path"])):
        if name in sys.modules:
            raise RuntimeError("Numerical source imported before frozen source selection")
        sys.path.insert(0, root)
    os.environ["PYTHONPATH"] = ":".join(
        [str(PRIVATE_SOURCE), plan["source_refs"]["rlinf"]["path"], "/opt/eb-native"]
    )
    os.chdir(OUTPUT_ROOT)


def verify_training_data(plan: dict, root: Path = Path("/training-data")) -> dict:
    path = root / "manifest.json"
    if path.is_symlink() or not 0 < path.stat().st_size <= 4 * 1024**2:
        raise ValueError("P-BFM dataset manifest is missing or unbounded")
    identity = plan["identity"]
    if file_sha256(path) != identity["dataset_manifest"]:
        raise ValueError("P-BFM dataset manifest changed")
    data = json.loads(path.read_text())
    payload = root / "train.pkl"
    if (
        data.get("schema_version") != "eb/pbfm-rpo-motion/v1"
        or data.get("split") != "train"
        or data.get("train_file") != "train.pkl"
        or data.get("dataset_id") != identity["dataset_id"]
        or data.get("native_loader", {}).get("all_motions_loaded") is not True
        or data.get("fk", {}).get("verified") is not True
        or {p.name for p in root.iterdir()} != {"train.pkl", "manifest.json"}
        or payload.is_symlink()
        or file_sha256(payload) != data.get("file_sha256")
    ):
        raise ValueError("P-BFM mount must contain only the frozen FK-verified training split")
    adaptation = json.loads((PRIVATE_SOURCE / ".eb-pbfm-source.json").read_text())
    from pbfm.robots import load_robot_training_spec

    from eb.runners.pbfm_source import rpo_robot_config

    native = load_robot_training_spec(rpo_robot_config(PRIVATE_SOURCE))
    if (
        data.get("pbfm_patch_sha256") != adaptation["patch_sha256"]
        or data.get("pbfm_snapshot") != identity["pbfm_snapshot"]
        or data.get("asset_snapshot") != identity["asset_snapshot"]
        or data.get("robot_xml_sha256") != file_sha256(Path(native.robot.xml_path))
        or data.get("joint_order") != native.robot.control_joint_names
        or data.get("body_order") != native.robot.body_names
    ):
        raise ValueError("P-BFM training data robot/source/asset/patch identity differs")
    return data


def parse_native_args(plan: dict) -> dict:
    validate_plan(plan)
    argv = native_arguments(plan)
    return {
        argv[i][2:].replace("-", "_"): (argv[i + 1] if i < 6 else int(argv[i + 1]))
        for i in range(0, len(argv), 2)
    }


def build_native_config(args: dict, rank: int, world: int):
    from pbfm.train import build_ufo_mjlab_config

    cfg = build_ufo_mjlab_config(
        device="cuda:0",
        work_dir=args["work_dir"],
        num_envs=args["num_envs"],
        num_env_steps=args["num_env_steps"],
        seed=args["seed"] + rank,
        use_wandb=False,
        checkpoint_every_steps=args["checkpoint_every_steps"],
        tracking_eval_every_steps=args["num_env_steps"] + 1,
        same_z_eval_every_steps=0,
        distributed_rank=rank,
        distributed_world_size=world,
        data_path=args["data_path"],
        buffer_size=args["buffer_size"],
        prior_plane_envs=0,
        robot_config=args["robot_config"],
        terrain_mode="training_mix",
        gradient_sync="auto",
        cache_expert_buffer=True,
    )
    overrides = dict(item.split("=", 1) for item in cfg.env.hydra_overrides)
    camera = CAMERA_CONTRACT
    overrides.update(
        {
            "simulator.config.sim.fps": "400",
            "simulator.config.sim.control_decimation": "8",
            "terrain.direct_depth.mount_body": camera["mount_body"],
            "terrain.direct_depth.mount_pos": json.dumps(camera["mount_pos"]),
            "terrain.direct_depth.camera_frame_quat_wxyz": json.dumps(camera["camera_frame_quat_wxyz"]),
        }
    )
    env = cfg.env.model_copy(
        update={
            "terrain_curriculum_start_global_step": 0,
            "terrain_curriculum_interval": 19200000,
            "hydra_overrides": [f"{key}={value}" for key, value in overrides.items()],
        }
    )
    cfg = cfg.model_copy(
        update={
            "env": env,
            "checkpoint_buffer": True,
            "checkpoint_rank_buffers": True,
            "log_every_updates": 1024 * world,
        }
    )
    # Revalidate the complete resolved structure; model_copy intentionally skips validation.
    return type(cfg).model_validate(cfg.model_dump(), strict=False)


def prepare_segment(plan: dict, args: dict) -> None:
    if STAGING_ROOT.exists() or NATIVE_ROOT.exists():
        raise ValueError("P-BFM segment output must be new")
    resume = plan.get("resume")
    if resume:
        from eb.pbfm_training_checkpoint import validate_pbfm_checkpoint

        source = Path("/resume-native")
        binding = resume.get("source_binding") or {}
        if (
            binding.get("run_id") != resume.get("source_run_id")
            or "sha256:" + file_sha256(source / "checkpoint-manifest.json") != binding.get("manifest_sha256")
            or "sha256:" + file_sha256(Path("/resume-receipt.json")) != binding.get("receipt_sha256")
        ):
            raise ValueError("P-BFM immutable resume manifest binding differs")
        manifest = json.loads((source / "checkpoint-manifest.json").read_text())
        source_receipt = json.loads(Path("/resume-receipt.json").read_text())
        periodic = bool(plan.get("recovery"))
        receipt_manifest = source_receipt.get("manifest_sha256" if periodic else "native_manifest_sha256")
        if (
            source_receipt.get("status") != "complete"
            or source_receipt.get("run_id") != resume["source_run_id"]
            or receipt_manifest != file_sha256(source / "checkpoint-manifest.json")
            or periodic and (source_receipt.get("schema_version") != "eb/pbfm-periodic-recovery/v1"
                             or source_receipt.get("training_complete") is not False
                             or source_receipt.get("parent_plan_hash")
                             != plan["recovery"]["parent_plan_hash"])
        ):
            raise ValueError("P-BFM resume source receipt does not bind the complete manifest")
        from eb.pbfm_longrun import checkpoint_source_plan

        checkpoint = validate_pbfm_checkpoint(
            source, manifest, run_id=resume["source_run_id"], plan=checkpoint_source_plan(plan),
            verify_hashes=True, resume=True
        )
        if (
            checkpoint["step"] != plan["start_updates"]
            or checkpoint["counts"]["local_time"] != plan["start_local_time"]
        ):
            raise ValueError("P-BFM resume counters differ")
        STAGING_ROOT.mkdir(parents=True)
        shutil.copytree(source / "checkpoint", STAGING_ROOT / "checkpoint")
    else:
        STAGING_ROOT.mkdir(parents=True)
    shutil.copyfile(OUTPUT_ROOT / "outputs/source-adaptation.json", STAGING_ROOT / "source-adaptation.json")


def verify_native_state(
    agent, checkpoint: Path | None = None, *, expected_updates: int | None = None
) -> dict:
    """Check every optimizer and complete model keys, including targets/normalizers."""
    import torch

    def finite(value):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("P-BFM state has NaN/Inf")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                finite(child)

    state, optimizers = agent._model.state_dict(), agent.optimizer_dict
    if set(optimizers) != OPTIMIZERS:
        raise ValueError("P-BFM requires all eight optimizers")
    if not state or not any("target" in key for key in state) or not any("normal" in key for key in state):
        raise ValueError("P-BFM model target/normalizer state is missing")
    finite(state)
    finite(optimizers)
    steps = {}
    for name, optimizer in optimizers.items():
        if not optimizer.get("state") or not optimizer.get("param_groups"):
            raise ValueError("P-BFM optimizer has no initialized state")
        steps[name] = sorted(
            {int(item["step"].item()) for item in optimizer["state"].values() if "step" in item}
        )
        if not steps[name] or min(steps[name]) <= 0:
            raise ValueError("P-BFM optimizer has not updated")
        parameter_ids = {parameter for group in optimizer["param_groups"] for parameter in group["params"]}
        if set(optimizer["state"]) != parameter_ids:
            raise ValueError("P-BFM optimizer parameter state is incomplete")
        if expected_updates is not None and steps[name] != [expected_updates]:
            raise ValueError("P-BFM optimizer step differs from native joint updates")
    if checkpoint is not None:
        from safetensors.torch import load_file

        saved = load_file(str(checkpoint / "model/model.safetensors"), device="cpu")
        if set(saved) != set(state) or any(saved[key].shape != state[key].shape for key in state):
            raise ValueError("P-BFM serialized model key/shape differs from complete model")
        agent._model.load_state_dict(saved, strict=True)
        saved_optimizers = torch.load(checkpoint / "optimizers.pth", map_location="cpu", weights_only=True)
        if set(saved_optimizers) != OPTIMIZERS:
            raise ValueError("P-BFM serialized optimizer is incomplete")
        finite(saved)
        finite(saved_optimizers)

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                return (
                    isinstance(right, torch.Tensor)
                    and left.shape == right.shape
                    and left.dtype == right.dtype
                    and torch.equal(left.cpu(), right.cpu())
                )
            if isinstance(left, dict):
                return (
                    isinstance(right, dict)
                    and left.keys() == right.keys()
                    and all(equal(left[k], right[k]) for k in left)
                )
            if isinstance(left, (list, tuple)):
                return (
                    isinstance(right, (list, tuple))
                    and len(left) == len(right)
                    and all(equal(a, b) for a, b in zip(left, right))
                )
            return left == right

        if not equal(saved_optimizers, optimizers):
            raise ValueError("P-BFM optimizer state/parameter groups changed on native loader roundtrip")
    return {
        "optimizer_names": sorted(OPTIMIZERS),
        "optimizer_state_steps": steps,
        "model_state_keys": len(state),
        "strict_state_verified": checkpoint is not None,
    }


def parameter_hash(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        cpu = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(cpu).all()):
            raise ValueError("P-BFM parameters have NaN/Inf")
        digest.update(name.encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def rng_state_hash() -> str:
    """Observe native RNG state without drawing or resetting any randomness."""
    import pickle
    import random

    import numpy as np
    import torch

    digest = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())))
    digest.update(torch.get_rng_state().numpy().tobytes())
    digest.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def configure_simulation_execution(plan: dict) -> bool:
    """Select MJLab's supported native eager path for the frozen E2048 profile."""
    eager = plan["identity"]["num_envs_per_rank"] == 2048
    if eager:
        import warp as wp
        from mujoco_warp._src.collision_convex import eb_enable_dynamic_scratch

        wp.set_mempool_enabled("cuda:0", False)
        eb_enable_dynamic_scratch()
    return eager


def simulation_execution_evidence(env, *, eager: bool) -> dict:
    import warp as wp
    from mujoco_warp._src.collision_convex import eb_dynamic_scratch_evidence

    sim = env.base_env.mjlab_env.sim
    evidence = {
        "mode": "cuda_graph" if sim.use_cuda_graph else "native_eager",
        "mempool_enabled": bool(wp.is_mempool_enabled(sim.wp_device)),
        "use_cuda_graph": bool(sim.use_cuda_graph),
        "graphs": {name: getattr(sim, name + "_graph") is not None
                   for name in ("step", "forward", "reset", "sense")},
        "num_envs": sim.num_envs, "nconmax": sim.cfg.nconmax, "njmax": sim.cfg.njmax,
        "convex_scratch": eb_dynamic_scratch_evidence(),
    }
    if (evidence["nconmax"] != 512 or evidence["njmax"] != 4096
            or eager and (evidence["mempool_enabled"] or evidence["use_cuda_graph"]
                          or any(evidence["graphs"].values()) or not evidence["convex_scratch"]["enabled"])):
        raise ValueError("P-BFM native simulation execution layout or constraint capacity differs")
    return evidence


def install_simulation_memory_handoff(trainer, rank: int, snapshots: list, execution: dict) -> None:
    """Return idle Torch memory before the unchanged native main sampling step."""
    import gc
    from functools import wraps

    import torch

    original_step = trainer.train_env.step
    sample = 0

    @wraps(original_step)
    def step(*args, **kwargs):
        nonlocal sample
        sample += 1
        if sample == 1:
            prior = getattr(trainer, "_priority_eval_env", None)
            execution["priority_eval"] = (
                simulation_execution_evidence(prior, eager=True)
                if prior is not None else {"available": False}
            )
            write_json(OUTPUT_ROOT / f"outputs/runtime/execution-rank-{rank}.json", execution)
        torch.cuda.synchronize()
        record_cuda_memory(rank, "before_sample", snapshots, sample=sample)
        gc.collect()
        torch.cuda.empty_cache()
        record_cuda_memory(rank, "after_sample_cleanup", snapshots, sample=sample)
        return original_step(*args, **kwargs)

    trainer.train_env.step = step


class MemorySnapshots(list):
    """Bound debug history while preserving cumulative allocator peaks."""

    def __init__(self):
        super().__init__()
        self.total_snapshots = 0
        self.peak_memory = {}

    def append(self, snapshot):
        self.total_snapshots += 1
        for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes"):
            if key in snapshot:
                self.peak_memory[key] = max(self.peak_memory.get(key, 0), snapshot[key])
        if len(self) == 256:
            del self[8]
        super().append(snapshot)


def record_cuda_memory(rank: int, stage: str, snapshots: list, **details) -> None:
    """Best-effort telemetry; an OOM or output failure must retain its first error."""
    import time

    import torch

    snapshot = {"stage": stage, "time_ns": time.time_ns(), **details}
    convex = sys.modules.get("mujoco_warp._src.collision_convex")
    if convex is not None and hasattr(convex, "eb_dynamic_scratch_evidence"):
        snapshot["convex_scratch"] = convex.eb_dynamic_scratch_evidence()
    try:
        free, total = torch.cuda.mem_get_info()
        snapshot.update(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_bytes=free, total_bytes=total,
        )
    except Exception as exc:
        snapshot["snapshot_error"] = type(exc).__name__ + ": " + str(exc)[:500]
    snapshots.append(snapshot)
    try:
        value = {"schema_version": "eb/pbfm-cuda-memory/v1", "rank": rank, "snapshots": snapshots}
        if isinstance(snapshots, MemorySnapshots):
            value.update(total_snapshots=snapshots.total_snapshots,
                         dropped_snapshots=snapshots.total_snapshots - len(snapshots),
                         peak_memory=snapshots.peak_memory)
        write_json(OUTPUT_ROOT / f"outputs/runtime/memory-rank-{rank}.json", value)
    except Exception as exc:
        try:
            print(f"P-BFM memory telemetry unavailable: {type(exc).__name__}: {str(exc)[:500]}",
                  file=sys.stderr, flush=True)
        except Exception:
            pass


def build_training_evidence(root: Path, plan: dict, counts: dict) -> dict:
    log = root / "outputs/native/train_log.txt"
    if not log.is_file() or log.stat().st_size > 32 * 1024**2:
        raise ValueError("P-BFM raw training log missing or too large for bounded analysis")
    with log.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("P-BFM training log has no metrics")
    numeric, quality = {}, {}
    for key in rows[0]:
        values, missing, invalid, nonfinite = [], 0, 0, 0
        for row in rows:
            raw = row.get(key)
            if raw in (None, ""):
                values.append(None)
                missing += 1
                continue
            try:
                value = float(raw)
            except (ValueError, TypeError):
                values.append(None)
                invalid += 1
                continue
            if not math.isfinite(value):
                values.append(None)
                nonfinite += 1
            else:
                values.append(value)
        if any(value is not None for value in values):
            numeric[key] = values
            quality[key] = {
                "finite_samples": sum(v is not None for v in values),
                "missing_samples": missing,
                "invalid_samples": invalid,
                "nonfinite_samples": nonfinite,
            }
    axis = "distributed/optimizer_steps"
    if axis not in numeric or any(value is None for value in numeric[axis]):
        raise ValueError("P-BFM training analysis requires actual native optimizer update counts")
    xs = numeric[axis]
    if any(value < 0 or value != int(value) or value > counts["optimizer_steps"] for value in xs) or any(
        a > b for a, b in zip(xs, xs[1:])
    ):
        raise ValueError("P-BFM native optimizer update axis is invalid")
    series = [
        key for key in numeric if any(word in key.lower() for word in ("loss", "critic", "actor", "fps"))
    ]
    series.sort(key=lambda key: ("loss" not in key.lower(), key))
    if not series:
        raise ValueError("P-BFM log has no numeric learning series")
    analysis = root / "outputs/analysis"
    analysis.mkdir(parents=True)
    shutil.copyfile(log, analysis / "train-timeseries.csv")
    width, panel_height = 960, 180
    charts = []
    for index, key in enumerate(series[:24]):
        values = numeric[key]
        finite_values = [v for v in values if v is not None]
        low, high = min(finite_values), max(finite_values)
        commands, markers, previous = [], [], False
        for x, value in zip(xs, values):
            if value is None:
                previous = False
                continue
            px = 65 + (x - min(xs)) * 845 / max(1, max(xs) - min(xs))
            py = index * panel_height + 125 - (value - low) * 90 / max(high - low, 1e-12)
            commands.append(f"{'L' if previous else 'M'}{px:.2f},{py:.2f}")
            markers.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.5" fill="#2266bb"/>')
            previous = True
        charts.append(
            f'<text x="65" y="{index * panel_height + 22}">{html.escape(key)} '
            f"[min {low:.4g}, max {high:.4g}]</text>"
            f'<path d="{" ".join(commands)}" fill="none" stroke="#2266bb"/>'
            f'<text x="65" y="{index * panel_height + 150}">{min(xs):g}</text>'
            f'<text x="860" y="{index * panel_height + 150}">{max(xs):g}</text>'
            f'<text x="360" y="{index * panel_height + 170}">Agent joint updates</text>' + "".join(markers)
        )
    (analysis / "train-timeseries.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_height * len(charts)}">'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(charts) + "</svg>"
    )
    summary = {
        "schema_version": "eb/pbfm-training-analysis/v1",
        "run_id": plan["run_id"],
        "counts": counts,
        "decision": "stop_nonfinite_metrics"
        if any(quality[key]["nonfinite_samples"] for key in series)
        else "wait_for_independent_validation",
        "source_log": "native/train_log.txt",
        "x_axis": axis,
        "series_names": list(numeric),
        "plotted_series_names": series[:24],
        "series": quality,
        "rows": len(rows),
        "limitations": [
            "Native CSV uses the collection-round start; checkpoint counters define segment progress.",
            "Independent terrain validation and media are required before continuing.",
        ],
    }
    write_json(analysis / "summary.json", summary)
    files = []
    for path in sorted(analysis.iterdir()):
        limit = 32 if plan.get("long_exploration") and path.suffix in {".csv", ".svg"} else 4
        if path.stat().st_size > limit * 1024**2:
            raise ValueError("P-BFM analysis file exceeds bounded evidence size")
        files.append(
            {
                "path": str(path.relative_to(root / "outputs")),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    index = {
        "schema_version": "eb/pbfm-training-evidence/v1",
        "run_id": plan["run_id"],
        "target_updates": plan["target_updates"],
        "files": files,
    }
    write_json(analysis / "index.json", index)
    return index
