"""Bounded P-BFM completion manifests; payload loaders run only in task workers."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath

from eb.pbfm_training import TEMPLATE, training_fingerprint
from eb.runners.pbfm_source import file_sha256

CHECKPOINT_SCHEMA = "eb/pbfm-native-checkpoint/v1"
RECEIPT_SCHEMA = "eb/pbfm-native-train-receipt/v1"
MODE = "pbfm_native_state"
MAX_FILES = 10000
STATE_COVERAGE = {
    "model": "restored",
    "optimizer": "restored",
    "replay_per_rank": "restored",
    "rng_per_rank": "not_available",
    "environment": "not_available",
    "command_context": "not_available",
    "expert_priorities": "not_available",
}
RESUME_LIMITATIONS = [
    "Environment, RNG, command context and dynamic expert priorities are initialized again.",
    "Native Replay closes episodes on reset; this is not exact continuation.",
]
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


def canonical_hash(value: dict) -> str:
    return training_fingerprint({key: item for key, item in value.items() if key != "hash"})


def _root(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)) or not path.is_dir():
        raise ValueError("P-BFM checkpoint root is missing, symlinked or escapes its path")
    return path


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024**2:
        raise ValueError("P-BFM checkpoint JSON missing or unbounded: " + path.name)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("P-BFM checkpoint JSON must be an object")
    return value


def _tree(root: Path) -> dict[str, Path]:
    files = {}
    count = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        count += len(dirs) + len(names)
        if count > MAX_FILES * 2:
            raise ValueError("P-BFM checkpoint file tree is unbounded")
        for name in dirs + names:
            path = Path(current) / name
            if path.is_symlink() or name == ".saving" or ".tmp" in name:
                raise ValueError("P-BFM checkpoint save is incomplete or contains a symlink")
        for name in names:
            path = Path(current) / name
            if name == "checkpoint-manifest.json":
                continue
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("P-BFM checkpoint members must be regular files")
            files[path.relative_to(root).as_posix()] = path
            if len(files) > MAX_FILES:
                raise ValueError("P-BFM checkpoint contains too many files")
    return files


def _metadata(root: Path, files: dict[str, Path]) -> dict:
    required = {
        "config.json",
        "effective_config.json",
        "effective-environment.json",
        "config.yaml",
        "train_log.txt",
        "source-adaptation.json",
        "checkpoint/config.json",
        "checkpoint/init_kwargs.json",
        "checkpoint/optimizers.pth",
        "checkpoint/model/config.json",
        "checkpoint/model/init_kwargs.json",
        "checkpoint/model/model.safetensors",
        "checkpoint/train_status.json",
    }
    if not required <= files.keys() or any(files[key].stat().st_size == 0 for key in required):
        raise ValueError("P-BFM required native state/configuration is incomplete")
    agent, model = _json(files["checkpoint/config.json"]), _json(files["checkpoint/model/config.json"])
    adaptation = _json(files["source-adaptation.json"])
    if adaptation.get("schema_version") != "eb/pbfm-source-adaptation/v1" or not re.fullmatch(
        r"[a-f0-9]{64}", str(adaptation.get("patch_sha256", ""))
    ):
        raise ValueError("P-BFM source adaptation/patch identity is incomplete")
    kwargs, model_kwargs = (
        _json(files["checkpoint/init_kwargs.json"]),
        _json(files["checkpoint/model/init_kwargs.json"]),
    )
    saved, effective = _json(files["config.json"]), _json(files["effective_config.json"])
    actor = (model.get("archi") or {}).get("actor") or {}
    if (
        agent.get("name") != "FBcprAuxAgent"
        or model.get("name") != "FBcprAuxModel"
        or agent.get("model") != model
        or kwargs != model_kwargs
        or kwargs.get("action_dim") != 23
        or model.get("archi", {}).get("z_dim") != 256
        or saved.get("agent") != agent
        or effective.get("agent") != agent
        or effective.get("prior_plane_envs") != 0
        or effective.get("checkpoint_buffer") is not True
        or effective.get("checkpoint_rank_buffers") is not True
        or actor.get("name") != "direct_depth"
        or actor.get("input_filter", {}).get("key")
        != ["state", "last_action", "history_actor", "heading", "velocity_command"]
    ):
        raise ValueError("P-BFM agent/model/effective configuration differs")
    shapes = {
        "state": [52],
        "last_action": [23],
        "history_actor": [300],
        "privileged_state": [358],
        "heading": [2],
        "velocity_command": [3],
        "depth_image": [8, 18, 32],
        "velocity_target_yaw": [3],
    }
    spaces = (kwargs.get("obs_space") or {}).get("spaces")
    if (
        not isinstance(spaces, dict)
        or set(spaces) != set(shapes) | {"terrain_priv"}
        or any((spaces.get(key) or {}).get("shape") != shape for key, shape in shapes.items())
        or not isinstance((spaces.get("terrain_priv") or {}).get("shape"), list)
        or len(spaces["terrain_priv"]["shape"]) != 1
        or spaces["terrain_priv"]["shape"][0] <= 0
    ):
        raise ValueError("P-BFM observation spaces differ")
    status = _json(files["checkpoint/train_status.json"])
    counts = {key: status.get(key) for key in ("local_time", "global_time", "optimizer_steps")}
    world = status.get("world_size")
    if (
        type(world) is not int
        or world not in (1, 2, 4)
        or any(type(value) is not int or not 0 <= value < 2**63 for value in counts.values())
        or counts["global_time"] != counts["local_time"] * world
        or effective.get("distributed_world_size") != world
    ):
        raise ValueError("P-BFM saved counters/world size differ")
    expected = {"train"} if world == 1 else {f"train_rank_{rank}" for rank in range(world)}
    buffers = root / "checkpoint/buffers"
    if not buffers.is_dir() or {path.name for path in buffers.iterdir()} != expected:
        raise ValueError("P-BFM checkpoint requires all and only main rank Replay shards")
    replay = []
    for name in sorted(expected):
        prefix = "checkpoint/buffers/" + name
        if not {prefix + "/config.json", prefix + "/buffer.hdf5"} <= files.keys():
            raise ValueError("P-BFM rank Replay is incomplete")
        cfg = _json(files[prefix + "/config.json"])
        if (
            cfg.get("explicit_next") is not True
            or cfg.get("sparse_next") is not True
            or cfg.get("compact_depth_history") is not True
            or cfg.get("n_dim") != 2
            or cfg.get("depth_history_offsets") != [35, 30, 25, 20, 15, 10, 5, 0]
            or files[prefix + "/buffer.hdf5"].stat().st_size == 0
        ):
            raise ValueError("P-BFM Replay successor/depth contract differs")
        replay.append({"rank": 0 if name == "train" else int(name.rsplit("_", 1)[1]), "path": prefix})
    return {
        "world_size": world,
        "counts": counts,
        "replay": replay,
        "patch_sha256": adaptation["patch_sha256"],
        "source_adaptation_sha256": file_sha256(files["source-adaptation.json"]),
        "agent_config_sha256": file_sha256(files["checkpoint/config.json"]),
        "effective_config_sha256": file_sha256(files["effective_config.json"]),
        "effective_environment_sha256": file_sha256(files["effective-environment.json"]),
    }


def _ranks(results: list[dict], metadata: dict, refs: dict) -> None:
    world, counts = metadata["world_size"], metadata["counts"]
    if not isinstance(results, list) or len(results) != world:
        raise ValueError("P-BFM completion requires all rank results")
    ranks, devices = set(), set()
    for result in results:
        if (
            not isinstance(result, dict)
            or type(result.get("rank")) is not int
            or not 0 <= result["rank"] < world
            or result["rank"] in ranks
            or result.get("world_size") != world
            or result.get("status") != "succeeded"
            or result.get("loader_verified") is not True
            or result.get("replay_loader_verified") is not True
            or result.get("strict_state_verified") is not True
            or result.get("full_state_finite_verified") is not True
            or set(result.get("optimizer_names", [])) != OPTIMIZERS
            or type(result.get("pid")) is not int
            or result["pid"] <= 0
            or not result.get("gpu_uuid")
            or result["gpu_uuid"] in devices
            or result.get("source_refs") != refs
            or any(result.get(key) != value for key, value in counts.items())
        ):
            raise ValueError("P-BFM rank native loader/state/source/counter verification is incomplete")
        ranks.add(result["rank"])
        devices.add(result["gpu_uuid"])


def validate_pbfm_checkpoint(
    native_root: Path,
    manifest: dict,
    *,
    run_id: str,
    plan: dict | None = None,
    verify_hashes: bool = False,
    resume: bool = False,
) -> dict:
    root = _root(native_root)
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("run_id") != run_id
        or manifest.get("template_id") != TEMPLATE
        or manifest.get("mode") != MODE
        or manifest.get("state_coverage") != STATE_COVERAGE
        or manifest.get("loader_verified") is not True
        or manifest.get("full_state_finite_verified") is not True
    ):
        raise ValueError("P-BFM checkpoint completion/producer/state coverage differs")
    for key in ("compatibility_fingerprint", "plan_hash", "image_digest"):
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(manifest.get(key, ""))):
            raise ValueError("P-BFM checkpoint frozen identity is missing")
    refs = manifest.get("source_refs")
    if not isinstance(refs, dict) or set(refs) != {"pbfm", "rlinf", "asset"}:
        raise ValueError("P-BFM checkpoint source references are missing")
    actual = _tree(root)
    records = manifest.get("files")
    if not isinstance(records, list) or not 0 < len(records) <= MAX_FILES:
        raise ValueError("P-BFM checkpoint file index is unbounded or empty")
    seen, total = set(), 0
    for row in records:
        relative = row.get("path")
        pure = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in str(pure)
            or str(pure) != relative
            or relative in seen
            or relative not in actual
            or type(row.get("bytes")) is not int
            or row["bytes"] != actual[relative].stat().st_size
            or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", "")))
        ):
            raise ValueError("P-BFM checkpoint file path/size/index differs")
        if verify_hashes and file_sha256(actual[relative]) != row["sha256"]:
            raise ValueError("P-BFM checkpoint payload hash differs")
        seen.add(relative)
        total += row["bytes"]
    if seen != actual.keys():
        raise ValueError("P-BFM checkpoint has unindexed or missing files")
    metadata = _metadata(root, actual)
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError("P-BFM manifest counters/configuration/Replay differ")
    _ranks(manifest.get("rank_results"), metadata, refs)
    if plan is not None:
        if (
            manifest["compatibility_fingerprint"] != plan.get("fingerprint")
            or manifest["image_digest"] != plan.get("image_digest")
            or refs != plan.get("source_refs")
            or metadata["world_size"] != plan.get("world_size")
            or manifest.get("identity") != plan.get("identity")
            or not resume
            and manifest["plan_hash"] != canonical_hash(plan)
            or not resume
            and metadata["counts"]["optimizer_steps"] != plan.get("target_updates")
        ):
            raise ValueError("P-BFM checkpoint training identity/target/source is incompatible")
    return {
        **manifest,
        "manifest": manifest,
        "path": str(root / "checkpoint"),
        "native_path": str(root),
        "bytes": total,
        "step": metadata["counts"]["optimizer_steps"],
        "limitations": RESUME_LIMITATIONS,
    }


def build_pbfm_checkpoint_manifest(
    native_root: Path, plan: dict, run_id: str, *, rank_results: list[dict], loader_verified: bool,
    allow_intermediate: bool = False,
) -> dict:
    root = _root(native_root)
    files = _tree(root)
    metadata = _metadata(root, files)
    updates = metadata["counts"]["optimizer_steps"]
    if allow_intermediate:
        from eb.pbfm_longrun import validate_long_plan
        from eb.pbfm_training import segment_endpoint

        validate_long_plan(plan)
        if not plan["start_updates"] < updates < plan["target_updates"]:
            raise ValueError("P-BFM periodic checkpoint is outside its parent segment")
        endpoint = segment_endpoint(updates, plan["world_size"], start_updates=plan["start_updates"],
                                    start_local_time=plan["start_local_time"], num_envs=2048)
        if metadata["counts"]["global_time"] != endpoint:
            raise ValueError("P-BFM periodic checkpoint counters differ")
        if any(result.get("optimizer_state_steps") != {name: [updates] for name in OPTIMIZERS}
               for result in rank_results):
            raise ValueError("P-BFM periodic optimizer proof differs from its actual update count")
    if loader_verified is not True or not allow_intermediate and updates != plan["target_updates"]:
        raise ValueError("P-BFM native loader verification/endpoint is incomplete")
    _ranks(rank_results, metadata, plan["source_refs"])
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "template_id": TEMPLATE,
        "mode": MODE,
        "state_coverage": STATE_COVERAGE,
        "plan_hash": canonical_hash(plan),
        "compatibility_fingerprint": plan["fingerprint"],
        "image_digest": plan["image_digest"],
        "identity": plan["identity"],
        "source_refs": plan["source_refs"],
        "loader_verified": True,
        "full_state_finite_required": True,
        "full_state_finite_verified": True,
        "rank_results": rank_results,
        **metadata,
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in sorted(files.items())
        ],
    }
    validate_pbfm_checkpoint(root, manifest, run_id=run_id, plan=plan, resume=allow_intermediate)
    return manifest


def write_pbfm_checkpoint_receipt(
    native_root: Path, plan: dict, run_id: str, *, rank_results: list[dict], loader_verified: bool
) -> dict:
    """Publish completion after the immutable snapshot and all evidence exist."""
    from eb.runners.rlinf_pbfm_runtime import write_json

    root = _root(native_root)
    manifest_path, receipt_path = root / "checkpoint-manifest.json", root.parent / "receipt.json"
    if any(p.exists() or p.is_symlink() for p in (manifest_path, receipt_path)):
        raise ValueError("P-BFM immutable publication already exists")
    manifest = build_pbfm_checkpoint_manifest(
        root, plan, run_id, rank_results=rank_results, loader_verified=loader_verified
    )
    write_json(manifest_path, manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "operation": "pbfm_native_train",
        "run_id": run_id,
        "native_manifest_sha256": file_sha256(manifest_path),
        **{
            key: manifest[key]
            for key in (
                "plan_hash",
                "compatibility_fingerprint",
                "image_digest",
                "world_size",
                "rank_results",
                "loader_verified",
                "full_state_finite_required",
                "full_state_finite_verified",
            )
        },
    }
    write_json(receipt_path, receipt)
    return receipt


def publish_pbfm_checkpoint(
    staging: Path, destination: Path, plan: dict, run_id: str, *, rank_results: list[dict]
) -> dict:
    """Copy a stopped native workspace; a failed copy cannot become a snapshot."""
    from eb.runners.rlinf_pbfm_runtime import build_training_evidence

    _root(staging)
    if destination.exists() or destination.is_symlink():
        raise ValueError("P-BFM snapshot destination already exists")
    temporary = destination.with_name(destination.name + ".publishing")
    if temporary.exists():
        raise ValueError("P-BFM prior publication is incomplete")
    _tree(staging)
    shutil.copytree(staging, temporary)
    manifest = build_pbfm_checkpoint_manifest(
        temporary, plan, run_id, rank_results=rank_results, loader_verified=True
    )
    temporary.replace(destination)
    build_training_evidence(destination.parent.parent, plan, manifest["counts"])
    if _json(destination.parent / "analysis/summary.json")["decision"] != "wait_for_independent_validation":
        raise ValueError("P-BFM nonfinite learning metrics prevent completion publication")
    return write_pbfm_checkpoint_receipt(
        destination, plan, run_id, rank_results=rank_results, loader_verified=True
    )


def read_pbfm_checkpoint(run_root: Path, spec, run_id: str, verify_hashes: bool = False) -> dict | None:
    root = _root(run_root)
    native = root / "outputs/native"
    if not native.exists():
        return None
    native = _root(native)
    manifest_path = native / "checkpoint-manifest.json"
    if not manifest_path.exists():
        return None
    plan = {**spec.params["native_train_plan"], "run_id": run_id}
    plan["hash"] = canonical_hash(plan)
    manifest = _json(manifest_path)
    checkpoint = validate_pbfm_checkpoint(
        native, manifest, run_id=run_id, plan=plan, verify_hashes=verify_hashes
    )
    receipt = _json(root / "outputs/receipt.json")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("operation") != "pbfm_native_train"
        or receipt.get("run_id") != run_id
        or receipt.get("native_manifest_sha256") != file_sha256(manifest_path)
        or any(
            receipt.get(key) != manifest.get(key)
            for key in (
                "plan_hash",
                "compatibility_fingerprint",
                "image_digest",
                "world_size",
                "rank_results",
                "loader_verified",
                "full_state_finite_verified",
            )
        )
    ):
        raise ValueError("P-BFM completion receipt differs from immutable manifest")
    return {**checkpoint, "receipt": receipt}
