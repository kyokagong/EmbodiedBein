"""Native UFO checkpoint receipts, shared with Python 3.10 task runners.

Only the task runner enables payload hashing and verifies native agent/Replay
loaders. Control-plane callers inspect bounded JSON, paths and file sizes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

CHECKPOINT_SCHEMA = "eb/ufo-native-checkpoint/v1"
RECEIPT_SCHEMA = "eb/ufo-native-train-receipt/v1"
TEMPLATE_ID = "rpo_ufo_fb_native"
MODE = "ufo_native_state"
MAX_FILES = 10000
MAX_JSON_BYTES = 4 * 1024 * 1024
STATE_COVERAGE = {
    "model": "restored", "optimizer": "restored", "replay_per_rank": "restored",
    "scheduler": "not_applicable", "rng_per_rank": "not_available",
    "dataloader": "not_available", "environment": "not_available", "rollout": "not_available",
}
RESUME_LIMITATIONS = [
    "Simulation environments and random seeds are initialized again.",
    "RNG, rollout context, ZBuffer and expert sampling priorities are not fully restored.",
    "Native state restoration is not bitwise exact continuation.",
]


def validate_reconfigured_resume(plan: dict, source_plan: dict) -> None:
    """One explicit scale change; agent, data, world size and sources stay frozen."""
    if not isinstance(source_plan, dict):
        raise ValueError("Native reconfiguration source plan is missing")
    resume = plan.get("resume") or {}
    old, new = source_plan.get("identity", {}), plan.get("identity", {})
    allowed = {"image_digest", "num_envs_per_rank", "buffer_size"}
    if (resume.get("replay_policy") != "fresh"
            or resume.get("source_plan") != source_plan
            or old.get("num_envs_per_rank", 1024) != 1024
            or new.get("num_envs_per_rank") != 2048
            or new.get("buffer_size") not in (old.get("buffer_size"), 2 * old.get("buffer_size", 0))
            or {k: v for k, v in old.items() if k not in allowed}
            != {k: v for k, v in new.items() if k not in allowed}
            or plan.get("source_refs") != source_plan.get("source_refs")
            or plan.get("world_size") != source_plan.get("world_size")
            or source_plan.get("full_state_finite_required") is not True
            or plan.get("full_state_finite_required") is not True):
        raise ValueError("Native fresh-Replay reconfiguration changes unsupported training semantics")


def canonical_hash(value: dict) -> str:
    encoded = json.dumps({k: v for k, v in value.items() if k != "hash"}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, label, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"Invalid native checkpoint {label}")
    return value


def _digest(value, label, *, prefixed=True):
    pattern = ("sha256:" if prefixed else "") + "[a-f0-9]{64}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid native checkpoint {label}")
    return value


def _root(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Native checkpoint root contains a symlink or traversal")
    if not path.is_dir():
        raise ValueError("Native checkpoint root is unavailable")
    return path


def _member(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or len(relative) > 1024 or "\\" in relative:
        raise ValueError("Invalid native checkpoint member path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or str(parts) != relative or relative == ".":
        raise ValueError("Native checkpoint member path escapes its root")
    path = root
    for component in parts.parts:
        path /= component
        if path.is_symlink():
            raise ValueError("Native checkpoint members must not be symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Native checkpoint member escapes its root")
    return path


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError(f"Missing or unbounded native checkpoint JSON: {path.name}")
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"Invalid native checkpoint JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Native checkpoint JSON must be an object: {path.name}")
    return value


def _source_refs(refs):
    if not isinstance(refs, dict) or set(refs) != {"ufo", "rlinf"}:
        raise ValueError("Native checkpoint source identities are missing")
    for name, ref in refs.items():
        if not isinstance(ref, dict):
            raise ValueError("Native checkpoint source identity is invalid")
        _digest(ref.get("snapshot_hash"), name + " source hash")
        for key in ("path", "manifest_path"):
            value = ref.get(key)
            if (not isinstance(value, str) or not value.startswith("/sources/")
                    or ".." in PurePosixPath(value).parts
                    or "\\" in value or len(value) > 1024):
                raise ValueError("Native checkpoint source path is invalid")
    return refs


def _tree(root: Path) -> dict[str, Path]:
    files, entries = {}, 0
    for current, directories, names in os.walk(root, followlinks=False):
        entries += len(directories) + len(names)
        if entries > MAX_FILES * 2:
            raise ValueError("Native checkpoint tree exceeds bounded file index")
        for name in directories + names:
            path = Path(current) / name
            if path.is_symlink():
                raise ValueError("Native checkpoint tree contains a symlink")
            if name == ".saving":
                raise ValueError("Native checkpoint save is incomplete (.saving)")
        for name in names:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if relative == "checkpoint-manifest.json":
                continue
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Native checkpoint member must be a regular file")
            files[relative] = path
            if len(files) > MAX_FILES:
                raise ValueError("Native checkpoint has too many files")
    return files


def _native_metadata(root: Path, files: dict[str, Path]) -> dict:
    required = {"config.json", "config.yaml", "effective_config.json", "train_log.txt",
                "checkpoint/config.json", "checkpoint/init_kwargs.json", "checkpoint/optimizers.pth",
                "checkpoint/model/config.json", "checkpoint/model/init_kwargs.json",
                "checkpoint/model/model.safetensors", "checkpoint/train_status.json"}
    if not required.issubset(files) or any(files[name].stat().st_size == 0 for name in required):
        raise ValueError("Native checkpoint required state/configuration member is missing or empty")
    agent = _json(files["checkpoint/config.json"])
    model = _json(files["checkpoint/model/config.json"])
    kwargs = _json(files["checkpoint/init_kwargs.json"])
    model_kwargs = _json(files["checkpoint/model/init_kwargs.json"])
    saved, effective = _json(files["config.json"]), _json(files["effective_config.json"])
    if (agent.get("name") != "FBcprAuxAgent" or model.get("name") != "FBcprAuxModel"
            or agent.get("model") != model or kwargs != model_kwargs or kwargs.get("action_dim") != 23
            or (model.get("archi") or {}).get("z_dim") != 256
            or effective.get("agent") != agent or saved.get("agent") != agent):
        raise ValueError("Native checkpoint agent/model configuration or init_kwargs differ")
    spaces = (kwargs.get("obs_space") or {}).get("spaces")
    dimensions = {"state": 52, "last_action": 23, "history_actor": 300, "privileged_state": 358}
    if (not isinstance(spaces, dict) or set(spaces) != set(dimensions)
            or any(not isinstance(spaces[key], dict) or spaces[key].get("shape") != [dim]
                   for key, dim in dimensions.items())):
        raise ValueError("Native checkpoint observation dimensions differ")
    status = _json(files["checkpoint/train_status.json"])
    counts = {key: _integer(status.get(key), key) for key in ("local_time", "global_time", "optimizer_steps")}
    world_size = _integer(status.get("world_size"), "world_size", 1)
    if (world_size > 64 or counts["global_time"] != counts["local_time"] * world_size
            or status.get("buffer_global_time") != counts["global_time"]
            or not isinstance(status.get("replay_id"), str) or not 0 < len(status["replay_id"]) <= 128
            or (effective.get("training") or {}).get("distributed_world_size") != world_size):
        raise ValueError("Native checkpoint world size, Replay identity or state counts differ")
    _integer(status.get("effective_batch_size"), "effective_batch_size", 1)
    if status.get("loss_mode") != ("single_rank" if world_size == 1 else "local_loss_average"):
        raise ValueError("Native checkpoint loss mode differs from world size")
    replay = []
    expected_ranks = {f"train_rank_{rank}" for rank in range(world_size)}
    buffers = _member(root, "checkpoint/buffers")
    if not buffers.is_dir() or {path.name for path in buffers.iterdir()} != expected_ranks:
        raise ValueError("Native checkpoint must contain all and only the declared rank Replay shards")
    for rank in range(world_size):
        prefix = f"checkpoint/buffers/train_rank_{rank}"
        names = {prefix + "/" + name for name in ("snapshot.json", "config.json", "buffer.hdf5")}
        if not names.issubset(files) or any(files[name].stat().st_size == 0 for name in names):
            raise ValueError(f"Native checkpoint rank {rank} Replay member is missing")
        snapshot = _json(files[prefix + "/snapshot.json"])
        if (snapshot.get("transition_format") != 2 or snapshot.get("id") != status["replay_id"]
                or snapshot.get("global_time") != counts["global_time"]
                or snapshot.get("world_size") != world_size):
            raise ValueError(f"Native checkpoint rank {rank} Replay snapshot differs")
        config = _json(files[prefix + "/config.json"])
        capacity = _integer(config.get("capacity"), "Replay capacity", 1)
        index = _integer(config.get("_idx"), "Replay cursor")
        if (index >= capacity or type(config.get("_is_full")) is not bool
                or not config["_is_full"] and index == 0 or config.get("n_dim") != 2):
            raise ValueError(f"Native checkpoint rank {rank} Replay configuration is invalid")
        replay.append({"rank": rank, "path": prefix, "snapshot": snapshot,
                       "capacity": capacity, "cursor": index, "full": config["_is_full"]})
    if world_size > 1:
        sync = _json(_member(root, "checkpoint/distributed_sync.json"))
        difference = sync.get("max_abs_diff_from_rank0")
        if (sync.get("world_size") != world_size or sync.get("rank") != 0
                or type(difference) not in (int, float) or not math.isfinite(difference)
                or not 0 <= difference <= 1e-5):
            raise ValueError("Native checkpoint distributed model synchronization is invalid")
    return {"counts": counts, "world_size": world_size, "replay_id": status["replay_id"], "replay": replay}


def _ranks(results, metadata, source_refs):
    world_size, counts = metadata["world_size"], metadata["counts"]
    if not isinstance(results, list) or len(results) != world_size:
        raise ValueError("Native checkpoint requires successful completion of every rank")
    ranks, devices = set(), set()
    for result in results:
        if (not isinstance(result, dict) or type(result.get("rank")) is not int
                or not 0 <= result["rank"] < world_size or result["rank"] in ranks
                or result.get("status") != "succeeded" or type(result.get("world_size")) is not int
                or result["world_size"] != world_size or result.get("loader_verified") is not True
                or result.get("replay_loader_verified") is not True
                or result.get("source_refs") != source_refs
                or any(result.get(key) != value for key, value in counts.items())):
            raise ValueError("Native checkpoint rank result, loader or source identity differs")
        _integer(result.get("pid"), "worker PID", 1)
        device = result.get("gpu_uuid")
        if not isinstance(device, str) or not device or device in devices:
            raise ValueError("Native checkpoint rank GPU identities are missing or duplicate")
        if ("full_state_finite_verified" in result
                and type(result["full_state_finite_verified"]) is not bool):
            raise ValueError("Native checkpoint rank full-state finite evidence is invalid")
        ranks.add(result["rank"])
        devices.add(device)
    return all(result.get("full_state_finite_verified") is True for result in results)


def validate_native_checkpoint(native_root: Path, manifest: dict, *, run_id: str,
                               plan: dict | None = None, verify_hashes: bool = False) -> dict:
    root = _root(native_root)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != CHECKPOINT_SCHEMA
            or manifest.get("status") != "complete" or manifest.get("run_id") != run_id
            or manifest.get("template_id") != TEMPLATE_ID or manifest.get("mode") != MODE
            or manifest.get("state_coverage") != STATE_COVERAGE
            or manifest.get("loader_verified") is not True):
        raise ValueError("Native checkpoint manifest completion, coverage or producer differs")
    _digest(manifest.get("compatibility_fingerprint"), "compatibility fingerprint")
    _digest(manifest.get("plan_hash"), "plan hash")
    _digest(manifest.get("image_digest"), "image digest")
    refs = _source_refs(manifest.get("source_refs"))
    actual = _tree(root)
    records = manifest.get("files")
    if not isinstance(records, list) or not 0 < len(records) <= MAX_FILES:
        raise ValueError("Native checkpoint manifest file index is invalid")
    seen, total = set(), 0
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Native checkpoint file record is invalid")
        path = _member(root, item.get("path"))
        relative = item["path"]
        size = _integer(item.get("bytes"), "file size")
        _digest(item.get("sha256"), "file digest", prefixed=False)
        if relative not in actual or relative in seen or not path.is_file() or path.stat().st_size != size:
            raise ValueError("Native checkpoint manifest member is missing, duplicate or changed")
        if verify_hashes and sha256_file(path) != item["sha256"]:
            raise ValueError("Native checkpoint file digest changed")
        total += size
        seen.add(relative)
    if seen != set(actual):
        raise ValueError("Native checkpoint tree differs from its immutable manifest index")
    metadata = _native_metadata(root, actual)
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError("Native checkpoint manifest counters or Replay identities differ")
    all_finite = _ranks(manifest.get("rank_results"), metadata, refs)
    finite_required = manifest.get("full_state_finite_required", False)
    finite_verified = manifest.get("full_state_finite_verified", False)
    plan_finite_required = plan.get("full_state_finite_required", False) if plan is not None else False
    if (type(finite_required) is not bool or type(finite_verified) is not bool
            or type(plan_finite_required) is not bool
            or "full_state_finite_verified" in manifest and finite_verified != all_finite
            or plan_finite_required and not finite_required
            or (finite_required or plan_finite_required) and not (finite_verified and all_finite)):
        raise ValueError("Native checkpoint full-state finite verification is incomplete or inconsistent")
    if plan is not None:
        if (manifest["compatibility_fingerprint"] != plan.get("fingerprint")
                or manifest["image_digest"] != plan.get("image_digest")
                or refs != plan.get("source_refs") or metadata["world_size"] != plan.get("world_size")):
            raise ValueError("Native checkpoint configuration/source/image/data/world size is incompatible")
    return {**manifest, "manifest": manifest, "path": str(root / "checkpoint"), "native_path": str(root),
            "full_state_finite_verified": finite_verified, "full_state_finite_required": finite_required,
            "bytes": total, "step": metadata["counts"]["optimizer_steps"], "limitations": RESUME_LIMITATIONS}


def build_native_checkpoint_manifest(native_root: Path, plan: dict, run_id: str, *,
                                     rank_results: list[dict], loader_verified: bool) -> dict:
    root = _root(native_root)
    files = _tree(root)
    metadata = _native_metadata(root, files)
    refs = _source_refs(plan.get("source_refs"))
    target_updates = _integer(plan.get("target_updates"), "target updates", 1)
    if (loader_verified is not True or metadata["world_size"] != plan.get("world_size")
            or metadata["counts"]["optimizer_steps"] < target_updates):
        raise ValueError("Native checkpoint target updates, loader verification or world size is incomplete")
    finite_verified = _ranks(rank_results, metadata, refs)
    finite_required = plan.get("full_state_finite_required", False)
    if type(finite_required) is not bool or finite_required and not finite_verified:
        raise ValueError("Native checkpoint requires full-state finite verification from every rank")
    manifest = {"schema_version": CHECKPOINT_SCHEMA, "status": "complete", "run_id": run_id,
                "template_id": TEMPLATE_ID, "mode": MODE, "plan_hash": canonical_hash(plan),
                "compatibility_fingerprint": plan.get("fingerprint"),
                "image_digest": plan.get("image_digest"),
                "source_refs": refs, **metadata, "state_coverage": STATE_COVERAGE,
                "loader_verified": True, "rank_results": rank_results,
                "full_state_finite_required": finite_required, "full_state_finite_verified": finite_verified,
                "files": [{"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                          for name, path in sorted(files.items())]}
    validate_native_checkpoint(root, manifest, run_id=run_id, plan=plan)
    return manifest


def _write_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError("Native checkpoint publication already exists or is incomplete")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_native_checkpoint_receipt(native_root: Path, plan: dict, run_id: str, *,
                                    rank_results: list[dict], loader_verified: bool) -> dict:
    root = _root(native_root)
    receipt_path = root.parent / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Native training completion receipt already exists")
    manifest = build_native_checkpoint_manifest(root, plan, run_id, rank_results=rank_results,
                                                loader_verified=loader_verified)
    manifest_path = root / "checkpoint-manifest.json"
    _write_json(manifest_path, manifest)
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": "complete", "run_id": run_id,
               "operation": "ufo_native_train", "plan_hash": manifest["plan_hash"],
               "compatibility_fingerprint": manifest["compatibility_fingerprint"],
               "native_manifest_sha256": sha256_file(manifest_path), "image_digest": manifest["image_digest"],
               "full_state_finite_required": manifest["full_state_finite_required"],
               "full_state_finite_verified": manifest["full_state_finite_verified"],
               "world_size": manifest["world_size"], "rank_results": rank_results, "loader_verified": True}
    _write_json(receipt_path, receipt)
    return receipt


def read_native_checkpoint(run_root: Path, run_id: str, *, plan: dict | None = None,
                           verify_hashes: bool = False, require_receipt: bool = True) -> dict:
    root = _root(run_root)
    native = _root(_member(root, "outputs/native"))
    manifest_path = _member(native, "checkpoint-manifest.json")
    manifest = _json(manifest_path)
    checkpoint = validate_native_checkpoint(native, manifest, run_id=run_id, plan=plan,
                                            verify_hashes=verify_hashes)
    if require_receipt:
        receipt = _json(_member(root, "outputs/receipt.json"))
        for key in ("full_state_finite_verified", "full_state_finite_required"):
            value = receipt.get(key, False)
            if type(value) is not bool or value != checkpoint[key]:
                raise ValueError("Native training receipt full-state finite evidence differs")
        if (receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "complete"
                or receipt.get("run_id") != run_id or receipt.get("operation") != "ufo_native_train"
                or receipt.get("native_manifest_sha256") != sha256_file(manifest_path)
                or receipt.get("loader_verified") is not True
                or any(receipt.get(key) != manifest.get(key) for key in (
                    "plan_hash", "compatibility_fingerprint", "image_digest", "world_size", "rank_results"))):
            raise ValueError("Native training completion receipt or rank evidence differs")
        checkpoint["receipt"] = receipt
    return checkpoint
