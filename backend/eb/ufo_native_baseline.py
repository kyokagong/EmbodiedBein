"""Fixed, diagnostic-only comparison protocol (also imported by Python 3.10).

This is one RPO v2/112-update native-launch comparison, not a generic command
runner or a second training recipe. Numerical differences are reported, never
converted to an equivalence claim by a configurable tolerance.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OPERATION = "ufo_native_baseline_v1"
SCHEMA = "eb/ufo-native-baseline-plan/v1"
RECEIPT_SCHEMA = "eb/ufo-native-baseline-receipt/v1"
ARTIFACT_KIND = "ufo_native_training_baseline"
SCOPED_ARTIFACT_KIND = "ufo_native_baseline_scoped_verification"
SCOPED_SCHEMA = "eb/ufo-native-baseline-scoped-verification/v1"
CHECKPOINT_SCHEMA = "eb/ufo-native-baseline-checkpoint/v1"
IMAGE = "eb/ufo-native:rpo-baseline-v2"
BASE_IMAGE = "sha256:e122bb0c96f28c3d2006dd0c466bf392f2f4dc8f0e09fba0e5f2e36f6cefdfe6"
SOURCE_RUN = "run_648c7193afd64a188100"
SOURCE_JOB = "job_9d046c6692df4b96862a"
DATASET = "ds_6541f726d7804007a93a"
RESOURCES = {"gpus": 4, "gpu_mode": "exclusive", "cpus": 32, "memory_gb": 80, "shm_gb": 8, "devices": []}
RULES = {
    "configuration": "All leaves exact; only env.device cuda:0 -> cuda:<rank> is allowed.",
    "paths": "Identical logical source, data, derived robot and work paths in both containers.",
    "counts": {"optimizer_steps": 112, "local_time": 18432, "global_time": 73728},
    "world_size": 4,
    "seed_per_rank": [42, 43, 44, 45],
    "initial_hash": "Exact same parameter_snapshot actor/forward/backward SHA256 per rank.",
    "final_weights": "Must change and remain finite; final cross-launch hashes need not be bitwise equal.",
    "synchronization": "Native saved distributed_sync must report all four ranks and max difference <= 1e-5.",
    "checkpoint": (
        "Identical member names, model/optimizer schema and all-rank Replay structure; UUIDs differ."
    ),
    "metrics": "Retain every numeric column and aligned-step difference; no numerical-equivalence threshold.",
    "qualification": "Diagnostic structural comparison only; not numerical or task-quality equivalence.",
    "observer": (
        "Read-only entry observation; sys.setprofile(None) before hashing, no training-loop profiling."
    ),
}
VERIFICATION_SCOPE = {
    "shared_checkpoint_origin_rank": 0,
    "shared_checkpoint_model_finite": "observed",
    "shared_checkpoint_five_optimizers_finite": "observed",
    "shared_checkpoint_native_agent_loader": "observed",
    "all_rank_model_and_normalizer_synchronization": "observed",
    "replay_native_loader_observed_ranks": [0, 1, 2, 3],
    "live_optimizer_finite_per_rank": {str(rank): "not_observed" for rank in range(4)},
    "rank_1_to_3_optimizer_checkpoint_readback": "not_observed",
    "full_per_rank_training_state_verified": False,
    "training_restore_supported": False,
    "numerical_equivalence_claimed": False,
}


def build_baseline_checkpoint_manifest(native_root, training, run_id, *, rank_observations):
    """Index actual baseline state without claiming live rank optimizer checks."""
    import copy

    from eb.ufo_training_checkpoint import _native_metadata, _root, _tree, canonical_hash, sha256_file

    root = _root(native_root)
    files = _tree(root)
    metadata = _native_metadata(root, files)
    if metadata["counts"] != RULES["counts"] or metadata["world_size"] != 4:
        raise ValueError("Incomplete baseline checkpoint counters")
    if (
        len(rank_observations) != 4
        or {row.get("rank") for row in rank_observations} != set(range(4))
        or len({row.get("gpu_uuid") for row in rank_observations}) != 4
        or any(
            row.get("replay_loader_verified") is not True
            or any(key in row for key in ("full_state_finite_verified", "loader_verified"))
            for row in rank_observations
        )
    ):
        raise ValueError("Baseline rank observations cannot claim complete rank optimizer verification")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "diagnostic_only": True,
        "training_restore_supported": False,
        "image_digest": training["image_digest"],
        "plan_hash": canonical_hash(training),
        "source_refs": training["source_refs"],
        **metadata,
        "verification_scope": copy.deepcopy(VERIFICATION_SCOPE),
        "rank_observations": rank_observations,
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(files.items())
        ],
    }


def execution_training_plan(plan: dict, run_id: str) -> dict:
    """New diagnostic image identity; original e122 source plan stays immutable."""
    from eb.ufo_training_checkpoint import canonical_hash

    source = plan["native_train_plan"]
    identity = {**source["identity"], "image_digest": plan["image_digest"]}
    result = {
        **source,
        "run_id": run_id,
        "image_digest": plan["image_digest"],
        "identity": identity,
        "fingerprint": canonical_hash(identity),
        # Baseline observes the shared saved state, not every live optimizer.
        # The original source plan's training requirement remains untouched.
        "full_state_finite_required": False,
    }
    result["hash"] = canonical_hash(result)
    return result


def leaves(value, prefix=""):
    if value == {} or value == []:
        return {prefix: value}
    if isinstance(value, dict):
        return {
            key: leaf
            for name in sorted(value)
            for key, leaf in leaves(value[name], f"{prefix}.{name}" if prefix else name).items()
        }
    if isinstance(value, list):
        return {
            key: leaf
            for index, item in enumerate(value)
            for key, leaf in leaves(item, f"{prefix}[{index}]").items()
        }
    return {prefix: value}


def config_differences(expected: dict, actual: dict, rank: int) -> list[dict]:
    left, right = leaves(expected), leaves(actual)
    differences = []
    for key in sorted(left.keys() | right.keys()):
        if key == "env.device" and left.get(key) == "cuda:0" and right.get(key) == f"cuda:{rank}":
            continue
        if key not in left or key not in right or left[key] != right[key]:
            differences.append({"path": key, "worker": left.get(key), "native": right.get(key)})
    return differences


def numeric_csv(path: Path) -> dict[int, dict[str, float]]:
    """Final rewritten native CSV, keyed by actual update count, including NaN rejection."""
    rows = {}
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("Missing or unbounded baseline CSV")
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if None in row:
                raise ValueError("Incomplete baseline CSV row")
            values = {}
            for key, raw in row.items():
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite baseline metric: {key}")
                values[key] = value
            step = values.get("distributed/optimizer_steps")
            if step is not None and step.is_integer():
                rows[int(step)] = values
    if not rows or max(rows) != 112:
        raise ValueError("Baseline CSV does not reach 112 actual updates")
    return rows


def metric_differences(worker: Path, native: Path) -> dict:
    left, right = numeric_csv(worker), numeric_csv(native)
    return {
        "worker_only_steps": sorted(left.keys() - right.keys()),
        "native_only_steps": sorted(right.keys() - left.keys()),
        "aligned": [
            {
                "optimizer_steps": step,
                "metrics": {
                    key: {
                        "worker": left[step].get(key),
                        "native": right[step].get(key),
                        "native_minus_worker": right[step][key] - left[step][key]
                        if key in left[step] and key in right[step]
                        else None,
                    }
                    for key in sorted(left[step].keys() | right[step].keys())
                },
            }
            for step in sorted(left.keys() & right.keys())
        ],
        "numerical_equivalence_claimed": False,
    }
