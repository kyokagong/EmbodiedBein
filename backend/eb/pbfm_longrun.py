"""A scoped, image-only compatibility bridge for the10000→100000 experiment."""
import hashlib
import re
from copy import deepcopy
from pathlib import Path

from eb.pbfm_training import training_fingerprint

LONG_MODE = "training_long_exploration"
LONG_IMAGE = "eb/rlinf-pbfm:rpo-depth-long-v1"
BASE_IMAGE = "sha256:8a1fea0322cee1af4ba8e98d76c1af20c88ff885057a49ab63ceaf2d6ec2c7de"
LONG_TIMEOUT_SECONDS = 108000
CHECKPOINT_GLOBAL_INTERVAL = 5120000
RUNTIME_FILES = (
    "pbfm_training.py", "pbfm_training_checkpoint.py", "pbfm_longrun.py",
    "runners/rlinf_pbfm_runtime.py", "runners/rlinf_pbfm_worker.py",
    "runners/pbfm_longrun.py",
    "pbfm_recovery.py", "runners/pbfm_numerics.py",
)


def runtime_compatibility_receipt() -> dict:
    root = Path(__file__).parent
    return {
        "schema_version": "eb/pbfm-long-runtime/v1", "base_image": BASE_IMAGE,
        "runtime_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in RUNTIME_FILES},
    }


def long_exploration_metadata(source_identity: dict, image: str) -> dict:
    if (source_identity.get("image_digest") != BASE_IMAGE
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(image)) or image == BASE_IMAGE):
        raise ValueError("P-BFM long exploration requires its frozen base and a derived image")
    return {
        "schema_version": "eb/pbfm-long-exploration/v1", "protocol": "direct100000",
        "experiment_target_updates": 100000, "source_identity": deepcopy(source_identity),
        "source_fingerprint": training_fingerprint(source_identity),
        "runtime_receipt_sha256": training_fingerprint(runtime_compatibility_receipt()),
        "checkpoint_interval_global_steps": CHECKPOINT_GLOBAL_INTERVAL,
        "checkpoint_retention": 2,
    }


def validate_long_plan(plan: dict) -> None:
    if plan.get("recovery"):
        from eb.pbfm_recovery import validate_recovery_plan

        return validate_recovery_plan(plan)
    identity = plan["identity"]
    long = plan.get("long_exploration") or {}
    source = {**identity, "image_digest": BASE_IMAGE}
    expected = long_exploration_metadata(source, plan["image_digest"])
    if long.get("runtime_receipt_sha256") == (
        "sha256:e78890fecc1dcdd8cc030e17b82e6bc078308a85922abb9bdc5d171a7490edbb"
    ):
        expected["runtime_receipt_sha256"] = long["runtime_receipt_sha256"]
    if (
        long != expected
        or plan.get("start_updates") != 10000 or plan.get("start_local_time") != 1298432
        or not 10000 < plan["target_updates"] <= 100000
        or identity.get("world_size") != 4 or identity.get("num_envs_per_rank") != 2048
        or identity.get("buffer_size") != 131072
        or plan.get("communication_timeout_seconds") != 900
        or not 60 <= plan["segment_timeout_seconds"] <= LONG_TIMEOUT_SECONDS
    ):
        raise ValueError("P-BFM long exploration source/image/identity/budget differs")


def checkpoint_source_plan(plan: dict) -> dict:
    """Validate old files against their actual identity, without relabeling their producer."""
    if not plan.get("long_exploration"):
        return plan
    validate_long_plan(plan)
    return {
        "identity": plan["long_exploration"]["source_identity"],
        "fingerprint": plan["long_exploration"]["source_fingerprint"],
        "image_digest": plan["long_exploration"]["source_identity"]["image_digest"],
        "world_size": plan["world_size"],
        "source_refs": plan["source_refs"],
    }
