"""Native UFO state with strict metadata, complete RNG/replay and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path, PurePosixPath

try:
    from eb.runners.bfm_ufo_agent import OPTIMIZERS, TARGETS, optimizer_steps
except ModuleNotFoundError:
    from bfm_ufo_agent import OPTIMIZERS, TARGETS, optimizer_steps

FORMAT = "ufo_agent_state_v1"
STREAM_FORMAT = "ufo_agent_state_v2"
COVERAGE = {
    "model": "restored",
    "optimizer": "restored",
    "scheduler": "not_applicable",
    "scaler": "not_applicable",
    "rng_per_rank": "restored",
    "replay": "restored",
    "expert_sampler": "restored",
    "z_pool": "restored",
    "rollout": "restored",
    "environment": "not_available",
}


def ufo_checkpoint_root(output: Path, cfg) -> Path:
    if cfg.actor.model.get("learning_profile") in {
        "physical_limits_stable_latent_v3",
        "physical_limits_stable_slew_v4",
    }:
        return output / "model" / ".checkpoint-source"
    return output / str(cfg.runner.logger.experiment_name) / "checkpoints"


def sampler_ranks(binding):
    topology = binding.get("topology_id")
    if topology not in {"single_learner", "one_learner_three_samplers"}:
        raise ValueError("Unknown native checkpoint topology")
    return {f"rank_{i}" for i in range(3 if topology == "one_learner_three_samplers" else 1)}


def validate_sampler_identities(expected, receipts, *, ranks):
    if (
        set(expected) != {"state_sha256", "version", "online_transitions"}
        or not re.fullmatch(r"[a-f0-9]{64}", str(expected.get("state_sha256", "")))
        or any(
            type(expected.get(key)) is not int or expected[key] < 0
            for key in ("version", "online_transitions")
        )
        or len(receipts) != ranks
        or any(type(item.get("rank")) is not int for item in receipts)
        or {item.get("rank") for item in receipts} != set(range(ranks))
        or any({key: value for key, value in item.items() if key != "rank"} != expected for item in receipts)
    ):
        raise ValueError("UFO sampler actor/B/norm/pool state or version differs")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def validate_ufo_checkpoint(path: Path, *, run_id=None, _staging=False, verify_hashes=True) -> dict:
    manifest = path / "resume-manifest.json"
    if (
        path.is_symlink()
        or manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size > 256 * 1024
    ):
        raise ValueError("Native checkpoint is partial or manifest invalid")
    receipt = json.loads(manifest.read_text())
    if set(receipt) != {
        "schema_version",
        "status",
        "format",
        "mode",
        "step",
        "primary_unit",
        "run_id",
        "execution",
        "compatibility_fingerprint",
        "state_coverage",
        "optimizer_steps",
        "model_keys",
        "files",
        "counters",
    }:
        raise ValueError("Native checkpoint has missing/unknown manifest fields")
    step = receipt.get("step")
    if (
        receipt.get("schema_version") != "eb/resume-checkpoint/v1"
        or receipt.get("format") not in {FORMAT, STREAM_FORMAT}
        or receipt.get("status") != "complete"
        or receipt.get("mode") != "trainer_state"
        or type(step) is not int
        or step < 0
        or path.name != f"global_step_{step}" + (".staging" if _staging else "")
        or (run_id is not None and receipt.get("run_id") != run_id)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(receipt.get("compatibility_fingerprint", "")))
        or receipt.get("state_coverage") != COVERAGE
        or receipt.get("optimizer_steps") != {name: step for name in OPTIMIZERS}
        or receipt.get("primary_unit") != "learner_joint_update"
    ):
        raise ValueError("Native checkpoint identity/count/coverage differs")
    binding = receipt.get("execution") or {}
    if (
        binding.get("checkpoint_format") != receipt["format"]
        or binding.get("provider_id") != "ufo"
        or binding.get("progress_unit") != "learner_joint_update"
        or binding.get("runtime_id") != "rlinf"
        or binding.get("topology_id") not in {"single_learner", "one_learner_three_samplers"}
    ):
        raise ValueError("Unknown native checkpoint execution binding")
    counters = receipt.get("counters") or {}
    if (
        set(counters) != {"learner_joint_updates", "sampling_rounds", "online_transitions", "next_trigger"}
        or any(type(value) is not int or value < 0 for value in counters.values())
        or counters["learner_joint_updates"] != step
        or counters["next_trigger"] < 11264
        or counters["next_trigger"] % 1024
    ):
        raise ValueError("Native checkpoint progress counters invalid")
    if binding.get("recipe_id") in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9"} and (
        counters["next_trigger"] < 34816 or (counters["next_trigger"] - 10240) % 24576
        or counters["online_transitions"] != counters["sampling_rounds"] * 24576
    ):
        raise ValueError("Scaled native checkpoint sampling counters invalid")
    keys = receipt.get("model_keys") or []
    for prefix in (
        *TARGETS,
        "_actor",
        "_backward_map",
        "_forward_map",
        "_discriminator",
        "_critic",
        "_aux_critic",
        "_obs_normalizer",
    ):
        if not any(key.startswith(prefix + ".") for key in keys):
            raise ValueError(f"Native checkpoint lacks network/target/norm {prefix}")
    files = receipt.get("files")
    if not isinstance(files, list) or not files or len(files) > 100:
        raise ValueError("Native checkpoint member list invalid")
    seen, total = set(), 0
    for item in files:
        name = item.get("path", "")
        relative = PurePosixPath(name)
        if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or name in seen:
            raise ValueError("Native checkpoint path escapes or repeats")
        member = path
        for part in relative.parts:
            member = member / part
            if member.is_symlink():
                raise ValueError("Native checkpoint symlink forbidden")
        if (
            not member.resolve().is_relative_to(path.resolve())
            or not member.is_file()
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            or member.stat().st_size != item["bytes"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
            or (verify_hashes and digest(member) != item.get("sha256"))
        ):
            raise ValueError("Native checkpoint member size/hash differs")
        seen.add(name)
        total += item["bytes"]
    if not {"config.json", "optimizers.pth", "model/model.safetensors", "trainer-state.pt"} <= seen:
        raise ValueError("Native checkpoint required state is missing")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p != manifest}
    if actual != seen or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Native checkpoint has unknown state files or symlinks")
    return {**receipt, "path": str(path.resolve()), "bytes": total}


def tensorize(value):
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {k: tensorize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensorize(v) for v in value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def numpyize(value):
    import torch

    if isinstance(value, dict):
        return {k: numpyize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(numpyize(v) for v in value)
    return value.numpy().copy() if torch.is_tensor(value) else value


def rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": tensorize(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch

    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Native RNG coverage incomplete")
    random.setstate(state["python"])
    np.random.set_state(numpyize(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG rank coverage differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_ufo_checkpoint(agent, replay: dict, rollout_state: dict, output: Path, *, receipt: dict) -> dict:
    import torch

    try:
        from eb.runners.bfm_ufo_policy import pool_state
    except ModuleNotFoundError:
        from bfm_ufo_policy import pool_state
    step = receipt["learner_joint_updates"]
    if (
        output.exists()
        or output.is_symlink()
        or output.name != f"global_step_{step}"
        or set(replay) != {"train", "expert_slicer"}
        or set(rollout_state) != sampler_ranks(receipt["execution"])
        or optimizer_steps(agent) != {name: step for name in OPTIMIZERS}
        or receipt["online_transitions"] != replay["train"].received
    ):
        raise ValueError("Checkpoint output/counters/optimizer/rank coverage invalid")
    state = agent._model.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite native model state")
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        agent.save(str(staging))
        checkpoint_format=receipt["execution"]["checkpoint_format"]
        if checkpoint_format not in {FORMAT, STREAM_FORMAT}:
            raise ValueError("Unknown native checkpoint format")
        stream_replay=checkpoint_format == STREAM_FORMAT
        payload = {
            "schema_version": "eb/ufo-trainer-state/v2" if stream_replay else "eb/ufo-trainer-state/v1",
            "receipt": dict(receipt),
            "replay": {k: (v.save_arrays(staging / "replay") if stream_replay and k=="train"
                            else v.state_dict()) for k,v in replay.items()},
            "pool": pool_state(agent),
            "rng": rng_state(),
            "rollout": rollout_state,
        }
        torch.save(tensorize(payload), staging / "trainer-state.pt")
        if receipt.get('training_gate') is not None:
            (staging/'training-gate.json').write_text(json.dumps(receipt['training_gate'],allow_nan=False))
            (staging/'training-gate-records.json').write_text(json.dumps(receipt['training_gate_records'],allow_nan=False))
        files = []
        for member in sorted(staging.rglob("*")):
            if member.is_file():
                with member.open("rb") as stream:
                    os.fsync(stream.fileno())
                files.append(
                    {
                        "path": member.relative_to(staging).as_posix(),
                        "bytes": member.stat().st_size,
                        "sha256": digest(member),
                    }
                )
        manifest = {
            "schema_version": "eb/resume-checkpoint/v1",
            "status": "complete",
            "format": checkpoint_format,
            "mode": "trainer_state",
            "step": step,
            "primary_unit": "learner_joint_update",
            "run_id": receipt["run_id"],
            "execution": receipt["execution"],
            "compatibility_fingerprint": receipt["compatibility_fingerprint"],
            "state_coverage": COVERAGE,
            "optimizer_steps": optimizer_steps(agent),
            "model_keys": sorted(state),
            "files": files,
            "counters": {
                k: receipt[k]
                for k in ("learner_joint_updates", "sampling_rounds", "online_transitions", "next_trigger")
            },
        }
        manifest_path = staging / "resume-manifest.json"
        with manifest_path.open("w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        result = validate_ufo_checkpoint(staging, run_id=receipt["run_id"], _staging=True)
        for directory in [*sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True), staging]:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        staging.rename(output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {**result, "path": str(output.resolve())}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_ufo_checkpoint(path: Path, *, agent, replay: dict, expected: dict, device: str) -> dict:
    import safetensors.torch
    import torch

    try:
        from eb.runners.bfm_ufo_policy import load_pool
    except ModuleNotFoundError:
        from bfm_ufo_policy import load_pool
    manifest = validate_ufo_checkpoint(path)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Native checkpoint compatibility differs: {key}")
    saved_cfg = json.loads((path / "config.json").read_text())
    saved_cfg["model"]["device"] = agent.cfg.model.device
    if saved_cfg != agent.cfg.model_dump():
        raise ValueError("Native restore config differs from checkpoint algorithm")
    payload = torch.load(path / "trainer-state.pt", map_location="cpu", weights_only=True)
    if (
        set(payload) != {"schema_version", "receipt", "replay", "pool", "rng", "rollout"}
        or payload["schema_version"] != ("eb/ufo-trainer-state/v2" if manifest["format"]==STREAM_FORMAT else "eb/ufo-trainer-state/v1")
    ):
        raise ValueError("Missing or unknown trainer state keys")
    receipt = payload["receipt"]
    if (
        receipt.get("execution") != manifest["execution"]
        or receipt.get("learner_joint_updates") != manifest["step"]
        or receipt.get("compatibility_fingerprint") != manifest["compatibility_fingerprint"]
        or any(receipt.get(k) != v for k, v in manifest["counters"].items())
        or set(payload["replay"]) != set(replay)
        or set(payload["rollout"]) != sampler_ranks(manifest["execution"])
    ):
        raise ValueError("Native trainer identity/counters/rank coverage differs")
    state = safetensors.torch.load_file(path / "model/model.safetensors", device=device)
    if sorted(state) != manifest["model_keys"] or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Native network/norm state differs")
    optimizers = torch.load(path / "optimizers.pth", map_location=device, weights_only=True)
    if set(optimizers) != set(OPTIMIZERS):
        raise ValueError("Native checkpoint requires exactly six optimizer states")
    if any(
        int(v["step"].item()) != manifest["step"]
        for optimizer in optimizers.values()
        for v in optimizer["state"].values()
    ):
        raise ValueError("Native optimizer steps differ from learner count")
    agent._model.load_state_dict(state, strict=True)
    for name in OPTIMIZERS:
        getattr(agent, name).load_state_dict(optimizers[name])
    if optimizer_steps(agent) != manifest["optimizer_steps"]:
        raise ValueError("Native optimizer state incomplete")
    for name in replay:
        if manifest["format"]==STREAM_FORMAT and name=="train":
            replay[name].load_arrays(path / "replay", numpyize(payload["replay"][name]))
        else:
            replay[name].load_state_dict(numpyize(payload["replay"][name]))
    if replay["train"].received != receipt["online_transitions"]:
        raise ValueError("Native replay received count differs")
    load_pool(agent, payload["pool"])
    restore_rng(payload["rng"])
    # Torch 2.7 legacy pickle cannot transport uint32 storage; NumPy owns this RNG.
    for state in payload["rollout"].values():
        state["rng_numpy"] = numpyize(state["rng_numpy"])
    return {"receipt": receipt, "rollout": payload["rollout"], "mode": "trainer_state"}
