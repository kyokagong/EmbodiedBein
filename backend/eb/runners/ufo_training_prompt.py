"""Derive new task latents with native UFO RewardInference in a task container."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from eb.ufo_native import TASKS, plan_hash, read_json, safe_file, sha256_file, validate_config
from eb.ufo_training_checkpoint import canonical_hash, validate_native_checkpoint
from eb.ufo_training_prompt import INFERENCE_FILES, PROMPT_SCHEMA, validate_prompt_plan


def reward_inference_arguments(root, buffer, plan):
    return dict(model_folder=root, data_path=root / "sim-data/motion.pkl",
        robot_config=root / "assets/robot.yaml", device="cpu", save_mp4=False,
        disable_dr=True, disable_obs_noise=True, episode_length=1,
        num_samples=plan["num_samples"], n_inferences=1, skip_rollouts=True,
        tasks=plan["tasks"], buffer_rank=0, buffer_path=buffer,
        max_workers=plan["cpu_workers"], process_executor=True,
        render_size=64, camera_distance=3.0, camera_azimuth=135.0, camera_elevation=-15.0,
        fps=25, max_episode_length_s=1.0, export_onnx=False)


def run_prompt(output: Path, plan: dict):
    import joblib
    import numpy as np
    import torch
    import yaml
    from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir
    from ufo.inference.reward_inference import run_reward_inference

    from eb.runners.rlinf_ufo_runtime import verify_source_refs
    from eb.runners.ufo_native_package import file_index, receipt, verify_files, write_json

    validate_prompt_plan(plan)
    verify_source_refs(plan)
    source = Path("/training-source")
    manifest_path = safe_file(source, "checkpoint-manifest.json")
    manifest = read_json(manifest_path, limit=4 * 1024 * 1024)
    if (sha256_file(manifest_path) != plan["native_manifest_sha256"]
            or canonical_hash(manifest) != plan["source_resume_artifact_sha256"]):
        raise ValueError("Prompt source training manifest changed")
    checkpoint = validate_native_checkpoint(source, manifest, run_id=plan["source_run_id"],
                                            verify_hashes=True)
    if (checkpoint["compatibility_fingerprint"] != plan["source_fingerprint"]
            or checkpoint["replay_id"] != plan["replay_id"]
            or checkpoint["replay"][0]["snapshot"] != plan["snapshot"]
            or checkpoint["source_refs"] != plan["source_refs"]
            or checkpoint["image_digest"] != plan["image_digest"]):
        raise ValueError("Prompt model and rank 0 Replay belong to different native states")
    model_record = next(item for item in manifest["files"]
                        if item["path"] == "checkpoint/model/model.safetensors")
    if model_record["sha256"] != plan["source_model_sha256"]:
        raise ValueError("Prompt source model weight identity differs")
    if output.is_symlink() or (output / "receipt.json").exists():
        raise ValueError("Prompt output already exists or is unsafe")
    output.mkdir(parents=True, exist_ok=True)
    root = output / "model"
    root.mkdir()
    verify_files(source, plan["files"])
    if {item["path"] for item in plan["files"]} != set(INFERENCE_FILES):
        raise ValueError("Prompt inference inputs differ from the bounded native file set")
    for item in plan["files"]:
        path = safe_file(source, item["path"])
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for index, item in enumerate(plan["extras"]):
        path = Path(f"/source-extra/{index}")
        if path.is_symlink() or sha256_file(path) != item["sha256"]:
            raise ValueError("Prompt auxiliary input changed after preview")
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if destination.stat().st_size != item["bytes"]:
            raise ValueError("Prompt auxiliary input size differs")
    codec = validate_config(read_json(root / "config.json"), read_json(root / "effective_config.json"),
                            read_json(root / "checkpoint/model/config.json"),
                            read_json(root / "checkpoint/model/init_kwargs.json"))
    if codec != plan["codec"]:
        raise ValueError("Prompt native codec changed")
    tree = ET.parse(root / "assets/rpo.xml")
    tree.getroot().find("compiler").set("meshdir", "meshes")
    tree.write(root / "assets/rpo.xml", encoding="unicode")
    robot = yaml.safe_load((root / "assets/robot.yaml").read_text())
    robot["xml_path"] = "rpo.xml"
    (root / "assets/robot.yaml").write_text(yaml.safe_dump(robot, sort_keys=False))
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    buffer = source / "checkpoint/buffers/train_rank_0"
    run_reward_inference(**reward_inference_arguments(root, buffer, plan))
    raw = safe_file(root, "reward_inference/reward_locomotion.pkl")
    values = joblib.load(raw)
    tasks = {}
    if not isinstance(values, dict) or set(values) != set(plan["tasks"]):
        raise ValueError("Native RewardInference did not return all frozen tasks")
    for task in plan["tasks"]:
        if not isinstance(values[task], list) or len(values[task]) != 1:
            raise ValueError("Native RewardInference returned an unexpected inference count")
        z = np.asarray(values[task][0], dtype=np.float32).reshape(-1)
        if z.shape != (256,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-6:
            raise ValueError("Native RewardInference produced an invalid task latent")
        tasks[task] = z.tolist()
    prompts = {"schema_version": "eb/ufo-native-prompts/v1", "derivation": "native_reward_inference",
               "tasks": tasks, "source_model_sha256": plan["source_model_sha256"],
               "buffer_snapshot_id": plan["replay_id"], "seed": plan["seed"]}
    prompt_root = output / "prompts"
    prompt_root.mkdir()
    write_json(prompt_root / "prompts.json", prompts)
    derivation = {"schema_version": PROMPT_SCHEMA, "operation": plan["operation"],
        "source_model_id": plan["model_id"], "source_model_sha256": plan["source_model_sha256"],
        "source_artifact_sha256": plan["source_artifact_sha256"],
        "source_resume_artifact_sha256": plan["source_resume_artifact_sha256"],
        "source_run_id": plan["source_run_id"], "snapshot": plan["snapshot"],
        "buffer_rank": 0, "seed": plan["seed"], "num_samples_per_inference": plan["num_samples"],
        "n_inferences": 1, "cpu_workers": plan["cpu_workers"], "process_executor": True,
        "tasks": plan["tasks"], "native_entrypoint": "ufo.inference.reward_inference.run_reward_inference",
        "source_refs": plan["source_refs"], "image_digest": plan["image_digest"],
        "torch_rng_before_sha256": rng_before,
        "torch_rng_after_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "raw_native_prompt_sha256": sha256_file(raw),
        "prompts_sha256": sha256_file(prompt_root / "prompts.json"),
        "qualification": "diagnostic", "model_quality_verified": False}
    write_json(prompt_root / "derivation.json", derivation)
    shutil.copyfile(raw, prompt_root / "reward_locomotion.pkl")
    prompt_manifest = {"schema_version": "eb/ufo-native-prompt-artifact/v1",
                       "derivation": derivation, "files": file_index(prompt_root)}
    write_json(prompt_root / "manifest.json", prompt_manifest)
    shutil.rmtree(root / "reward_inference")
    shutil.copyfile(prompt_root / "prompts.json", root / "prompts.json")
    write_json(root / "codec.json", codec)
    model = load_model_from_checkpoint_dir(root / "checkpoint", device="cpu")
    model.eval()
    obs = {key: torch.zeros((1, dim)) for key, dim in (
        ("state", 52), ("last_action", 23), ("history_actor", 300))}
    with torch.inference_mode():
        action = model.act(obs, torch.tensor([tasks[TASKS[0]["id"]]]), mean=True)
    if tuple(action.shape) != (1, 23) or not torch.isfinite(action).all():
        raise ValueError("Derived prompt model loader or actor validation failed")
    package = {"schema_version": "eb/ufo-native-import/v1", "codec": codec,
        "source": {"kind": "managed_training", "model_id": plan["model_id"],
                   "artifact_sha256": plan["source_artifact_sha256"], "run_id": plan["source_run_id"]},
        "source_model_id": plan["model_id"], "training_provenance": "platform",
        "prompt_derivation": "native_reward_inference", "prompt_binding": derivation,
        "source_refs": plan["source_refs"],
        "code_sources": {"ufo": plan["ufo_source_ref"], "rlinf": plan["rlinf_source_ref"]},
        "prompt_artifact_sha256": plan_hash(prompt_manifest), "loader_verified": True,
        "image_digest": plan["image_digest"], "files": file_index(root), "simulation_only": True}
    if sum(item["bytes"] for item in package["files"]) > 2 * 1024**3:
        raise ValueError("Derived native inference package exceeds 2 GiB")
    write_json(root / "manifest.json", package)
    return receipt(output, plan, manifest=package, prompt_manifest=prompt_manifest, package="model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan, limit=4 * 1024 * 1024)
    validate_prompt_plan(plan)
    output = Path("/outputs")
    output.mkdir(exist_ok=True)
    for name in ("tmp", "cache"):
        (output / name).mkdir(exist_ok=True)
    os.chdir(output)
    run_prompt(output, plan)


if __name__ == "__main__":
    main()
