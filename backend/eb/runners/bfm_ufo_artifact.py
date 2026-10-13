"""Export pinned native actor/prompt through the existing RPO ONNX protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from eb.rpo_bfm import CODEC_HASH, codec_hash_for_execution
    from eb.runners.bfm_ufo_agent import AGENT_CONFIG_KEYS, agent_config, to_ufo_observation
    from eb.runners.bfm_ufo_checkpoint import ufo_checkpoint_root, validate_ufo_checkpoint
    from eb.runners.bfm_ufo_policy import observation_space
    from eb.runners.rpo_bfm_motion import MotionView
    from eb.runners.rpo_ufo_motion import digest, load_robot_state
except ModuleNotFoundError:
    from bfm_ufo_agent import AGENT_CONFIG_KEYS, agent_config, to_ufo_observation
    from bfm_ufo_checkpoint import ufo_checkpoint_root, validate_ufo_checkpoint
    from bfm_ufo_policy import observation_space
    from rpo_bfm import CODEC_HASH, codec_hash_for_execution
    from rpo_bfm_motion import MotionView
    from rpo_ufo_motion import digest, load_robot_state


def native_checkpoint_path(root: Path):
    if (root / "resume-manifest.json").is_file():
        return root
    receipt_path = root / "native-checkpoint.json"
    if root.is_symlink() or receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Native model package receipt missing")
    receipt = json.loads(receipt_path.read_text())
    path = root / str(receipt.get("path", ""))
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Native checkpoint escapes model package")
    return path


class NativeExportPolicy:
    def __init__(self, model, view, codec_sha256=CODEC_HASH):
        self.model = model.eval()
        self.actor, self.backward_map = model._actor, model._backward_map
        self.obs_normalizer = model._obs_normalizer.eval()
        self.motion_view = view
        self._reference = view.reference()
        self.codec_sha256 = codec_sha256

    def normalize(self, obs):
        return self.model._normalize(to_ufo_observation(obs))

    def goal_inference(self, goal_obs):
        return self.model.goal_inference(to_ufo_observation(goal_obs))


def load_ufo_export_policy(checkpoint, config_path, motion_root, asset_sha256):
    import safetensors.torch
    import yaml

    native_path = native_checkpoint_path(checkpoint)
    manifest = validate_ufo_checkpoint(native_path)
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg["actor"]["model"]
    codec_sha256 = codec_hash_for_execution(manifest["execution"])
    if model_cfg.get("codec_sha256", CODEC_HASH) != codec_sha256:
        raise ValueError("Native export codec differs from frozen recipe")
    try:
        from eb.rpo_bfm import auxiliary_profile_for_execution
    except ModuleNotFoundError:
        from rpo_bfm import auxiliary_profile_for_execution
    if model_cfg.get("auxiliary_profile") != auxiliary_profile_for_execution(manifest["execution"]):
        raise ValueError("Native export auxiliary profile differs from frozen recipe")
    if (
        model_cfg.get("policy_setup") != "rpo-v2-bfm-ufo"
        or model_cfg.get("obs_dim") != 375
        or model_cfg.get("critic_obs_dim") != 733
        or model_cfg.get("action_dim") != 23
        or model_cfg.get("num_action_chunks") != 1
        or model_cfg.get("asset_sha256") != asset_sha256
    ):
        raise ValueError("Native export config/interface/asset differs")
    native_cfg = agent_config(
        {key: model_cfg[key] for key in AGENT_CONFIG_KEYS if key in model_cfg}, device="cpu"
    )
    saved_cfg = json.loads((native_path / "config.json").read_text())
    # Device is a placement choice; all algorithm and architecture fields are frozen.
    saved_cfg["model"]["device"] = "cpu"
    if saved_cfg != native_cfg.model_dump():
        raise ValueError("Native export resolved config differs from checkpoint algorithm")
    model = native_cfg.model.build(observation_space(), 23)
    weights = native_path / "model/model.safetensors"
    state = safetensors.torch.load_file(weights, device="cpu")
    inference_keys = set(model.state_dict())
    if not inference_keys <= state.keys() or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Native inference state incomplete/nonfinite")
    model.load_state_dict({key: state[key] for key in inference_keys}, strict=True)
    _, expert = load_robot_state(motion_root, asset_sha256=asset_sha256, expected_codec_sha256=codec_sha256)
    for clip in expert.clips:
        if "reward_features" not in clip["arrays"]:
            raise ValueError("Native command prompt needs frozen train reward features")
    view = MotionView(
        [(clip["id"], clip["arrays"]) for clip in expert.clips], expert.manifest_sha256, seed=42
    )
    return NativeExportPolicy(model, view, codec_sha256), weights, cfg


def export_ufo_policy(
    checkpoint: Path, resolved_config: Path, motion_root: Path, asset_sha256: str, output: Path
) -> dict:
    try:
        from eb.runners.rpo_bfm_artifact import export
    except ModuleNotFoundError:
        from rpo_bfm_artifact import export
    manifest = validate_ufo_checkpoint(native_checkpoint_path(checkpoint))
    result = export(checkpoint, resolved_config, motion_root, asset_sha256, output)
    with np.load(output / result["golden"]["path"], allow_pickle=False) as traces:
        for left, right in (
            ("torch_action", "onnx_action"),
            ("torch_action_single", "onnx_action_single"),
            ("torch_command_z", "onnx_command_z"),
        ):
            np.testing.assert_allclose(traces[left], traces[right], atol=1e-5, rtol=1e-4)
    result.update(
        execution=manifest["execution"],
        checkpoint_format=manifest["format"],
        learner_joint_updates=manifest["step"],
        native_manifest_sha256=digest(native_checkpoint_path(checkpoint) / "resume-manifest.json"),
        primary_unit="learner_joint_update",
        io_signature={
            "actor_fields": [
                {"name": "state", "width": 52, "layout": "q-minus-default23,dq23,gravity_b3,omega_b/4"},
                {"name": "last_action", "width": 23, "units": "policy [-1,1]"},
                {"name": "history_actor", "width": 300, "layout": "4x75 oldest-first, frame-major"},
            ],
            "privileged_width": 358,
            "latent_dim": 256,
            "action_dim": 23,
            "normalization": "frozen native batch norm embedded in ONNX",
            "action_decoder": "RPO v1 default_q + .25*ACTION_GAIN*action",
        },
    )
    try:
        from eb.rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
    except ModuleNotFoundError:
        from rpo_bfm import action_control_for_execution, auxiliary_profile_for_execution
    control = action_control_for_execution(manifest["execution"])
    if control:
        result["action_control"] = control
        if control.get("control_profile") == "reachable_v2":
            result["io_signature"]["action_decoder"] = "URDF finite midpoint + halfwidth*action"
    auxiliary = auxiliary_profile_for_execution(manifest["execution"])
    if auxiliary:
        result["auxiliary_profile"] = auxiliary
    (output / "export-manifest.json").write_text(json.dumps(result, indent=2))
    return result


def publish_ufo_checkpoint(output, cfg, receipt):
    import os
    import shutil

    from rlinf_entry import atomic_json

    root = ufo_checkpoint_root(output, cfg)
    candidates = []
    for path in root.glob("global_step_*"):
        if path.name.removeprefix("global_step_").isdigit():
            candidates.append(validate_ufo_checkpoint(path, run_id=os.environ["EB_RUN_ID"]))
    if not candidates:
        raise ValueError("UFO training exited without a complete native checkpoint")
    chosen = max(candidates, key=lambda c: c["step"])
    if chosen["step"] != receipt["target_learner_updates"] or chosen["execution"] != receipt["execution"]:
        raise ValueError("UFO completed native checkpoint differs from frozen target/binding")
    destination = output / "model"
    destination.mkdir(exist_ok=True)
    staging = destination / ".ufo.staging"
    staging.mkdir(exist_ok=False)
    try:
        relative = f"native/global_step_{chosen['step']}"
        # Stable checkpoints originate within the model bind mount, so finalized bytes can be shared.
        copy_file = os.link if root.is_relative_to(destination) else shutil.copy2
        shutil.copytree(chosen["path"], staging / relative, copy_function=copy_file)
        shutil.copyfile(output / "resolved-config.yaml", staging / "resolved-config.yaml")
        shutil.copyfile("/opt/ufo-source-receipt.json", staging / "source-receipt.json")
        atomic_json(
            staging / "native-checkpoint.json",
            {"path": relative, "step": chosen["step"], "format": chosen["format"]},
        )
        atomic_json(
            staging / "actor-normalization.json",
            {
                "schema_version": "eb/rpo-bfm-normalization/v1",
                "kind": "frozen_running_stats",
                "observation_dim": 733,
                "history_length": 4,
                "latent_dim": 256,
            },
        )
        files = [
            {
                "path": "model/" + p.relative_to(staging).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        ]
        for member in staging.iterdir():
            if (destination / member.name).exists():
                raise ValueError("Native model publication refuses to overwrite outputs")
            member.rename(destination / member.name)
        staging.rmdir()
        atomic_json(
            output / "checkpoint-manifest.json",
            {
                "schema_version": "eb/checkpoint/v1",
                "status": "complete",
                "format": chosen["format"],
                "run_id": chosen["run_id"],
                "template_id": receipt["template_id"],
                "execution": chosen["execution"],
                "step": chosen["step"],
                "model_path": "model",
                "weights_path": "model/" + relative + "/model/model.safetensors",
                "native_checkpoint_path": "model/" + relative,
                "files": files,
                "source_checkpoint": str(Path(chosen["path"]).relative_to(output)),
                "primary_unit": "learner_joint_update",
                "bfm_progress": chosen["counters"],
            },
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
