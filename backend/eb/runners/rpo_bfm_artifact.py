"""Export a frozen BFM actor and train-only prompt latents for platform smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OBS_DIM, LATENT_DIM, ACTION_DIM = 375, 256, 23
COMMAND_REFERENCE_LIMIT = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ActorGraph(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.actor = policy.actor.eval()
        self.normalizer = policy.obs_normalizer.eval()

    def forward(self, observation, z):
        obs = {"state":observation[:, :52],
               "last_action":observation[:, 52:75],
               "history_actor":observation[:, 75:]}
        normalized = self.normalizer(obs)
        latent = 16 * torch.nn.functional.normalize(z, p=2, dim=-1)
        return self.actor(normalized, latent, .05).mean.reshape(-1, ACTION_DIM)


class _CommandLatentGraph(torch.nn.Module):
    def __init__(self, beta, features):
        super().__init__()
        self.register_buffer("beta", beta.detach().float())
        self.register_buffer("features", features.detach().float())

    def forward(self, commands):
        translation = (self.features[None, :, :2] - commands[:, None, :2]).square().sum(-1)
        rotation = (self.features[None, :, 2] - commands[:, None, 2]).square()
        upright = self.features[:, 3].clamp(0, 1)
        height = ((self.features[:, 4] - .5) / .25).clamp(0, 1)
        rewards = torch.exp(-translation / .25) * torch.exp(-rotation / .25) * upright * height
        latent = rewards @ self.beta / self.beta.shape[0]
        return 16 * torch.nn.functional.normalize(latent, p=2, dim=-1)


def _load_policy(checkpoint: Path, config_path: Path, motion_root: Path, asset_sha256: str):
    import yaml
    from bfm_zero_policy import BFMZeroPolicy
    from rpo_bfm import CODEC_HASH

    for path in (checkpoint, config_path, motion_root):
        if path.is_symlink():
            raise ValueError("BFM export input must not be a symlink")
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("BFM export config exceeds budget")
    if ((yaml.safe_load(config_path.read_text()) or {}).get("actor") or {}).get("model", {}).get("policy_setup") == "rpo-v2-bfm-ufo":
        from bfm_ufo_artifact import load_ufo_export_policy
        return load_ufo_export_policy(checkpoint, config_path, motion_root, asset_sha256)
    weights_path = checkpoint / "model_state_dict/full_weights.pt" if checkpoint.is_dir() else checkpoint
    if (not weights_path.is_file() or weights_path.is_symlink()
            or config_path.stat().st_size > 1024 * 1024):
        raise ValueError("BFM export checkpoint/config is missing or unbounded")
    config = yaml.safe_load(config_path.read_text()) or {}
    model_cfg = dict((config.get("actor") or {}).get("model") or {})
    if (model_cfg.get("policy_setup") != "rpo-v2-bfm-zero"
            or model_cfg.get("obs_dim") != OBS_DIM or model_cfg.get("action_dim") != ACTION_DIM
            or model_cfg.get("critic_obs_dim") != 733 or model_cfg.get("num_action_chunks") != 1
            or model_cfg.get("asset_sha256") != asset_sha256):
        raise ValueError("BFM source config differs from the frozen actor/asset contract")
    model_cfg["motion_root"] = str(motion_root)
    model_cfg["asset_sha256"] = asset_sha256
    policy = BFMZeroPolicy(model_cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    for wrapper in ("model", "model_state_dict", "state_dict", "module"):
        if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
            state = state[wrapper]
    if not isinstance(state, dict):
        raise ValueError("BFM checkpoint does not contain a state dict")
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_fsdp_wrapped_module.")):
            key = key.removeprefix("module.").removeprefix("_fsdp_wrapped_module.")
        if key in cleaned:
            raise ValueError("BFM checkpoint has duplicate model state keys")
        cleaned[key] = value
    policy.load_state_dict(cleaned, strict=True)
    policy.eval()
    if any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("BFM checkpoint contains non-finite model/normalizer state")
    if policy.codec_sha256 != CODEC_HASH:
        raise ValueError("BFM checkpoint codec differs from the exporter")
    return policy, weights_path, config


def export(checkpoint: Path, resolved_config: Path, motion_root: Path,
           asset_sha256: str, output: Path) -> dict:
    import onnx
    import onnxruntime as ort
    from bfm_zero_policy import command_to_latent, project_z
    from rpo_bfm import ACTION_GAIN, CODEC_HASH, CODEC_V2_HASH, ACTION_HALFWIDTH_V2, PROMPT_COMMAND

    policy, weights_path, config = _load_policy(checkpoint, resolved_config, motion_root, asset_sha256)
    codec_sha256 = policy.codec_sha256
    action_scale = np.asarray(ACTION_HALFWIDTH_V2) if codec_sha256 == CODEC_V2_HASH else np.asarray(ACTION_GAIN) * .25
    output.mkdir(parents=True, exist_ok=True)
    graph = _ActorGraph(policy).eval()
    reference = policy._reference
    goal_obs = {"states":torch.from_numpy(reference["states"][:1]),
                "privileged_state":torch.from_numpy(reference["privileged_state"][:1])}
    goal_z = policy.goal_inference(goal_obs)
    expert = policy.motion_view.sample_sequences(8, length=8)
    tracking_obs = {"states":torch.from_numpy(expert["states"]),
                    "privileged_state":torch.from_numpy(expert["privileged_state"])}
    tracking_beta = policy.backward_map(policy.normalize(tracking_obs)).mean(0, keepdim=True)
    tracking_z = project_z(tracking_beta)
    reference_count = len(reference["states"])
    if not 1 <= reference_count:
        raise ValueError("BFM command front end requires train reference states")
    reference_indices = np.linspace(0, reference_count - 1,
                                    min(reference_count, COMMAND_REFERENCE_LIMIT), dtype=np.int64)
    command_observation = {key:torch.from_numpy(reference[key][reference_indices])
                           for key in ("states", "privileged_state")}
    with torch.inference_mode():
        beta = policy.backward_map(policy.normalize(command_observation)).detach().cpu()
    features = torch.from_numpy(reference["reward_features"][reference_indices]).float()
    command_graph = _CommandLatentGraph(beta, features).eval()
    minimum = torch.tensor(PROMPT_COMMAND["minimum"], dtype=torch.float32)
    maximum = torch.tensor(PROMPT_COMMAND["maximum"], dtype=torch.float32)
    commands = torch.stack((minimum, torch.zeros(3), torch.tensor([.3, 0., 0.]), maximum,
                            torch.tensor([.4, -.2, .7]), torch.tensor([-.3, .25, -1.1])))
    command_z = command_graph(commands)
    direct_z = command_to_latent(commands, beta, features)
    if not torch.allclose(command_z, direct_z, atol=1e-6, rtol=1e-6):
        raise ValueError("BFM command front end differs from the frozen command-to-latent function")
    reward_z = command_graph(torch.tensor([[0., 0., 0.], [.3, 0., 0.]], dtype=torch.float32))
    prompts = torch.cat((goal_z, tracking_z, reward_z), dim=0).detach().cpu().numpy().astype(np.float32)
    prompt_names = ["goal_train_reference", "tracking_train_sequence", "reward_stand", "reward_walk_0p3mps"]
    if (prompts.shape != (len(prompt_names), LATENT_DIM) or not np.isfinite(prompts).all()
            or not np.allclose(np.linalg.norm(prompts, axis=1), 16., atol=1e-4)):
        raise ValueError("BFM prompt construction produced an invalid latent")
    prompt_path = output / "prompts.npz"
    np.savez_compressed(prompt_path, **{name:prompts[index] for index, name in enumerate(prompt_names)})
    command_path = output / "command-to-latent.onnx"
    torch.onnx.export(
        command_graph, (commands[:1],), command_path,
        input_names=["command"], output_names=["z"],
        dynamic_axes={"command":{0:"batch"}, "z":{0:"batch"}}, opset_version=17,
    )
    command_model = onnx.load(str(command_path), load_external_data=False)
    command_input_dims = command_model.graph.input[0].type.tensor_type.shape.dim
    command_output_dims = command_model.graph.output[0].type.tensor_type.shape.dim
    if (len(command_model.graph.input) != 1 or command_model.graph.input[0].name != "command"
            or len(command_input_dims) != 2 or command_input_dims[-1].dim_value != 3
            or len(command_model.graph.output) != 1 or command_model.graph.output[0].name != "z"
            or len(command_output_dims) != 2 or not command_output_dims):
        raise ValueError("BFM command ONNX graph does not expose command[3] to z[256]")
    command_output_dims[-1].ClearField("dim_param")
    command_output_dims[-1].dim_value = LATENT_DIM
    onnx.save(command_model, command_path)
    command_session = ort.InferenceSession(str(command_path), providers=["CPUExecutionProvider"])
    command_expected = command_graph(commands).detach().cpu().numpy()
    command_actual = command_session.run(None, {"command":commands.numpy()})[0]
    command_actual_single = np.vstack([
        command_session.run(None, {"command":commands[index:index+1].numpy()})[0]
        for index in range(len(commands))])
    command_error = float(max(np.abs(command_expected-command_actual).max(initial=0),
                              np.abs(command_expected-command_actual_single).max(initial=0)))
    if (command_session.get_inputs()[0].shape[-1] != 3
            or command_session.get_outputs()[0].shape[-1] != LATENT_DIM
            or command_error > 1e-4 or not np.isfinite(command_actual).all()
            or not np.allclose(np.linalg.norm(command_actual, axis=-1), 16., atol=1e-4)):
        raise ValueError("BFM command ONNX failed range, norm, shape or Torch golden comparison")
    prompt_manifest = {
        "schema_version":"eb/rpo-bfm-prompts/v1", "status":"diagnostic_smoke",
        "model_weights_sha256":sha256(weights_path), "codec_sha256":codec_sha256,
        "asset_sha256":asset_sha256, "motion_manifest_sha256":policy.motion_view.manifest_sha256,
        "reference_split":"train", "reference_limit":len(reference["states"]),
        "prompts":{name:{"index":index, "z_norm":float(np.linalg.norm(prompts[index]))}
                   for index, name in enumerate(prompt_names)},
        "commands":{"reward_stand":[0., 0., 0.], "reward_walk_0p3mps":[.3, 0., 0.],
                    "field":PROMPT_COMMAND["field"], "units":PROMPT_COMMAND["units"]},
        "command_frontend":{"path":command_path.name, "sha256":sha256(command_path),
                             "inputs":["command"], "outputs":["z"], "reference_split":"train",
                             "reference_count":len(reference_indices),
                             "reference_limit":COMMAND_REFERENCE_LIMIT,
                             "minimum":PROMPT_COMMAND["minimum"],
                             "maximum":PROMPT_COMMAND["maximum"], "formula":"rpo-bfm-reward-inference/v1"},
        "artifact":{"path":prompt_path.name, "sha256":sha256(prompt_path)},
    }
    (output / "prompt-manifest.json").write_text(
        json.dumps(prompt_manifest, indent=2, allow_nan=False) + "\n")

    rng = np.random.default_rng(42)
    observation = rng.normal(0, .05, (4, OBS_DIM)).astype(np.float32)
    observation[:, :52] = reference["states"][:4]
    z = prompts.copy()
    observation_tensor, z_tensor = torch.from_numpy(observation), torch.from_numpy(z)
    onnx_path = output / "policy.onnx"
    torch.onnx.export(
        graph, (observation_tensor[:1], z_tensor[:1]), onnx_path,
        input_names=["observation", "z"], output_names=["action"],
        dynamic_axes={"observation":{0:"batch"}, "z":{0:"batch"}, "action":{0:"batch"}},
        opset_version=17,
    )
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    graph_io = onnx_model.graph
    input_dims = {value.name: value.type.tensor_type.shape.dim[-1].dim_value
                  for value in graph_io.input if value.type.tensor_type.shape.dim}
    output_dims = graph_io.output[0].type.tensor_type.shape.dim
    if (input_dims != {"observation":OBS_DIM, "z":LATENT_DIM}
            or len(graph_io.input) != 2 or len(graph_io.output) != 1
            or graph_io.output[0].name != "action" or not output_dims):
        raise ValueError("BFM ONNX graph does not expose static observation/z/action dimensions")
    # Torch leaves the distribution's fixed-width output symbolic after Expand/Reshape.
    output_dims[-1].ClearField("dim_param")
    output_dims[-1].dim_value = ACTION_DIM
    onnx.save(onnx_model, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if (len(session.get_inputs()) != 2 or len(session.get_outputs()) != 1
            or session.get_inputs()[0].shape[-1] != OBS_DIM
            or session.get_inputs()[1].shape[-1] != LATENT_DIM
            or session.get_outputs()[0].shape[-1] != ACTION_DIM):
        raise ValueError("BFM ONNX measured I/O differs from observation/z/action contract")
    with torch.inference_mode():
        expected = graph(observation_tensor, z_tensor).numpy()
        expected_single = np.vstack([graph(observation_tensor[i:i+1], z_tensor[i:i+1]).numpy()
                                     for i in range(len(prompts))])
    actual = session.run(None, {"observation":observation, "z":z})[0]
    actual_single = np.vstack([session.run(None, {"observation":observation[i:i+1], "z":z[i:i+1]})[0]
                               for i in range(len(prompts))])
    delta = np.abs(expected - actual)
    single_delta = np.abs(expected_single - actual_single)
    error = float(max(delta.max(initial=0), single_delta.max(initial=0)))
    target_error = float(max((delta * action_scale).max(initial=0),
                             (single_delta * action_scale).max(initial=0)))
    if (not np.isfinite(expected).all() or not np.isfinite(actual).all()
            or error > 1e-4 or target_error > 1e-4
            or np.max(np.abs(expected)) > 1.0001):
        raise ValueError("BFM ONNX failed finite/bounds/PyTorch golden comparison")
    golden_path = output / "golden-traces.npz"
    np.savez_compressed(golden_path, observation=observation, z=z,
                        torch_action=expected, onnx_action=actual,
                        torch_action_single=expected_single, onnx_action_single=actual_single,
                        commands=commands.numpy(), torch_command_z=command_expected,
                        onnx_command_z=command_actual, onnx_command_z_single=command_actual_single)
    manifest = {
        "schema_version":"eb/rpo-bfm-export/v1", "status":"verified",
        "readiness":"diagnostic_smoke", "model_weights_sha256":sha256(weights_path),
        "resolved_config_sha256":sha256(resolved_config), "codec_sha256":codec_sha256,
        "asset_sha256":asset_sha256, "motion_manifest_sha256":policy.motion_view.manifest_sha256,
        "onnx":{"path":onnx_path.name, "sha256":sha256(onnx_path), "obs":OBS_DIM,
                "latent":LATENT_DIM, "act":ACTION_DIM, "opset":17,
                "inputs":["observation", "z"], "batch_sizes":[1, len(prompts)],
                "max_abs_error":error, "max_joint_target_error_rad":target_error,
                "normalization":"embedded"},
        "prompts":{"path":prompt_path.name, "sha256":sha256(prompt_path),
                   "manifest":"prompt-manifest.json", "names":prompt_names,
                   "manifest_sha256":sha256(output / "prompt-manifest.json")},
        "command_latent":{"path":command_path.name, "sha256":sha256(command_path),
                           "command_dim":3, "latent_dim":LATENT_DIM, "opset":17,
                           "reference_split":"train", "reference_count":len(reference_indices),
                           "reference_limit":COMMAND_REFERENCE_LIMIT,
                           "batch_sizes":[1, len(commands)], "max_abs_error":command_error},
        "golden":{"path":golden_path.name, "sha256":sha256(golden_path),
                  "n_inputs":len(prompts), "batch_sizes":[1, len(prompts)],
                  "n_command_inputs":len(commands), "command_batch_sizes":[1, len(commands)]},
        "training_profile":(config.get("actor") or {}).get("model", {}).get("bfm_profile"),
    }
    (output / "export-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--resolved-config", type=Path, required=True)
    export_parser.add_argument("--motion-root", type=Path, required=True)
    export_parser.add_argument("--asset-sha256", required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        import yaml
        config = yaml.safe_load(args.resolved_config.read_text())
        exporter = export
        if config.get("actor", {}).get("model", {}).get("policy_setup") == "rpo-v2-bfm-ufo":
            from bfm_ufo_artifact import export_ufo_policy
            exporter = export_ufo_policy
        manifest = exporter(args.checkpoint, args.resolved_config, args.motion_root,
                            args.asset_sha256, args.output)
        print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
