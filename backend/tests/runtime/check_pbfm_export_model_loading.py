"""CPU regression for the real export boundary and a frozen full checkpoint.

Mount native model/prompt read-only at /readonly-model and /readonly-prompt,
and the prepared source at /outputs/runtime/pbfm. No simulation or CUDA work.
"""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from eb.runners import pbfm_eval, pbfm_export, pbfm_source, rlinf_pbfm_runtime


def check_prepare(model_root):
    from pbfm.agents.fb_cpr_aux.config import FBcprAuxModelConfig
    from pbfm.inference import runtime as native_runtime

    saved_config = json.loads((model_root / "checkpoint/model/config.json").read_text())
    calls = []
    policy = NS()
    policy.to = lambda device: calls.append(("to", device)) or policy
    policy.eval = lambda: calls.append(("eval",)) or policy

    def load(path, *, device):
        # The real serialized model schema rejects cuda:0 before any allocation.
        FBcprAuxModelConfig.model_validate({**saved_config, "device": device}, strict=False)
        assert device == "cuda"
        calls.append(("load", device))
        return policy

    receipt = {"patch_sha256": "a" * 64}
    plan = {
        "camera_contract": pbfm_source.CAMERA_CONTRACT,
        "evaluation_mode": {
            "domain_randomization": False, "observation_noise": False,
            "camera_installation_randomization": False,
        },
        "source_refs": {}, "pbfm_patch_sha256": receipt["patch_sha256"], "model_files": [],
        "prompt_manifest_sha256": "b" * 64, "prompt_file_sha256": "b" * 64,
        "robot_xml_sha256": "b" * 64,
    }
    robot = NS(robot=NS(xml_path="/robot.xml"), config_path="/robot.yaml")
    manifest = {"split": "validation", "file_sha256": "b" * 64}
    with tempfile.TemporaryDirectory(prefix="eb-export-load-") as directory:
        with (
            patch.object(rlinf_pbfm_runtime, "verify_source_refs"),
            patch.object(pbfm_source, "prepare_pbfm_source", return_value=receipt),
            patch.object(pbfm_source, "rpo_robot_config", return_value=Path("/robot.yaml")),
            patch.object(pbfm_export.os, "chdir"),
            patch.object(pbfm_export.sys, "path", list(pbfm_export.sys.path)),
            patch.object(pbfm_export, "verify_index"),
            patch.object(pbfm_export, "safe_file", side_effect=lambda root, name: root / name),
            patch.object(pbfm_export, "sha256", return_value="b" * 64),
            patch.object(pbfm_export, "read_json", side_effect=lambda path: (
                receipt if path.name == "source-adaptation.json" else manifest
            )),
            patch("pbfm.robots.load_robot_training_spec", return_value=robot),
            patch("pbfm.agents.load_utils.load_model_from_checkpoint_dir", side_effect=load),
            patch.object(native_runtime, "load_mjlab_env_cfg", return_value=("env-config", False)) as env,
            patch.object(torch.cuda, "set_device") as set_device,
        ):
            result = pbfm_export.prepare_runtime(plan, Path(directory))
        assert result == (policy, robot, "env-config", False, receipt)
        assert calls == [("load", "cuda"), ("to", "cuda:0"), ("eval",)]
        set_device.assert_called_once_with(torch.device("cuda:0"))
        assert env.call_args.kwargs["device"] == "cuda:0"
    assert pbfm_eval.prepare_runtime is pbfm_export.prepare_runtime
    print("EXPORT_AND_EVAL_NATIVE_CHECKPOINT_DEVICE_BOUNDARY_VERIFIED", flush=True)


def check_checkpoint(model_root, prompt_root, robot_config):
    from pbfm.agents.load_utils import load_model_from_checkpoint_dir
    from pbfm.inference.export.depth_policy import export_direct_depth_policy
    from pbfm.inference.runtime import _terrain_env_cfg, _tracking_z, load_mjlab_env_cfg
    from safetensors import safe_open

    model = load_model_from_checkpoint_dir(model_root / "checkpoint", device="cpu").eval()
    assert all(p.device.type == "cpu" for p in model.parameters())
    with safe_open(model_root / "checkpoint/model/model.safetensors", framework="pt", device="cpu") as saved:
        state = model.state_dict()
        assert set(state) == set(saved.keys())
        for name, tensor in state.items():
            assert torch.equal(tensor, saved.get_tensor(name)), name
            assert bool(torch.isfinite(tensor).all()), name
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 600527007, parameters
    print(json.dumps({"stage": "full_checkpoint_loaded", "parameters": parameters,
                      "state_keys": len(state), "all_saved_tensors_equal_and_finite": True}), flush=True)
    config, _ = load_mjlab_env_cfg(
        model_root, data_path=prompt_root / "validation.pkl", robot_config=robot_config,
        device="cuda:0", headless=True, disable_dr=True, disable_obs_noise=True,
        max_episode_length_s=21.0,
    )
    assert config.device == "cuda:0" and config.disable_domain_randomization and config.disable_obs_noise
    for family in ("flat", "perlin_rough", "low_stairs_up", "low_stairs_down", "low_platforms",
                   "hf_pyramid_slope_inv", "boxes"):
        assert _terrain_env_cfg(config, family, seed=71000).device == "cuda:0"

    backward = {key: torch.zeros((9, *model.obs_space[key].shape))
                for key in model.cfg.archi.b.input_filter.key}
    z = _tracking_z(model, backward)
    assert z.shape == (9, 256) and bool(torch.isfinite(z).all())
    with tempfile.TemporaryDirectory(prefix="eb-full-policy-cpu-") as directory:
        path = Path(directory) / "policy.onnx"
        exported = export_direct_depth_policy(model, path, verify=True)
        assert exported["format"] == "direct_depth_full_v1"
        assert exported["output_names"] == ["action", "estimated_velocity"]
        assert exported["depth_shape"] == [8, 18, 32] and exported["z_dim"] == 256
        assert sum(shape[0] for key, shape in exported["input_shapes"].items()
                   if key != "z" and len(shape) == 1) == 380
        assert len(exported["verification"]) == 12
        obs = {key: torch.zeros((1, *space.shape)) for key, space in model.obs_space.items()}
        obs["depth_image"] = (torch.arange(8 * 18 * 32) % 256).to(torch.uint8).reshape(1, 8, 18, 32)
        obs["velocity_command"][:] = torch.tensor([0.8, -0.2, 1.0])
        session = pbfm_export.parity_session(path)
        actual = pbfm_export.onnx_action(session, obs, z[:1])
        expected = model.act(obs, z[:1], mean=True)
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-4, rtol=2e-4)
        print(json.dumps({"parameters": parameters, "state_keys": len(state),
                          "onnx_bytes": path.stat().st_size,
                          "native_verification_cases": len(exported["verification"]),
                          "consumer_action_max_error": float(np.abs(actual - expected.numpy()).max())}),
              flush=True)
    print("FROZEN_FULL_MODEL_CPU_LOAD_BACKWARD_ENV_CONFIG_FULL_ONNX_AND_ORT_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/readonly-model"))
    parser.add_argument("--prompt-root", type=Path, default=Path("/readonly-prompt"))
    parser.add_argument("--robot-config", type=Path,
                        default=Path("/outputs/runtime/pbfm/pbfm/configs/robots/rpo_23dof.yaml"))
    parser.add_argument("--boundary-only", action="store_true")
    args = parser.parse_args()
    assert not torch.cuda.is_available(), "this check must not have CUDA access"
    check_prepare(args.model_root)
    if not args.boundary_only:
        check_checkpoint(args.model_root, args.prompt_root, args.robot_config)
