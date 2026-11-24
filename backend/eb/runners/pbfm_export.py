"""Original direct-depth export plus current-model tracking prompt and real-trace parity."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from eb.pbfm_evaluation import validate_action_plan
from eb.pbfm_io import file_index, read_json, safe_file, sha256, verify_index, write_json


def prepare_runtime(plan: dict, output: Path):
    from eb.runners.pbfm_source import CAMERA_CONTRACT, prepare_pbfm_source, rpo_robot_config
    from eb.runners.rlinf_pbfm_runtime import verify_source_refs

    if plan["camera_contract"] != CAMERA_CONTRACT:
        raise ValueError("PBFM camera contract drift")
    if plan["evaluation_mode"] != {
        "domain_randomization": False,
        "observation_noise": False,
        "camera_installation_randomization": False,
    }:
        raise ValueError("PBFM inference must use frozen non-randomized evaluation inputs")
    verify_source_refs(
        {
            "source_refs": {
                name: {
                    **ref,
                    "path": "/sources/pbfm" if name == "pbfm" else "/robot-assets",
                    "manifest_path": f"/sources/{name}-manifest.json",
                }
                for name, ref in plan["source_refs"].items()
            }
        }
    )
    private = output / "runtime/pbfm"
    source_receipt = prepare_pbfm_source(Path("/sources/pbfm"), private)
    if source_receipt["patch_sha256"] != plan["pbfm_patch_sha256"]:
        raise ValueError("model-bound PBFM adaptation changed")
    sys.path.insert(0, str(private))
    os.chdir(private)
    model = output / "runtime/model"
    verify_index(Path("/model"), plan["model_files"], {row["path"] for row in plan["model_files"]})
    for row in plan["model_files"]:
        destination = model / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(Path("/model"), row["path"]), destination)
    prompt_root = Path("/prompt-data")
    manifest = read_json(safe_file(prompt_root, "manifest.json"))
    if (
        sha256(prompt_root / "manifest.json") != plan["prompt_manifest_sha256"]
        or manifest.get("split") != "validation"
        or manifest.get("file_sha256") != plan["prompt_file_sha256"]
        or sha256(safe_file(prompt_root, "validation.pkl")) != plan["prompt_file_sha256"]
    ):
        raise ValueError("frozen validation prompt data changed")
    from pbfm.agents.load_utils import load_model_from_checkpoint_dir
    from pbfm.inference.runtime import checkpoint_load_device, load_mjlab_env_cfg
    from pbfm.robots import load_robot_training_spec

    robot = load_robot_training_spec(rpo_robot_config(private))
    if sha256(Path(robot.robot.xml_path)) != plan["robot_xml_sha256"]:
        raise ValueError("model-bound RPO XML changed")
    producer_source = read_json(model / "source-adaptation.json")
    if producer_source["patch_sha256"] != source_receipt["patch_sha256"]:
        raise ValueError("export adaptation differs from the training producer")
    policy = load_model_from_checkpoint_dir(
        model / "checkpoint", device=checkpoint_load_device("cuda:0")
    ).to("cuda:0").eval()
    config, root_height = load_mjlab_env_cfg(
        model,
        data_path=prompt_root / "validation.pkl",
        robot_config=robot.config_path,
        device="cuda:0",
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=21.0,
    )
    return policy, robot, config, root_height, source_receipt


def write_inference_package(
    destination: Path, plan: dict, robot, core, policy, export: dict, source_receipt: dict
):
    """Make all simulation inference assets and explicit deployment codecs portable."""
    import xml.etree.ElementTree as ET

    from omegaconf import OmegaConf

    from eb.runners import pbfm_codec

    package_robot = destination / "robot"
    package_robot.mkdir()
    tree = ET.parse(robot.robot.xml_path)
    compiler = tree.getroot().find("compiler")
    meshroot = Path(compiler.get("meshdir"))
    compiler.set("meshdir", "meshes")
    for mesh in tree.findall(".//mesh"):
        name = mesh.get("file")
        if not name:
            continue
        source = safe_file(meshroot, name)
        target = package_robot / "meshes" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    tree.write(package_robot / "rpo.xml", encoding="utf-8", xml_declaration=True)
    import mujoco

    portable = mujoco.MjModel.from_xml_path(str(package_robot / "rpo.xml"))
    if portable.nq != 30 or portable.nu != 23:
        raise ValueError("portable RPO asset package has different control semantics")
    config = OmegaConf.load(robot.config_path)
    config.xml_path = "rpo.xml"
    OmegaConf.save(config, package_robot / "robot.yaml")
    private_root = Path(robot.config_path).parents[3]
    # Include original versioned inference/perception/history code and configs;
    # training weights, Replay, optimizer and unselected robot assets are excluded.
    for source in (private_root / "pbfm").rglob("*"):
        relative = source.relative_to(private_root)
        if (
            not source.is_file()
            or source.is_symlink()
            or source.suffix not in {".py", ".yaml", ".json"}
            or relative.parts[1] == "data"
        ):
            continue
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    local_source_robot = destination / "source/pbfm/configs/robots/rpo_23dof.yaml"
    config.xml_path = "../../../../robot/rpo.xml"
    OmegaConf.save(config, local_source_robot)
    shutil.copyfile(Path(pbfm_codec.__file__), destination / "pbfm_codec.py")
    obs = OmegaConf.to_container(core.config.obs, resolve=True)
    codec = {
        "schema_version": "eb/pbfm-rpo-codec/v1",
        "control_joint_names": list(robot.robot.control_joint_names),
        "state_order": [
            "dof_pos-default_pose-default_offset",
            "dof_vel",
            "projected_gravity",
            "base_ang_vel",
        ],
        "obs_scales": obs["obs_scales"],
        "history": obs["obs_auxiliary"]["history_actor"],
        "history_order": "terms alphabetical; newest first; previous frames only; zero on partial reset",
        "default_dof_pos": core.default_dof_pos[0].detach().cpu().tolist(),
        "action_target_scale": core.action_target_scale[0].detach().cpu().tolist(),
        "action_clip": float(core.config.robot.control.action_clip_value),
        "normalize_action_from": float(core.config.robot.control.normalize_action_from),
        "normalize_action_to": float(core.config.robot.control.normalize_action_to),
        "last_action": "clipped normalized command before motor delay",
        "default_offset": "evaluation zero; otherwise explicit native sampled offset",
        "pd_target": "default_pose + default_offset + delayed_clipped_command * action_target_scale",
        "physics_hz": 400,
        "evaluation_mode": plan["evaluation_mode"],
        "policy_hz": 50,
        "depth_input_contract": export["depth_input_contract"],
        "depth_preprocessing_implementation": "source/pbfm/perception/instinct_direct_depth.py",
        "input_names": export["input_names"],
        "input_shapes": export["input_shapes"],
        "normalizer": "normalizers.json; exact frozen deployed normalizers embedded in policy.onnx",
        "velocity_command": export["velocity_command"],
        "heading": "[valid*(1-cos(error)), valid*sin(error)]",
    }
    write_json(destination / "codec.json", codec)
    normalizers = {
        key: {
            name: tensor.detach().cpu().tolist()
            for name, tensor in policy._obs_normalizer._normalizers[key].state_dict().items()
        }
        for key in export["actor_input_keys"]
    }
    write_json(destination / "normalizers.json", {"embedded_in_onnx": True, "per_key_state": normalizers})
    write_json(destination / "camera.json", plan["camera_contract"])
    write_json(destination / "terrain.json", plan["terrain_contract"])
    write_json(
        destination / "task.json",
        {
            "schema_version": "eb/pbfm-inference-task/v1",
            "kind": "tracking",
            "motion_id": plan["motion_id"],
            "prompt": "tracking-prompt.npz",
            "contract": "tracking_heading0_yaw_velocity_v1",
            "model_hash": plan["model_hash"],
            "commands": ["no_command=[0,0,0]", "valid_stop=[0,0,1]", "valid_move=[vx,vy,1]"],
        },
    )
    licenses = destination / "licenses"
    licenses.mkdir()
    for source, name in (
        (private_root / "LICENSE", "P-BFM-LICENSE"),
        (Path("/robot-assets/robolab/LICENSE"), "robolab-LICENSE"),
        (Path("/robot-assets/rpo_hardware/LICENSE"), "rpo-hardware-LICENSE"),
        (Path("/robot-assets/rpo_hardware/LICENSES.md"), "rpo-hardware-LICENSES.md"),
        (Path("/robot-assets/rpo_hardware/THIRD_PARTY_NOTICES.md"), "rpo-hardware-THIRD_PARTY_NOTICES.md"),
    ):
        if source.is_file() and not source.is_symlink():
            shutil.copyfile(source, licenses / name)
    write_json(
        destination / "sources.json",
        {
            "schema_version": "eb/pbfm-policy-sources/v1",
            "source_refs": plan["source_refs"],
            "image_digest": plan["image_digest"],
            "pbfm_patch_sha256": source_receipt["patch_sha256"],
            "model_hash": plan["model_hash"],
            "prompt_source_manifest_sha256": plan["prompt_manifest_sha256"],
            "asset_manifest": read_json(Path("/robot-assets/asset-manifest.json")),
            "license_status": "source license files preserved; "
            "dataset ancestry/licensing remains limited to source declaration",
            "simulation_only": True,
            "hardware_release": False,
        },
    )
    freeze = Path("/opt/pbfm-runtime-freeze.txt")
    if not freeze.is_file():
        raise ValueError("inference package requires the producer image's exact runtime dependency freeze")
    shutil.copyfile(freeze, destination / "requirements.freeze.txt")
    (destination / "README.txt").write_text(
        "Simulation inference package only.\n"
        "policy.onnx contains frozen actor observation normalizers, depth CNN/GRU and estimator.\n"
        "Use input_names/input_shapes from codec.json. pbfm_codec.py implements scalar/history/action "
        "and the once-only ONNX depth conversion.\n"
        "Metric depth uses the included native source/pbfm/perception/instinct_direct_depth.py "
        "and camera.json.\n"
        "Use tracking-prompt.npz with heading0 alignment; robot/robot.yaml and robot/rpo.xml "
        "resolve packaged meshes locally.\n"
        "Install the exact requirements.freeze.txt environment and add source to PYTHONPATH "
        "for native simulation helpers.\n"
        "The package is not a hardware release or formal acceptance certificate.\n"
    )
    return codec


def archive_package(destination: Path, output: Path):
    import zipfile

    archive = output / "policy.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for entry in file_index(destination):
            stream.write(safe_file(destination, entry["path"]), "policy/" + entry["path"])
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None or not stream.namelist():
            raise ValueError("portable PBFM policy archive failed integrity verification")


def compare_paired_traces(reference: dict, exported: dict) -> dict:
    import numpy as np

    for value in (reference, exported):
        steps = len(value["actions"])
        if (
            not 1 <= steps <= 32
            or value["root_xyz"].shape != (steps + 1, 3)
            or value["actions"].shape != (steps, 23)
            or not np.isfinite(value["root_xyz"]).all()
            or not np.isfinite(value["actions"]).all()
            or (steps < 32 and not value["terminated"])
        ):
            raise ValueError("paired closed-loop trace is incomplete or nonfinite")
    count = min(len(reference["actions"]), len(exported["actions"]))
    return {
        "schema_version": "eb/pbfm-paired-closed-loop/v1",
        "complete": True,
        "torch_steps": len(reference["actions"]),
        "onnx_steps": len(exported["actions"]),
        "shared_prefix_steps": count,
        "torch_terminated": bool(reference["terminated"]),
        "onnx_terminated": bool(exported["terminated"]),
        "termination_matches": bool(reference["terminated"]) == bool(exported["terminated"]),
        "max_action_abs_difference": float(
            np.abs(reference["actions"][:count] - exported["actions"][:count]).max()
        ),
        "max_root_abs_difference_m": float(
            np.abs(reference["root_xyz"][: count + 1] - exported["root_xyz"][: count + 1]).max()
        ),
        "quality_passed": None,
        "comparison": "two independent closed-loop trajectories; finite clock checks only",
    }


def paired_closed_loop(plan, policy, config, prompt, session, destination: Path):
    import numpy as np
    import torch
    from pbfm.inference.runtime import _assign_training_tile, _terrain_env_cfg

    from eb.runners.pbfm_eval import advance_with_successor, reset_target

    frozen = plan["paired_rollout"]
    traces = []
    (destination / "paired").mkdir()
    for backend in ("torch", "onnx"):
        torch.manual_seed(frozen["seed"])
        np.random.seed(frozen["seed"])
        wrapped = _terrain_env_cfg(config, frozen["terrain"], seed=frozen["seed"]).build(num_envs=1)[0]
        try:
            core = wrapped._env
            if not np.isclose(core.dt, 0.02, rtol=0, atol=1e-8):
                raise ValueError("paired export rollout changed the native 50Hz clock")
            _assign_training_tile(core, family="flat", difficulty_row=0)
            core._motion_lib.load_all_motions()
            core.is_evaluating = True
            context = {
                key: value.copy() if isinstance(value, np.ndarray) else value for key, value in prompt.items()
            }
            context["heading_world_xy"][:] = frozen["heading_world_xy"]
            context["reference_velocity_world_xy"][:] = frozen["velocity_world_xy"]
            context["heading_valid"][:] = True
            context["velocity_valid"][:] = True
            observation, _ = wrapped.reset(to_numpy=False, target_states=reset_target(context, core))
            roots = [core.robot_root_states[0, :3].detach().cpu().numpy().copy()]
            actions, terminal = [], False
            for step in range(frozen["steps"]):
                observation = commanded_observation(observation, core, context, step)
                z = torch.as_tensor(context["z"][step : step + 1], device=core.device)
                with torch.no_grad():
                    action = (
                        policy.act(observation, z, mean=True).detach().cpu().numpy()
                        if backend == "torch"
                        else onnx_action(session, observation, z)
                    )
                result, successor = advance_with_successor(
                    wrapped, torch.as_tensor(action, device=core.device)
                )
                observation, _, terminated, truncated, _ = result
                actions.append(action[0])
                roots.append(successor["root"][:3])
                terminal = bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any())
                if terminal:
                    break
            trace = {"actions": np.asarray(actions), "root_xyz": np.asarray(roots), "terminated": terminal}
            np.savez_compressed(
                destination / f"paired/{backend}.npz",
                **trace,
                physics_time_s=np.arange(len(roots)) * 0.02,
                command_velocity_world_xy=frozen["velocity_world_xy"],
                seed=frozen["seed"],
                model_hash=plan["model_hash"],
            )
            traces.append(trace)
        finally:
            wrapped.close()
    summary = {
        **compare_paired_traces(*traces),
        "frozen_plan": frozen,
        "observation_normalization": "same frozen model normalizers, no updates",
        "model_hash": plan["model_hash"],
    }
    write_json(destination / "paired/summary.json", summary)
    return summary


def tracking_prompt(policy, env, motion_id: int, use_root_height_obs: bool) -> tuple[dict, dict]:
    import numpy as np
    import torch
    from pbfm.agents.behavior_context import root_heading_xy
    from pbfm.agents.inference_context import build_tracking_commands, tracking_context_arrays
    from pbfm.inference.runtime import _target_states_from_obs, _tracking_z
    from pbfm.utils.helpers import get_backward_observation
    from torch.utils._pytree import tree_map

    env._motion_lib.load_all_motions()
    if motion_id >= env._motion_lib.num_motions():
        raise ValueError("prompt motion ID is outside the frozen validation split")
    backward, reference = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
    with torch.no_grad():
        z = _tracking_z(policy, tree_map(lambda x: x[1:].to("cuda:0") if hasattr(x, "to") else x, backward))
    if z.ndim != 2 or z.shape[1] != 256 or not torch.isfinite(z).all():
        raise ValueError("tracking backward encoder did not yield finite 256D prompts")
    frames = len(z)
    repeats = int(np.ceil(1000 / frames))
    initial_heading = root_heading_xy(reference["ref_body_rots"][:1, 0].to("cuda:0").float())
    commands = build_tracking_commands(
        root_heading_xy(reference["ref_body_rots"][:, 0].to("cuda:0").float()),
        reference["ref_body_vels"][:, 0, :2].to("cuda:0").float(),
        initial_heading,
        motion_name=str(env._motion_lib._motion_data_keys[motion_id]),
        frames=frames,
        repeats=repeats,
        mode="auto",
        policy=getattr(policy.cfg, "reference_velocity_policy", "walk_run"),
        temporal_mode=getattr(policy.cfg, "reference_velocity_temporal_mode", "instantaneous"),
        temporal_horizon=int(getattr(policy.cfg, "reference_velocity_temporal_horizon", 1)),
    )
    arrays = tracking_context_arrays(z.repeat(repeats, 1), commands, initial_heading, env.dt)
    if not bool(getattr(policy.cfg, "heading_context_enabled", False)):
        arrays["heading_valid"] = np.zeros_like(arrays["heading_valid"])
    for key, value in list(arrays.items()):
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == frames * repeats:
            arrays[key] = value[:1000]
    target = _target_states_from_obs(reference, device="cuda:0", num_dof=23)
    arrays["reset_root_states_xyzw"] = target["root_states"].detach().cpu().numpy()
    arrays["reset_dof_states"] = target["dof_states"].detach().cpu().numpy()
    for source_key, target_key in (("dof_pos", "reference_dof_pos"), ("ref_body_pos", "reference_body_pos")):
        values = reference[source_key].detach().cpu().numpy()
        arrays[target_key] = np.concatenate([values[:frames]] * (repeats + 1), axis=0)[:1001]
    body = arrays["reference_body_pos"] - arrays["reference_body_pos"][:, :1]
    c, s = arrays["initial_heading_world_xy"][0]
    body[..., :2] = body[..., :2] @ np.asarray([[c, -s], [s, c]])
    arrays["reference_body_pos_heading0"] = body
    if len(arrays["z"]) != 1000:
        raise ValueError("tracking prompt does not cover all 20 seconds")
    return arrays, target


def commanded_observation(observation, env, prompt: dict, step: int):
    import torch
    from pbfm.agents.behavior_context import heading_observation, root_heading_xy
    from pbfm.agents.velocity_context import velocity_observation

    device = env.device
    heading = root_heading_xy(env.base_quat.float())
    if "heading" in observation:
        observation["heading"] = heading_observation(
            heading,
            torch.as_tensor(prompt["heading_world_xy"][step : step + 1], device=device),
            torch.as_tensor(prompt["heading_valid"][step : step + 1], device=device).bool(),
        )
    if "velocity_command" in observation:
        observation["velocity_command"] = velocity_observation(
            torch.as_tensor(prompt["reference_velocity_world_xy"][step : step + 1], device=device),
            heading,
            torch.as_tensor(prompt["velocity_valid"][step : step + 1], device=device).bool(),
        )
    return observation


def parity_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def onnx_action(session, observation: dict, z):
    import numpy as np
    from pbfm.inference.export.depth_policy import depth_to_onnx_float

    feed = {}
    for spec in session.get_inputs():
        tensor = z if spec.name == "z" else observation[spec.name]
        value = tensor.detach().cpu().numpy()
        feed[spec.name] = depth_to_onnx_float(value) if len(value.shape) == 4 else value.astype(np.float32)
    result = session.run(None, feed)[0]
    if result.shape != (1, 23) or not np.isfinite(result).all():
        raise ValueError("exported policy returned invalid RPO action")
    return result


def run(plan: dict, output: Path = Path("/outputs")):
    import numpy as np
    import torch

    validate_action_plan(plan)
    policy, robot, config, root_height, source_receipt = prepare_runtime(plan, output)
    from pbfm.inference.export.depth_policy import export_direct_depth_policy

    from eb.runners.pbfm_codec import PolicyCodec, depth_input
    from eb.runners.pbfm_eval import advance_with_successor

    wrapped = config.build(num_envs=1)[0]
    try:
        env = wrapped._env
        arrays, target = tracking_prompt(policy, env, plan["motion_id"], root_height)
        destination = output / "policy"
        destination.mkdir()
        export = export_direct_depth_policy(policy, destination / "policy.onnx", verify=True)
        if (
            export.get("format") != "direct_depth_full_v1"
            or export.get("depth_shape") != [8, 18, 32]
            or export.get("z_dim") != 256
            or export.get("output_action_dim") != 23
        ):
            raise ValueError("ONNX is not the complete native RPO direct-depth policy")
        scalar_dim = sum(
            shape[0] for key, shape in export["input_shapes"].items() if key != "z" and len(shape) == 1
        )
        if scalar_dim != 380:
            raise ValueError("exported RPO scalar observation semantics differ")
        codec_config = write_inference_package(destination, plan, robot, env, policy, export, source_receipt)
        codec = PolicyCodec(codec_config)
        session = parity_session(destination / "policy.onnx")
        obs, _ = wrapped.reset(to_numpy=False, target_states=target)
        errors = []
        trace_checks = []
        for step in range(32):
            obs = commanded_observation(obs, env, arrays, step)
            if step < 3:
                obs["velocity_command"] = torch.tensor(
                    [[0.0, 0.0, 0.0] if step == 0 else [0.0, 0.0, 1.0] if step == 1 else [0.8, -0.2, 1.0]],
                    device=env.device,
                )
            scalar = codec.observations(
                env.dof_pos.detach().cpu().numpy(),
                env.dof_vel.detach().cpu().numpy(),
                env.projected_gravity.detach().cpu().numpy(),
                env.base_ang_vel.detach().cpu().numpy(),
                env.actions.detach().cpu().numpy(),
                default_offset=env.default_dof_pos_offset.detach().cpu().numpy(),
            )
            for key, value in scalar.items():
                np.testing.assert_allclose(value, obs[key].detach().cpu().numpy(), atol=2e-5, rtol=2e-5)
            z = torch.as_tensor(arrays["z"][step : step + 1], device=env.device)
            with torch.no_grad():
                expected = policy.act(obs, z, mean=True)
            actual = onnx_action(session, obs, z)
            float_obs = dict(obs)
            float_obs["depth_image"] = torch.as_tensor(
                depth_input(obs["depth_image"].detach().cpu().numpy()), device=env.device
            )
            float_action = onnx_action(session, float_obs, z)
            np.testing.assert_allclose(actual, float_action, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(actual, expected.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
            errors.append(float(np.max(np.abs(actual - expected.detach().cpu().numpy()))))
            clipped, pd_target = codec.action_target(
                actual, default_offset=env.default_dof_pos_offset.detach().cpu().numpy()
            )
            np.testing.assert_allclose(
                clipped,
                env._normalized_action(torch.as_tensor(actual, device=env.device)).detach().cpu().numpy(),
                atol=2e-5,
                rtol=2e-5,
            )
            step_result, successor = advance_with_successor(
                wrapped, torch.as_tensor(actual, device=env.device)
            )
            obs, _, terminated, truncated, _ = step_result
            order = env._action_term_dof_indices.detach().cpu().numpy()
            if successor["pd_target_names"] != [env.dof_names[index] for index in order]:
                raise ValueError("MJLab actuator target names differ from the native control joint order")
            _, pd_target = codec.action_target(
                actual,
                default_offset=successor["default_offset"][None, :],
                delayed_command=successor["applied_actions"][None, :],
            )
            np.testing.assert_allclose(
                pd_target[:, env._action_term_dof_indices.detach().cpu().numpy()]
                - successor["encoder_bias_target"],
                successor["pd_target"],
                atol=2e-5,
                rtol=2e-5,
            )
            trace_checks.append(
                {
                    "step": step,
                    "uint8_float_single_normalization_verified": True,
                    "native_scalar_history_verified": True,
                    "native_pd_target_verified": True,
                }
            )
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                break
        np.savez_compressed(destination / "tracking-prompt.npz", **arrays)
        write_json(
            destination / "prompt.json",
            {
                "schema_version": "eb/pbfm-tracking-prompt/v1",
                "model_version_id": plan["model_version_id"],
                "model_hash": plan["model_hash"],
                "motion_id": plan["motion_id"],
                "source_manifest_sha256": plan["prompt_manifest_sha256"],
                "source_file_sha256": plan["prompt_file_sha256"],
                "frames": 1000,
                "dt": 0.02,
                "z_dim": 256,
                "method": "native _tracking_z backward_map/seq average/project_z",
            },
        )
        wrapped.close()
        wrapped = None
        paired = paired_closed_loop(plan, policy, config, arrays, session, destination)
        write_json(
            destination / "manifest.json",
            {
                "schema_version": "eb/rpo-pbfm-policy/v1",
                "model_version_id": plan["model_version_id"],
                "model_hash": plan["model_hash"],
                "onnx_sha256": sha256(destination / "policy.onnx"),
                "prompt_sha256": sha256(destination / "tracking-prompt.npz"),
                "camera_contract": plan["camera_contract"],
                "joint_order": list(robot.robot.control_joint_names),
                "trace_verified": True,
                "trace_steps": len(errors),
                "trace_max_abs_error": max(errors),
                "numeric_verification": export["verification"],
                "real_trace_checks": trace_checks,
                "paired_closed_loop": paired,
                "input_names": export["input_names"],
                "input_shapes": export["input_shapes"],
                "normalizer_file": "normalizers.json",
                "codec_file": "codec.json",
                "robot_config": "robot/robot.yaml",
                "robot_xml": "robot/rpo.xml",
                "camera_profile": "camera.json",
                "terrain_profile": "terrain.json",
                "task_context": "task.json",
                "sources_and_licenses": "sources.json",
                "files": file_index(destination),
                "capability": "simulation_diagnostic",
                "formal_acceptance": False,
            },
        )
        archive_package(destination, output)
        write_json(
            output / "export-receipt.json",
            {
                "schema_version": "eb/pbfm-export-receipt/v1",
                "run_id": os.environ["EB_RUN_ID"],
                "operation": plan["operation"],
                "plan_hash": plan["hash"],
                "source_receipt": source_receipt,
                "files": file_index(output),
            },
        )
    finally:
        if wrapped is not None:
            wrapped.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    run(json.loads(parser.parse_args().plan_json))


if __name__ == "__main__":
    main()
