"""Import/export/bundle entrypoints, executed only by ebd task containers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from eb.ufo_native import (
    TASKS,
    finite_tree,
    inference_overrides,
    plan_hash,
    read_json,
    safe_file,
    sha256_file,
    validate_bundle_source,
    validate_config,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_index(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in output package")
        if path.is_file() and path.name != "manifest.json":
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return result


def verify_files(root, files):
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("Empty or duplicate file manifest")
    for item in files:
        path = safe_file(root, item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {item['path']}")


def verified_manifest(root, expected_sha256):
    manifest = read_json(safe_file(root, "manifest.json"))
    if plan_hash(manifest) != expected_sha256:
        raise ValueError("Input manifest hash mismatch")
    verify_files(root, manifest["files"])
    return manifest


def receipt(output, plan, **fields):
    value = {
        "schema_version": "eb/ufo-native-receipt/v1",
        "operation": plan["operation"],
        "run_id": os.environ["EB_RUN_ID"],
        "plan_hash": plan["hash"],
        "image_digest": plan["image_digest"],
        **fields,
    }
    write_json(output / "receipt.json", value)
    print(
        json.dumps({"event": "receipt_complete", "operation": plan["operation"], "run_id": value["run_id"]}),
        flush=True,
    )
    return value


def import_model(output, plan):
    import joblib
    import numpy as np
    from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir

    source, root = plan["source"], output / "model"
    root.mkdir()
    for item in source["files"]:
        path = safe_file(Path("/source"), item["path"])
        before = path.stat()
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if (
            sha256_file(destination) != item["sha256"]
            or path.stat().st_mtime_ns != before.st_mtime_ns
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError("Import source changed after preview")
    for i, item in enumerate(source["extras"]):
        path = Path(f"/source-extra/{i}")
        if path.is_symlink() or sha256_file(path) != item["sha256"]:
            raise ValueError("Import auxiliary asset changed after preview")
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha256_file(destination) != item["sha256"]:
            raise ValueError("Auxiliary asset copy is corrupt")
    codec = validate_config(
        read_json(root / "config.json"),
        read_json(root / "effective_config.json"),
        read_json(root / "checkpoint/model/config.json"),
        read_json(root / "checkpoint/model/init_kwargs.json"),
    )
    if codec != source["codec"]:
        raise ValueError("Imported model codec changed")
    tree = ET.parse(root / "assets/rpo.xml")
    tree.getroot().find("compiler").set("meshdir", "meshes")
    tree.write(root / "assets/rpo.xml", encoding="unicode")
    import yaml

    robot = yaml.safe_load((root / "assets/robot.yaml").read_text())
    robot["xml_path"] = "rpo.xml"
    (root / "assets/robot.yaml").write_text(yaml.safe_dump(robot, sort_keys=False))
    embedded = joblib.load(root / "reward_inference/reward_locomotion.pkl")
    tasks = {}
    for task in TASKS:
        z = np.asarray(embedded[task["id"]][0], dtype=np.float32).reshape(-1)
        if z.shape != (256,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-6:
            raise ValueError("Invalid frozen task embedding")
        tasks[task["id"]] = z.tolist()
    write_json(
        root / "prompts.json",
        {
            "schema_version": "eb/ufo-native-prompts/v1",
            "derivation": "external_unverified",
            "tasks": tasks,
            "source_sha256": next(
                f["sha256"] for f in source["files"] if f["path"].endswith("reward_locomotion.pkl")
            ),
        },
    )
    write_json(root / "codec.json", codec)
    model = load_model_from_checkpoint_dir(root / "checkpoint", device="cpu")
    model.eval()
    import torch

    torch.set_num_threads(4)
    obs = {
        k: torch.zeros((1, size)) for k, size in (("state", 52), ("last_action", 23), ("history_actor", 300))
    }
    with torch.inference_mode():
        action = model.act(obs, torch.tensor([tasks[TASKS[0]["id"]]]), mean=True)
    if tuple(action.shape) != (1, 23) or not torch.isfinite(action).all():
        raise ValueError("Native actor load validation failed")
    # The pickle is a source receipt, not part of subsequent runtime inputs.
    (root / "reward_inference/reward_locomotion.pkl").unlink()
    manifest = {
        "schema_version": "eb/ufo-native-import/v1",
        "codec": codec,
        "source": source,
        "loader_verified": True,
        "files": file_index(root),
        "image_digest": plan["image_digest"],
        "training_provenance": "external_declared",
    }
    write_json(root / "manifest.json", manifest)
    return receipt(output, plan, manifest=manifest, package="model")


def native_env_config(root, *, device="cuda:0", seconds=21):
    from ufo.envs.config import MJLabEnvConfig

    saved = read_json(root / "config.json")
    env = saved["env"]
    env["task_overrides"] = inference_overrides(read_json(root / "effective_config.json"))
    env.update(
        device=device,
        mjcf_path=str(root / "assets/rpo.xml"),
        robot_config_path=str(root / "assets/robot.yaml"),
        motion_data_path=str(root / "sim-data/motion.pkl"),
        disable_domain_randomization=True,
        disable_obs_noise=True,
        max_episode_length_s=seconds,
        auto_reset=False,
    )
    env["robot_training"]["robot"]["xml_path"] = str(root / "assets/rpo.xml")
    return MJLabEnvConfig.model_validate(env)


def verify_export(model, onnx_path, codec, prompts):
    if not onnx_path.is_file():
        raise FileNotFoundError("Exported ONNX is missing")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(71000)
    max_action, max_target = 0.0, 0.0
    samples = []
    for task, context in prompts.items():
        for scale in (0.0, 0.1, 1.0, 3.0):
            obs = {
                k: torch.tensor(rng.normal(size=(8, size)).astype("float32") * scale)
                for k, size in (("state", 52), ("last_action", 23), ("history_actor", 300))
            }
            z = torch.tensor([context], dtype=torch.float32).expand(8, -1)
            inputs = np.concatenate(
                [obs[k].numpy() for k in ("state", "last_action", "history_actor")] + [z.numpy()], axis=1
            )
            with torch.inference_mode():
                expected = model.act(obs, z, mean=True).numpy()
            actual = session.run(None, {"actor_obs": inputs})[0]
            error = np.abs(expected - actual)
            target_error = np.abs(np.clip(expected, -1, 1) - np.clip(actual, -1, 1)) * np.asarray(
                codec["action_gain"]
            )
            if (
                not np.isfinite(actual).all()
                or not np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                or target_error.max() > 1e-4
            ):
                raise ValueError("Native actor and ONNX export are not equivalent")
            max_action = max(max_action, float(error.max()))
            max_target = max(max_target, float(target_error.max()))
            samples.append(inputs)
    np.savez_compressed(onnx_path.parent / "golden-inputs.npz", actor_obs=np.concatenate(samples))
    return {
        "verified": True,
        "samples": sum(len(s) for s in samples),
        "atol": 1e-5,
        "rtol": 1e-4,
        "max_action_error": max_action,
        "max_target_error_rad": max_target,
    }


def export_model(output, plan):
    import torch
    import yaml
    from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir
    from ufo.tools.export_onnx import export_policy_model
    from ufo.tools.robot_config import load_robot_training_spec

    from eb.runners import ufo_native_codec

    torch.set_num_threads(4)
    root, payload = Path("/input"), output / "payload"
    verified_manifest(root, plan["source_artifact_sha256"])
    payload.mkdir()
    shutil.copytree(root / "assets", payload / "assets")
    for name in ("codec.json", "prompts.json"):
        shutil.copyfile(root / name, payload / name)
    codec = read_json(root / "codec.json")
    prompts = read_json(root / "prompts.json")["tasks"]
    model = load_model_from_checkpoint_dir(root / "checkpoint", device="cpu")
    model.eval()
    cfg = native_env_config(root, device="cpu")
    from dataclasses import asdict

    from ufo.envs.config import make_mjlab_ufo_env_cfg, resolve_task_config

    native_task = resolve_task_config(cfg, 1)
    physics_cfg = make_mjlab_ufo_env_cfg(
        native_task,
        num_envs=1,
        seed=cfg.seed,
        mjcf_path=cfg.mjcf_path,
        auto_reset=False,
        robot_training=cfg.robot_training,
    )
    write_json(payload / "physics.json", asdict(physics_cfg.sim.mujoco))
    robot_yaml = yaml.safe_load((root / "assets/robot.yaml").read_text())
    robot_yaml["xml_path"] = str(root / "assets/rpo.xml")
    with tempfile.TemporaryDirectory() as temp:
        robot_path = Path(temp) / "robot.yaml"
        robot_path.write_text(yaml.safe_dump(robot_yaml))
        robot = load_robot_training_spec(robot_path)
        metadata = export_policy_model(model, payload, robot, env_config=cfg)
    (payload / "FBcprAuxModel.onnx").rename(payload / "policy.onnx")
    (payload / "FBcprAuxModel.deploy.yaml").rename(payload / "deploy.yaml")
    (payload / "FBcprAuxModel.meta.json").unlink()
    verification = verify_export(model, payload / "policy.onnx", codec, prompts)
    shutil.copyfile(ufo_native_codec.__file__, payload / "runtime.py")
    write_json(
        payload / "example-observation.json",
        {"q": codec["default_pos"], "dq": [0.0] * 23, "gravity": [0.0, 0.0, -1.0], "omega": [0.0] * 3},
    )
    shutil.copyfile(root / "licenses/UFO.txt", payload / "UFO-LICENSE.txt")
    write_json(payload / "runtime-receipt.json", read_json(Path("/opt/eb-native/runtime-receipt.json")))
    (payload / "README.txt").write_text(
        "RPO UFO native simulation-only inference package.\n"
        "No robot/motor connection. No hardware qualification or signature.\n"
        "Run in the pinned native task image with /bundle mounted read-only:\n"
        "python /bundle/runtime.py --package /bundle --input /bundle/example-observation.json "
        "--task move-ego-0-0\n"
        "Call Policy.codec.reset() before a new episode. Task embeddings are fixed; no interpolation.\n"
    )
    export = {
        "schema_version": "eb/ufo-native-export/v1",
        "files": file_index(payload),
        "source_artifact_sha256": plan["source_artifact_sha256"],
        "source_model_id": plan["model_id"],
        "codec": codec,
        "verification": verification,
        "image_digest": plan["image_digest"],
        "input_name": "actor_obs",
        "input_dim": metadata["actor_obs_dim"],
        "output_dim": 23,
        "simulation_only": True,
        "training_provenance": "external_declared",
        "prompt_derivation": "external_unverified",
    }
    write_json(payload / "manifest.json", export)
    return receipt(output, plan, manifest=export, package="payload")


def bundle_model(output, plan):
    root = Path("/input")
    payload = verified_manifest(root, plan["source_artifact_sha256"])
    source = validate_bundle_source(payload, plan)
    evidence = []
    for i, item in enumerate(plan["reports"]):
        report = Path(f"/reports/{i}/receipt.json")
        value = read_json(report)
        if (
            sha256_file(report) != item["receipt_sha256"]
            or value["plan_hash"] != item["plan_hash"]
            or value["payload_sha256"] != plan["source_artifact_sha256"]
        ):
            raise ValueError("Bundle report is not bound to this exact payload")
        evidence.append(
            {
                "report_id": item["id"],
                "backend": item["backend"],
                "qualification": item["qualification"],
                "receipt_sha256": item["receipt_sha256"],
                "receipt": value,
            }
        )
    manifest = {
        "schema_version": "eb/rpo-ufo-native-bundle/v1",
        "simulation_only": True,
        "signed": False,
        "hardware_readiness": "unverified",
        "model_id": plan["model_id"],
        "payload_sha256": plan["source_artifact_sha256"],
        "payload_manifest": payload,
        "source_import_manifest": source,
        "source_model_id": plan["source_model_id"],
        "coverage": {
            r["backend"]: {"report_id": r["report_id"], "qualification": r["qualification"]} for r in evidence
        },
        "evidence": evidence,
        "image_digest": plan["image_digest"],
    }
    write_json(output / "bundle-manifest.json", manifest)
    destination = output / "policy.ebp"
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(output / "bundle-manifest.json", arcname="manifest.json")
        archive.add(root / "manifest.json", arcname="payload/manifest.json")
        for item in payload["files"]:
            archive.add(safe_file(root, item["path"]), arcname="payload/" + item["path"])
    with tarfile.open(destination, "r:gz") as archive:
        expected = {
            "manifest.json",
            "payload/manifest.json",
            *("payload/" + f["path"] for f in payload["files"]),
        }
        if set(archive.getnames()) != expected:
            raise ValueError("Bundle archive file list differs")
    return receipt(
        output,
        plan,
        manifest=manifest,
        package="policy.ebp",
        files=[
            {"path": "policy.ebp", "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/outputs"))
    args = parser.parse_args()
    plan = read_json(args.plan)
    if plan_hash(plan) != plan["hash"] or not finite_tree(plan):
        raise ValueError("Native task plan changed")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "receipt.json").exists():
        raise ValueError("Output receipt already exists")
    if plan["operation"] == "ufo_native_live":
        from eb.runners.ufo_native_live import run_live

        run_live(args.output, plan)
    elif plan["operation"] == "ufo_native_eval":
        from eb.runners.ufo_native_sim import evaluate

        evaluate(args.output, plan)
    else:
        {
            "ufo_native_import": import_model,
            "ufo_native_export": export_model,
            "ufo_native_bundle": bundle_model,
        }[plan["operation"]](args.output, plan)


if __name__ == "__main__":
    main()
