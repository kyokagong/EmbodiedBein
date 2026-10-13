import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eb.runners.bfm_ufo_agent import build_agent  # noqa: E402
from eb.runners.bfm_ufo_artifact import export_ufo_policy  # noqa: E402
from eb.runners.bfm_ufo_checkpoint import save_ufo_checkpoint  # noqa: E402
from eb.runners.bfm_ufo_policy import UFOBFMPolicy, observation_space  # noqa: E402
from eb.runners.bfm_ufo_replay import UFOReplayBridge  # noqa: E402


@pytest.mark.parametrize("tracking,velocity", [(True, False), (False, False), (False, True)])
def test_native_actor_and_prompts_execute_in_onnx(tmp_path, monkeypatch, tracking, velocity):
    import hashlib

    import yaml

    from eb.domain.training_recipe import RECIPES, UFO_ISAAC_RECIPE, UFO_VELOCITY_RECIPE
    from eb.rpo_bfm import BODY_ORDER, CODEC_HASH, RPO_LAB_JOINT_ORDER

    recipe_id = UFO_VELOCITY_RECIPE if velocity else UFO_ISAAC_RECIPE
    model_settings = {
        "hidden_dim": 64,
        "hidden_layers": 2,
        "batch_size": 16,
        "rollout_expert_trajectories": tracking,
    }
    if velocity:
        from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig

        from eb.runners import bfm_ufo_agent, bfm_ufo_artifact, bfm_ufo_policy

        original_config = bfm_ufo_agent.agent_config

        # CPU fixture reduces allocation only; native nine-term update/export methods stay unchanged.
        def small_config(cfg, *, device):
            values = original_config(cfg, device=device).model_dump()
            for architecture in values["model"]["archi"].values():
                if isinstance(architecture, dict) and "hidden_dim" in architecture:
                    architecture.update(hidden_dim=64, hidden_layers=2)
            values["train"]["batch_size"] = 16
            return FBcprAuxAgentConfig.model_validate(values)

        for module in (bfm_ufo_agent, bfm_ufo_artifact, bfm_ufo_policy):
            monkeypatch.setattr(module, "agent_config", small_config)
        import importlib

        for name in ("bfm_ufo_agent", "bfm_ufo_artifact", "bfm_ufo_policy"):
            monkeypatch.setattr(importlib.import_module(name), "agent_config", small_config)
        model_settings.update(
            hidden_dim=2048,
            hidden_layers=6,
            batch_size=1024,
            architecture_profile="native_fb_v1",
            auxiliary_profile="soft_velocity_v5",
        )
    torch.set_num_threads(1)
    motion = tmp_path / "motion"
    motion.mkdir()
    arrays = {
        "root_pos": np.ones((270, 3), np.float32),
        "root_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (270, 1)),
        "q": np.zeros((270, 23), np.float32),
        "dq": np.zeros((270, 23), np.float32),
        "root_lin_vel_w": np.zeros((270, 3), np.float32),
        "root_ang_vel_w": np.zeros((270, 3), np.float32),
        "states": np.zeros((270, 52), np.float32),
        "privileged_state": np.ones((270, 358), np.float32),
        "reward_features": np.tile(np.array([0, 0, 0, 1, 0.7, 0], np.float32), (270, 1)),
        "valid": np.ones(270, bool),
    }
    clips = []
    for source in ("official_rpo", "g1"):
        path = motion / (source + ".npz")
        np.savez(path, **arrays)
        clips.append(
            {
                "id": source,
                "source": source,
                "source_group": source,
                "split": "train",
                "fps": 50,
                "frames": 270,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "eb/rpo-robot-state/v1",
        "asset_sha256": "a" * 64,
        "codec_sha256": CODEC_HASH,
        "quaternion": "wxyz",
        "velocity_frame": "world",
        "velocity_point": "root_link_origin",
        "joint_order": list(RPO_LAB_JOINT_ORDER),
        "body_order": list(BODY_ORDER),
        "clips": clips,
        "parents": [{"manifest_sha256": "b" * 64}],
        "source_refs": [{"sha256": "c" * 64, "license": "fixture only"}],
    }
    (motion / "manifest.json").write_text(json.dumps(manifest))
    from eb.runners.rpo_ufo_motion import load_robot_state

    _, expert = load_robot_state(motion, asset_sha256="a" * 64)
    agent = build_agent(observation_space(), model_settings, device="cpu")
    replay = {
        "train": UFOReplayBridge(capacity=32, aux_names=tuple(agent.cfg.aux_rewards)),
        "expert_slicer": expert,
    }
    policy = UFOBFMPolicy(model_settings)
    rollout = (
        {
            f"rank_{rank}": UFOBFMPolicy({**model_settings, "seed": 42 + rank}).rollout_state_dict()
            for rank in range(3)
        }
        if velocity
        else {"rank_0": policy.rollout_state_dict()}
    )
    checkpoint = tmp_path / "global_step_0"
    save_ufo_checkpoint(
        agent,
        replay,
        rollout,
        checkpoint,
        receipt={
            "run_id": "run_abcdef",
            "learner_joint_updates": 0,
            "sampling_rounds": 0,
            "online_transitions": 0,
            "next_trigger": 11264,
            "compatibility_fingerprint": "sha256:" + "b" * 64,
            "execution": RECIPES[recipe_id].binding.model_dump(),
        },
    )
    config = tmp_path / "resolved-config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "actor": {
                    "model": {
                        "policy_setup": "rpo-v2-bfm-ufo",
                        "obs_dim": 375,
                        "critic_obs_dim": 733,
                        "action_dim": 23,
                        "num_action_chunks": 1,
                        "asset_sha256": "a" * 64,
                        **model_settings,
                    }
                }
            }
        )
    )
    result = export_ufo_policy(checkpoint, config, motion, "a" * 64, tmp_path / "export")
    assert result["status"] == "verified"
    assert result["execution"]["provider_id"] == "ufo"
    assert result["onnx"]["max_abs_error"] <= 1e-4
    assert result["command_latent"]["reference_split"] == "train"
    if velocity:
        assert result["auxiliary_profile"] == "soft_velocity_v5"
        from eb.runners.bfm_ufo_checkpoint import load_ufo_checkpoint

        restored = build_agent(observation_space(), model_settings, device="cpu")
        buffers = {
            "train": UFOReplayBridge(capacity=32, aux_names=tuple(restored.cfg.aux_rewards)),
            "expert_slicer": expert,
        }
        load_ufo_checkpoint(
            checkpoint,
            agent=restored,
            replay=buffers,
            expected={"compatibility_fingerprint": "sha256:" + "b" * 64},
            device="cpu",
        )
        legacy = build_agent(
            observation_space(),
            {key: value for key, value in model_settings.items() if key != "auxiliary_profile"},
            device="cpu",
        )
        before = {key: value.clone() for key, value in legacy._model.state_dict().items()}
        with pytest.raises(ValueError, match="config differs"):
            load_ufo_checkpoint(checkpoint, agent=legacy, replay=buffers, expected={}, device="cpu")
        for key, value in before.items():
            torch.testing.assert_close(legacy._model.state_dict()[key], value, atol=0, rtol=0)
    from eb.runners.bfm_ufo_artifact import load_ufo_export_policy

    tampered = yaml.safe_load(config.read_text())
    tampered["actor"]["model"]["rollout_expert_trajectories"] = not tracking
    mismatched = tmp_path / "mismatched-config.yaml"
    mismatched.write_text(yaml.safe_dump(tampered))
    with pytest.raises(ValueError, match="config differs from checkpoint algorithm"):
        load_ufo_export_policy(checkpoint, mismatched, motion, "a" * 64)
    from eb.runners.rpo_bfm_artifact import main

    cli_output = tmp_path / "cli-export"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rpo_bfm_artifact.py",
            "export",
            "--checkpoint",
            str(checkpoint),
            "--resolved-config",
            str(config),
            "--motion-root",
            str(motion),
            "--asset-sha256",
            "a" * 64,
            "--output",
            str(cli_output),
        ],
    )
    main()
    cli_manifest = json.loads((cli_output / "export-manifest.json").read_text())
    assert cli_manifest["execution"] == result["execution"]
    assert cli_manifest["checkpoint_format"] == "ufo_agent_state_v1"
    import shutil
    import sys
    from types import ModuleType, SimpleNamespace

    from omegaconf import OmegaConf

    from eb.runners.bfm_ufo_artifact import publish_ufo_checkpoint
    from eb.runners.bfm_ufo_checkpoint import ufo_checkpoint_root

    # The task image intentionally excludes API settings dependencies.
    settings = ModuleType("eb.settings")
    settings.get_settings = lambda: SimpleNamespace(models_root=tmp_path / "models")
    monkeypatch.setitem(sys.modules, "eb.settings", settings)
    import eb.checkpoints

    monkeypatch.setattr(eb.checkpoints, "get_settings", settings.get_settings)
    from eb.checkpoints import checkpoint_model

    output = tmp_path / "output"
    publish_cfg = OmegaConf.create(
        {
            "runner": {"logger": {"experiment_name": "fixture"}},
            "actor": {
                "model": {"learning_profile": None if tracking else "physical_limits_stable_latent_v3"}
            },
        }
    )
    native_output = ufo_checkpoint_root(output, publish_cfg) / "global_step_0"
    shutil.copytree(checkpoint, native_output)
    shutil.copyfile(config, output / "resolved-config.yaml")
    monkeypatch.setenv("EB_RUN_ID", "run_abcdef")
    publish_ufo_checkpoint(
        output,
        publish_cfg,
        {
            "template_id": recipe_id,
            "execution": RECIPES[recipe_id].binding.model_dump(),
            "target_learner_updates": 0,
        },
    )
    published = checkpoint_model(tmp_path, run_id="run_abcdef", verify_hashes=True)
    assert published["step"] == 0
    assert published["manifest"]["execution"] == cli_manifest["execution"]
    assert not (output / "model/model_state_dict/full_weights.pt").exists()
    source_weights = native_output / "model/model.safetensors"
    published_weights = output / "model/native/global_step_0/model/model.safetensors"
    if not tracking:
        assert source_weights.stat().st_ino == published_weights.stat().st_ino
    source_weights.unlink()
    assert checkpoint_model(tmp_path, run_id="run_abcdef", verify_hashes=True)["step"] == 0
    model_root = tmp_path / "models/runs/run_abcdef"
    model_root.parent.mkdir(parents=True)
    shutil.move(output / "model", model_root)
    from eb.checkpoints import resume_checkpoint

    restored = resume_checkpoint(tmp_path, "run_abcdef", step=0)
    assert restored["format"] == "ufo_agent_state_v1"
    assert Path(restored["path"]).is_relative_to(model_root)
    if not tracking:
        raw_root = model_root / ".checkpoint-source/global_step_0"
        shutil.rmtree(raw_root)
        shutil.copytree(model_root / "native/global_step_0", raw_root)
        manifest_path = output / "checkpoint-manifest.json"
        published_receipt = manifest_path.read_text()
        manifest_path.unlink()
        failed_progress = resume_checkpoint(tmp_path, "run_abcdef", step=0)
        assert Path(failed_progress["path"]) == raw_root
        assert failed_progress["format"] == "ufo_agent_state_v1"
        manifest_path.write_text(published_receipt)
