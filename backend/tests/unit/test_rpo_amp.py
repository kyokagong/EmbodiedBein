from pathlib import Path
import hashlib
import json
import sys
import types

import numpy as np
import pytest
import yaml

from eb.domain.contract import load_env
from eb.rpo_wbc import AMP_MOTION_CONVERTER_REVISION, JOINT_ORDER, RPO_LAB_JOINT_ORDER


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "contracts/ee_rpo_v2_amp_locomotion_23dof_v1.yaml"
CONFIG = ROOT / "backend/eb/runners/config/rpo_amp_stand_walk_ppo.yaml"


def test_amp_locomotion_contract_has_its_own_history_and_joint_codec():
    env = load_env(CONTRACT)
    assert env.validate_self().passed
    assert env.status == "draft"
    assert env.joint_order == list(JOINT_ORDER)
    assert len(set(env.joint_order)) == 23
    assert env.observation.history_length == 10
    assert (env.observation.dim, env.observation.critic_dim, env.action.dim) == (780, 1390, 23)
    assert (env.control.policy_hz, env.control.control_hz, env.control.decimation) == (50, 200, 4)
    assert env.control.command_dim == 3
    assert env.sim_bindings.isaac_task == "RPO-AMP-StandWalk-v1"


def test_amp_recipe_resolves_to_one_rollout_and_divisible_batches():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["runner"]["max_epochs"] == 1000
    assert config["runner"]["use_training_pipeline"] is False
    assert config["runner"]["overlap_env_bootstrap"] is False
    train = config["env"]["train"]
    assert (train["total_num_envs"], train["max_steps_per_rollout_epoch"],
            train["rollout_epoch"], train["max_episode_steps"]) == (512, 24, 1, 1000)
    assert train["auto_reset"] and not train["ignore_terminations"]
    assert train["init_params"]["id"] == "RPO-AMP-StandWalk-v1"
    assert (config["actor"]["model"]["obs_dim"],
            config["actor"]["model"]["critic_obs_dim"],
            config["actor"]["model"]["action_dim"]) == (780, 1390, 23)
    assert (config["actor"]["global_batch_size"],
            config["actor"]["micro_batch_size"]) == (4096, 1024)
    assert 512 * 24 % config["actor"]["global_batch_size"] == 0
    assert config["actor"]["global_batch_size"] % config["actor"]["micro_batch_size"] == 0
    assert config["algorithm"]["update_epoch"] == 5


def test_rlinf_model_factory_passes_privileged_critic_dimension():
    patch = (ROOT / "deploy/patches/rlinf/patch_rpo_wbc.py").read_text()
    assert 'if cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof" else cfg.obs_dim' in patch


def test_amp_windows_never_cross_clips_or_invalid_frames():
    from eb.runners.rpo_amp_motion import valid_window_ends

    assert valid_window_ends([0, 0, 0, 1, 1, 1], [True] * 6) == [2, 5]
    assert valid_window_ends([0, 0, 0, 1, 1, 1],
                             [True, False, True, True, True, True]) == [5]


def _motion_fixture(root: Path, *, split: str = "train", changed_order: bool = False):
    clips = []
    for label in ("stand", "walk"):
        filename = f"{label}.npy"
        np.save(root / filename, np.ones((4, 74), dtype=np.float32))
        sha = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        clips.append({"id": label, "source_group": label, "behavior": label,
                      "split": split, "file": filename, "sha256": sha, "fps": 50,
                      "source_license": "fixture-only"})
    order = list(JOINT_ORDER)
    if changed_order:
        order[0], order[1] = order[1], order[0]
    manifest = {"schema_version": "eb/rpo-amp-motion/v1", "usage": "train",
                "purpose": "motion_prior", "asset_sha256": "a" * 64,
                "feature_schema": "rpo_amp_features_v1", "joint_order": order,
                "source_joint_order": list(RPO_LAB_JOINT_ORDER),
                "converter_revision": AMP_MOTION_CONVERTER_REVISION,
                "clips": clips}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_motion_loader_only_samples_verified_train_windows(tmp_path):
    from eb.runners.rpo_amp_motion import load_motion_prior

    _motion_fixture(tmp_path)
    sampler = load_motion_prior(tmp_path, asset_sha256="a" * 64)
    windows, ids = sampler.sample(6)
    assert windows.shape == (6, 3, 74)
    assert all(clip in {"stand", "walk"} and end in {2, 3} for clip, end in ids)
    state = sampler.state_dict()
    assert sampler.sample(5)[1] == (
        sampler.load_state_dict(state) or sampler.sample(5)[1]
    )


def test_motion_normalizer_keeps_constant_features_in_physical_range(tmp_path):
    from eb.runners.rpo_amp_motion import load_motion_prior

    _motion_fixture(tmp_path)
    sampler = load_motion_prior(tmp_path, asset_sha256="a" * 64)
    expected = np.array([0.1] * 3 + [0.05] * 23 + [0.2] * 23
                        + [0.05] * 18 + [0.05] * 3 + [0.05] + [0.1] * 3)
    np.testing.assert_allclose(sampler.std, expected)


@pytest.mark.parametrize("change", ["wrong_order", "eval_split", "tampered_file"])
def test_motion_loader_rejects_wrong_provenance_or_data(tmp_path, change):
    from eb.runners.rpo_amp_motion import load_motion_prior

    _motion_fixture(tmp_path, split="eval" if change == "eval_split" else "train",
                    changed_order=change == "wrong_order")
    if change == "tampered_file":
        np.save(tmp_path / "walk.npy", np.zeros((4, 74), dtype=np.float32))
    with pytest.raises(ValueError):
        load_motion_prior(tmp_path, asset_sha256="a" * 64)


def test_amp_task_registration_keeps_existing_rpo_tasks(monkeypatch):
    from eb.runners.rpo_wbc_extension import register_task

    registry = {}
    fake_gym = types.SimpleNamespace(
        registry=registry,
        register=lambda id, **kwargs: registry.setdefault(id, kwargs),
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    register_task()
    assert set(registry) == {
        "RPO-WBC-WalkUpper-v0", "RPO-WBC-StandWalk-v1", "RPO-AMP-StandWalk-v1"
    }
    assert registry["RPO-AMP-StandWalk-v1"]["entry_point"] == "rpo_amp_extension:RPOAMPIsaacEnv"
