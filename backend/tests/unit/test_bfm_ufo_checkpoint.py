import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from eb.runners.bfm_ufo_agent import AUX_NAMES, OBS_DIMS, build_agent, optimizer_steps
from eb.runners.bfm_ufo_policy import UFOBFMPolicy, observation_space
from eb.runners.bfm_ufo_replay import UFOExpertSlicer, UFOReplayBridge
from eb.runners.bfm_ufo_checkpoint import save_ufo_checkpoint, load_ufo_checkpoint, validate_ufo_checkpoint


def objects():
    torch.set_num_threads(1)
    agent = build_agent(observation_space(), {"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16}, device="cpu")
    expert = UFOExpertSlicer([{"id": source, "source": source, "source_group": source, "split": "train",
        "arrays": {"states": np.zeros((270, 52), np.float32), "privileged_state": np.zeros((270, 358), np.float32),
            "valid": np.ones(270, bool)}} for source in ("official_rpo", "g1")], manifest_sha256="a" * 64, seed=1)
    online = UFOReplayBridge(capacity=32, seed=1)
    online.append({"observation": {k: np.zeros((16, v), np.float32) for k, v in OBS_DIMS.items()},
        "action": np.zeros((16, 23), np.float32), "z": np.ones((16, 256), np.float32),
        "aux_rewards": {k: np.zeros((16, 1), np.float32) for k in AUX_NAMES},
        "episode_id": np.zeros((16, 1), np.int64), "step_id": np.zeros((16, 1), np.int64),
        "policy_version": np.zeros((16, 1), np.int64),
        "next": {"observation": {k: np.zeros((16, v), np.float32) for k, v in OBS_DIMS.items()},
            "terminated": np.zeros((16, 1), bool), "truncated": np.zeros((16, 1), bool)}})
    return agent, {"train": online, "expert_slicer": expert}


def receipt(step=1):
    from eb.domain.training_recipe import RECIPES, UFO_ISAAC_RECIPE
    return {"run_id": "run_fixture", "learner_joint_updates": step, "sampling_rounds": 1,
        "online_transitions": 16, "next_trigger": 11264,
        "compatibility_fingerprint": "sha256:" + "b" * 64,
        "execution": RECIPES[UFO_ISAAC_RECIPE].binding.model_dump()}


def test_contact_checkpoint_rejects_unscaled_counters(tmp_path):
    agent, replay = objects()
    checkpoint=tmp_path/'global_step_0'
    policy=UFOBFMPolicy({'hidden_dim':64,'hidden_layers':2,'batch_size':16})
    save_ufo_checkpoint(agent,replay,{'rank_0':policy.rollout_state_dict()},checkpoint,receipt=receipt(0))
    path=checkpoint/'resume-manifest.json';value=json.loads(path.read_text())
    value['execution']['recipe_id']='rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Scaled native checkpoint sampling counters'):
        validate_ufo_checkpoint(checkpoint)


def test_trainer_state_restores_next_update(tmp_path):
    agent, replay = objects()
    agent.update(replay, 11264)
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    checkpoint = tmp_path / "global_step_1"
    save_ufo_checkpoint(agent, replay, {"rank_0": policy.rollout_state_dict()}, checkpoint, receipt=receipt())
    expected = agent.update(replay, 12288)
    restored, buffers = objects()
    result = load_ufo_checkpoint(checkpoint, agent=restored, replay=buffers,
        expected={"compatibility_fingerprint": receipt()["compatibility_fingerprint"]}, device="cpu")
    actual = restored.update(buffers, 12288)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], atol=1e-6, rtol=1e-5)
    for key, value in agent._model.state_dict().items():
        torch.testing.assert_close(value, restored._model.state_dict()[key], atol=1e-6, rtol=1e-5)
    assert set(optimizer_steps(restored).values()) == {2}
    assert result["receipt"]["learner_joint_updates"] == 1


def test_four_gpu_checkpoint_restores_all_three_sampler_rngs(tmp_path):
    from eb.domain.training_recipe import RECIPES, UFO_FOUR_GPU_RECIPE
    agent, replay = objects()
    agent.update(replay, 11264)
    policies = [UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, "seed": 42 + i}) for i in range(3)]
    ranks = {f"rank_{i}": policy.rollout_state_dict() for i, policy in enumerate(policies)}
    metadata = {**receipt(), "execution": RECIPES[UFO_FOUR_GPU_RECIPE].binding.model_dump()}
    checkpoint = tmp_path / "global_step_1"
    with pytest.raises(ValueError):
        save_ufo_checkpoint(agent, replay, {"rank_0": ranks["rank_0"]}, checkpoint, receipt=metadata)
    save_ufo_checkpoint(agent, replay, ranks, checkpoint, receipt=metadata)
    restored, buffers = objects()
    result = load_ufo_checkpoint(checkpoint, agent=restored, replay=buffers,
        expected={"compatibility_fingerprint": metadata["compatibility_fingerprint"]}, device="cpu")
    assert set(result["rollout"]) == set(ranks)
    for key in ranks:
        torch.testing.assert_close(result["rollout"][key]["rng_torch"], ranks[key]["rng_torch"])
    assert not torch.equal(result["rollout"]["rank_0"]["rng_torch"], result["rollout"]["rank_2"]["rng_torch"])
    import pickle
    transferred = pickle.loads(pickle.dumps(result["rollout"]))
    for index, policy in enumerate(policies):
        policy.load_rollout_state_dict(transferred[f"rank_{index}"], reset_episode=True)
        expected_rng = np.random.RandomState()
        expected_rng.set_state(ranks[f"rank_{index}"]["rng_numpy"])
        actual_rng = np.random.RandomState()
        actual_rng.set_state(policy.rng_numpy)
        np.testing.assert_array_equal(actual_rng.rand(8), expected_rng.rand(8))


def test_ufo_checkpoint_rejects_partial_or_unknown_state(tmp_path):
    agent, replay = objects()
    agent.update(replay, 11264)
    checkpoint = tmp_path / "global_step_1"
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    save_ufo_checkpoint(agent, replay, {"rank_0": policy.rollout_state_dict()}, checkpoint, receipt=receipt())
    manifest = checkpoint / "resume-manifest.json"
    value = json.loads(manifest.read_text())
    original = manifest.read_text()
    value["model_keys"] = [k for k in value["model_keys"] if not k.startswith("_target_aux_critic")]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="target"):
        validate_ufo_checkpoint(checkpoint, run_id="run_fixture")
    manifest.write_text(original)
    file = checkpoint / "trainer-state.pt"
    content = file.read_bytes()
    file.write_bytes(content[:-1] + bytes([(content[-1] + 1) % 256]))
    with pytest.raises(ValueError, match="hash"):
        validate_ufo_checkpoint(checkpoint, run_id="run_fixture")
    assert validate_ufo_checkpoint(checkpoint, run_id="run_fixture", verify_hashes=False)["step"] == 1
    with pytest.raises(ValueError, match="hash"):
        load_ufo_checkpoint(checkpoint, agent=agent, replay=replay, expected={}, device="cpu")


def test_recipe_change_rejects_full_resume(tmp_path):
    agent, replay = objects()
    checkpoint = tmp_path / "global_step_0"
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    save_ufo_checkpoint(agent, replay, {"rank_0": policy.rollout_state_dict()}, checkpoint, receipt=receipt(0))
    with pytest.raises(ValueError, match="compatibility"):
        load_ufo_checkpoint(checkpoint, agent=agent, replay=replay,
            expected={"compatibility_fingerprint": "sha256:" + "c" * 64}, device="cpu")


@pytest.mark.parametrize("changed", [{"batch_size": 32}, {"rollout_expert_trajectories": False}])
def test_same_shape_different_algorithm_config_cannot_restore(tmp_path, changed):
    agent, replay = objects()
    checkpoint = tmp_path / "global_step_0"
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    save_ufo_checkpoint(agent, replay, {"rank_0": policy.rollout_state_dict()}, checkpoint, receipt=receipt(0))
    restored = build_agent(observation_space(),
        {"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, **changed}, device="cpu")
    with pytest.raises(ValueError, match="config differs"):
        load_ufo_checkpoint(checkpoint, agent=restored, replay=replay,
            expected={"compatibility_fingerprint": receipt()["compatibility_fingerprint"]}, device="cpu")


@pytest.mark.parametrize("damage", ["unknown_manifest_key", "missing_optimizer", "wrong_step",
    "missing_file", "extra_file", "escaping_path", "symlink"])
def test_native_checkpoint_structure_rejects_incomplete_or_unsafe_members(tmp_path, damage):
    agent, replay = objects()
    checkpoint = tmp_path / "global_step_0"
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    save_ufo_checkpoint(agent, replay, {"rank_0": policy.rollout_state_dict()}, checkpoint, receipt=receipt(0))
    manifest = checkpoint / "resume-manifest.json"
    value = json.loads(manifest.read_text())
    if damage == "unknown_manifest_key":
        value["unknown"] = True
    elif damage == "missing_optimizer":
        value["optimizer_steps"].pop("actor_optimizer")
    elif damage == "wrong_step":
        value["step"] = 1
    elif damage == "missing_file":
        (checkpoint / "trainer-state.pt").unlink()
    elif damage == "extra_file":
        (checkpoint / "unexpected.pt").write_bytes(b"unexpected")
    elif damage == "escaping_path":
        value["files"][0]["path"] = "../outside.pt"
    else:
        member = checkpoint / "trainer-state.pt"
        outside = tmp_path / "outside.pt"
        member.rename(outside)
        member.symlink_to(outside)
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_ufo_checkpoint(checkpoint, verify_hashes=False)


@pytest.mark.parametrize("damage", ["missing_pool", "missing_rng", "unknown_payload_key",
    "incomplete_pool", "incomplete_rng", "missing_optimizer_payload", "missing_network", "extra_network"])
def test_native_restore_rejects_invalid_payload_even_with_matching_file_hash(tmp_path, damage):
    import safetensors.torch
    from eb.runners.bfm_ufo_checkpoint import digest

    agent, replay = objects()
    checkpoint = tmp_path / "global_step_0"
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    save_ufo_checkpoint(agent, replay, {"rank_0": policy.rollout_state_dict()}, checkpoint, receipt=receipt(0))
    member = checkpoint / "trainer-state.pt"
    payload = torch.load(member, weights_only=True)
    if damage in {"missing_pool", "missing_rng"}:
        payload.pop(damage.removeprefix("missing_"))
    elif damage == "unknown_payload_key":
        payload["unknown"] = True
    elif damage == "incomplete_pool":
        payload["pool"].pop("storage")
    elif damage == "incomplete_rng":
        payload["rng"].pop("torch_cpu")
    elif damage == "missing_optimizer_payload":
        member = checkpoint / "optimizers.pth"
        payload = torch.load(member, weights_only=True)
        payload.pop("actor_optimizer")
    else:
        member = checkpoint / "model/model.safetensors"
        payload = safetensors.torch.load_file(member)
        if damage == "missing_network":
            payload.pop(next(key for key in payload if key.startswith("_actor.")))
        else:
            payload["unknown.weight"] = torch.zeros(1)
    if damage in {"missing_network", "extra_network"}:
        safetensors.torch.save_file(payload, member)
    else:
        torch.save(payload, member)
    manifest = checkpoint / "resume-manifest.json"
    value = json.loads(manifest.read_text())
    for item in value["files"]:
        if item["path"] == member.relative_to(checkpoint).as_posix():
            item.update(bytes=member.stat().st_size, sha256=digest(member))
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_ufo_checkpoint(checkpoint, agent=agent, replay=replay, expected={}, device="cpu")


def test_native_diagnostics_preserve_update_math():
    from eb.runners.bfm_ufo_agent import instrument_updates
    original, replay = objects()
    measured, measured_replay = objects()
    measured._model.load_state_dict(original._model.state_dict(), strict=True)
    diagnostics = instrument_updates(measured)
    torch.manual_seed(123)
    np.random.seed(123)
    expected = original.update(replay, 11264)
    torch.manual_seed(123)
    np.random.seed(123)
    actual = measured.update(measured_replay, 11264)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], atol=1e-6, rtol=1e-5)
    for key, value in original._model.state_dict().items():
        torch.testing.assert_close(value, measured._model.state_dict()[key], atol=1e-6, rtol=1e-5)
    assert len(diagnostics) == 18
    assert all(np.isfinite(value) for value in diagnostics.values())
    assert any(value > 0 for key, value in diagnostics.items() if key.endswith("parameter_delta_l2"))
