import pytest
torch = pytest.importorskip("torch")


def test_sampler_checkpoint_preserves_its_own_python_rng():
    torch.set_num_threads(1)
    import random
    from eb.runners.bfm_ufo_policy import UFOBFMPolicy
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, "seed": 43})
    state = policy.rollout_state_dict()
    expected = random.Random(43)
    expected.setstate(state["rng_python"])
    policy.rng_python = random.Random(100).getstate()
    policy.load_rollout_state_dict(state, reset_episode=True)
    actual = random.Random()
    actual.setstate(policy.rng_python)
    assert [actual.random() for _ in range(8)] == [expected.random() for _ in range(8)]
from eb.runners.bfm_ufo_policy import UFOBFMPolicy, pool_state
from eb.runners.bfm_ufo_agent import OBS_DIMS


def test_mixed_pool_sync_and_repeated_read():
    torch.set_num_threads(1)
    source = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, "seed": 1})
    target = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, "seed": 2})
    source.native.z_buffer.add(torch.ones(8, 256))
    target.native.cfg = target.native.cfg.model_copy(update={"train": target.native.cfg.train.model_copy(
        update={"rollout_expert_trajectories": False})})
    target.apply_learner_packet({"model": source.model.state_dict(), "pool": pool_state(source.native),
        "version": 3, "online_transitions": 11264})
    assert target.version == 3 and len(target.native.z_buffer) == 8
    torch.testing.assert_close(target.native.z_buffer._storage, source.native.z_buffer._storage)
    obs = {("states" if k == "state" else k): torch.zeros(4, v) for k, v in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(4, 1, dtype=torch.long), step_id=torch.zeros(4, 1, dtype=torch.long))
    first = target.predict_action_batch(env_obs=obs)
    state = target.rollout_state_dict()
    second = target.predict_action_batch(env_obs=obs)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(state["rng_torch"], target.rollout_state_dict()["rng_torch"])
    assert target.native.env_idx_with_expert_rollout is None
    assert not hasattr(target.native, "actor_optimizer")


def test_behavior_episode_identity_is_unique_across_environments_and_ranks():
    ids = []
    for rank in range(3):
        policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16, "sampler_rank": rank})
        policy.native.cfg = policy.native.cfg.model_copy(update={"train": policy.native.cfg.train.model_copy(
            update={"rollout_expert_trajectories": False})})
        obs = {("states" if key == "state" else key): torch.zeros(2, dim) for key, dim in OBS_DIMS.items()}
        obs.update(episode_id=torch.zeros(2, 1, dtype=torch.long), step_id=torch.zeros(2, 1, dtype=torch.long))
        _, info = policy.predict_action_batch(env_obs=obs)
        ids.extend(info["forward_inputs"]["episode_id"].flatten().tolist())
    assert len(set(ids)) == 6


def test_rollout_resume_resets_context_for_new_environment():
    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    state = policy.rollout_state_dict()
    state.update(z=torch.ones(4, 256), context_key=((0, 0, 0, 0), (8, 8, 8, 8)))
    policy.load_rollout_state_dict(state, reset_episode=True)
    assert policy.z is None and policy.context_key is None


def test_native_tracking_expert_follows_model_device(monkeypatch):
    import numpy as np
    from eb.runners.bfm_ufo_replay import UFOExpertSlicer

    policy = UFOBFMPolicy({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16})
    policy.expert = UFOExpertSlicer([
        {"id": source, "source": source, "source_group": source, "split": "train",
         "arrays": {"states": np.zeros((270, 52), np.float32),
                    "privileged_state": np.zeros((270, 358), np.float32), "valid": np.ones(270, bool)}}
        for source in ("official_rpo", "g1")], manifest_sha256="a" * 64)
    policy.to("cpu")
    policy.online_transitions = 10240
    calls = []
    native_act = policy.native.act
    def act(*args, **kwargs):
        calls.append(1)
        return native_act(*args, **kwargs)
    monkeypatch.setattr(policy.native, "act", act)
    obs = {("states" if key == "state" else key): torch.zeros(16, dim) for key, dim in OBS_DIMS.items()}
    obs.update(episode_id=torch.zeros(16, 1, dtype=torch.long), step_id=torch.zeros(16, 1, dtype=torch.long))
    action, _ = policy.predict_action_batch(env_obs=obs)
    assert action.shape == (16, 1, 23) and calls == [1]
    # Meta exercises placement without initializing CUDA or allocating GPU memory.
    policy.to("meta")
    assert str(policy.native.device) == "meta"
    sampled = policy.expert.sample(250, seq_length=250)
    assert all(value.device.type == "meta" for value in sampled["next"]["observation"].values())
