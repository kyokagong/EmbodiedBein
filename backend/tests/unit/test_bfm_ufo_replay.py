import numpy as np
import pytest


def test_native_aligned_replay_capacity_keeps_restore_and_capacity_boundary():
    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    replay = UFOReplayBridge(capacity=1_000_000)
    restored = UFOReplayBridge(capacity=1_000_000)
    restored.load_state_dict(replay.state_dict())
    assert restored.capacity == 1_000_000 and restored.received == 0
    with pytest.raises(ValueError, match="capacity"):
        UFOReplayBridge(capacity=5_120_001)

from eb.runners.bfm_ufo_replay import UFOExpertSlicer, UFOReplayBridge


def clips():
    return [{"id": source, "source": source, "source_group": source, "split": "train",
        "arrays": {"states": np.arange(270 * 52, dtype=np.float32).reshape(270, 52),
            "privileged_state": np.zeros((270, 358), np.float32),
            "valid": np.ones(270, bool)}} for source in ("official_rpo", "g1")]


def test_expert_windows_respect_valid_and_ancestor_split():
    data = clips()
    data[0]["arrays"]["valid"][100] = False
    sampler = UFOExpertSlicer(data, manifest_sha256="a" * 64, seed=17)
    assert len(sampler.windows[250]["official_rpo"]) == 0
    with pytest.raises(ValueError, match="eligible"):
        sampler.sample_numpy(250, seq_length=250)
    samples = sampler.sample_numpy(800, seq_length=8)
    for sample in samples["sample_ids"].reshape(-1, 8, 2):
        assert np.all(sample[:, 0] == sample[0, 0])
        assert np.all(np.diff(sample[:, 1]) == 1)
        assert data[int(sample[0, 0])]["arrays"]["valid"][sample[:, 1]].all()
    data[1]["source_group"] = data[0]["source_group"]
    data[1]["split"] = "acceptance"
    with pytest.raises(ValueError, match="split"):
        UFOExpertSlicer(data, manifest_sha256="a" * 64, seed=17)


def test_sampler_resume_next_draw():
    first = UFOExpertSlicer(clips(), manifest_sha256="a" * 64, seed=8)
    first.sample_numpy(128)
    state = first.state_dict()
    expected = first.sample_numpy(128)
    second = UFOExpertSlicer(clips(), manifest_sha256="a" * 64, seed=9)
    second.load_state_dict(state)
    actual = second.sample_numpy(128)
    np.testing.assert_array_equal(expected["sample_ids"], actual["sample_ids"])
    with pytest.raises(ValueError):
        second.set_priorities({"acceptance": 2})


def transition():
    from eb.runners.bfm_ufo_agent import OBS_DIMS, AUX_NAMES
    return {"observation": {key: np.zeros((2, dim), np.float32) for key, dim in OBS_DIMS.items()},
        "action": np.full((2, 23), .7, np.float32), "z": np.full((2, 256), .8, np.float32),
        "aux_rewards": {key: np.zeros((2, 1), np.float32) for key in AUX_NAMES},
        "episode_id": np.array([[0], [0]], np.int64), "step_id": np.array([[3], [3]], np.int64),
        "policy_version": np.array([[1], [1]], np.int64),
        "next": {"observation": {key: np.full((2, dim), 4, np.float32) for key, dim in OBS_DIMS.items()},
            "terminated": np.array([[True], [False]]), "truncated": np.array([[False], [True]])}}


def test_replay_keeps_behavior_and_terminal():
    replay = UFOReplayBridge(capacity=3, seed=2)
    replay.append(transition())
    batch = replay.sample_numpy(32)
    assert (batch["action"] == .7).all() and (batch["z"] == .8).all()
    assert (batch["next"]["observation"]["state"] == 4).all()
    state = replay.state_dict()
    expected = replay.sample_numpy(16)
    restored = UFOReplayBridge(capacity=3, seed=10)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(expected["next"]["terminated"], restored.sample_numpy(16)["next"]["terminated"])
    malformed = transition()
    malformed["next"]["terminated"] = np.ones((2, 1), np.float32)
    with pytest.raises(ValueError):
        replay.append(malformed)
