import hashlib
import json
import pickle

import numpy as np
import pytest

from eb import rpo_amp_convert
from eb.rpo_amp_convert import _source_features, _straight_walk_segments
from eb.runners.rpo_amp_motion import load_motion_prior
from eb.rpo_wbc import JOINT_ORDER, RPO_LAB_JOINT_ORDER


def test_official_motion_conversion_resamples_and_preserves_forward_speed():
    count = 101
    root = np.zeros((count, 3))
    root[:, 0] = np.arange(count) * 0.003
    root[:, 2] = 0.8
    quat = np.zeros((count, 4))
    quat[:, 0] = 1
    features = _source_features({
        "fps": 100, "root_pos": root, "root_rot": quat,
        "dof_pos": np.zeros((count, 23)),
        "key_body_pos": np.broadcast_to(root[:, None], (count, 6, 3)).copy(),
    })
    assert features.shape == (51, 74)
    np.testing.assert_allclose(features[1:-1, 71], 0.3, atol=1e-5)
    np.testing.assert_allclose(features[:, 67:70], np.tile([0, 0, -1], (51, 1)), atol=1e-5)
    np.testing.assert_allclose(features[:, 70], 0.8, atol=1e-5)


def test_official_lab_joint_order_is_mapped_to_canonical_order():
    count = 5
    root = np.zeros((count, 3))
    root[:, 2] = 0.8
    quat = np.zeros((count, 4))
    quat[:, 0] = 1
    lab_q = np.broadcast_to(np.arange(23, dtype=float), (count, 23)).copy()
    features = _source_features({
        "fps": 50, "root_pos": root, "root_rot": quat, "dof_pos": lab_q,
        "key_body_pos": np.broadcast_to(root[:, None], (count, 6, 3)).copy(),
    })
    expected = [RPO_LAB_JOINT_ORDER.index(name) for name in JOINT_ORDER]
    np.testing.assert_array_equal(features[0, 3:26], expected)


def test_official_conversion_creates_verified_train_dataset_and_rejects_mutation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    official = {}
    for name, speed, behavior in (("standing", 0.0, "stand"), ("walking", 0.3, "walk")):
        root = np.zeros((501, 3))
        root[:, 0] = np.arange(501) * speed / 100
        root[:, 2] = 0.8
        quat = np.zeros((501, 4))
        quat[:, 0] = 1
        key_pos = np.broadcast_to(root[:, None], (501, 6, 3)).copy()
        key_pos[:, :2, 2] = 0.0
        path = source / f"{name}.pkl"
        with path.open("wb") as stream:
            pickle.dump({"fps": 100, "root_pos": root, "root_rot": quat,
                         "dof_pos": np.zeros((501, 23)),
                         "key_body_pos": key_pos}, stream)
        official[name] = (behavior, hashlib.sha256(path.read_bytes()).hexdigest(), 1)
    monkeypatch.setattr(rpo_amp_convert, "OFFICIAL", official)
    asset = tmp_path / "asset-manifest.json"
    asset.write_text(json.dumps({"status": "verified"}))
    destination = tmp_path / "managed"

    report = rpo_amp_convert.convert(source, destination, asset)
    assert report["validation"]["training_ready"] is True
    assert report["validation"]["storage_ref"] == str(destination)
    assert report["source_groups_count"] == 2
    assert report["behavior_seconds"]["walk"] >= 4
    assert json.loads((destination / "manifest.json").read_text())["selection"]["smoothing_frames"] == 25
    sampler = load_motion_prior(destination, asset_sha256=report["asset_sha256"])
    windows, ids = sampler.sample(4)
    assert windows.shape == (4, 3, 74)
    assert [item[0] for item in ids] == ["standing", "walking-straight-00", "standing", "walking-straight-00"]

    with pytest.raises(FileExistsError):
        rpo_amp_convert.convert(source, destination, asset)
    with (destination / "walking-straight-00.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_motion_prior(destination, asset_sha256=report["asset_sha256"])


def test_straight_walk_segments_drop_turns_and_reject_insufficient_gait():
    features = np.zeros((320, 74), dtype=np.float32)
    features[:, 71] = 0.3
    features[130:190, 2] = 2.0
    segments = _straight_walk_segments(features)
    assert len(segments) == 2
    assert sum(map(len, segments)) >= 200
    assert all(np.max(np.abs(segment[:, 2])) <= 0.5 for segment in segments)
    with pytest.raises(ValueError, match="four seconds"):
        _straight_walk_segments(features[:110])
    features[:, 67:70] = [0, 0, -1]
    features[:, 70] = 1.0
    features[:, 51] = -0.6
    features[:, 54] = -0.6
    with pytest.raises(ValueError, match="four seconds"):
        _straight_walk_segments(features)


def test_walk_segments_bound_net_turn_without_crossing_gait_boundaries():
    features = np.zeros((500, 74), dtype=np.float32)
    features[:, 71] = 0.3
    features[:, 2] = 0.1
    segments = _straight_walk_segments(features)
    assert len(segments) >= 2
    assert sum(len(segment) for segment in segments) >= 200
    assert all(abs(float(segment[:, 2].sum() * 0.02)) <= 0.26 for segment in segments)


def test_motion_audit_recomputes_velocity_under_time_scaling(tmp_path):
    count = 101
    root = np.zeros((count, 3))
    root[:, 0] = np.arange(count) * 0.012
    root[:, 2] = 0.7
    quat = np.zeros((count, 4))
    quat[:, 0] = 1
    key_pos = np.broadcast_to(root[:, None], (count, 6, 3)).copy()
    key_pos[:, :2, 2] = 0
    with (tmp_path / "walk.pkl").open("wb") as stream:
        pickle.dump({"fps":100, "root_pos":root, "root_rot":quat,
                     "dof_pos":np.zeros((count, 23)), "key_body_pos":key_pos}, stream)
    row = rpo_amp_convert.audit(tmp_path)["candidates"][0]
    assert row["grounded_forward_frames"] == 0
    assert row["time_scales"]["3"]["grounded_forward_frames_1s"] >= 100
    assert row["time_scales"]["3"]["median_forward_mps"] == pytest.approx(0.4)
