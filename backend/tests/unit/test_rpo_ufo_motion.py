import numpy as np
import pytest

from eb.runners.rpo_ufo_motion import validate_robot_state_arrays, validate_robot_state_manifest


def test_robot_state_cannot_be_rebuilt_from_codec_view():
    with pytest.raises(ValueError, match="raw RobotState"):
        validate_robot_state_arrays({"states": np.zeros((10, 52), np.float32),
            "privileged_state": np.zeros((10, 358), np.float32)}, frames=10)


def test_robot_state_quaternion_and_world_translation_are_preserved():
    n = 10
    arrays = {"root_pos": np.full((n, 3), 8., np.float32),
        "root_quat": np.tile(np.array([1., 0., 0., 0.], np.float32), (n, 1)),
        "q": np.zeros((n, 23), np.float32), "dq": np.zeros((n, 23), np.float32),
        "root_lin_vel_w": np.ones((n, 3), np.float32), "root_ang_vel_w": np.zeros((n, 3), np.float32),
        "states": np.zeros((n, 52), np.float32), "privileged_state": np.zeros((n, 358), np.float32),
        "valid": np.ones(n, bool)}
    validate_robot_state_arrays(arrays, frames=n)
    assert (arrays["root_pos"] == 8).all()
    arrays["root_quat"][0] *= 2
    with pytest.raises(ValueError, match="quaternion"):
        validate_robot_state_arrays(arrays, frames=n)


def test_robot_state_manifest_rejects_ancestor_leakage():
    from eb.rpo_bfm import BODY_ORDER, RPO_LAB_JOINT_ORDER, CODEC_HASH
    manifest = {"schema_version": "eb/rpo-robot-state/v1", "asset_sha256": "a" * 64,
        "codec_sha256": CODEC_HASH, "quaternion": "wxyz", "velocity_frame": "world", "velocity_point": "root_link_origin",
        "joint_order": list(RPO_LAB_JOINT_ORDER), "body_order": list(BODY_ORDER),
        "source_refs": [{"sha256": "b" * 64, "license": "local research"}],
        "parents": [{"manifest_sha256": "c" * 64}],
        "clips": [{"id": "x", "source_group": "take", "source": "g1", "split": "train",
            "file": "x.npz", "sha256": "d" * 64, "fps": 50, "frames": 12},
            {"id": "y", "source_group": "take", "source": "g1", "split": "acceptance",
            "file": "y.npz", "sha256": "e" * 64, "fps": 50, "frames": 12}]}
    with pytest.raises(ValueError, match="split"):
        validate_robot_state_manifest(manifest, asset_sha256="a" * 64)


def test_frozen_raw_parent_conversion_preserves_test_split(tmp_path):
    import hashlib
    import json
    from eb.rpo_bfm import MIXED_POOL_CONVERTER_REVISION
    from eb.runners.rpo_ufo_motion import convert_robot_state, load_robot_state
    torch = pytest.importorskip("torch")
    from eb.runners.rpo_bfm_codec import encode_privileged, encode_state

    asset = tmp_path / "asset-manifest.json"
    asset.write_text('{"status":"verified"}')
    asset_sha = hashlib.sha256(asset.read_bytes()).hexdigest()
    arrays = {"root_pos": np.full((270, 3), 8., np.float32),
        "root_quat": np.tile(np.array([1., 0., 0., 0.], np.float32), (270, 1)),
        "q": np.full((270, 23), .2, np.float32), "dq": np.full((270, 23), .4, np.float32),
        "body_lin_vel": np.full((270, 24, 3), .5, np.float32),
        "body_ang_vel": np.full((270, 24, 3), .6, np.float32),
        "states": np.zeros((270, 52), np.float32), "privileged_state": np.ones((270, 358), np.float32),
        "valid": np.ones(270, bool)}
    arrays["body_pos"] = np.full((270, 24, 3), 8., np.float32)
    arrays["body_quat"] = np.tile(np.array([1, 0, 0, 0], np.float32), (270, 24, 1))
    arrays["states"] = encode_state(torch.from_numpy(arrays["q"]), torch.from_numpy(arrays["dq"]),
        torch.tensor([0., 0., -1.]).expand(270, 3), torch.full((270, 3), .6)).numpy()
    arrays["privileged_state"] = encode_privileged(*[torch.from_numpy(arrays[key])
        for key in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel")]).numpy()
    clips = []
    for group, split in [("official-rpo-unverified-ancestry", "train"), ("g1-train", "train"),
                         ("g1-val", "validation"), ("g1-test", "test")]:
        path = tmp_path / (group + ".npz")
        np.savez(path, **arrays)
        clips.append({"id": group, "source_group": group, "split": split, "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frames": 270, "fps": 50})
    parent = tmp_path / "manifest.json"
    parent.write_text(json.dumps({"schema_version": "eb/rpo-bfm-motion/v1", "asset_sha256": asset_sha,
        "converter_revision": MIXED_POOL_CONVERTER_REVISION, "output_quaternion": "wxyz",
        "source_refs": [{"sha256": "a" * 64, "license": "fixture"}], "clips": clips,
        "velocity_method": "central_finite_difference_50hz"}))
    output = tmp_path / "output"
    manifest = convert_robot_state(parent, output, asset_manifest=asset)
    _, expert = load_robot_state(output, asset_sha256=asset_sha)
    assert [clip["split"] for clip in manifest["clips"]] == ["train", "train", "validation", "test"]
    assert all(clip["split"] == "train" for clip in expert.clips)
    for clip in expert.clips:
        assert (clip["arrays"]["root_pos"] == 8).all()
        assert (clip["arrays"]["root_lin_vel_w"] == .5).all()
        assert (clip["arrays"]["root_ang_vel_w"] == .6).all()
    with pytest.raises(ValueError, match="new output"):
        convert_robot_state(parent, output, asset_manifest=asset)
    arrays["states"][0, 0] += 1
    corrupt = tmp_path / clips[1]["file"]
    np.savez(corrupt, **arrays)
    parent_value = json.loads(parent.read_text())
    parent_value["clips"][1]["sha256"] = hashlib.sha256(corrupt.read_bytes()).hexdigest()
    parent.write_text(json.dumps(parent_value))
    with pytest.raises(ValueError, match="codec"):
        convert_robot_state(parent, tmp_path / "corrupt-output", asset_manifest=asset)
    assert not (tmp_path / "corrupt-output").exists()
