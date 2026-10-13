import copy
import hashlib
import json
from pathlib import Path

import pytest


def test_bfm_contract_freezes_native_order_and_real_body_dimensions():
    from eb.rpo_bfm import ACTION_GAIN, BODY_ORDER, CODEC_HASH, DEFAULT_Q, ENV_ID, spec
    from eb.rpo_wbc import RPO_LAB_JOINT_ORDER

    value = spec()
    assert ENV_ID == "ee_rpo_v2_bfm_23dof_v1"
    assert tuple(value["joint_order"]) == RPO_LAB_JOINT_ORDER
    assert len(BODY_ORDER) == 24 and BODY_ORDER[0] == "base_link"
    assert value["observation_dims"] == {
        "states": 52, "last_action": 23, "history_actor": 300, "privileged_state": 358,
    }
    assert value["prompt_command"] == {
        "field": "commands", "dim": 3, "units": ["m/s", "m/s", "rad/s"],
        "minimum": [-.6, -.5, -1.57], "maximum": [1., .5, 1.57],
        "consumer": "command_to_latent",
    }
    assert len(ACTION_GAIN) == len(DEFAULT_Q) == 23
    assert max(ACTION_GAIN) > 5  # Measured knees require more than G1's gain=5.
    assert CODEC_HASH == hashlib.sha256(json.dumps(value, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
    import yaml
    contract = yaml.safe_load((Path(__file__).resolve().parents[3] /
                               "contracts/ee_rpo_v2_bfm_23dof_v1.yaml").read_text())
    assert contract["observation"]["groups"] == value["observation_dims"]
    assert contract["action"]["gain"] == list(ACTION_GAIN)
    assert contract["joint_order"] == value["joint_order"]
    assert contract["control"]["command_dim"] == 3
    assert contract["prompt"]["command_input"] == value["prompt_command"]


def test_bfm_motion_manifest_rejects_wrong_order_split_and_missing_provenance():
    from eb.rpo_bfm import BODY_ORDER, CODEC_HASH, MOTION_CONVERTER_REVISION, validate_motion_manifest
    from eb.rpo_wbc import RPO_LAB_JOINT_ORDER

    manifest = {
        "schema_version": "eb/rpo-bfm-motion/v1", "purpose": "behavior_prior", "usage": "train",
        "joint_order": list(RPO_LAB_JOINT_ORDER), "body_order": list(BODY_ORDER),
        "converter_revision": MOTION_CONVERTER_REVISION,
        "codec_sha256": CODEC_HASH, "asset_sha256": "a" * 64,
        "qualification": "diagnostic", "dynamic_replay_verified": False,
        "source_rights": "repository_license_only",
        "source_refs": [{"uri": "https://github.com/Roboparty/robolab", "revision": "b" * 40,
                         "sha256": "c" * 64, "license": "BSD-3-Clause"}],
        "clips": [{"id": "stand", "file": "stand.npz", "sha256": "d" * 64,
                   "fps": 50, "split": "train", "source_group": "original-stand", "frames": 20}],
    }
    result = validate_motion_manifest(manifest, asset_sha256="a" * 64)
    assert result["smoke_ready"] and not result["training_ready"]
    from eb.rpo_bfm import REFERENCE_POOL_CONVERTER_REVISION
    candidate = validate_motion_manifest({**manifest, "converter_revision":REFERENCE_POOL_CONVERTER_REVISION,
                                         "qualification":"reference_candidate"}, asset_sha256="a"*64)
    assert candidate["prior_candidate_ready"] and not candidate["training_ready"]
    assert not candidate["controller_evaluation_pass"]
    from eb.rpo_bfm import G1_CONVERTER_REVISION
    g1={**manifest,"converter_revision":G1_CONVERTER_REVISION,"qualification":"reference_candidate",
        "source_use":"local_noncommercial_no_redistribution","calibration_verified":True,
        "calibration_applied_to_solver":True,"calibration_sha256":"f"*64}
    assert validate_motion_manifest(g1,asset_sha256="a"*64)["prior_candidate_ready"]
    assert not validate_motion_manifest({**g1,"qualification":"qualified","dynamic_replay_verified":True,
                                        "source_rights":"verified"},asset_sha256="a"*64)["training_ready"]
    for flag in ("calibration_verified","calibration_applied_to_solver"):
        with pytest.raises(ValueError,match="G1 BFM"):
            validate_motion_manifest({**g1,flag:False},asset_sha256="a"*64)
    legacy = {**manifest, "converter_revision":"isaac-fk-raw-resample-v1",
              "qualification":"qualified", "dynamic_replay_verified":True, "source_rights":"verified"}
    checked = validate_motion_manifest(legacy, asset_sha256="a"*64)
    assert not checked["smoke_ready"] and not checked["training_ready"]
    for mutate in (
        lambda m: m["joint_order"].reverse(),
        lambda m: m.update(usage="eval"),
        lambda m: m["clips"][0].update(file="../escape.npz"),
        lambda m: m["clips"].append({**m["clips"][0], "id": "leak", "file": "leak.npz", "split": "test"}),
        lambda m: m.update(source_refs=[]),
        lambda m: m.update(codec_sha256="e" * 64),
    ):
        invalid = copy.deepcopy(manifest)
        mutate(invalid)
        with pytest.raises(ValueError):
            validate_motion_manifest(invalid, asset_sha256="a" * 64)


def test_bfm_converter_refuses_unpinned_pickle_before_deserialization(tmp_path):
    from eb.runners.rpo_bfm_motion import verify_source_files

    (tmp_path / "A1-_Stand_stageii.pkl").write_bytes(b"untrusted pickle")
    (tmp_path / "36_11.pkl").write_bytes(b"untrusted pickle")
    with pytest.raises(ValueError, match="hash"):
        verify_source_files(tmp_path)


def test_bfm_local_pool_is_pinned_and_uses_complete_timelines():
    from pathlib import Path
    from eb.runners.rpo_bfm_motion import source_profile, resample_times, verify_source_files

    sources, cap, revision = source_profile("local_pool")
    assert len(sources) == 30 and cap is None
    assert sum(item["loader"] == "joblib" for item in sources.values()) == 4
    assert revision != "isaac-fk-raw-resample-v1"
    verify_source_files(Path("/data/kk/assets/roboparty/rpo-v2/robolab/data/motions/rpo_lab"), profile="local_pool")
    assert len(resample_times(4170, 120, cap)) == 1738
    assert len(resample_times(4170, 120, 500)) == 500
    with pytest.raises(ValueError):
        source_profile("untrusted_path")


def test_bfm_lab_root_quaternion_is_wxyz_not_original_gmr_xyzw():
    import numpy as np
    from eb.runners.rpo_bfm_motion import lab_quaternions_xyzw

    upright = np.array([[.7167328, -.01515419, .007939, .697138]])
    xyzw = lab_quaternions_xyzw(upright)
    assert np.allclose(xyzw, upright[:, [1, 2, 3, 0]])
    assert 1-2*(xyzw[0, 0]**2+xyzw[0, 1]**2) > .99
    assert np.array_equal(lab_quaternions_xyzw(np.array([[1., 0., 0., 0.]])), [[0., 0., 0., 1.]])
    with pytest.raises(ValueError):
        lab_quaternions_xyzw(np.zeros((1, 4)))


def test_reference_pool_preserves_hard_limits_but_does_not_filter_pd_range():
    import numpy as np
    from eb.runners.rpo_bfm_motion import reference_mask, source_profile, reference_group
    from eb.rpo_bfm import DEFAULT_Q, REFERENCE_POOL_CONVERTER_REVISION

    q = np.tile(DEFAULT_Q, (10, 1))
    q[:, 11] = 2.0  # Deliberately beyond the policy's fixed PD target range.
    limits = np.tile([-3., 3.], (23, 1))
    dq = np.zeros_like(q)
    height, upright = np.ones(10), np.ones(10)
    assert reference_mask(q, dq, height, upright, limits).all()
    q[0, 11] = 3.1
    dq[1, 0] = 51
    height[2] = .4
    upright[3] = .2
    assert reference_mask(q, dq, height, upright, limits).tolist() == [False]*4 + [True]*6
    sources, cap, rev = source_profile("reference_pool")
    assert len(sources) == 30 and cap is None and rev == REFERENCE_POOL_CONVERTER_REVISION
    assert reference_group("run_start_180_R_001__A345_M") == reference_group("run_stop_180_R_001__A345")
    assert reference_group("B9_-__Walk_turn_left_90_stageii") == reference_group("C3_-_run_stageii")


def test_bfm_probe_tracks_separate_continuous_intervals_and_never_qualifies_static_data():
    import numpy as np
    from eb.runners.rpo_bfm_motion_probe import continuous_intervals, tracking_pass

    mask = np.array([True]*10 + [False] + [True]*8 + [False] + [True]*12)
    assert continuous_intervals(mask) == [(0, 10), (20, 32)]
    good = dict(steps=9, available_steps=9, finite=True, hard_limit_frames=0,
                joint_rmse_rad=.05, root_position_rmse_m=.02, root_angle_rmse_rad=.03,
                contact_foot_speed_rms_mps=.01, contact_samples=10, ended_early=False)
    assert tracking_pass(good)
    for bad in ({"steps":8}, {"finite":False}, {"hard_limit_frames":1},
                {"joint_rmse_rad":.5}, {"root_position_rmse_m":.5},
                {"contact_samples":0}, {"ended_early":True}):
        assert not tracking_pass({**good, **bad})


def test_rlinf_observation_keeps_bfm_commands_and_auxiliary_reward():
    from eb.runners.rpo_bfm_extension import rlinf_observation

    raw = {key: (key, width) for key, width in {
        "states": 52, "last_action": 23, "history_actor": 300,
        "privileged_state": 358, "commands": 3, "bfm_aux_rewards": 3,
        "episode_id": 1, "step_id": 1,
    }.items()}
    forwarded = rlinf_observation(raw)
    assert forwarded.keys() == raw.keys()
    assert forwarded["commands"] == raw["commands"]
    assert forwarded["bfm_aux_rewards"] == raw["bfm_aux_rewards"]
