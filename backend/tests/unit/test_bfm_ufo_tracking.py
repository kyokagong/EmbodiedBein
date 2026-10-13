import numpy as np
import pytest


def test_held_out_tracking_never_uses_training_or_invalid_windows():
    from eb.runners.bfm_ufo_tracking import tracking_cases

    clips = []
    for source in ("official_rpo", "g1"):
        for split in ("train", "validation"):
            clips.append(
                {
                    "id": source + split,
                    "source": source,
                    "split": split,
                    "frames": 260,
                "valid_frames": 260,
                "arrays": {"valid": np.ones(260, bool)},
                }
            )
    clips.insert(0, {"id": "a_invalid", "source": "g1", "split": "validation", "frames": 260,
        "valid_frames": 259, "arrays": {"valid": np.r_[False, np.ones(259, bool)]}})
    cases = tracking_cases(clips)
    assert len(cases) == 2
    assert all(clip["split"] == "validation" and start == 0 for clip, start in cases)
    with pytest.raises(ValueError):
        tracking_cases([clip for clip in clips if clip["source"] != "g1"])


def test_tracking_prompt_is_native_and_does_not_update_normalizer():
    torch = pytest.importorskip("torch")
    from eb.runners.bfm_ufo_agent import agent_config
    from eb.runners.bfm_ufo_policy import observation_space
    from eb.runners.bfm_ufo_tracking import tracking_latents

    torch.set_num_threads(1)
    model = (
        agent_config({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16}, device="cpu")
        .model.build(observation_space(), 23)
        .eval()
    )
    arrays = {"states": np.zeros((260, 52), np.float32), "privileged_state": np.zeros((260, 358), np.float32)}
    before = {key: value.clone() for key, value in model.state_dict().items()}
    actual = tracking_latents(model, arrays, 1)
    expected = model.tracking_inference(
        {
            "state": torch.from_numpy(arrays["states"][1:251]),
            "privileged_state": torch.from_numpy(arrays["privileged_state"][1:251]),
        }
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (250, 256) and torch.isfinite(actual).all()
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)


def test_tracking_collector_rejects_missing_or_changed_media(tmp_path):
    import hashlib

    from eb.runners.bfm_ufo_tracking import SUITE, validate_tracking_receipt

    plan = {"clip_ids": ["heldout_a", "heldout_b"]}
    cases = []
    for clip_id, seed in zip(plan["clip_ids"], SUITE["case_ids"]):
        video, trace = f"tracking-{seed}.mp4", f"tracking-{seed}.jsonl"
        (tmp_path / video).write_bytes(b"test fixture")
        (tmp_path / trace).write_bytes(b"physical trace")
        cases.append(
            {
                "clip_id": clip_id,
                "case_id": seed,
                "scene_seed": SUITE["scene_seed"],
                "split": "validation",
                "steps": 3,
                "q_rmse_rad": 0.1,
                "video": video,
                "trace": trace,
                "trace_sha256": hashlib.sha256(b"physical trace").hexdigest(),
                "video_integrity": {
                    "resolution": [320, 240],
                    "fps": 10,
                    "decoded_frames": 2,
                    "sha256": hashlib.sha256(b"test fixture").hexdigest(),
                },
            }
        )
    receipt = {
        "schema_version": "eb/ufo-heldout-tracking/v1",
        "suite": SUITE,
        "plan": plan,
        "cases": cases,
        "qualification": "not_requested",
    }
    validate_tracking_receipt(receipt, plan, tmp_path)
    (tmp_path / cases[0]["video"]).write_bytes(b"damaged")
    with pytest.raises(ValueError):
        validate_tracking_receipt(receipt, plan, tmp_path)
    (tmp_path / cases[0]["video"]).unlink()
    with pytest.raises(ValueError):
        validate_tracking_receipt(receipt, plan, tmp_path)


def test_tracking_action_is_mutable_at_physics_boundary_and_masks_inactive_cases():
    torch = pytest.importorskip("torch")
    from eb.runners.bfm_ufo_agent import OBS_DIMS, agent_config
    from eb.runners.bfm_ufo_policy import observation_space
    from eb.runners.bfm_ufo_tracking import tracking_action

    torch.set_num_threads(1)
    model = (
        agent_config({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16}, device="cpu")
        .model.build(observation_space(), 23)
        .eval()
    )
    obs = {key: torch.zeros(2, width) for key, width in OBS_DIMS.items()}
    z = model.sample_z(2)
    with torch.inference_mode():
        expected = model.act(obs, z, mean=True)
    assert torch.is_inference(expected)
    with pytest.raises(RuntimeError, match="Inplace update to inference tensor"):
        expected[1] = 0
    action = tracking_action(model, obs, z, [True, False])
    assert not torch.is_inference(action) and not action.requires_grad
    action.add_(.1)
    torch.testing.assert_close(action[0], expected[0] + .1)
    torch.testing.assert_close(action[1], torch.full((23,), .1))
