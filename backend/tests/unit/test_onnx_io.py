from pathlib import Path

from eb.onnx_io import read_signature

CKPT = Path("/data/kk/Berkeley-Humanoid-Lite/checkpoints")


def test_biped_onnx_dims():
    p = CKPT / "policy_biped_25hz_a.onnx"
    sig = read_signature(p)
    assert sig["act"] == 12
    assert sig["obs"] == 45


def test_six_checkpoint_onnx_measured():
    expected_act = {
        "policy_biped_25hz_a.onnx": 12,
        "policy_biped_25hz_b.onnx": 12,
        "policy_biped_50hz.onnx": 12,
        "policy_humanoid.onnx": 22,
        "policy_humanoid_legs.onnx": 12,
        "policy_video.onnx": 12,
    }
    for name, act in expected_act.items():
        sig = read_signature(CKPT / name)
        assert sig["act"] == act, (name, sig)
        assert sig["obs"] in (45, 75, 10, 85)
