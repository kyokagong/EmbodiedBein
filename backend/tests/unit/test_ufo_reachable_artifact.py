import json

import numpy as np
import pytest

pytest.importorskip("torch")


def test_reachable_native_export_records_codec_decoder_and_control(tmp_path, monkeypatch):
    from eb.domain.training_recipe import RECIPES
    from eb.rpo_bfm import CODEC_V2_HASH, REACHABLE_RECIPE, REACHABLE_CONTROL
    from eb.runners import bfm_ufo_artifact, rpo_bfm_artifact
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "resume-manifest.json").write_text("{}")
    output = tmp_path / "export"
    output.mkdir()
    values = np.zeros((1, 23), np.float32)
    np.savez(output / "golden.npz", torch_action=values, onnx_action=values,
        torch_action_single=values, onnx_action_single=values,
        torch_command_z=values, onnx_command_z=values)
    binding = RECIPES[REACHABLE_RECIPE].binding.model_dump()
    monkeypatch.setattr(bfm_ufo_artifact, "validate_ufo_checkpoint", lambda path: {
        "execution": binding, "format": "ufo_agent_state_v1", "step": 24})
    monkeypatch.setattr(rpo_bfm_artifact, "export", lambda *args: {
        "codec_sha256": CODEC_V2_HASH, "golden": {"path": "golden.npz"}})
    result = bfm_ufo_artifact.export_ufo_policy(checkpoint, tmp_path / "config", tmp_path / "motion", "a" * 64, output)
    assert result["codec_sha256"] == CODEC_V2_HASH
    assert result["action_control"] == REACHABLE_CONTROL
    assert result["io_signature"]["action_decoder"] == "URDF finite midpoint + halfwidth*action"
    assert result["execution"] == binding and result["learner_joint_updates"] == 24
    assert json.loads((output / "export-manifest.json").read_text()) == result
