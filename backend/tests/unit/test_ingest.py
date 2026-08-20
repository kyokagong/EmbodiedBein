from pathlib import Path

from eb.ingest import infer_contract


def test_infer_biped_and_humanoid():
    cid, io = infer_contract(Path("/data/kk/Berkeley-Humanoid-Lite/checkpoints/policy_biped_25hz_a.onnx"))
    assert cid == "ec_bhl_biped_12dof_v1"
    assert io == {"obs": 45, "act": 12}
    cid, io = infer_contract(Path("logs/rsl_rl/humanoid/x/events"))
    assert cid == "ec_bhl_humanoid_22dof_v1"
    assert io["obs"] == 75
