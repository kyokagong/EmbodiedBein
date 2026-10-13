"""CPU update differential; GPU execution is only the fixed platform probe Job."""

import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from bfm_ufo_agent import (
    AUX_NAMES,
    OBS_DIMS,
    OPTIMIZERS,
    TARGETS,
    agent_config,
    build_agent,
    optimizer_steps,
    to_ufo_observation,
    validate_source_receipt,
)


class FixedBatch:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, batch_size, seq_length=8):
        return copy.deepcopy(self.batch)


def compare(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    else:
        assert left == right


def test_native_update_matches_bridge(device="cpu"):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    np.random.seed(71)
    spaces = gym.spaces.Dict(
        {k: gym.spaces.Box(-np.inf, np.inf, (v,), np.float32) for k, v in OBS_DIMS.items()}
    )
    cfg = (
        {"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16}
        if device == "cpu"
        else {"hidden_dim": 1024, "hidden_layers": 3, "batch_size": 128}
    )
    batch_size = cfg["batch_size"]
    native = agent_config(cfg, device=device).build(spaces, 23)
    bridge = build_agent(spaces, cfg, device=device)
    bridge._model.load_state_dict(native._model.state_dict(), strict=True)
    before = {k: v.clone() for k, v in native._model.named_parameters()}
    obs = {k: torch.randn(batch_size, v) for k, v in OBS_DIMS.items()}
    legacy = {("states" if k == "state" else k): v for k, v in obs.items()}
    online = {
        "observation": obs,
        "action": torch.rand(batch_size, 23) * 2 - 1,
        "z": native._model.sample_z(batch_size).cpu(),
        "aux_rewards": {k: torch.rand(batch_size, 1) for k in AUX_NAMES},
        "next": {
            "observation": {k: torch.randn_like(v) for k, v in obs.items()},
            "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
        },
    }
    expert = {
        "observation": {k: torch.randn(batch_size, OBS_DIMS[k]) for k in ("state", "privileged_state")},
        "next": {
            "observation": {k: torch.randn(batch_size, OBS_DIMS[k]) for k in ("state", "privileged_state")}
        },
    }
    bridged = copy.deepcopy(online)
    bridged["observation"] = to_ufo_observation(legacy)
    results = []
    for agent, batch in ((native, online), (bridge, bridged)):
        torch.manual_seed(91)
        np.random.seed(91)
        results.append(agent.update({"train": FixedBatch(batch), "expert_slicer": FixedBatch(expert)}, 11264))
    compare(results[0], results[1])
    compare(native._model.state_dict(), bridge._model.state_dict())
    for name in OPTIMIZERS:
        compare(getattr(native, name).state_dict(), getattr(bridge, name).state_dict())
    for (_, a), (_, b) in zip(native._model.named_parameters(), bridge._model.named_parameters()):
        if a.grad is not None:
            compare(a.grad, b.grad)
            assert torch.isfinite(a.grad).all()
    assert set(optimizer_steps(native).values()) == {1}
    for target in TARGETS:
        assert hasattr(native._model, target)
    compare(native.z_buffer._storage, bridge.z_buffer._storage)
    assert len(native.z_buffer) == batch_size and not native.z_buffer.empty()
    delta = sum(float((p - before[k]).abs().sum()) for k, p in native._model.named_parameters())
    assert delta > 0 and np.isfinite(delta)
    assert all(torch.isfinite(value).all() for value in native._model.state_dict().values())
    return {
        "status": "passed",
        "device": device,
        "joint_updates": 1,
        "configuration": cfg,
        "optimizer_steps": optimizer_steps(native),
        "parameter_delta_l1": delta,
        "model_bytes": sum(v.numel() * v.element_size() for v in native._model.state_dict().values()),
        "atol": 1e-6,
        "rtol": 1e-5,
        "targets": list(TARGETS),
        "z_pool_count": len(native.z_buffer),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = test_native_update_matches_bridge(args.device)
    result["source"] = validate_source_receipt(json.loads(Path("/opt/ufo-source-receipt.json").read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
