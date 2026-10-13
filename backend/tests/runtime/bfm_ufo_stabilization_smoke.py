"""Exercise all six real native updates and v7 loss bounds with a small CPU fixture."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from bfm_ufo_agent import OBS_DIMS, VELOCITY_AUX_NAMES, agent_config, instrument_updates, optimizer_steps
from bfm_ufo_agent_smoke import FixedBatch


def run():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    np.random.seed(71)
    cfg = agent_config({"hidden_dim": 64, "hidden_layers": 2, "batch_size": 16}, device="cpu").model_dump()
    cfg["train"]["lr_discriminator"] = 3e-6
    cfg["aux_rewards"] = list(VELOCITY_AUX_NAMES)
    cfg["aux_rewards_scaling"]["limits_dof_vel"] = -25.
    spaces = gym.spaces.Dict({key: gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
                             for key, dim in OBS_DIMS.items()})
    agent = FBcprAuxAgentConfig.model_validate(cfg).build(spaces, 23)
    agent._model._forward_map.register_forward_hook(lambda module, args, value: value * 100.)
    metrics = instrument_updates(agent, optimizer_profile="stabilized_v7")
    obs = {key: torch.randn(16, dim) for key, dim in OBS_DIMS.items()}
    online = {"observation": obs, "action": torch.rand(16, 23) * 2 - 1,
        "z": agent._model.sample_z(16),
        "aux_rewards": {key: torch.rand(16, 1) for key in VELOCITY_AUX_NAMES},
        "next": {"observation": {key: torch.randn_like(value) for key, value in obs.items()},
                 "terminated": torch.zeros(16, 1, dtype=torch.bool)}}
    expert = {"observation": {key: torch.randn(16, OBS_DIMS[key]) for key in ("state", "privileged_state")},
        "next": {"observation": {key: torch.randn(16, OBS_DIMS[key]) for key in ("state", "privileged_state")}}}
    agent._model._discriminator.compute_reward = lambda **kwargs: torch.full((16, 1), -10.)
    result = agent.update({"train": FixedBatch(online), "expert_slicer": FixedBatch(expert)}, 11264)
    assert set(optimizer_steps(agent).values()) == {1}
    assert float(result["mean_disc_reward_raw"]) == -10.
    assert float(result["mean_disc_reward"]) == -2.
    assert float(result["disc_reward_clip_fraction"]) == 1.
    assert float(result["actor_reg_weight_raw"]) > 50.
    assert float(result["actor_reg_weight_applied"]) == 50.
    assert all(torch.isfinite(value).all() for value in result.values())
    for name in optimizer_steps(agent):
        limit = 1. if name == "actor_optimizer" else 5.
        assert metrics[name + "/grad_norm_after_clip_l2"] <= limit + 1e-5
    output = {"status": "passed", "six_adam_steps": optimizer_steps(agent),
              "metrics": {key: float(value) for key, value in result.items()}, "optimizer_metrics": metrics}
    Path("/work/reachable-native-step.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == "__main__":
    run()
