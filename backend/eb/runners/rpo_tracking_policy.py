"""RLinf PPO interface for a reference-conditioned residual policy."""

import torch
from rpo_flat_policy import RPOFlatPolicy


class RPOTrackingPolicy(RPOFlatPolicy):
    ACTOR_DIM, CRITIC_DIM = 547, 905

    def __init__(self, cfg):
        super().__init__(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.num_action_chunks,
            cfg.add_value_head,
            False,
            critic_obs_dim=cfg.critic_obs_dim,
        )
        with torch.no_grad():
            self.actor_mean.weight.zero_()
            self.actor_mean.bias.zero_()
            self.actor_std.fill_(0.1)
