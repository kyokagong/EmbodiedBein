"""Official RPO-Flat networks with RLinf's PPO model interface.

Statistics are frozen throughout each rollout/update and updated by the actor
once at the round boundary. RoboLab/RSL updates them each environment step;
this RLinf cadence retains the normalized rollout inputs and their old logprobs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.networks.normalization import EmpiricalNormalization
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


@torch.no_grad()
def update_global_normalizer(norm, states):
    """Merge learner moments without gathering the full rollout on every GPU."""
    if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
        norm.update(states)
        return
    if not norm.training or (norm.until is not None and norm.count >= norm.until):
        return
    width, count = states.shape[-1], states.shape[0]
    moments = torch.zeros(1 + 2 * width, dtype=torch.float64, device=states.device)
    moments[0] = count
    if count:
        variance, mean = torch.var_mean(states, dim=0, unbiased=False)
        mean, variance = mean.double(), variance.double()
        moments[1:1 + width] = count * mean
        moments[1 + width:] = count * (variance + mean.square())
    torch.distributed.all_reduce(moments)
    if moments[0] == 0:
        return
    mean = moments[1:1 + width] / moments[0]
    variance = (moments[1 + width:] / moments[0] - mean.square()).clamp_min(0)
    mean = mean.to(norm._mean.dtype).unsqueeze(0)
    variance = variance.to(norm._var.dtype).unsqueeze(0)
    norm.count.add_(moments[0].long())
    rate = moments[0].to(norm._mean.dtype) / norm.count
    delta = mean - norm._mean
    norm._mean.add_(rate * delta)
    norm._var.add_(rate * (variance - norm._var + delta * (mean - norm._mean)))
    norm._var.clamp_min_(0)
    norm._std.copy_(norm._var.sqrt())


class RPOFlatPolicy(MLPPolicy):
    def __init__(self, obs_dim, action_dim, num_action_chunks, add_value_head,
                 add_q_head, q_head_type="default", value_granularity="action_level",
                 critic_obs_dim=None):
        # Build the official network directly; the generic MLP has different
        # layers, activation, initialization and optional AMP-only pose biases.
        nn.Module.__init__(self)
        if (obs_dim, action_dim, num_action_chunks, critic_obs_dim) != (780, 23, 1, 1390):
            raise ValueError("RPO Flat requires actor780/critic1390/action23 and one chunk")
        if not add_value_head or add_q_head or value_granularity != "action_level":
            raise ValueError("RPO Flat requires its privileged PPO value head")
        self.obs_dim, self.critic_obs_dim = obs_dim, critic_obs_dim
        self.action_dim, self.num_action_chunks = action_dim, num_action_chunks
        self.value_granularity = value_granularity
        self.independent_std, self.final_tanh = True, False
        self.action_scale = None
        self.torch_compile_enabled = False
        self.cuda_graph_manager = None
        self.actor_norm = EmpiricalNormalization(obs_dim)
        self.critic_norm = EmpiricalNormalization(critic_obs_dim)
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 512), nn.ELU(),
                                      nn.Linear(512, 256), nn.ELU(),
                                      nn.Linear(256, 128), nn.ELU())
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim))
        self.value_head = nn.Sequential(nn.Linear(critic_obs_dim, 512), nn.ELU(),
                                        nn.Linear(512, 256), nn.ELU(),
                                        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        actor, critic = env_obs.get("states"), env_obs.get("critic_states")
        if (actor is None or critic is None or actor.ndim != 2 or critic.ndim != 2
                or actor.shape[-1] != 780 or critic.shape[-1] != 1390
                or actor.shape[0] != critic.shape[0]):
            raise ValueError("RPO Flat needs states780 and privileged critic_states1390")
        return {"states": actor.to(device), "critic_states": critic.to(device)}

    def _sample_actions(self, states):
        mean = self.actor_mean(self.backbone(states))
        return mean, self.actor_std.log().expand_as(mean)

    @torch.no_grad()
    def predict_action_batch(self, env_obs, calculate_logprobs=True, calculate_values=True,
                             return_obs=True, mode="train", **kwargs):
        raw = self.preprocess_env_obs(env_obs)
        states = self.actor_norm(raw["states"])
        critic = self.critic_norm(raw["critic_states"])
        mean = self.actor_mean(self.backbone(states))
        std = self.actor_std.expand_as(mean)
        distribution = Normal(mean, std)
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported RPO Flat prediction mode: {mode}")
        action = distribution.sample() if mode == "train" else mean.clone()
        inputs = {"action": action, "model_action": action,
                  "states": states, "critic_states": critic,
                  "raw_states": raw["states"], "raw_critic_states": raw["critic_states"],
                  "old_mean": mean, "old_std": std.clone()}
        return action.unsqueeze(1), {
            # RLinf sums these 23 terms in action_level loss preprocessing.
            "prev_logprobs": distribution.log_prob(action),
            "prev_values": self.value_head(critic) if calculate_values else torch.zeros_like(action[:, :1]),
            "forward_inputs": inputs,
        }

    def default_forward(self, forward_inputs, compute_logprobs=True, compute_entropy=True,
                        compute_values=True, **kwargs):
        mean = self.actor_mean(self.backbone(forward_inputs["states"]))
        std = self.actor_std.expand_as(mean)
        distribution = Normal(mean, std)
        result = {}
        if compute_logprobs:
            result["logprobs"] = distribution.log_prob(forward_inputs["action"])
        if compute_entropy:
            result["entropy"] = distribution.entropy()
        if compute_values:
            result["values"] = self.value_head(forward_inputs["critic_states"])
        if "old_mean" in forward_inputs:
            old_mean, old_std = forward_inputs["old_mean"], forward_inputs["old_std"]
            result["flat_kl"] = (torch.log(std / old_std + 1e-5)
                                 + (old_std.square() + (old_mean - mean).square())
                                 / (2 * std.square()) - .5).sum(-1).detach()
        return result

    @torch.no_grad()
    def update_normalizers(self, actor_states, critic_states):
        update_global_normalizer(self.actor_norm,
                                 actor_states.reshape(-1, 780).to(self.actor_norm._mean.device))
        update_global_normalizer(self.critic_norm,
                                 critic_states.reshape(-1, 1390).to(self.critic_norm._mean.device))
