"""BFM-Zero networks and prompt inference; optimization belongs to the RLinf worker.

Network modules are reused verbatim from the pinned CC BY-NC 4.0 source package.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import torch
from bfm_zero_upstream.nn_filters import DictInputFilterConfig
from bfm_zero_upstream.nn_models import (
    EMA,
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
)
from bfm_zero_upstream.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from rpo_bfm import CODEC_HASH, OBS_DIMS, PROMPT_COMMAND
from torch import nn

PROFILES = {"smoke": (128, 3), "bringup": (1024, 3), "research": (2048, 6)}


def project_z(value):
    if (value.ndim < 2 or value.shape[-1] != 256 or value.numel() == 0
            or not bool(torch.isfinite(value).all()) or bool((value.norm(dim=-1) < 1e-8).any())):
        raise ValueError("BFM prompt is empty, nonfinite, wrong-dimensional or degenerate")
    return 16 * torch.nn.functional.normalize(value, dim=-1)


class BFMZeroPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        profile = str(cfg.get("bfm_profile", "bringup"))
        if profile not in PROFILES:
            raise ValueError("unknown BFM capacity profile")
        hidden, blocks = PROFILES[profile]
        self.obs_dim, self.action_dim, self.z_dim = 375, 23, 256
        self.codec_sha256 = CODEC_HASH
        keys = {"state": 52, "privileged_state": 358, "last_action": 23, "history_actor": 300}
        space = gym.spaces.Dict({key:gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
                                 for key,width in keys.items()})
        actor_filter = DictInputFilterConfig(key=["state", "last_action", "history_actor"])
        critic_filter = DictInputFilterConfig(key=["state", "privileged_state", "last_action", "history_actor"])
        backward_filter = DictInputFilterConfig(key=["state", "privileged_state"])
        self.actor = ActorArchiConfig(model="residual", hidden_dim=hidden, hidden_layers=blocks,
                                      embedding_layers=2, input_filter=actor_filter).build(space, 256, 23)
        forward_cfg = ForwardArchiConfig(model="residual", hidden_dim=hidden, hidden_layers=blocks,
                                        embedding_layers=2, num_parallel=2, input_filter=critic_filter)
        self.forward_map = forward_cfg.build(space, 256, 23)
        self.backward_map = BackwardArchiConfig(hidden_dim=256, hidden_layers=1, norm=True,
                                               input_filter=backward_filter).build(space, 256)
        self.discriminator = DiscriminatorArchiConfig(hidden_dim=1024, hidden_layers=3,
                                                     input_filter=backward_filter).build(space, 256)
        self.critic = forward_cfg.build(space, 256, 23, output_dim=1)
        self.aux_critic = forward_cfg.build(space, 256, 23, output_dim=1)
        self.obs_normalizer = ObsNormalizerConfig(
            normalizers={key:BatchNormNormalizerConfig(momentum=.01) for key in keys},
            allow_mismatching_keys=True).build(space)
        self.aux_reward_normalizer = EMA(translate=False, scale=True)
        self.targets = nn.ModuleDict({name:copy.deepcopy(module) for name,module in (
            ("forward_map", self.forward_map), ("backward_map", self.backward_map),
            ("critic", self.critic), ("aux_critic", self.aux_critic))})
        self.targets.requires_grad_(False)
        self.register_buffer("z_pool", torch.zeros(8192, 256))
        self.register_buffer("z_pool_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("z_pool_position", torch.zeros((), dtype=torch.long))
        self._rollout_rng = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 42)))
        self._global_step = 0
        self._contexts = {}
        self.motion_view = None
        self._reference = None
        self._reward_cache = None
        if cfg.get("motion_root"):
            from pathlib import Path

            from rpo_bfm_motion import load_motion_view
            self.motion_view = load_motion_view(Path(cfg["motion_root"]), asset_sha256=str(cfg["asset_sha256"]),
                                               seed=int(cfg.get("seed", 42)))
            self._reference = self.motion_view.reference()
        self.eval()

    def online_modules(self):
        return {"actor":self.actor, "forward":self.forward_map, "backward":self.backward_map,
                "discriminator":self.discriminator, "critic":self.critic, "aux_critic":self.aux_critic}

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def normalize(self, obs):
        result = {}
        for key,width in OBS_DIMS.items():
            internal = "state" if key == "states" else key
            value = obs.get(key, obs.get(internal))
            if value is not None:
                if value.ndim != 2 or value.shape[-1] != width:
                    raise ValueError(f"BFM {key} must have shape [batch,{width}]")
                result[internal] = value.to(self.device, dtype=torch.float32)
        if "state" not in result:
            raise ValueError("BFM observation is missing states")
        self.obs_normalizer.eval()
        return self.obs_normalizer(result)

    @torch.no_grad()
    def update_normalizers(self, current, following):
        self.obs_normalizer.train()
        try:
            for obs in (current, following):
                internal = {"state" if key == "states" else key:value.to(self.device) for key,value in obs.items()
                            if key in OBS_DIMS}
                self.obs_normalizer(internal)
        finally:
            self.obs_normalizer.eval()

    @torch.no_grad()
    def goal_inference(self, observation):
        return project_z(self.backward_map(self.normalize(observation)))

    @torch.no_grad()
    def tracking_inference(self, observation, *, lookahead=8):
        if not 1 <= lookahead <= 250:
            raise ValueError("BFM tracking lookahead is outside the supported range")
        encodings = self.backward_map(self.normalize(observation))
        return project_z(torch.stack([encodings[i:i+lookahead].mean(0) for i in range(len(encodings))]))

    @torch.no_grad()
    def reward_inference(self, observation, rewards):
        embeddings = self.backward_map(self.normalize(observation))
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).reshape(-1, 1)
        if rewards.shape[0] != embeddings.shape[0] or not bool(torch.isfinite(rewards).all()):
            raise ValueError("BFM reward inference samples and rewards do not match")
        return project_z((embeddings * rewards).mean(0, keepdim=True))

    @torch.no_grad()
    def command_latent(self, commands):
        if self._reference is None:
            raise ValueError("BFM command inference requires a frozen train calibration view")
        if self._reward_cache is None:
            obs = {key:torch.from_numpy(self._reference[key]).to(self.device)
                   for key in ("states", "privileged_state")}
            self._reward_cache = (self.backward_map(self.normalize(obs)).detach(),
                                  torch.from_numpy(self._reference["reward_features"]).to(self.device))
        beta, features = self._reward_cache
        return command_to_latent(commands.to(self.device), beta, features)

    @torch.no_grad()
    def add_mixed_z(self, z):
        z = z.detach()[-len(self.z_pool):]
        position = int(self.z_pool_position)
        indices = (torch.arange(len(z), device=self.device)+position) % len(self.z_pool)
        self.z_pool[indices] = z
        self.z_pool_position.fill_((position+len(z)) % len(self.z_pool))
        self.z_pool_count.fill_(min(len(self.z_pool), int(self.z_pool_count)+len(z)))

    def set_global_step(self, step):
        if int(step) != self._global_step:
            self._reward_cache = None
        self._global_step = int(step)

    @torch.no_grad()
    def _sample_rollout_z(self):
        count = int(self.z_pool_count)
        if count:
            index = torch.randint(count, (), generator=self._rollout_rng).to(self.device)
            return self.z_pool[index:index+1].clone()
        prior = torch.randn(1, self.z_dim, generator=self._rollout_rng)
        return project_z(prior.to(self.device))

    @torch.no_grad()
    def _rollout_latents(self, observation, batch, mode):
        supplied = observation.get("z")
        if supplied is not None:
            if supplied.shape != (batch, self.z_dim):
                raise ValueError("BFM supplied rollout latent has invalid shape")
            return project_z(supplied.to(self.device, dtype=torch.float32))
        if mode == "eval":
            commands = observation.get("commands")
            if commands is None:
                commands = torch.zeros(batch, 3, device=self.device)
            return self.command_latent(torch.as_tensor(commands, device=self.device, dtype=torch.float32))

        episode_ids = torch.as_tensor(observation.get("episode_id", torch.zeros(batch)), device="cpu").reshape(-1)
        step_ids = torch.as_tensor(observation.get("step_id", torch.zeros(batch)), device="cpu").reshape(-1)
        if len(episode_ids) != batch or len(step_ids) != batch or bool((step_ids < 0).any()):
            raise ValueError("BFM rollout episode/step metadata does not match the batch")
        interval = int(self.cfg.get("update_z_every_step", 100))
        if interval < 1:
            raise ValueError("BFM rollout latent refresh interval must be positive")
        result = []
        for index, (episode, step) in enumerate(zip(episode_ids.tolist(), step_ids.tolist(), strict=True)):
            previous = self._contexts.get(index)
            if previous is not None and previous["episode_id"] == episode and step < previous["step_id"]:
                raise ValueError("BFM rollout step moved backwards within an episode")
            resample = (previous is None or previous["episode_id"] != episode
                        or (step > previous["step_id"] and step % interval == 0))
            latent = self._sample_rollout_z() if resample else previous["z"].to(self.device).unsqueeze(0)
            self._contexts[index] = {"episode_id":int(episode), "step_id":int(step), "z":latent[0].cpu().clone()}
            result.append(latent)
        return torch.cat(result, dim=0)

    @torch.no_grad()
    def predict_action_batch(self, observation=None, *, env_obs=None, mode="train", **kwargs):
        if mode not in {"train", "eval"}:
            raise ValueError("BFM action mode must be train or eval")
        observation = env_obs if observation is None else observation
        if not isinstance(observation, dict):
            raise ValueError("BFM rollout observation is missing")
        normalized = self.normalize(observation)
        batch = normalized["state"].shape[0]
        z = self._rollout_latents(observation, batch, mode)
        distribution = self.actor(normalized, z, .05)
        if mode == "eval":
            action = distribution.mean
        else:
            noise = torch.randn(distribution.mean.shape, generator=self._rollout_rng, dtype=torch.float32)
            noise = (noise.to(self.device) * distribution.scale).clamp(-.3, .3)
            action = (distribution.mean + noise).clamp(-1+1e-6, 1-1e-6)
        placeholder = action.new_zeros((batch, 1))
        return action[:, None, :], {"prev_logprobs":placeholder, "prev_values":placeholder.clone(),
                                    "forward_inputs":{"action":action.detach().clone(),
                                                      "model_action":action.detach().clone(),
                                                      "bfm_z_pool_count":self.z_pool_count.reshape(1, 1).expand(batch, 1).clone(),
                                                      "z":z.detach().clone()}}

    def rollout_state_dict(self):
        return {"schema_version":"eb/bfm-rollout/v1", "codec_sha256":self.codec_sha256,
                "global_step":self._global_step, "rng":self._rollout_rng.get_state().clone(),
                "contexts":{index:{"episode_id":value["episode_id"], "step_id":value["step_id"],
                                   "z":value["z"].clone()}
                            for index,value in self._contexts.items()},
                "motion_sampler":self.motion_view.state_dict() if self.motion_view is not None else None}

    def load_rollout_state_dict(self, state):
        if (state.get("schema_version") != "eb/bfm-rollout/v1" or state.get("codec_sha256") != self.codec_sha256
                or type(state.get("global_step")) is not int or state["global_step"] < 0
                or not isinstance(state.get("contexts"), dict)):
            raise ValueError("BFM rollout checkpoint schema/codec mismatch")
        rng = state.get("rng")
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError("BFM rollout checkpoint RNG state is invalid")
        restored_rng = torch.Generator(device="cpu")
        try:
            restored_rng.set_state(rng.cpu())
        except RuntimeError as exc:
            raise ValueError("BFM rollout checkpoint RNG state is invalid") from exc
        contexts = {}
        for index, value in state["contexts"].items():
            z = value.get("z") if isinstance(value, dict) else None
            if (type(index) is not int or index < 0 or type(value.get("episode_id")) is not int
                    or type(value.get("step_id")) is not int or value["step_id"] < 0
                    or not isinstance(z, torch.Tensor) or z.shape != (self.z_dim,)
                    or not bool(torch.isfinite(z).all()) or not torch.isclose(z.norm(), z.new_tensor(16.), atol=1e-4)):
                raise ValueError("BFM rollout checkpoint latent context is invalid")
            contexts[index] = {"episode_id":value["episode_id"], "step_id":value["step_id"],
                               "z":z.detach().cpu().clone()}
        motion_state = state.get("motion_sampler")
        if motion_state is not None:
            if self.motion_view is None:
                raise ValueError("BFM rollout checkpoint requires its frozen motion view")
            self.motion_view.load_state_dict(motion_state)
        self._rollout_rng = restored_rng
        self._global_step = state["global_step"]
        self._contexts = contexts
        self._reward_cache = None

    def forward(self, observation, z):
        return self.actor(self.normalize(observation), z, .05).mean

    def default_forward(self, **kwargs):
        raise ValueError("BFM requires loss_type=bfm_zero and its dedicated RLinf worker")

    def is_cuda_graph_enabled(self):
        return False

    def release_cuda_graph(self):
        pass


def command_to_latent(commands, beta, features):
    if not isinstance(commands, torch.Tensor) or not isinstance(beta, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise ValueError("BFM command/reference inputs must be tensors")
    minimum = commands.new_tensor(PROMPT_COMMAND["minimum"])
    maximum = commands.new_tensor(PROMPT_COMMAND["maximum"])
    if (commands.ndim != 2 or commands.shape[-1] != PROMPT_COMMAND["dim"] or beta.ndim != 2 or beta.shape[1] != 256
            or features.shape != (len(beta), 6) or len(beta) == 0
            or not bool(torch.isfinite(commands).all()) or not bool(torch.isfinite(beta).all())
            or not bool(torch.isfinite(features).all())
            or bool((commands < minimum).any()) or bool((commands > maximum).any())):
        raise ValueError("BFM command/reference shape, finite values or velocity range is invalid")
    translation = (features[None, :, :2] - commands[:, None, :2]).square().sum(-1)
    rotation = (features[None, :, 2] - commands[:, None, 2]).square()
    upright = features[:, 3].clamp(0, 1)
    height = ((features[:, 4]-.5)/.25).clamp(0, 1)
    rewards = torch.exp(-translation/.25) * torch.exp(-rotation/.25) * upright * height
    return project_z(rewards @ beta / len(beta))
