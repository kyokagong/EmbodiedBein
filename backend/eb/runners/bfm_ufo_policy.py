"""Optimizer-free native UFO rollout model and explicit Python prompt state."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

try:
    from eb.runners.bfm_ufo_agent import AGENT_CONFIG_KEYS, OBS_DIMS, agent_config, to_ufo_observation
    from eb.runners.bfm_ufo_checkpoint import validate_sampler_identities as validate_sampler_identities
    from eb.runners.rpo_ufo_motion import load_robot_state
except ModuleNotFoundError:
    from bfm_ufo_agent import AGENT_CONFIG_KEYS, OBS_DIMS, agent_config, to_ufo_observation
    from bfm_ufo_checkpoint import validate_sampler_identities as validate_sampler_identities
    from rpo_ufo_motion import load_robot_state


def observation_space():
    return gym.spaces.Dict(
        {k: gym.spaces.Box(-np.inf, np.inf, (v,), np.float32) for k, v in OBS_DIMS.items()}
    )


def sync_identity(packet):
    hasher = hashlib.sha256()
    for key, value in sorted(packet["model"].items()):
        value = value.detach().cpu().contiguous()
        hasher.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        hasher.update(value.numpy().tobytes())
    pool = packet["pool"]
    hasher.update(pool["storage"].detach().cpu().contiguous().numpy().tobytes())
    hasher.update(
        json.dumps({key: pool[key] for key in ("cursor", "full", "capacity")}, sort_keys=True).encode()
    )
    return {
        "state_sha256": hasher.hexdigest(),
        "version": packet["version"],
        "online_transitions": packet["online_transitions"],
    }


def pool_state(agent):
    pool = agent.z_buffer
    return {
        "storage": pool._storage.detach().cpu().clone(),
        "cursor": pool._idx,
        "full": pool._is_full,
        "capacity": pool.capacity,
    }


def load_pool(agent, state):
    pool = agent.z_buffer
    if (
        set(state) != {"storage", "cursor", "full", "capacity"}
        or state["capacity"] != pool.capacity
        or state["storage"].shape != pool._storage.shape
        or not torch.isfinite(state["storage"]).all()
        or type(state["cursor"]) is not int
        or not 0 <= state["cursor"] < pool.capacity
        or type(state["full"]) is not bool
    ):
        raise ValueError("Malformed UFO z pool")
    pool._storage.copy_(state["storage"].to(pool._storage))
    pool._idx, pool._is_full = state["cursor"], state["full"]


class UFOBFMPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
        from humanoidverse.agents.misc.zbuffer import ZBuffer

        config = agent_config(
            {key: cfg[key] for key in AGENT_CONFIG_KEYS if key in cfg},
            device="cpu",
        )
        # Construct only the inference model; calling agent.__init__ would create six more Adams.
        self.native = FBcprAuxAgent.__new__(FBcprAuxAgent)
        self.native.cfg = config
        self.model = config.model.build(observation_space(), 23)
        self.native._model = self.model
        self.native.z_buffer = ZBuffer(config.train.z_buffer_size, 256, "cpu")
        self.native.env_idx_with_expert_rollout = None
        self.expert = None
        if cfg.get("motion_root"):
            try:
                from eb.rpo_bfm import CODEC_HASH
            except ModuleNotFoundError:
                from rpo_bfm import CODEC_HASH
            _, self.expert = load_robot_state(
                Path(cfg["motion_root"]), asset_sha256=cfg["asset_sha256"], seed=int(cfg.get("seed", 42)),
                expected_codec_sha256=cfg.get("codec_sha256", CODEC_HASH)
            )
        self.version, self.online_transitions = 0, 0
        self.z, self.context_key, self.cached = None, None, None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(cfg.get("seed", 42)))
            self.rng_torch = torch.get_rng_state()
        self.rng_numpy = np.random.RandomState(int(cfg.get("seed", 42))).get_state()
        self.rng_python = random.Random(int(cfg.get("seed", 42))).getstate()
        self.rng_cuda = None
        self.seed = int(cfg.get("seed", 42))
        self.sampler_rank = int(cfg.get("sampler_rank", 0))
        self.context_profile = cfg.get("context_profile")
        self.ankle_control_profile = cfg.get("ankle_control_profile")
        if self.context_profile not in (None, "environment_tracking_v10"):
            raise ValueError("Unknown UFO rollout context profile")
        if self.context_profile:
            self.rollout_context = None
        self.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        if self.expert is not None:
            self.expert.device = self.device
        if self.device.type == "cuda" and self.rng_cuda is None:
            with torch.random.fork_rng(devices=[self.device.index or 0]):
                torch.cuda.manual_seed(self.seed)
                self.rng_cuda = torch.cuda.get_rng_state(self.device)
        return self

    def set_global_step(self, global_step):
        # Runner rounds are never used as learner progress.
        pass

    @torch.no_grad()
    def predict_action_batch(self, observation=None, *, env_obs=None, mode="train", **kwargs):
        obs = observation if observation is not None else env_obs
        if obs is None or "episode_id" not in obs or "step_id" not in obs:
            raise ValueError("UFO rollout requires episode/control-step identity")
        key = (tuple(obs["episode_id"].flatten().tolist()), tuple(obs["step_id"].flatten().tolist()))
        cache_key = (key, self.version, mode)
        if self.cached is not None and self.cached[0] == cache_key:
            return copy.deepcopy(self.cached[1])
        native_obs = {k: v.to(self.device) for k, v in to_ufo_observation(obs).items() if k in OBS_DIMS}
        if {k: tuple(v.shape[1:]) for k, v in native_obs.items()} != {k: (v,) for k, v in OBS_DIMS.items()}:
            raise ValueError("UFO rollout observation schema differs")
        before_np = np.random.get_state()
        before_python = random.getstate()
        before_torch = torch.get_rng_state()
        before_cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        try:
            np.random.set_state(self.rng_numpy)
            random.setstate(self.rng_python)
            torch.set_rng_state(self.rng_torch)
            if before_cuda is not None:
                torch.cuda.set_rng_state(self.rng_cuda, self.device)
            if key != self.context_key:
                if self.native.cfg.train.rollout_expert_trajectories and self.expert is None:
                    raise ValueError("UFO continuous rollout requires train-only 250-frame expert windows")
                if self.context_profile:
                    try:
                        from eb.runners.bfm_ufo_rollout_context import EnvironmentRolloutContext
                    except ModuleNotFoundError:
                        from bfm_ufo_rollout_context import EnvironmentRolloutContext
                    if self.rollout_context is None:
                        self.rollout_context = EnvironmentRolloutContext(len(native_obs["state"]), seed=self.seed, device=self.device)
                    if self.ankle_control_profile == "kd25":
                        tracking_mode=obs.get("ufo_tracking_mode")
                        if (tracking_mode is None or tracking_mode.dtype != torch.bool or tracking_mode.shape != (len(native_obs['state']),1)
                                or not torch.equal(tracking_mode.flatten().to(self.device),self.rollout_context.mask)):
                            raise ValueError("Tracking partition differs between environment and sampler")
                    self.z = self.rollout_context.update(self.native, self.expert, obs["episode_id"],
                        obs["step_id"], obs["ufo_reset_reference"], self.version)
                else:
                    self.z = self.native.maybe_update_rollout_context(
                        self.z, obs["step_id"].reshape(-1, 1).to(self.device), {"expert_slicer": self.expert}
                    )
                self.context_key = key
            if mode == "train" and self.online_transitions < 10240:
                action = torch.rand(len(native_obs["state"]), 23, device=self.device) * 2 - 1
            else:
                action = self.native.act(native_obs, self.z, mean=mode != "train")
            self.rng_numpy, self.rng_torch = np.random.get_state(), torch.get_rng_state()
            self.rng_python = random.getstate()
            if before_cuda is not None:
                self.rng_cuda = torch.cuda.get_rng_state(self.device)
        finally:
            np.random.set_state(before_np)
            random.setstate(before_python)
            torch.set_rng_state(before_torch)
            if before_cuda is not None:
                torch.cuda.set_rng_state(before_cuda, self.device)
        result = {
            "prev_logprobs": torch.zeros_like(action[:, None]),
            "prev_values": torch.zeros(len(action), 1, device=self.device),
            "forward_inputs": {
                "action": action.detach().clone(),
                "z": self.z.detach().clone(),
                "policy_version": torch.full(
                    (len(action), 1), self.version, dtype=torch.long, device=self.device
                ),
                "episode_id": obs["episode_id"].clone()
                + self.sampler_rank * 2**40
                + torch.arange(len(action), device=obs["episode_id"].device).reshape(-1, 1) * 2**24,
                "step_id": obs["step_id"].clone(),
            },
        }
        if self.context_profile:
            context=self.rollout_context
            phase=context.last_steps-context.starts
            q_error=torch.zeros(len(action),1,device=self.device)
            ids=context.tracking_ids
            expected_q=context.tracking_q[context.cache_index[ids],phase[ids]]
            q_error[ids]=((native_obs["state"][ids,:23]-expected_q).square().mean(-1).sqrt())[:,None]
            result["forward_inputs"]["ufo_tracking_trace"]=torch.cat((context.mask[:,None].float(),
                phase[:,None].float(),context.references.float(),context.encoding_versions[:,None].float(),
                context.starts[:,None].float(),q_error),-1)
        output = (action[:, None], result)
        self.cached = (cache_key, copy.deepcopy(output))
        return output

    def rollout_state_dict(self):
        state = {
            "schema_version": "eb/ufo-rollout/v2",
            "pool": pool_state(self.native),
            "version": self.version,
            "online_transitions": self.online_transitions,
            "rng_torch": self.rng_torch.clone(),
            "rng_numpy": copy.deepcopy(self.rng_numpy),
            "rng_python": copy.deepcopy(self.rng_python),
            "rng_cuda": self.rng_cuda,
            "z": None if self.z is None else self.z.detach().cpu().clone(),
            "context_key": self.context_key,
            "tracking_z": getattr(self.native, "tracking_z", None),
            "expert_env_ids": self.native.env_idx_with_expert_rollout,
            "expert": None if self.expert is None else self.expert.state_dict(),
        }
        if self.context_profile:
            state["schema_version"] = "eb/ufo-rollout/v3"
            state["environment_context"] = None if self.rollout_context is None else self.rollout_context.state_dict()
        return state

    def load_rollout_state_dict(self, state, *, reset_episode=False):
        if set(state) != set(self.rollout_state_dict()) or state["schema_version"] != self.rollout_state_dict()["schema_version"]:
            raise ValueError("Unknown UFO rollout state")
        load_pool(self.native, state["pool"])
        self.version, self.online_transitions = state["version"], state["online_transitions"]
        self.rng_torch, self.rng_numpy = state["rng_torch"].clone(), copy.deepcopy(state["rng_numpy"])
        self.rng_python = copy.deepcopy(state["rng_python"])
        self.rng_cuda = state["rng_cuda"]
        self.z = None if reset_episode or state["z"] is None else state["z"].to(self.device)
        self.context_key = None if reset_episode else state["context_key"]
        self.native.env_idx_with_expert_rollout = None if reset_episode else state["expert_env_ids"]
        if state["tracking_z"] is not None:
            self.native.tracking_z = state["tracking_z"].to(self.device)
        if self.expert is not None:
            self.expert.load_state_dict(state["expert"])
        elif state["expert"] is not None:
            raise ValueError("Rollout expert view missing")
        if self.context_profile:
            saved = state["environment_context"]
            self.rollout_context = None
            if saved is not None and not reset_episode:
                try:
                    from eb.runners.bfm_ufo_rollout_context import EnvironmentRolloutContext
                except ModuleNotFoundError:
                    from bfm_ufo_rollout_context import EnvironmentRolloutContext
                self.rollout_context = EnvironmentRolloutContext(len(saved["mask"]), seed=self.seed, device=self.device)
                self.rollout_context.load_state_dict(saved)
        self.cached = None

    def apply_learner_packet(self, packet):
        if set(packet) != {"model", "pool", "version", "online_transitions"}:
            raise ValueError("Unknown UFO sync packet")
        self.model.load_state_dict(packet["model"], strict=True)
        self.model.eval()
        load_pool(self.native, packet["pool"])
        self.version, self.online_transitions = packet["version"], packet["online_transitions"]
        self.cached = None

    def sync_receipt(self):
        return {
            "rank": self.sampler_rank,
            **sync_identity(
                {
                    "model": self.model.state_dict(),
                    "pool": pool_state(self.native),
                    "version": self.version,
                    "online_transitions": self.online_transitions,
                }
            ),
        }
