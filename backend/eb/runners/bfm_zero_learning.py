# Copyright (c) Meta Platforms, Inc. and affiliates.
# Loss equations adapted from LeCAR-Lab/BFM-Zero@318cf44 under CC BY-NC 4.0.
# Attribution and full license: bfm_zero_upstream/UPSTREAM.md and LICENSE.
"""Differentiable BFM objectives and bounded storage; no training loop or optimizer."""

from __future__ import annotations

import torch
from bfm_zero_policy import project_z
from rpo_bfm import CODEC_HASH, OBS_DIMS
from torch.nn import functional as F


def ensemble_value(values, pessimism=.5):
    if values.shape[0] != 2:
        raise ValueError("BFM expects two independent ensemble members")
    # Equivalent to upstream's sum of ordered pairwise differences / (K²-K), K=2.
    return values.mean(0) - pessimism * (values[0]-values[1]).abs()


def bellman_discount(terminated, gamma=.98):
    if terminated.dtype != torch.bool or terminated.ndim != 2 or terminated.shape[-1] != 1:
        raise ValueError("BFM terminated must be bool [batch,1]")
    return gamma * (~terminated).float()


def fb_objective(forward, backward, target_forward, target_backward, discount,
                 *, ortho_coef=100., pessimism=0., z=None, q_loss_coef=0.):
    count = backward.shape[0]
    if count < 2 or forward.shape != target_forward.shape or forward.shape[1:] != backward.shape or discount.shape != (count, 1):
        raise ValueError("BFM pairwise FB batch shape is invalid; microbatching is unsupported")
    target_forward, target_backward = target_forward.detach(), target_backward.detach()
    target_m = ensemble_value(target_forward @ target_backward.T, pessimism).detach()
    matrices = forward @ backward.T
    off_diag = 1-torch.eye(count, device=backward.device, dtype=backward.dtype)
    denominator = count*(count-1)
    difference = matrices - discount * target_m
    off_loss = .5 * (difference * off_diag).square().sum() / denominator
    diag_loss = -difference.diagonal(dim1=1, dim2=2).mean() * forward.shape[0]
    covariance = backward @ backward.T
    orth_diag = -covariance.diagonal().mean()
    orth_off = .5 * (covariance * off_diag).square().sum() / denominator
    orth_loss = orth_diag + orth_off
    loss = off_loss + diag_loss + ortho_coef * orth_loss
    q_loss = loss.new_zeros(())
    if q_loss_coef:
        if z is None or z.shape != backward.shape:
            raise ValueError("BFM Q regularizer requires the relabeled latent")
        with torch.no_grad():
            latent_covariance = backward.T @ backward / count
            reward = (torch.linalg.solve(latent_covariance, backward, left=False) * z).sum(-1)
            next_q = ensemble_value((target_forward*z).sum(-1), pessimism)
            q_target = reward + discount[:, 0]*next_q
        q_loss = .5 * forward.shape[0] * F.mse_loss((forward*z).sum(-1), q_target.expand(forward.shape[0], -1))
        loss = loss + q_loss_coef*q_loss
    return loss, {"fb_loss":loss.detach(), "fb_offdiag":off_loss.detach(), "fb_diag":diag_loss.detach(),
                  "orth_loss":orth_loss.detach(), "q_loss":q_loss.detach(),
                  "B_norm":backward.detach().norm(dim=-1).mean(), "target_M":target_m.mean()}


def discriminator_objective(discriminator, expert_obs, expert_z, online_obs, behavior_z,
                            *, gradient_penalty=10.):
    expert_logits = discriminator.compute_logits(expert_obs, expert_z)
    online_logits = discriminator.compute_logits(online_obs, behavior_z)
    loss = (-F.logsigmoid(expert_logits)+F.softplus(online_logits)).mean()
    gp = loss.new_zeros(())
    if gradient_penalty:
        alpha = torch.rand(len(expert_z), 1, device=expert_z.device)
        interpolated = {}
        for key in expert_obs:
            interpolated[key] = (alpha*expert_obs[key]+(1-alpha)*online_obs[key]).detach().requires_grad_(True)
        latent = (alpha*expert_z+(1-alpha)*behavior_z).detach().requires_grad_(True)
        inputs = [*interpolated.values(), latent]
        prediction = discriminator.compute_logits(interpolated, latent)
        gradients = torch.autograd.grad(prediction, inputs, torch.ones_like(prediction),
                                        create_graph=True, retain_graph=True, allow_unused=True)
        used = torch.cat([gradient for gradient in gradients if gradient is not None], dim=-1)
        gp = (used.norm(dim=-1)-1).square().mean()
        loss = loss + gradient_penalty*gp
    return loss, {"disc_loss":loss.detach(), "disc_wgan_gp_loss":gp.detach(),
                  "disc_expert_logit":expert_logits.detach().mean(),
                  "disc_online_logit":online_logits.detach().mean(),
                  "disc_expert_loss":(-F.logsigmoid(expert_logits)).detach().mean(),
                  "disc_train_loss":F.softplus(online_logits).detach().mean()}


def critic_objective(values, next_values, reward, discount, *, pessimism=.5):
    if values.ndim != 3 or values.shape[-1] != 1 or next_values.shape != values.shape or reward.shape != values.shape[1:]:
        raise ValueError("BFM critic/reward batch shape is invalid")
    target = (reward.detach()+discount*ensemble_value(next_values.detach(), pessimism)).detach()
    loss = .5*values.shape[0]*F.mse_loss(values, target.expand_as(values))
    return loss, {"critic_loss":loss.detach(), "target_Q":target.mean(), "Q":values.detach().mean()}


def actor_objective(forward, discriminator_q, auxiliary_q, z, *, reg_coeff=.05, reg_coeff_aux=.02,
                    pessimism=.5, scale_reg=True):
    fb_value = ensemble_value((forward*z).sum(-1), pessimism)
    discriminator_value = ensemble_value(discriminator_q, pessimism)
    auxiliary_value = ensemble_value(auxiliary_q, pessimism)
    weight = fb_value.abs().mean().detach() if scale_reg else 1.
    loss = (-fb_value.mean()-weight*reg_coeff*discriminator_value.mean()
            -weight*reg_coeff_aux*auxiliary_value.mean())
    return loss, {"actor_loss":loss.detach(), "Q_fb":fb_value.detach().mean(),
                  "Q_discriminator":discriminator_value.detach().mean(), "Q_aux":auxiliary_value.detach().mean(),
                  "reg_weight":torch.as_tensor(weight).detach(),
                  "discriminator_contribution":(-weight*reg_coeff*discriminator_value.mean()).detach(),
                  "aux_contribution":(-weight*reg_coeff_aux*auxiliary_value.mean()).detach()}


@torch.no_grad()
def encode_expert(model, next_obs, *, sequence_length=8):
    embeddings = model.backward_map(next_obs)
    if len(embeddings) % sequence_length:
        raise ValueError("BFM expert encodings must be contiguous divisible sequences")
    z = project_z(embeddings.reshape(-1, sequence_length, 256).mean(1))
    return z.repeat_interleave(sequence_length, dim=0)


@torch.no_grad()
def mixed_z(model, train_next_obs, expert_z, *, goal_ratio=.2, expert_ratio=.6):
    if min(goal_ratio, expert_ratio) < 0 or goal_ratio+expert_ratio > 1:
        raise ValueError("BFM latent mixture probabilities are invalid")
    count = len(expert_z)
    z = project_z(torch.randn(count, 256, device=expert_z.device))
    choice = torch.multinomial(z.new_tensor([goal_ratio, expert_ratio, 1-goal_ratio-expert_ratio]),
                               count, replacement=True)[:, None]
    order = torch.randperm(count, device=z.device)
    goals = project_z(model.backward_map({key:value[order] for key,value in train_next_obs.items()}))
    z = torch.where(choice == 0, goals, z)
    return torch.where(choice == 1, expert_z[torch.randperm(count, device=z.device)], z)


@torch.no_grad()
def soft_update_targets(model, *, fb_tau=.01, critic_tau=.005):
    for name, online in (("forward_map", model.forward_map), ("backward_map", model.backward_map),
                         ("critic", model.critic), ("aux_critic", model.aux_critic)):
        tau = fb_tau if name in {"forward_map", "backward_map"} else critic_tau
        if not 0 <= tau <= 1:
            raise ValueError("BFM target tau is outside [0,1]")
        for target, current in zip(model.targets[name].parameters(), online.parameters(), strict=True):
            target.mul_(1-tau).add_(current, alpha=tau)


class TransitionReplay:
    """CPU ring counted in transitions. Snapshot tensors never alias the live buffer."""

    def __init__(self, *, capacity=100000, seed=42):
        if type(capacity) is not int or not 1 <= capacity <= 200000:
            raise ValueError("BFM replay capacity must be 1..200000 transitions")
        self.capacity, self.size, self.position, self.total_seen = capacity, 0, 0, 0
        self.storage = {}
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        return self.size

    def _flatten(self, batch):
        count = len(batch["action"])
        if count == 0:
            raise ValueError("BFM replay cannot add an empty batch")
        flat = {}
        for prefix in ("obs", "next_obs"):
            if set(batch.get(prefix, {})) != set(OBS_DIMS):
                raise ValueError("BFM replay observation/terminal fields are missing or unexpected")
            for key,width in OBS_DIMS.items():
                value = batch[prefix][key]
                if value.shape != (count, width) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"BFM replay {prefix}/{key} is invalid")
                flat[prefix+"/"+key] = value.detach().cpu()
        for key,width in (("action", 23), ("z", 256), ("aux_rewards", 3), ("terminated", 1), ("truncated", 1)):
            value = batch.get(key)
            expected_dtype = torch.bool if key in {"terminated", "truncated"} else torch.float32
            if (not isinstance(value, torch.Tensor) or value.shape != (count, width)
                    or value.dtype != expected_dtype or not bool(torch.isfinite(value).all())):
                raise ValueError(f"BFM replay {key} is invalid")
            flat[key] = value.detach().cpu()
        for key,value in batch.items():
            if key not in {"obs", "next_obs", *flat}:
                if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) != count:
                    raise ValueError(f"BFM replay metadata {key} must be batch-aligned tensor")
                flat[key] = value.detach().cpu()
        flat["transition_id"] = torch.arange(self.total_seen, self.total_seen+count, dtype=torch.long)[:, None]
        return flat

    def add(self, batch):
        flat = self._flatten(batch)
        count = len(flat["action"])
        if not self.storage:
            self.storage = {key:torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype) for key,value in flat.items()}
        if set(flat) != set(self.storage) or any(value.shape[1:] != self.storage[key].shape[1:]
                                               or value.dtype != self.storage[key].dtype for key,value in flat.items()):
            raise ValueError("BFM replay schema changed during collection")
        if count >= self.capacity:
            for key,value in flat.items():
                self.storage[key].copy_(value[-self.capacity:])
            self.size, self.position = self.capacity, 0
        else:
            indices = (torch.arange(count)+self.position) % self.capacity
            for key,value in flat.items():
                self.storage[key][indices] = value
            self.size = min(self.capacity, self.size+count)
            self.position = (self.position+count) % self.capacity
        self.total_seen += count

    def sample(self, batch_size):
        if not self.size or type(batch_size) is not int or batch_size < 1:
            raise ValueError("BFM replay is empty or sample size is invalid")
        indices = torch.randint(self.size, (batch_size,), generator=self.rng)
        flat = {key:value[indices].clone() for key,value in self.storage.items()}
        result = {prefix:{key:flat.pop(prefix+"/"+key) for key in OBS_DIMS} for prefix in ("obs", "next_obs")}
        return {**result, **flat, "sample_ids":indices[:, None]}

    def state_dict(self):
        return {"schema_version":"eb/bfm-replay/v1", "codec_sha256":CODEC_HASH, "capacity":self.capacity,
                "size":self.size, "position":self.position, "total_seen":self.total_seen,
                "storage":{key:value[:self.size].clone() for key,value in self.storage.items()},
                "rng":self.rng.get_state().clone()}

    def load_state_dict(self, state):
        size, position = state.get("size", -1), state.get("position", -1)
        total_seen = state.get("total_seen", -1)
        if (state.get("schema_version") != "eb/bfm-replay/v1" or state.get("codec_sha256") != CODEC_HASH
                or state.get("capacity") != self.capacity or type(size) is not int or not 1 <= size <= self.capacity
                or type(position) is not int or not 0 <= position < self.capacity
                or type(total_seen) is not int or total_seen < size
                or (size < self.capacity and position != size)):
            raise ValueError("BFM replay checkpoint schema/capacity/counters mismatch")

        required = {f"{prefix}/{key}" for prefix in ("obs", "next_obs") for key in OBS_DIMS}
        required.update({"action", "z", "aux_rewards", "terminated", "truncated", "transition_id"})
        source = state.get("storage")
        if not isinstance(source, dict) or not required.issubset(source):
            raise ValueError("BFM replay checkpoint is missing required schema tensors")
        shapes = {f"{prefix}/{key}":(width, torch.float32)
                  for prefix in ("obs", "next_obs") for key,width in OBS_DIMS.items()}
        shapes.update({"action":(23, torch.float32), "z":(256, torch.float32),
                       "aux_rewards":(3, torch.float32), "terminated":(1, torch.bool),
                       "truncated":(1, torch.bool), "transition_id":(1, torch.long)})
        restored_storage = {}
        for key,value in source.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) != size:
                raise ValueError("BFM replay checkpoint tensor shape mismatch")
            expected = shapes.get(key)
            if expected and (value.shape[1] != expected[0] or value.dtype != expected[1]):
                raise ValueError(f"BFM replay checkpoint {key} shape/dtype mismatch")
            if key not in shapes and (value.dtype.is_floating_point and not bool(torch.isfinite(value).all())):
                raise ValueError(f"BFM replay checkpoint metadata {key} contains nonfinite values")
            restored_storage[key] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            restored_storage[key][:size] = value.cpu()

        rng_state = state.get("rng")
        if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("BFM replay checkpoint RNG state is invalid")
        restored_rng = torch.Generator(device="cpu")
        try:
            restored_rng.set_state(rng_state.cpu())
        except RuntimeError as exc:
            raise ValueError("BFM replay checkpoint RNG state is invalid") from exc

        self.storage = restored_storage
        self.size, self.position, self.total_seen = size, position, total_seen
        self.rng = restored_rng
