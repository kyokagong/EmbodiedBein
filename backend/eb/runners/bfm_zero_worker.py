"""RLinf-owned BFM learner and checkpoint state."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from bfm_zero_learning import (
    TransitionReplay,
    actor_objective,
    bellman_discount,
    critic_objective,
    discriminator_objective,
    encode_expert,
    fb_objective,
    mixed_z,
    soft_update_targets,
)
from rpo_bfm import CODEC_HASH, OBS_DIMS
from torch import nn

MODULE_LRS = {
    "actor": 3e-4,
    "forward_map": 1e-4,
    "backward_map": 1e-4,
    "discriminator": 1e-4,
    "critic": 1e-4,
    "aux_critic": 1e-4,
}


def build_optimizer(model, *, learning_rates=None, adam_eps=1e-8):
    rates = {**MODULE_LRS, **(learning_rates or {})}
    modules = {"actor":model.actor, "forward_map":model.forward_map,
               "backward_map":model.backward_map, "discriminator":model.discriminator,
               "critic":model.critic, "aux_critic":model.aux_critic}
    if set(modules) != set(MODULE_LRS):
        raise ValueError("BFM optimizer requires six independent online modules")
    groups = [{"params":list(modules[name].parameters()), "lr":float(rates[name]), "name":name}
              for name in MODULE_LRS]
    if any(not group["params"] or group["lr"] <= 0 for group in groups):
        raise ValueError("BFM optimizer group is empty or has a nonpositive rate")
    return torch.optim.Adam(groups, eps=adam_eps)


def verify_optimizer_steps(optimizer, update_count):
    counts = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            step = optimizer.state.get(parameter, {}).get("step")
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or not bool(torch.isfinite(step)) or float(step) != update_count):
                raise ValueError(f"BFM optimizer {group['name']} step differs from learner counter")
        counts[group["name"]] = update_count
    return counts


class BFMZeroUpdater:
    """One optimizer step per BFM objective, backed by RLinf's optimizer state."""

    def __init__(self, model, *, batch_size=128, updates_per_round=1,
                 replay_capacity=100000, optimizer=None, config=None):
        if type(batch_size) is not int or batch_size < 8 or batch_size % 8:
            raise ValueError("BFM batch size must be a positive multiple of expert sequence length 8")
        if type(updates_per_round) is not int or updates_per_round < 1:
            raise ValueError("BFM updates_per_round must be positive")
        self.model, self.config = model, dict(config or {})
        self.batch_size, self.updates_per_round = batch_size, updates_per_round
        self.replay = TransitionReplay(capacity=replay_capacity, seed=int(self.config.get("seed", 42)))
        self.optimizer = optimizer or build_optimizer(model)
        self.update_count = 0

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def _obs(values, device):
        if not isinstance(values, dict):
            raise ValueError("BFM transition is missing an observation dictionary")
        result = {}
        for key, width in OBS_DIMS.items():
            value = values.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != width:
                raise ValueError(f"BFM transition observation {key} has an invalid shape")
            result[key] = value.to(device=device, dtype=torch.float32)
        return result

    def _step(self, loss, module):
        if loss.ndim or not bool(torch.isfinite(loss)):
            raise ValueError("BFM objective must be a finite scalar")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(module.parameters(), float(self.config.get("max_grad_norm", 10.)))
        if not bool(torch.isfinite(norm)):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("BFM objective produced a nonfinite gradient")
        self.optimizer.step()
        return float(norm.detach())

    def train_round(self, transitions):
        current = self._obs(transitions["obs"], self.device)
        following = self._obs(transitions["next_obs"], self.device)
        if len(current["states"]) != len(transitions["action"]):
            raise ValueError("BFM transition fields are not batch-aligned")
        self.model.update_normalizers(current, following)
        self.replay.add(transitions)
        self.model.train()
        metrics = {}
        updates = 0
        try:
            for _ in range(self.updates_per_round):
                if len(self.replay) < self.batch_size:
                    break
                values = self._update()
                for key, value in values.items():
                    metrics[key] = metrics.get(key, 0.) + value
                self.update_count += 1
                updates += 1
        finally:
            self.model.eval()
        metrics = {key:value / updates for key,value in metrics.items()} if updates else {}
        metrics["replay_size"] = len(self.replay)
        metrics["update_count"] = self.update_count
        metrics["updates_this_round"] = updates
        metrics["replay_total_seen"] = self.replay.total_seen
        metrics["replay_total_draws"] = self.update_count * self.batch_size
        metrics["replay_draws_per_new_transition"] = updates * self.batch_size / len(transitions["action"])
        metrics["z_pool_count"] = int(self.model.z_pool_count)
        metrics["rollout_action_saturation"] = float((transitions["action"].abs() >= .99).float().mean())
        metrics["termination_fraction"] = float(transitions["terminated"].float().mean())
        for index, name in enumerate(("joint_limit", "torque", "action_smoothness")):
            metrics[f"aux_reward/{name}"] = float(transitions["aux_rewards"][:, index].mean())
        return metrics

    def _update(self):
        model = self.model
        batch = self.replay.sample(self.batch_size)
        data = {key:value.to(self.device) for key, value in batch.items()
                if isinstance(value, torch.Tensor)}
        obs = model.normalize(batch["obs"] | {})
        next_obs = model.normalize(batch["next_obs"] | {})
        action, behavior_z = data["action"], data["z"]
        terminated = data["terminated"].bool()
        discount = bellman_discount(terminated, float(self.config.get("gamma", .98)))
        expert_raw = model.motion_view.sample_sequences(self.batch_size, length=8)
        expert = {"states":torch.from_numpy(expert_raw["states"]).to(self.device),
                  "privileged_state":torch.from_numpy(expert_raw["privileged_state"]).to(self.device)}
        expert_next = {"states":torch.from_numpy(expert_raw["next_states"]).to(self.device),
                       "privileged_state":torch.from_numpy(expert_raw["next_privileged_state"]).to(self.device)}
        expert_obs, expert_next_obs = model.normalize(expert), model.normalize(expert_next)
        expert_z = encode_expert(model, expert_next_obs, sequence_length=8)
        z = mixed_z(model, next_obs, expert_z,
                    goal_ratio=float(self.config.get("goal_ratio", .2)),
                    expert_ratio=float(self.config.get("expert_ratio", .6)))
        model.add_mixed_z(z)

        disc_loss, disc_metrics = discriminator_objective(
            model.discriminator, expert_obs, expert_z, obs, behavior_z,
            gradient_penalty=float(self.config.get("gradient_penalty", 10.)))
        disc_norm = self._step(disc_loss, model.discriminator)

        with torch.no_grad():
            next_action = model.actor(next_obs, z, .05).sample(clip=.3)
            target_forward = model.targets["forward_map"](next_obs, z, next_action)
            target_backward = model.targets["backward_map"](next_obs)
        forward = model.forward_map(obs, z, action)
        backward = model.backward_map(next_obs)
        fb_loss, fb_metrics = fb_objective(
            forward, backward, target_forward, target_backward, discount,
            ortho_coef=float(self.config.get("ortho_coef", 100.)),
            pessimism=float(self.config.get("pessimism", 0.)), z=z,
            q_loss_coef=float(self.config.get("q_loss_coef", 0.)))
        fb_norm = self._step(fb_loss, nn.ModuleList([model.forward_map, model.backward_map]))

        discriminator_reward = model.discriminator.compute_logits(obs, z).detach()
        aux_reward = model.aux_reward_normalizer(data["aux_rewards"].sum(-1, keepdim=True))
        critic_losses, critic_metrics, critic_norms = {}, {}, {}
        for name, reward in (("critic", discriminator_reward), ("aux_critic", aux_reward)):
            values = getattr(model, name)(obs, z, action)
            next_values = model.targets[name](next_obs, z, next_action)
            loss, detail = critic_objective(values, next_values, reward, discount,
                                            pessimism=float(self.config.get("pessimism", .5)))
            critic_norms[name] = self._step(loss, getattr(model, name))
            critic_losses[name] = float(loss.detach())
            critic_metrics[name] = detail

        frozen = [*model.forward_map.parameters(), *model.critic.parameters(), *model.aux_critic.parameters()]
        requires_grad = [parameter.requires_grad for parameter in frozen]
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            actor_dist = model.actor(obs, z, .05)
            actor_action = actor_dist.sample(clip=.3)
            actor_loss, actor_metrics = actor_objective(
                model.forward_map(obs, z, actor_action), model.critic(obs, z, actor_action),
                model.aux_critic(obs, z, actor_action), z,
                reg_coeff=float(self.config.get("reg_coeff", .05)),
                reg_coeff_aux=float(self.config.get("reg_coeff_aux", .02)),
                pessimism=float(self.config.get("pessimism", .5)))
            actor_norm = self._step(actor_loss, model.actor)
        finally:
            for parameter, enabled in zip(frozen, requires_grad, strict=True):
                parameter.requires_grad_(enabled)

        soft_update_targets(model, fb_tau=float(self.config.get("fb_tau", .01)),
                            critic_tau=float(self.config.get("critic_tau", .005)))
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {
            "disc_loss":float(disc_loss.detach()), "disc_grad_norm":disc_norm,
            "fb_loss":float(fb_loss.detach()), "fb_grad_norm":fb_norm,
            "critic_loss":critic_losses["critic"], "critic_grad_norm":critic_norms["critic"],
            "aux_critic_loss":critic_losses["aux_critic"],
            "aux_critic_grad_norm":critic_norms["aux_critic"],
            "actor_loss":float(actor_loss.detach()), "actor_grad_norm":actor_norm,
            "expert_z_norm":float(expert_z.norm(dim=-1).mean()),
            "target_update_count":1,
        }
        metrics.update({key:float(value) for key,value in {**disc_metrics, **fb_metrics}.items()})
        metrics.update({f"{name}/{key}":float(value) for name,detail in critic_metrics.items()
                        for key,value in detail.items()})
        metrics.update({f"actor_{key}":float(value) for key,value in actor_metrics.items()})
        metrics["actor_action_saturation"] = float((actor_dist.mean.detach().abs() >= .99).float().mean())
        metrics["replay_sample_age"] = float((self.replay.total_seen - 1 - data["transition_id"]).float().mean())
        with torch.no_grad():
            singular = torch.linalg.svdvals(backward.detach() - backward.detach().mean(0))
            probability = singular.square() / singular.square().sum().clamp_min(1e-12)
            metrics["B_effective_rank"] = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
        threshold = float(self.config.get("max_grad_norm", 10.))
        for name in ("disc", "fb", "critic", "aux_critic", "actor"):
            norm = metrics[f"{name}_grad_norm"]
            metrics[f"{name}_grad_clipped"] = float(norm > threshold)
            metrics[f"{name}_grad_scale"] = min(1., threshold / (norm + 1e-6))
        return metrics

    def state_dict(self):
        return {"schema_version":"eb/bfm-updater/v1", "codec_sha256":CODEC_HASH,
                "update_count":self.update_count, "replay":self.replay.state_dict()}

    def load_state_dict(self, state):
        count = state.get("update_count")
        if (state.get("schema_version") != "eb/bfm-updater/v1"
                or state.get("codec_sha256") != CODEC_HASH or type(count) is not int or count < 0):
            raise ValueError("BFM updater checkpoint schema/codec/counter mismatch")
        restored = TransitionReplay(capacity=self.replay.capacity)
        restored.load_state_dict(state["replay"])
        self.replay, self.update_count = restored, count


try:
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
except ModuleNotFoundError as exc:
    if exc.name != "rlinf":
        raise
    EmbodiedFSDPActor = object


class RPOBFMActor(EmbodiedFSDPActor):
    """RLinf owns rollout scheduling, sync, optimizer and checkpoint lifecycle."""

    def build_optimizer(self, model, enable_critic_warmup=False):
        module = model.module if hasattr(model, "module") else model
        configured = self.cfg.actor.model.get("learning_rates", {})
        optimizer = build_optimizer(module, learning_rates=dict(configured),
                                    adam_eps=float(self.cfg.actor.optim.get("adam_eps", 1e-8)))
        from rlinf.utils.utils import warmup_optimizer_state
        warmup_optimizer_state(optimizer)
        return optimizer

    def init_worker(self):
        super().init_worker()
        if self._world_size != 1 or self.cfg.actor.model.num_action_chunks != 1:
            raise ValueError("RPO BFM smoke/bring-up requires one learner and one action chunk")
        model = self.model.module if hasattr(self.model, "module") else self.model
        if model.motion_view is None:
            raise ValueError("RPO BFM learner requires a hash-pinned observation-only motion view")
        cfg = self.cfg.actor.model
        self.bfm = BFMZeroUpdater(model, batch_size=int(cfg.get("batch_size", 128)),
                                  updates_per_round=int(cfg.get("updates_per_round", 1)),
                                  replay_capacity=int(cfg.get("replay_capacity", 100000)),
                                  optimizer=self.optimizer, config=dict(cfg.get("learning", {})))

    def _process_received_rollout_batch(self, batch):
        return batch

    def compute_advantages_and_returns(self):
        batch = self.rollout_batch
        if not batch.get("curr_obs") or not batch.get("next_obs"):
            raise ValueError("RPO BFM requires RLinf transition observations")
        return {"bfm/collected_transitions":int(batch["actions"].shape[0] * batch["actions"].shape[1])}

    @staticmethod
    def _aligned_flag(value, steps, batch, name):
        if value is None:
            return torch.zeros(steps, batch, 1, dtype=torch.bool)
        if value.shape[0] == steps + 1:
            value = value[1:]
        if value.shape[:2] != (steps, batch):
            raise ValueError(f"RPO BFM {name} flags do not align with actions")
        return value.reshape(steps, batch, -1)[..., :1].bool()

    def run_training(self):
        batch = self.rollout_batch
        actions = batch["actions"]
        steps, envs = actions.shape[:2]
        if actions.ndim == 4:
            actions = actions[:, :, 0]
        forward = batch["forward_inputs"]
        if "z" not in forward or "action" not in forward:
            raise ValueError("RPO BFM rollout omitted its behavior latent/action")
        curr, nxt = batch["curr_obs"], batch["next_obs"]
        current = {key:curr[key].reshape(-1, width).float() for key,width in OBS_DIMS.items()}
        following = {key:nxt[key].reshape(-1, width).float() for key,width in OBS_DIMS.items()}
        aux = nxt.get("bfm_aux_rewards")
        if aux is None or aux.shape[:2] != (steps, envs):
            raise ValueError("RPO BFM transition is missing aligned auxiliary rewards")
        terminations = self._aligned_flag(batch.get("terminations"), steps, envs, "termination")
        truncations = self._aligned_flag(batch.get("truncations"), steps, envs, "truncation")
        transitions = {"obs":current, "next_obs":following,
                       "action":actions.reshape(-1, 23).float(),
                       "z":forward["z"].reshape(-1, 256).float(),
                       "aux_rewards":aux.reshape(-1, 3).float(),
                       "terminated":terminations.reshape(-1, 1),
                       "truncated":truncations.reshape(-1, 1)}
        metrics = self.bfm.train_round(transitions)
        if "bfm_z_pool_count" in forward:
            metrics["rollout_z_pool_count_min"] = float(forward["bfm_z_pool_count"].min())
            metrics["rollout_z_pool_count_max"] = float(forward["bfm_z_pool_count"].max())
        metrics["learner_peak_allocated_mb"]=torch.cuda.max_memory_allocated()/1024**2
        metrics["learner_peak_reserved_mb"]=torch.cuda.max_memory_reserved()/1024**2
        self.lr_scheduler.step()
        return {f"bfm/{key}":value for key,value in metrics.items()}

    def save_checkpoint(self, save_path: str, step: int = 0):
        optimizer_steps = verify_optimizer_steps(self.optimizer, self.bfm.update_count)
        super().save_checkpoint(save_path, step)
        model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = Path(save_path).parent
        target = checkpoint / "bfm/state.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".pt.tmp")
        try:
            torch.save({"updater":self.bfm.state_dict(), "rollout":model.rollout_state_dict()}, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        from rlinf_entry import atomic_json, prune_old_checkpoints, write_resume_manifest
        atomic_json(checkpoint / "bfm/progress.json", {
            "schema_version":"eb/bfm-learner-progress/v1", "runner_round":step,
            "learner_updates":self.bfm.update_count, "updates_per_round":self.bfm.updates_per_round,
            "optimizer_steps":optimizer_steps,
            "replay_total_seen":self.bfm.replay.total_seen, "z_pool_count":int(model.z_pool_count)})
        write_resume_manifest(checkpoint, "trainer_state")
        prune_old_checkpoints(checkpoint.parent)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        model = self.model.module if hasattr(self.model, "module") else self.model
        path = Path(load_path).parent / "bfm/state.pt"
        if path.is_symlink() or not path.is_file():
            raise ValueError("RPO BFM checkpoint is missing replay and rollout state")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.bfm.load_state_dict(state["updater"])
        rounds = int(Path(load_path).parent.name.removeprefix("global_step_"))
        if self.bfm.update_count != rounds * self.bfm.updates_per_round:
            raise ValueError("BFM checkpoint learner updates differ from runner rounds")
        verify_optimizer_steps(self.optimizer, self.bfm.update_count)
        model.load_rollout_state_dict(state["rollout"])
        from rlinf_entry import mark_resume_loaded
        mark_resume_loaded(Path(self.cfg.runner.logger.log_path), Path(load_path).parent)
