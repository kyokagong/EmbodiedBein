"""RLinf PPO actor worker with a training-only adversarial motion prior."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from .rpo_amp_motion import load_motion_prior
except ImportError:  # Image copies runners as top-level modules.
    from rpo_amp_motion import load_motion_prior

try:
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
except ModuleNotFoundError as exc:
    if exc.name != "rlinf":
        raise
    EmbodiedFSDPActor = object  # CPU module checks run without the RLinf image.


class AMPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(222, 1024), nn.ELU(),
                                 nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, 1))

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def style_reward(scores, valid, weight, dt=0.02):
    return dt * weight * (1.0 - 0.25 * (scores - 1.0).square()).clamp_min(0) * valid


def discriminator_loss(discriminator, expert, policy, grad_penalty_scale=10.0):
    expert = expert.detach().requires_grad_(True)
    expert_score = discriminator(expert)
    policy_score = discriminator(policy.detach())
    grad = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)[0]
    return (0.5 * (expert_score - 1).square().mean()
            + 0.5 * (policy_score + 1).square().mean()
            + grad_penalty_scale * grad.square().sum(dim=-1).mean())


class RPOAMPActor(EmbodiedFSDPActor):
    """Owns D/replay/normalizer while delegating PPO and FSDP to RLinf."""

    def init_worker(self):
        super().init_worker()
        if self._world_size != 1 or self.cfg.actor.model.num_action_chunks != 1:
            raise ValueError("RPO AMP requires one learner and one action chunk")
        motion = self.cfg.motion_prior
        self.motion = load_motion_prior(
            Path(motion.manifest_path).parent,
            asset_sha256=motion.asset_sha256,
            seed=self.cfg.actor.seed,
        )
        self.mean = torch.from_numpy(self.motion.mean).to(self.device).repeat(3)
        self.std = torch.from_numpy(self.motion.std).to(self.device).repeat(3)
        self.discriminator = AMPDiscriminator().to(self.device)
        cfg = self.cfg.amp_discriminator
        self.discriminator_optim = torch.optim.Adam([
            {"params": self.discriminator.net[:-1].parameters(),
             "weight_decay": cfg.trunk_weight_decay},
            {"params": self.discriminator.net[-1].parameters(),
             "weight_decay": cfg.output_weight_decay},
        ], lr=cfg.learning_rate)
        self.history = torch.empty((cfg.history_capacity, 222), dtype=torch.float32)
        self.history_size = 0
        self.history_cursor = 0
        self.empty_rounds = 0
        self.amp_round = 0
        self._current = None

    def set_global_step(self, global_step):
        super().set_global_step(global_step)
        self.amp_round = global_step

    def _style_weight(self):
        cfg = self.cfg.amp_discriminator
        if self.amp_round < cfg.style_warmup_rounds:
            return 0.0
        return cfg.style_weight_max * min(
            1.0, (self.amp_round - cfg.style_warmup_rounds)
            / (cfg.style_full_rounds - cfg.style_warmup_rounds)
        )

    def compute_advantages_and_returns(self):
        sidecar = self.rollout_batch["forward_inputs"]
        windows = sidecar.pop("amp_window")
        valid = sidecar.pop("amp_valid").bool()
        if windows.shape[-2:] != (3, 74) or valid.shape != windows.shape[:-2] + (1,):
            raise ValueError("AMP sidecar is misaligned with PPO transitions")
        if windows.shape[:2] != self.rollout_batch["rewards"].shape[:2]:
            raise ValueError("AMP reward rows differ from PPO reward rows")
        raw = windows.reshape(-1, 222)
        self._current = raw[valid.reshape(-1)].detach().cpu().contiguous()
        with torch.no_grad():
            normalized = (raw.to(self.device) - self.mean) / self.std
            scores = self.discriminator(normalized)
            style = style_reward(scores, valid.reshape(-1).to(self.device), self._style_weight())
        self.rollout_batch["rewards"] += style.cpu().reshape_as(self.rollout_batch["rewards"])
        metrics = super().compute_advantages_and_returns()
        metrics["amp/style_reward_mean"] = style.mean().item()
        metrics["amp/valid_windows"] = int(valid.sum().item())
        return metrics

    def run_training(self):
        metrics = super().run_training()
        current = self._current
        if current is None:
            raise ValueError("AMP PPO update has no aligned motion windows")
        count = len(current)
        if count == 0:
            self.empty_rounds += 1
            if self.empty_rounds >= 2:
                raise ValueError("AMP has no valid policy windows for two rounds")
            return {**metrics, "amp/discriminator_skipped": 1}
        self.empty_rounds = 0
        cfg = self.cfg.amp_discriminator
        self.discriminator.train()
        losses = []
        mini_size = (count + cfg.minibatches - 1) // cfg.minibatches
        for _ in range(cfg.epochs):
            for _ in range(cfg.minibatches):
                batch_size = min(mini_size, count)
                expert, _ = self.motion.sample(batch_size)
                expert = torch.from_numpy(expert.reshape(batch_size, 222)).to(self.device)
                neg_current = current[torch.randint(count, (batch_size,), device="cpu")]
                if self.history_size:
                    half = batch_size // 2
                    neg_current[:half] = self.history[
                        torch.randint(self.history_size, (half,), device="cpu")
                    ]
                policy = neg_current.to(self.device)
                loss = discriminator_loss(
                    self.discriminator, (expert - self.mean) / self.std,
                    (policy - self.mean) / self.std, cfg.grad_penalty_scale,
                )
                self.discriminator_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), cfg.max_grad_norm)
                self.discriminator_optim.step()
                losses.append(loss.detach().item())
        self._append_history(current)
        self._current = None
        return {**metrics, "amp/discriminator_loss": sum(losses) / len(losses),
                "amp/valid_windows": count}

    def _append_history(self, windows):
        capacity = len(self.history)
        if len(windows) >= capacity:
            self.history.copy_(windows[-capacity:])
            self.history_size = capacity
            self.history_cursor = 0
            return
        index = (torch.arange(len(windows)) + self.history_cursor) % capacity
        self.history[index] = windows
        self.history_cursor = (self.history_cursor + len(windows)) % capacity
        self.history_size = min(capacity, self.history_size + len(windows))

    def save_checkpoint(self, save_path: str, step: int = 0):
        super().save_checkpoint(save_path, step)
        path = Path(save_path).parent / "amp"
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.discriminator_optim.state_dict(),
            "mean": self.mean.cpu(), "std": self.std.cpu(),
            "history": self.history[:self.history_size].clone(),
            "history_size": self.history_size, "history_cursor": self.history_cursor,
            "empty_rounds": self.empty_rounds, "amp_round": self.amp_round,
            "motion_sampler": self.motion.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, path / "state.pt")
        from rlinf_entry import prune_old_checkpoints, write_resume_manifest
        write_resume_manifest(path.parent, "trainer_state")
        prune_old_checkpoints(path.parent.parent, keep=4)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        path = Path(load_path).parent / "amp/state.pt"
        if not path.is_file():
            raise ValueError("AMP checkpoint is missing discriminator and sampler state")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(state["mean"], self.mean.cpu()) or not torch.equal(state["std"], self.std.cpu()):
            raise ValueError("AMP motion normalizer differs on resume")
        self.discriminator.load_state_dict(state["discriminator"])
        self.discriminator_optim.load_state_dict(state["optimizer"])
        self.motion.load_state_dict(state["motion_sampler"])
        self.history_size = state["history_size"]
        self.history_cursor = state["history_cursor"]
        if self.history_size > len(self.history) or len(state["history"]) != self.history_size:
            raise ValueError("AMP history capacity differs on resume")
        self.history[:self.history_size] = state["history"]
        self.empty_rounds = state["empty_rounds"]
        self.amp_round = state["amp_round"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
