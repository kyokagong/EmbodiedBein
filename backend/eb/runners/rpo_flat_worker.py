"""Official Flat PPO settings, retaining RLinf's trainer and FSDP lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


def adaptive_kl_lr(lr, kl, desired_kl):
    if kl > 2 * desired_kl:
        return max(1e-5, lr / 1.5)
    if 0 < kl < desired_kl / 2:
        return min(1e-2, lr * 1.5)
    return lr


def clipped_value_mse(values, returns, previous, clip):
    clipped = previous + (values - previous).clamp(-clip, clip)
    return torch.maximum((returns - values).square(), (returns - clipped).square())


def distributed_kl_mean(total, count, device):
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(values)
    return (values[0] / values[1]).item() if values[1] > 0 else None


def consensus_adaptive_lrs(rates, kl, desired_kl, device):
    values = torch.tensor([adaptive_kl_lr(rate, kl, desired_kl) for rate in rates],
                          dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(values, src=0)
    return values.tolist()


def run_on_rank_zero(operation):
    """Wait for all shards, then publish one receipt and share write failures."""
    if not torch.distributed.is_initialized():
        return operation()
    torch.distributed.barrier()
    error = [None]
    if torch.distributed.get_rank() == 0:
        try:
            operation()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    torch.distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])


def _all_rng_ranks_valid(valid, device):
    flag = torch.tensor(int(valid), device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def save_rank_rng(checkpoint, device="cpu"):
    from rlinf.utils.utils import get_rng_state

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = Path(checkpoint) / "rng" / f"rank_{rank}.pt"
    temporary = path.with_suffix(".pt.tmp")
    error = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(get_rng_state(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if not _all_rng_ranks_valid(error is None, device):
        raise RuntimeError(f"per-rank RNG save failed: {error or 'another learner failed'}")


def load_rank_rng(checkpoint, device="cpu"):
    from rlinf.utils.utils import set_rng_state

    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    path = Path(checkpoint) / "rng" / f"rank_{rank}.pt"
    if not _all_rng_ranks_valid(path.is_file(), device):
        if world_size > 1:
            raise FileNotFoundError("per-rank RNG checkpoint is missing for one or more learners")
        return False  # Legacy single-learner DCP checkpoints already restored RNG.
    error = None
    try:
        set_rng_state(torch.load(path, map_location="cpu", weights_only=False))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if not _all_rng_ranks_valid(error is None, device):
        raise RuntimeError(f"per-rank RNG restore failed: {error or 'another learner failed'}")
    return True


def mark_warm_start_loaded(model, output):
    from rlinf_entry import atomic_json

    path = Path(output) / "warm-start-receipt.json"
    try:
        receipt = json.loads(path.read_text())
        expected = receipt["normalization_count"]
        valid = (receipt.get("schema_version") == "eb/warm-start/v1"
                 and receipt.get("status") == "validated" and expected > 0
                 and model.actor_norm.count.item() == model.critic_norm.count.item() == expected)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        valid = False
    flag = torch.tensor(int(valid), device=next(model.parameters()).device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if not flag.item():
        raise ValueError("warm-start normalization counts do not match the validated receipt")
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    run_on_rank_zero(lambda: atomic_json(path, {**receipt, "status": "loaded", "world_size": world_size}))


class AdaptiveKLScheduler(LRScheduler):
    """Preserve adaptive rates when RLinf advances/saves its round scheduler."""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class RPOFlatActor(EmbodiedFSDPActor):
    def init_worker(self):
        super().init_worker()
        if self.cfg.actor.model.num_action_chunks != 1:
            raise ValueError("RPO Flat requires one action chunk")
        if os.environ.get("EB_WARM_START_RECEIPT"):
            mark_warm_start_loaded(self.model, self.cfg.runner.logger.log_path)
        self._flat_kl_sum = 0.0
        self._flat_kl_count = 0
        self._flat_kl_metrics = []

    def build_lr_scheduler(self, optimizer, optim_config=None, last_epoch=-1):
        return AdaptiveKLScheduler(optimizer, last_epoch=last_epoch)

    def save_checkpoint(self, save_path: str, step: int = 0):
        super().save_checkpoint(save_path, step)
        save_rank_rng(save_path, next(self.model.parameters()).device)
        from rlinf_entry import prune_old_checkpoints, write_resume_manifest
        checkpoint = Path(save_path).parent
        def publish():
            write_resume_manifest(checkpoint, "trainer_state")
            prune_old_checkpoints(checkpoint.parent)
        run_on_rank_zero(publish)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        load_rank_rng(load_path, next(self.model.parameters()).device)
        from rlinf_entry import mark_resume_loaded
        run_on_rank_zero(lambda: mark_resume_loaded(Path(self.cfg.runner.logger.log_path),
                                                   Path(load_path).parent))

    def _record_flat_kl(self, kl, loss_mask=None):
        if loss_mask is not None:
            kl = kl[loss_mask.reshape(kl.shape).bool()]
        self._flat_kl_sum += kl.sum().item()
        self._flat_kl_count += kl.numel()

    def optimizer_step(self):
        device = next(self.model.parameters()).device
        kl = distributed_kl_mean(self._flat_kl_sum, self._flat_kl_count, device)
        if kl is not None:
            rates = consensus_adaptive_lrs([group["lr"] for group in self.optimizer.param_groups],
                                          kl, self.cfg.algorithm.desired_kl, device)
            for group, rate in zip(self.optimizer.param_groups, rates):
                group["lr"] = rate
            self._flat_kl_metrics.append(kl)
        self._flat_kl_sum, self._flat_kl_count = 0.0, 0
        return super().optimizer_step()

    def run_training(self):
        self._flat_kl_metrics = []
        metrics = super().run_training()
        inputs = self.rollout_batch["forward_inputs"]
        self.model.update_normalizers(inputs["raw_states"], inputs["raw_critic_states"])
        metrics["rpo_flat/normalizer_count"] = self.model.actor_norm.count.item()
        metrics["rpo_flat/adaptive_kl"] = sum(self._flat_kl_metrics) / max(1, len(self._flat_kl_metrics))
        return metrics
