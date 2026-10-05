"""Separate native DCP and full-weight save peaks without dropping training state."""
from pathlib import Path

import torch
import torch.distributed as dist

from libero_fsdp import release_cpu_heap
from rlinf_entry import prune_old_checkpoints, write_resume_manifest


def save_checkpoint(worker, save_path, mode):
    restore_weights = worker.is_weight_offloaded
    restore_optimizer = worker.is_optimizer_offloaded
    try:
        if restore_weights:
            worker.load_param_and_grad(worker.device)
        if restore_optimizer:
            worker.load_optimizer(worker.device)
        release_cpu_heap()
        worker._strategy.save_checkpoint(
            worker.model, worker.optimizer, worker.lr_scheduler, save_path,
            save_full_model_weights=False,
        )
        # DCP's CPU state dictionaries are now out of scope on every rank.
        release_cpu_heap()
        dist.barrier()
        if worker._cfg.fsdp_config.get("save_full_model_weights", True):
            state = worker.get_model_state_dict(cpu_offload=True, full_state_dict=True)
            try:
                if dist.get_rank() == 0:
                    destination = Path(save_path) / "model_state_dict"
                    destination.mkdir(parents=True, exist_ok=True)
                    torch.save(state, destination / "full_weights.pt")
            finally:
                del state
                release_cpu_heap()
            dist.barrier()
        if mode == "exact":
            from rlinf.utils.utils import get_rng_state
            from torchdata.stateful_dataloader import StatefulDataLoader

            if not isinstance(worker.data_loader, StatefulDataLoader):
                raise TypeError("exact SFT resume requires StatefulDataLoader")
            states = [None] * worker._world_size
            dist.all_gather_object(states, worker.data_loader.state_dict())
            rng_states = [None] * worker._world_size
            dist.all_gather_object(rng_states, get_rng_state())
            if dist.get_rank() == 0:
                torch.save(states, Path(save_path) / "data.pt")
                torch.save(rng_states, Path(save_path) / "rng.pt")
            dist.barrier()
    finally:
        if restore_weights:
            worker.offload_param_and_grad()
        if restore_optimizer:
            worker.offload_optimizer()
    if dist.get_rank() == 0:
        checkpoint = Path(save_path).parent
        write_resume_manifest(checkpoint, mode)
        prune_old_checkpoints(checkpoint.parent)
