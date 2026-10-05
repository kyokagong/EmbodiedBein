"""Bound CPU model-loading peaks while retaining upstream PPO/FSDP execution."""
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from libero_fsdp import load_model_sequentially, release_cpu_heap


def discard_bootstrap_only_validity(batch):
    # LIBERO supplies no per-step validity mask. Upstream's initial bootstrap
    # nevertheless adds one all-true row; it is not a time-major training field.
    mask = batch.get("valid_step_mask")
    if (
        mask is not None
        and mask.shape[0] == 1
        and mask.shape[1] == batch["prev_logprobs"].shape[1]
        and bool(mask.all())
    ):
        batch.pop("valid_step_mask")


class LiberoPpoActor(EmbodiedFSDPActor):
    def model_provider_func(self):
        return load_model_sequentially(super().model_provider_func)

    def _process_received_rollout_batch(self, rollout_batch):
        discard_bootstrap_only_validity(rollout_batch)
        return super()._process_received_rollout_batch(rollout_batch)

    def load_param_and_grad(self, *args, **kwargs):
        super().load_param_and_grad(*args, **kwargs)
        release_cpu_heap()

    def offload_param_and_grad(self, *args, **kwargs):
        super().offload_param_and_grad(*args, **kwargs)
        release_cpu_heap()

    def save_checkpoint(self, save_path, step=0):
        from libero_checkpoint import save_checkpoint
        save_checkpoint(self, save_path, "trainer_state")

    def load_checkpoint(self, load_path):
        super().load_checkpoint(load_path)
        import os
        import torch.distributed as dist
        if dist.get_rank() == 0:
            from rlinf_entry import mark_resume_loaded
            mark_resume_loaded(os.environ["EB_OUTPUT_DIR"], load_path)


from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class LiberoRolloutWorker(MultiStepRolloutWorker):
    def reload_model(self):
        super().reload_model()
        release_cpu_heap()

    def offload_model(self):
        super().offload_model()
        release_cpu_heap()
