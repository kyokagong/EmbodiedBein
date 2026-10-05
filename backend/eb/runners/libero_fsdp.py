"""Use FSDP boundaries that OpenPI's joint expert/VLM forward actually calls."""
from omegaconf import OmegaConf


def release_cpu_heap():
    # The pinned Linux image uses glibc. Model/offload tensors can leave GiBs
    # of freed allocations in its arenas; CUDA empty_cache does not release them.
    import ctypes
    import gc
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def load_model_sequentially(factory):
    import torch.distributed as dist

    model = None
    for owner in range(dist.get_world_size()):
        if dist.get_rank() == owner:
            model = factory()
            release_cpu_heap()
        # All ranks finish loading before collective FSDP wrapping begins.
        dist.barrier()
    return model


def configure_sft_wrapping(cfg):
    # Joint forward calls these children directly, bypassing DecoderLayer.forward.
    # Keep the tied token embedding/lm_head in the root FSDP unit.
    policy = {
        "transformer_layer_cls_to_wrap": [
            "GemmaMLP", "SiglipVisionEmbeddings", "GemmaRMSNorm", "GemmaRotaryEmbedding",
        ],
        "no_split_names": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "action_in_proj", "action_out_proj", "state_proj",
            "action_time_mlp_in", "action_time_mlp_out", "time_mlp_in", "time_mlp_out",
        ],
    }
    OmegaConf.update(cfg, "actor.fsdp_config.wrap_policy", policy, merge=False, force_add=True)
