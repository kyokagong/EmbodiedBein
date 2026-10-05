import torch
from omegaconf import OmegaConf
from libero_fsdp import configure_sft_wrapping
from torch.distributed.fsdp.wrap import _recursive_wrap
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config, OpenPi0ForRLActionPrediction
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
c=OpenPi0Config(**get_openpi_config('pi05_libero').model.__dict__)
c.__dict__['train_expert_only']=True
with torch.device('meta'):
    model=OpenPi0ForRLActionPrediction(c)
model.freeze_vlm()
model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
seen=set()
units=[]
class Audit(torch.nn.Module):
    def __init__(self,module,**kwargs):
        super().__init__()
        self.module=module
        own=[p for p in module.parameters() if id(p) not in seen]
        seen.update(map(id,own))
        if own:
            units.append((module.__class__.__name__,sum(p.numel() for p in own),{p.requires_grad for p in own},{str(p.dtype) for p in own}))
cfg=OmegaConf.create({})
configure_sft_wrapping(cfg)
_recursive_wrap(model,get_fsdp_wrap_policy(model,config=cfg.actor.fsdp_config,model_type='openpi'),Audit,set(),set())
failures=[unit for unit in units if len(unit[2])>1 or len(unit[3])>1]
print('FSDP exact policy units',len(units),'mixed units',failures)
assert not failures
assert not any(unit[0] == 'GemmaDecoderLayer' for unit in units)
