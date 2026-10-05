"""Measure actual checkpoint-loader heap retention without allocating a GPU."""
import gc
import json
import time
import torch
from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model
from libero_fsdp import release_cpu_heap

torch.set_num_threads(2)
def report(stage):
    with open('/proc/self/status') as stream:
        rss = next(line.strip() for line in stream if line.startswith('VmRSS:'))
    print(json.dumps({'stage': stage, 'rss': rss, 'elapsed': time.monotonic()-start}), flush=True)

start = time.monotonic()
cfg = OmegaConf.load('/previous/resolved-config.yaml').actor.model
report('before_load')
model = get_model(cfg, torch_dtype=torch.bfloat16)
report('live_model')
samples = [(p, p.detach().flatten()[:4].clone()) for p in model.parameters()]
release_cpu_heap()
for parameter, expected in samples:
    torch.testing.assert_close(parameter.detach().flatten()[:4], expected)
report('live_model_after_trim')
del samples
model.to('meta')
gc.collect()
report('released_cpu_storage')
release_cpu_heap()
report('released_cpu_storage_after_trim')
