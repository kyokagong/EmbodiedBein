"""Two-rank native DCP/model/optimizer roundtrip across separated save phases."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, root):
    from omegaconf import OmegaConf
    import libero_checkpoint as owned
    import rlinf.hybrid_engines.fsdp.strategy.base as base
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    # The real checkpoint implementation otherwise synchronizes CUDA; this
    # test has CPU models and real Gloo collectives, so only that hook is replaced.
    base.clear_memory = owned.release_cpu_heap
    dist.init_process_group('gloo', init_method='file://' + root + '/rendezvous', rank=rank, world_size=2)
    torch.manual_seed(17)
    model = torch.nn.Linear(4, 2)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 1.0)
    model(torch.ones(2, 4)).sum().backward()
    optim.step()
    scheduler.step()
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    expected_moments = [state['exp_avg'].clone() for state in optim.state.values()]
    expected_rng = torch.get_rng_state().clone()
    expected_epoch = scheduler.last_epoch
    events = []
    real_trim = owned.release_cpu_heap
    owned.release_cpu_heap = lambda: (events.append('trim'), real_trim())[-1]
    def full_state(**kwargs):
        assert events[-1] == 'trim'
        assert (Path(root) / 'global_step_1/actor/dcp_checkpoint/.metadata').is_file()
        events.append('full')
        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    worker = SimpleNamespace(
        is_weight_offloaded=False, is_optimizer_offloaded=False,
        model=model, optimizer=optim, lr_scheduler=scheduler, _strategy=FSDPStrategy,
        _cfg=OmegaConf.create({'fsdp_config': {'save_full_model_weights': True}}),
        get_model_state_dict=full_state,
    )
    os.environ['EB_RUN_ID'] = 'run_smoke'
    os.environ['EB_RESUME_FINGERPRINT'] = 'sha256:' + 'a' * 64
    checkpoint = root + '/global_step_1/actor'
    owned.save_checkpoint(worker, checkpoint, 'trainer_state')
    assert events == ['trim', 'trim', 'full', 'trim'], events
    full = torch.load(checkpoint + '/model_state_dict/full_weights.pt', weights_only=True)
    for key, value in expected.items():
        torch.testing.assert_close(full[key], value)
    for parameter in model.parameters():
        parameter.data.zero_()
    for state in optim.state.values():
        state['exp_avg'].zero_()
    torch.manual_seed(99)
    scheduler.last_epoch = 99
    FSDPStrategy.load_checkpoint(model, optim, scheduler, checkpoint)
    for key, value in expected.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    for state, expected_value in zip(optim.state.values(), expected_moments):
        torch.testing.assert_close(state['exp_avg'], expected_value)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert scheduler.last_epoch == expected_epoch
    if rank == 0:
        manifest = json.loads((Path(root) / 'global_step_1/resume-manifest.json').read_text())
        assert manifest['mode'] == 'trainer_state' and manifest['step'] == 1
    dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(run, args=(root,), nprocs=2, join=True)
    print('PASS: native DCP and full weights roundtrip on two ranks; cleanup separates save phases')
