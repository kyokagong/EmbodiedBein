"""Exercise real four-rank CPU collectives and non-overlapping model loading."""
from pathlib import Path
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, root):
    from libero_fsdp import load_model_sequentially

    dist.init_process_group("gloo", init_method="file://" + root + "/rendezvous", rank=rank, world_size=4)

    def load():
        with open(root + "/events", "a") as log:
            log.write(f"start {rank}\n")
        model = torch.nn.Linear(16, 16)
        model.load_state_dict(torch.load(root + "/weights.pt", weights_only=True))
        time.sleep(0.05)
        with open(root + "/events", "a") as log:
            log.write(f"end {rank}\n")
        return model

    model = load_model_sequentially(load)
    assert torch.all(model.weight == 0.5)
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        model = torch.nn.Linear(16, 16)
        model.weight.data.fill_(0.5)
        torch.save(model.state_dict(), root + "/weights.pt")
        mp.spawn(worker, args=(root,), nprocs=4, join=True)
        events = Path(root + "/events").read_text().splitlines()
        assert events == [f"{event} {rank}" for rank in range(4) for event in ("start", "end")], events
        print("PASS: four real gloo ranks load models sequentially and all retain loaded weights")
