"""CPU/Gloo checks for Flat's four-learner statistics and rank-zero receipts."""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from rpo_flat_policy import RPOFlatPolicy
import rpo_flat_worker as worker


def check_rank(rank, world_size, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/group", rank=rank,
                            world_size=world_size)
    try:
        torch.manual_seed(42)
        policy = RPOFlatPolicy(780, 23, 1, True, False, critic_obs_dim=1390)
        policy = FullyShardedDataParallel(policy, device_id=torch.device("cpu"),
                                         sharding_strategy=ShardingStrategy.NO_SHARD)
        reference = RPOFlatPolicy(780, 23, 1, True, False, critic_obs_dim=1390)
        actor = [torch.full((r + 1, 780), float(r)) for r in range(world_size)]
        critic = [torch.full((r + 1, 1390), float(2 * r + 1)) for r in range(world_size)]
        for norm in (policy.actor_norm, reference.actor_norm):
            norm.update(torch.ones(3, 780))
        for norm in (policy.critic_norm, reference.critic_norm):
            norm.update(torch.ones(3, 1390))
        reference.actor_norm.update(torch.cat(actor))
        reference.critic_norm.update(torch.cat(critic))
        policy.update_normalizers(actor[rank], critic[rank])
        for current, expected in ((policy.actor_norm, reference.actor_norm),
                                  (policy.critic_norm, reference.critic_norm)):
            assert current.count.item() == expected.count.item() == 13
            for key in ("_mean", "_var", "_std"):
                torch.testing.assert_close(getattr(current, key), getattr(expected, key))
        policy.eval()
        policy.update_normalizers(actor[rank], critic[rank])
        assert policy.actor_norm.count.item() == 13
        policy.train()

        # One learner has no selected KL samples; it must still join the reduction.
        count = rank if rank else 0
        kl = worker.distributed_kl_mean(.01 * rank * count, count, torch.device("cpu"))
        assert abs(kl - (.01 * 14 / 6)) < 1e-12
        assert worker.distributed_kl_mean(0., 0, torch.device("cpu")) is None
        rates = worker.consensus_adaptive_lrs([.001 * (rank + 1)], kl, .01, torch.device("cpu"))
        assert abs(rates[0] - .001 / 1.5) < 1e-12

        def write_once():
            path = Path(directory) / "receipt.json"
            assert not path.exists()
            path.write_text(json.dumps({"writer": rank}))

        worker.run_on_rank_zero(write_once)
        assert json.loads((Path(directory) / "receipt.json").read_text())["writer"] == 0
        try:
            worker.run_on_rank_zero(lambda: (_ for _ in ()).throw(ValueError("receipt failed")))
        except RuntimeError as exc:
            assert "receipt failed" in str(exc)
        else:
            raise AssertionError("all learners must observe a rank-zero write failure")

        output = Path(directory)
        receipt = output / "warm-start-receipt.json"
        worker.run_on_rank_zero(lambda: receipt.write_text(json.dumps({
            "schema_version": "eb/warm-start/v1", "status": "validated",
            "normalization_count": 13.,
        })))
        worker.mark_warm_start_loaded(policy, output)
        loaded = json.loads(receipt.read_text())
        assert loaded["status"] == "loaded" and loaded["world_size"] == 4
        worker.run_on_rank_zero(lambda: receipt.write_text(json.dumps({
            "schema_version": "eb/warm-start/v1", "status": "validated",
            "normalization_count": 12.,
        })))
        try:
            worker.mark_warm_start_loaded(policy, output)
        except ValueError as exc:
            assert "normalization" in str(exc)
        else:
            raise AssertionError("warm start accepted different normalization counts")
        worker.run_on_rank_zero(lambda: receipt.write_text(json.dumps({
            "schema_version": "eb/warm-start/v1", "status": "validated",
            "normalization_count": 13.,
        })))
        if rank == 3:
            policy.critic_norm.count.add_(1)
        try:
            worker.mark_warm_start_loaded(policy, output)
        except ValueError:
            pass
        else:
            raise AssertionError("a single mismatched learner must fail every rank")

        assert callable(getattr(worker, "save_rank_rng", None)), "per-rank RNG save is missing"
        assert callable(getattr(worker, "load_rank_rng", None)), "per-rank RNG load is missing"
        actor_checkpoint = output / "actor"
        torch.manual_seed(100 + rank)
        np.random.seed(200 + rank)
        random.seed(300 + rank)
        worker.save_rank_rng(actor_checkpoint)
        dist.barrier()
        assert all((actor_checkpoint / "rng" / f"rank_{r}.pt").is_file()
                   for r in range(world_size))
        expected = (torch.rand(5), np.random.random(5), [random.random() for _ in range(5)])
        samples = [None] * world_size
        dist.all_gather_object(samples, expected[0].tolist())
        assert len({tuple(sample) for sample in samples}) == world_size
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        worker.load_rank_rng(actor_checkpoint)
        torch.testing.assert_close(torch.rand(5), expected[0], rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.random(5), expected[1])
        assert [random.random() for _ in range(5)] == expected[2]
        blocked = output / "not-a-directory"
        worker.run_on_rank_zero(lambda: blocked.write_text("blocked"))
        try:
            worker.save_rank_rng(blocked if rank == 3 else actor_checkpoint)
        except RuntimeError as exc:
            assert "save failed" in str(exc)
        else:
            raise AssertionError("one learner's RNG save failure must fail every rank")
        worker.run_on_rank_zero(lambda: (actor_checkpoint / "rng" / "rank_3.pt").unlink())
        try:
            worker.load_rank_rng(actor_checkpoint)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("multi-rank resume accepted a missing RNG member")
        worker.run_on_rank_zero(lambda: (actor_checkpoint / "rng" / "rank_3.pt").write_bytes(b"invalid"))
        try:
            worker.load_rank_rng(actor_checkpoint)
        except RuntimeError as exc:
            assert "restore failed" in str(exc)
        else:
            raise AssertionError("one learner's invalid RNG file must fail every rank")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="rpo-flat-gloo-") as directory:
        mp.start_processes(check_rank, args=(4, directory), nprocs=4, start_method="fork")
    # Legacy single-learner checkpoints may retain native DCP RNG restoration.
    with tempfile.TemporaryDirectory(prefix="rpo-flat-rng-legacy-") as directory:
        assert worker.load_rank_rng(directory) is False
    print("Four-learner moments, KL/LR, receipts and independent RNG roundtrip checks passed")
