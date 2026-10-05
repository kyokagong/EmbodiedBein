"""Build-time self-check for the patched RLInf RoboTwin policy path."""
from concurrent.futures import Future as ConcurrentFuture

import torch
import numpy as np

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.scheduler.collective.async_work import AsyncChannelWork
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


worker = object.__new__(MultiStepRolloutWorker)
calls = []


def predict(obs, **_):
    calls.append(len(obs["states"]))
    marker = obs["states"][:, :1, None]
    return marker.expand(-1, 50, 14).clone(), {}


worker._predict_rollout_actions = predict
incoming = [{"obs": {
    "main_images": torch.zeros(4, 8, 8, 3, dtype=torch.uint8),
    "wrist_images": torch.zeros(4, 2, 8, 8, 3, dtype=torch.uint8),
    "states": torch.arange(rank * 4, rank * 4 + 4)[:, None],
    "task_descriptions": [f"rank-{rank}"] * 4,
}} for rank in range(4)]
merged = worker._merge_obs_batches(incoming)["obs"]
actions = worker._predict_eval_micro_batches(merged, 4)
parts = worker._split_actions(actions, [4, 4, 4, 4])
assert actions.shape == (16, 50, 14)
assert calls == [4, 4, 4, 4]
assert [part.shape for part in parts] == [(4, 50, 14)] * 4
assert [int(part[0, 0, 0]) for part in parts] == [0, 4, 8, 12]

calls.clear()
tail = worker._predict_eval_micro_batches(merged | {"states": merged["states"][:10],
    "main_images": merged["main_images"][:10], "wrist_images": merged["wrist_images"][:10],
    "task_descriptions": merged["task_descriptions"][:10]}, 4)
assert tail.shape == (10, 50, 14) and calls == [4, 4, 2]
try:
    worker._predict_eval_micro_batches(merged, 0)
except ValueError:
    pass
else:
    raise AssertionError("zero micro-batch must be rejected")

assert [EnvWorker.eval_rank_slice(6, rank, 4) for rank in range(4)] == [
    (0, 2), (2, 4), (4, 5), (5, 6)]
image = np.zeros((8, 8, 3), dtype=np.uint8)
assert RoboTwinEnv.center_and_crop(object.__new__(RoboTwinEnv), image) is image

channel_work = object.__new__(AsyncChannelWork)
channel_work._future = torch.futures.Future()
failed = ConcurrentFuture()
failed.set_exception(RuntimeError("channel worker died"))
channel_work._complete(failed)
try:
    channel_work._future.wait()
except RuntimeError as exc:
    assert "channel worker died" in str(exc)
else:
    raise AssertionError("channel failures must reach the waiting evaluation")
