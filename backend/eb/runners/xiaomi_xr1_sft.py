"""Xiaomi-Robotics-1 model and exact RoboTwin ALOHA FK data bridge for RLInf."""
import bisect
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rlinf.config import EMBODIED_MODEL, SupportedModel
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

from robotwin_aloha_fk import encode_xr1_action


XIAOMI_XR1 = SupportedModel.models.get("xiaomi_xr1") or SupportedModel.register("xiaomi_xr1")
SupportedModel.XIAOMI_XR1 = XIAOMI_XR1
EMBODIED_MODEL.add(XIAOMI_XR1)


def _episode_rows(root):
    return [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]


def _read_episode(root, episode):
    import pyarrow.parquet as pq

    path = root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["observation.state", "action"])
    return (np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            np.asarray(table["action"].to_pylist(), dtype=np.float32))


def _normalization(root, destination, action_length=30):
    states, actions = [], []
    for row in _episode_rows(root)[:16]:
        observation, target = _read_episode(root, int(row["episode_index"]))
        for frame in np.linspace(0, len(observation) - 1, min(64, len(observation)), dtype=int):
            future = target[frame:frame + action_length]
            if len(future) < action_length:
                future = np.concatenate((future, np.repeat(future[-1:], action_length - len(future), axis=0)))
            encoded, _ = encode_xr1_action(observation[frame], future)
            state = np.zeros((1, 60), dtype=np.float32)
            state[0, :6], state[0, 7] = observation[frame, :6], observation[frame, 6]
            state[0, 8:14], state[0, 15] = observation[frame, 7:13], observation[frame, 13]
            actions.append(encoded)
            states.append(state)
    actions, states = np.asarray(actions), np.asarray(states)
    mean, std = actions.mean(0), actions.std(0)
    q01, q99 = np.quantile(states, 0.01, axis=0)[0], np.quantile(states, 0.99, axis=0)[0]
    active = np.array([*range(0, 6), 7, *range(8, 14), 15])
    equal = q99[active] <= q01[active]
    q01[active[equal]] -= 1e-6
    q99[active[equal]] += 1e-6
    destination.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                       "q01": [q01.tolist()], "q99": [q99.tolist()]}) + "\n")


class RoboTwinXr1Dataset(Dataset):
    def __init__(self, root, stats):
        from mibot.utils.io import validate_quantiles, validate_stats

        self.root = Path(root)
        self.rows = _episode_rows(self.root)
        self.ends = np.cumsum([int(row["length"]) for row in self.rows]).tolist()
        self.mean, self.std = validate_stats(stats["mean"], stats["std"], 30)
        self.q01, self.q99 = validate_quantiles(stats["q01"], stats["q99"])

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=4)
    def _episode(self, episode):
        return _read_episode(self.root, episode)

    def __getitem__(self, index):
        import av
        from PIL import Image
        from mibot.data.datasets.json_dataset import JsonDataset
        from mibot.utils.io import normalize_action, normalize_quantile

        slot = bisect.bisect_right(self.ends, index)
        row = self.rows[slot]
        episode = int(row["episode_index"])
        frame = index - (self.ends[slot - 1] if slot else 0)
        observation, target = self._episode(episode)
        steps = min(30, len(target) - frame)
        future = target[frame:frame + steps]
        if steps < 30:
            future = np.concatenate((future, np.repeat(future[-1:], 30 - steps, axis=0)))
        action, action_mask = encode_xr1_action(observation[frame], future)
        action_mask[steps:] = False
        state = np.zeros((1, 60), dtype=np.float32)
        state[0, :6], state[0, 7] = observation[frame, :6], observation[frame, 6]
        state[0, 8:14], state[0, 15] = observation[frame, 7:13], observation[frame, 13]
        images = []
        chunk = episode // 1000
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            video = self.root / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode:06d}.mp4"
            with av.open(str(video)) as container:
                image = next(value for number, value in enumerate(container.decode(video=0)) if number == frame)
            images.append(image.to_image())
        instruction = row["tasks"][0]
        prompt = ("The following observations are captured from multiple views.\n# Ego View\n<image>\n"
                  "# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
                  f"Generate robot actions for the task:\n{instruction} /no_cot")
        messages = JsonDataset._messages([
            {"from": "human", "value": prompt}, {"from": "gpt", "value": "<cot></cot>"},
            {"from": "human", "value": "Robot state: <state>"},
            {"from": "gpt", "value": "".join(f"<a_{i}>" for i in range(steps)) + "<score>"},
        ], JsonDataset.__new__(JsonDataset)._augment(images))
        action = normalize_action(action, self.mean, self.std)
        state = normalize_quantile(state, self.q01, self.q99)
        return {"messages": messages, "action": torch.from_numpy(action),
                "action_mask": torch.from_numpy(action_mask), "state": torch.from_numpy(state),
                "vlm_action_target": torch.from_numpy(action[:steps]),
                "vlm_action_mask": torch.from_numpy(action_mask[:steps]),
                "vlm_action_actual_length": steps}


class XiaomiXr1SftWorker(FSDPVlaSftWorker):
    def _prepare_local_qwen(self):
        output = Path(os.environ["EB_OUTPUT_DIR"])
        alias = output / "Qwen/Qwen3-VL-4B-Instruct"
        if self._rank == 0:
            alias.parent.mkdir(exist_ok=True)
            if not alias.exists():
                alias.symlink_to(Path(str(self.cfg.actor.model.processor_path)), target_is_directory=True)
        torch.distributed.barrier()
        os.chdir(output)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset:
            raise ValueError("Xiaomi-Robotics-1 smoke recipe has no validation split")
        self._prepare_local_qwen()
        from mibot.data.collate.custom_collate import CustomCollate

        root = Path(str(data_paths))
        stats_path = Path(os.environ["EB_OUTPUT_DIR"]) / "xr1-normalization.json"
        if self._rank == 0 and not stats_path.exists():
            _normalization(root, stats_path)
        torch.distributed.barrier()
        dataset = RoboTwinXr1Dataset(root, json.loads(stats_path.read_text()))
        sampler = DistributedSampler(dataset, self._world_size, self._rank, shuffle=True,
                                     seed=int(self.cfg.actor.seed), drop_last=True)
        loader = DataLoader(dataset, batch_size=self.micro_batch_size, sampler=sampler,
                            collate_fn=CustomCollate(), num_workers=0, pin_memory=True, drop_last=True)
        return loader, {"format": "robotwin_aloha_xr1_60d", "normalization": str(stats_path)}

    def model_provider_func(self):
        self._prepare_local_qwen()
        from mibot.models.VLA.XR1 import xr1
        from mibot.models.VLM.qwen3vl import Qwen3VLForConditionalGeneration

        original = Qwen3VLForConditionalGeneration._from_config

        def local_from_config(config, **kwargs):
            kwargs["attn_implementation"] = "sdpa"
            return original(config, **kwargs)

        Qwen3VLForConditionalGeneration._from_config = local_from_config
        with torch.device("meta"):
            model = xr1(ffn_gradient_checkpointing=True, async_train=False)
        checkpoint = torch.load(Path(str(self.cfg.actor.model.model_path)) / "model_states.pt",
                                map_location="cpu", mmap=True, weights_only=False)["module"]
        state = {key.removeprefix("model."): value for key, value in checkpoint.items()}
        info = model.load_state_dict(state, strict=True, assign=True)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(f"XR-1 checkpoint mismatch: {info}")
        for module in model.modules():
            if type(module).__name__ == "Qwen3VLVisionRotaryEmbedding" and module.inv_freq.is_meta:
                dim = module.inv_freq.numel() * 2
                module.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            elif type(module).__name__ == "Qwen3VLTextRotaryEmbedding" and module.inv_freq.is_meta:
                module.inv_freq, module.attention_scaling = module.rope_init_fn(module.config, "cpu")
                module.original_inv_freq = module.inv_freq
        model.beta = torch.distributions.Beta(1.5, 1.0)
        sink = torch.nn.Module()
        sink.register_parameter("weight", model.sink.weight)
        model.sink = sink
        # RoboTwin smoke/post-training updates the XR-1 action stack while the
        # 4B vision-language backbone stays frozen.
        model.vlm.requires_grad_(False)
        return model

    def get_train_model_output(self, batch):
        batch = {key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
                 for key, value in batch.items()}
        with self.amp_context:
            output = self.model(batch, return_loss=True)
        return output["loss"], {key: value.detach().item() for key, value in output.items()}

    def get_eval_model_output(self, batch):
        raise NotImplementedError("Xiaomi-Robotics-1 smoke recipe has no validation split")
