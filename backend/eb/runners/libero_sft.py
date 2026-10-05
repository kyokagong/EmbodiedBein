"""Job-container adapter for official LeRobot v3 LIBERO and RLinf OpenPI."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from openpi import transforms
from openpi.models.model import Observation
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from torchdata.stateful_dataloader import StatefulDataLoader


class LiberoDataset:
    def __init__(self, root, config, episodes=None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info = json.loads((Path(root) / "meta/info.json").read_text())
        if not str(info.get("codebase_version", "")).startswith("v3"):
            raise ValueError("Official LIBERO adapter requires LeRobot v3")
        required = {"observation.images.image", "observation.images.image2", "observation.state", "action"}
        if not required.issubset(info["features"]):
            raise ValueError("LIBERO camera/state/action schema mismatch")
        self.dataset = LeRobotDataset(
            "lerobot/libero", root=root, episodes=episodes,
            delta_timestamps={"action": [i / info["fps"] for i in range(10)]},
            download_videos=False, video_backend="pyav",
        )
        self.transform = transforms.compose([
            transforms.RepackTransform({
                "observation/image": "observation.images.image",
                "observation/wrist_image": "observation.images.image2",
                "observation/state": "observation.state", "actions": "action", "prompt": "task",
            }),
            *config.data_transforms.inputs,
            transforms.Normalize(config.norm_stats, use_quantiles=config.use_quantile_norm),
            *config.model_transforms.inputs,
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = {key: value.numpy() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        return self.transform(sample)


class LiberoLoader(StatefulDataLoader):
    def __init__(self, dataset, data_config, **kwargs):
        self.config = data_config
        self._eb_epoch = 0
        super().__init__(dataset, **kwargs)

    def data_config(self):
        return self.config

    def __iter__(self):
        self.sampler.set_epoch(self._eb_epoch)
        for batch in super().__iter__():
            yield Observation.from_dict(batch), batch["actions"]
        self._eb_epoch += 1

    def state_dict(self):
        state = super().state_dict()
        state["_eb_epoch"] = self._eb_epoch
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        self._eb_epoch = int(state.pop("_eb_epoch", 0))
        super().load_state_dict(state)


def collate(samples):
    import jax
    return jax.tree.map(lambda *values: torch.from_numpy(np.stack(values)), *samples)


class LiberoSftWorker(FSDPVlaSftWorker):
    def model_provider_func(self):
        from libero_fsdp import load_model_sequentially
        return load_model_sequentially(super().model_provider_func)

    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset:
            raise ValueError("LIBERO held-out data is evaluation-only; upstream SFT evaluation is unsupported")
        model = self.cfg.actor.model
        if model.openpi.config_name != "pi05_libero":
            raise ValueError("LIBERO adapter requires pi05_libero (10-step raw delta actions)")
        config = get_openpi_config("pi05_libero", model_path=model.model_path)
        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.norm_stats is None:
            raise ValueError("Base model must include train-only normalization statistics")
        episodes = self.cfg.data.get("train_episode_ids")
        if episodes is not None:
            episodes = list(episodes)
            if not episodes or any(type(value) is not int or value < 0 for value in episodes):
                raise ValueError("train_episode_ids must be a nonempty list of episode indices")
        dataset = LiberoDataset(str(data_paths), data_config, episodes)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=self._world_size, rank=self._rank,
            shuffle=True, seed=int(self.cfg.actor.seed), drop_last=True,
        )
        loader = LiberoLoader(dataset, data_config, batch_size=self.cfg.actor.micro_batch_size,
            sampler=sampler, drop_last=True, num_workers=0, collate_fn=collate)
        if len(loader) == 0:
            raise ValueError("Training dataset is smaller than the distributed batch")
        return loader, data_config

    def get_max_steps_per_epoch(self):
        return max(1, len(self.data_loader) // self.gradient_accumulation)

    def save_checkpoint(self, save_path, step=0):
        from libero_checkpoint import save_checkpoint
        save_checkpoint(self, save_path, "exact")

    def load_checkpoint(self, load_path):
        super().load_checkpoint(load_path)
        if torch.distributed.get_rank() == 0:
            from rlinf_entry import mark_resume_loaded
            mark_resume_loaded(Path(os.environ["EB_OUTPUT_DIR"]), load_path)
