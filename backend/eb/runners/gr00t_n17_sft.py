"""NVIDIA GR00T-N1.7 hooks driven by RLInf's SFT loop."""
import json
import os
import shutil
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from rlinf.config import EMBODIED_MODEL, SupportedModel
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


GR00T_N1D7 = SupportedModel.models.get("gr00t_n1d7") or SupportedModel.register("gr00t_n1d7")
SupportedModel.GR00T_N1D7 = GR00T_N1D7
EMBODIED_MODEL.add(GR00T_N1D7)


def _pyav_frames(path, indices, **_kwargs):
    import av

    wanted = {int(value): slot for slot, value in enumerate(indices)}
    frames = [None] * len(indices)
    with av.open(str(path)) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number in wanted:
                frames[wanted[number]] = frame.to_ndarray(format="rgb24")
            if number >= max(wanted):
                break
    if any(frame is None for frame in frames):
        raise ValueError(f"Video {path} ended before requested frames")
    return np.stack(frames)


def _runtime_data_view(source: Path, output: Path) -> Path:
    if output.exists():
        return output
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    for name in ("data", "videos"):
        (output / name).symlink_to(source / name, target_is_directory=True)
    groups = {
        "left_arm": {"start": 0, "end": 6}, "left_gripper": {"start": 6, "end": 7},
        "right_arm": {"start": 7, "end": 13}, "right_gripper": {"start": 13, "end": 14},
    }
    modality = {
        "state": groups, "action": groups,
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (output / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    return output


def _runtime_model_view(source: Path, cosmos: Path, output: Path) -> Path:
    if output.exists():
        return output
    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.name != "config.json":
            (output / item.name).symlink_to(item, target_is_directory=item.is_dir())
    config = json.loads((source / "config.json").read_text())
    config["model_name"] = str(cosmos)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return output


class Gr00tN17SftWorker(FSDPVlaSftWorker):
    def _build_pipeline(self, data_paths):
        if hasattr(self, "_gr00t_pipeline"):
            return self._gr00t_pipeline
        from gr00t.configs.base_config import get_default_config
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.types import ModalityConfig
        import gr00t.model.gr00t_n1d7.gr00t_n1d7 as gr00t_model
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
        from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
        import gr00t.data.dataset.lerobot_episode_loader as episode_loader

        # Upstream recognizes the repository id by name before loading it. The
        # platform pins the same repository at a local, read-only path.
        gr00t_model.get_backbone_cls = lambda _config: Qwen3Backbone
        episode_loader.get_frames_by_indices = _pyav_frames

        output = Path(os.environ["EB_OUTPUT_DIR"])
        data = output / "gr00t-data"
        model = output / "gr00t-parent"
        if self._rank == 0:
            _runtime_data_view(Path(str(data_paths)), data)
            _runtime_model_view(Path(str(self.cfg.actor.model.model_path)),
                                Path(str(self.cfg.actor.model.cosmos_path)), model)
        torch.distributed.barrier()
        keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
        modality = {
            "video": ModalityConfig([0], ["cam_high", "cam_left_wrist", "cam_right_wrist"]),
            "state": ModalityConfig([0], keys),
            "action": ModalityConfig(list(range(40)), keys),
            "language": ModalityConfig([0], ["task"]),
        }
        # The upstream stats path reads the process-wide registry even when the
        # training config already carries a custom embodiment definition.
        MODALITY_CONFIGS["new_embodiment"] = modality
        config = get_default_config().load_dict({"data": {"datasets": [{
            "dataset_paths": [str(data)], "mix_ratio": 1.0, "embodiment_tag": "new_embodiment",
        }]}})
        config.data.modality_configs = {"new_embodiment": modality}
        config.data.num_shards_per_epoch = 64
        config.data.shard_size = 8
        config.data.episode_sampling_rate = 0.01
        config.data.video_backend = "pyav"
        config.model.model_name = str(self.cfg.actor.model.cosmos_path)
        config.model.use_relative_action = False
        config.model.use_flash_attention = False
        config.training.start_from_checkpoint = str(model)
        config.training.transformers_local_files_only = True
        config.training.eval_strategy = "no"
        config.validate()
        cfg_dir = output / "gr00t-experiment"
        cfg_dir.mkdir(exist_ok=True)
        pipeline = Gr00tN1d7Pipeline(config, cfg_dir)
        pipeline.setup()
        self._gr00t_pipeline = pipeline
        return pipeline

    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset:
            raise ValueError("GR00T-N1.7 smoke recipe has no validation split")
        pipeline = self._build_pipeline(data_paths)
        loader = DataLoader(pipeline.train_dataset, batch_size=self.micro_batch_size,
                            collate_fn=pipeline.data_collator, num_workers=0, pin_memory=True)
        return loader, {"format": "robotwin_aloha_lerobot", "modality": "new_embodiment"}

    def model_provider_func(self):
        return self._gr00t_pipeline.model

    def get_train_model_output(self, batch):
        batch = {key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
                 for key, value in batch.items()}
        with self.amp_context:
            output = self.model(**batch)
        loss = output["loss"]
        return loss, {"loss": loss.detach().item()}

    def get_max_steps_per_epoch(self):
        # NVIDIA's sharded mixture is iterable and intentionally has no __len__.
        return 64

    def get_eval_model_output(self, batch):
        raise NotImplementedError("GR00T-N1.7 smoke recipe has no validation split")
