"""OpenDM DM0.5 model and data hooks for RLInf's SFT worker."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import BuildAction, ChatTokenization, LoadImages, Normalize, PadAction, Pipeline, PixelTransform
from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
from opendm.model.dm05.dm05_lora import DM05LoraConfig, apply_lora_to_dm05_model
from rlinf.config import EMBODIED_MODEL, SupportedModel
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from robotwin_dm05_view import rebase

OPENDM_DM05 = SupportedModel.models.get("opendm_dm05") or SupportedModel.register("opendm_dm05")
SupportedModel.OPENDM_DM05 = OPENDM_DM05
EMBODIED_MODEL.add(OPENDM_DM05)


class Dm05SftWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        root = Path(str(data_paths))
        if not (root / "index_cache.json").is_file() or not (root / "norm_stats.json").is_file():
            raise ValueError("DM0.5 requires a frozen OpenDM JSONL/video view with norm_stats.json")
        runtime_root = Path("/workspace/output/dm05-runtime-view")
        if self._rank == 0:
            rebase(root, runtime_root)
        torch.distributed.barrier()
        processor = AutoProcessor.from_pretrained(self.cfg.actor.model.model_path, local_files_only=True)
        pipeline = Pipeline([
            BuildAction(50, ActionMode.ABSOLUTE),
            LoadImages(["images_1", "images_2", "images_3"], image_dir="/"),
            PixelTransform(NoAugmentationPipeline()),
            Normalize(str(root / "norm_stats.json"), ["state", "action"], use_quantiles=True),
            ChatTokenization(processor, 256, 1024, ["Head", "Left wrist", "Right wrist"], add_state=True),
            PadAction(32),
        ])
        dataset = JsonlDataset(str(runtime_root), pipeline, "robotwin2_generalist", {
            "robot_type": RobotType.ALOHA_ROBOTWIN2.value,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2],
        })
        sampler = DistributedSampler(dataset, self._world_size, self._rank, shuffle=not eval_dataset)
        loader = DataLoader(dataset, batch_size=self.micro_batch_size, sampler=sampler,
                            collate_fn=TrainingCollator(processor.tokenizer.pad_token_id, 1024))
        return loader, {"format": "opendm_robotwin2", "norm_stats": str(root / "norm_stats.json")}

    def model_provider_func(self):
        path = self.cfg.actor.model.model_path
        config = DM05Config.from_pretrained(path, local_files_only=True)
        config.vlm_config._attn_implementation = {"": "sdpa", "text_config": "sdpa", "vision_config": "sdpa"}
        config.vlm_config.text_config._attn_implementation = "sdpa"
        config.vlm_config.vision_config._attn_implementation = "sdpa"
        model = DM05ForConditionalGeneration.from_pretrained(
            path, config=config, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
        )
        model.set_attention_implementation(llm_attn_implementation="eager",
                                           vision_attn_implementation="sdpa",
                                           action_attn_implementation="sdpa", bf16=True)
        model.enable_gradient_checkpointing(vlm_gradient_checkpointing=True,
                                            ae_gradient_checkpointing=True)
        model.freeze_vlm_embedding()
        return apply_lora_to_dm05_model(
            model, DM05LoraConfig(dump_trainable_path=None), base_model_name_or_path=str(path)
        )

    def get_train_model_output(self, batch):
        batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
        with self.amp_context:
            output = self.model(**batch)
        return output.loss, {"loss": output.loss.detach().item(), "fm_loss": output.fm_loss.detach().item()}

    def get_max_steps_per_epoch(self):
        return max(1, len(self.data_loader) // self.gradient_accumulation) if self.data_loader else 0

    def get_eval_model_output(self, batch):
        raise NotImplementedError("DM0.5 smoke recipe has no validation split")
