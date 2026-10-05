"""Exact LingBot-VLA-2 model/data hooks driven by RLInf's SFT loop."""
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from rlinf.config import EMBODIED_MODEL, SupportedModel
from rlinf.utils.utils import collect_param_names_need_sync
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


LINGBOT_VLA2 = SupportedModel.models.get("lingbot_vla2") or SupportedModel.register("lingbot_vla2")
SupportedModel.LINGBOT_VLA2 = LINGBOT_VLA2
EMBODIED_MODEL.add(LINGBOT_VLA2)


def _official_config(model_path: str, tokenizer_path: str):
    from lingbotvla.models.config_registry import get_config_registry

    path = Path(__file__).parent / "config/lingbot_vla2_robotwin.yaml"
    value = yaml.safe_load(path.read_text())
    value["model"].update(model_path=model_path, config_path=model_path,
                          tokenizer_path=tokenizer_path, vit_attn_implementation="eager")
    value["train"].update(align_params={}, enable_fp32=False, enable_mixed_precision=True,
                          init_device="meta", use_compile=False)
    value["data"].update(use_future_image=False, num_workers=0)
    kwargs = {**value["model"], **value["train"]}
    cls = get_config_registry().get_config_cls_from_config_key(value["model"]["config_key"])
    return cls(**kwargs), value


class LingbotVla2SftWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset:
            raise ValueError("LingBot-VLA-2 smoke recipe has no validation split")
        from lingbotvla.data import VLADataCollatorWithPacking, build_vla_dataset
        from lingbotvla.models import build_processor

        root = Path(str(data_paths))
        if not (root / "meta/info.json").is_file():
            raise ValueError("LingBot-VLA-2 requires the validated RoboTwin ALOHA LeRobot view")
        model_path = str(self.cfg.actor.model.model_path)
        tokenizer_path = str(self.cfg.actor.model.tokenizer_path)
        model_config, frozen = _official_config(model_path, tokenizer_path)
        data = frozen["data"]
        output_dir = Path(os.environ["EB_OUTPUT_DIR"])
        listing = output_dir / "lingbot-robotwin.txt"
        robot_config_root = output_dir / "lingbot-robot-configs"
        if self._rank == 0:
            listing.write_text(f"robotwin {root}\n")
            robot_config_root.mkdir(exist_ok=True)
            robot_config = yaml.safe_load(
                Path("/opt/lingbot-vla2/configs/robot_configs/robotwin.yaml").read_text()
            )
            robot_config["norm_stats"] = "/opt/lingbot-vla2/assets/norm_stats/robotwin.json"
            (robot_config_root / "robotwin.yaml").write_text(yaml.safe_dump(robot_config))
        torch.distributed.barrier()
        data.update(train_path=str(listing), chunk_size=50,
                    robot_config_root=str(robot_config_root))
        dataset = build_vla_dataset(SimpleNamespace(**data),
            SimpleNamespace(**frozen["model"]), model_config,
            build_processor(tokenizer_path), use_depth_align=False)
        sampler = DistributedSampler(dataset, self._world_size, self._rank, shuffle=True,
                                     seed=int(self.cfg.actor.seed), drop_last=True)
        loader = DataLoader(dataset, batch_size=self.micro_batch_size, sampler=sampler,
            drop_last=True, num_workers=0, pin_memory=True,
            collate_fn=VLADataCollatorWithPacking())
        if len(loader) == 0:
            raise ValueError("LingBot-VLA-2 dataset is smaller than the distributed batch")
        return loader, {"format": "robotwin_aloha_lerobot", "mapping": "robotwin"}

    def model_provider_func(self):
        from lingbotvla.models import build_foundation_model

        model_path = str(self.cfg.actor.model.model_path)
        tokenizer_path = str(self.cfg.actor.model.tokenizer_path)
        config, frozen = _official_config(model_path, tokenizer_path)
        return build_foundation_model(config_path=model_path, config_cls=config,
            weights_path=model_path, torch_dtype="bfloat16", attn_implementation="eager",
            moe_implementation=frozen["model"]["moe_implementation"], init_device="meta",
            config_kwargs={**frozen["model"], **frozen["train"]})

    def init_worker(self):
        from lingbotvla.distributed.parallel_state import init_parallel_state
        from lingbotvla.distributed.torch_parallelize import build_parallelize_model

        init_parallel_state(dp_size=self._world_size, dp_replicate_size=1,
            dp_shard_size=self._world_size, dp_mode="fsdp2")
        module = self.model_provider_func()
        self.param_names_need_sync = collect_param_names_need_sync(module)
        self.model = build_parallelize_model(module,
            weights_path=str(self.cfg.actor.model.model_path), enable_full_shard=True,
            enable_mixed_precision=True, enable_fp32=False,
            enable_gradient_checkpointing=True, init_device="meta",
            basic_modules=module._no_split_modules, enable_reentrant=False,
            enable_forward_prefetch=True, fsdp_llm_blocks=False,
            use_depth_align=False, split_fused_experts_from_decoder_fsdp=False,
            vlm_fsdp=True, use_future_image=False)
        self.optimizer = self.build_optimizer(self.model)
        self.lr_scheduler = self.build_lr_scheduler(self.optimizer, self._cfg.optim)
        self.grad_scaler = self.build_grad_scaler(False)

    def get_train_model_output(self, batch):
        batch = {key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
                 for key, value in batch.items()}
        with self.amp_context:
            output = self.model(**batch)
        loss, vla_loss = output[:2]
        metrics = {"loss": loss.detach().item(), "vla_loss": vla_loss.detach().item()}
        if len(output) > 4 and isinstance(output[4], dict):
            metrics.update({key: value.detach().item() for key, value in output[4].items()
                            if torch.is_tensor(value) and value.numel() == 1})
        return loss, metrics

    def get_eval_model_output(self, batch):
        raise NotImplementedError("LingBot-VLA-2 smoke recipe has no validation split")
