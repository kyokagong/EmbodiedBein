"""One-step DM0.5/RoboTwin compatibility probe (RLInf adapter building block)."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import BuildAction, ChatTokenization, LoadImages, Normalize, PadAction, Pipeline, PixelTransform
from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
from opendm.model.dm05.dm05_lora import DM05LoraConfig, apply_lora_to_dm05_model


def main(model_path: Path, data_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    config = DM05Config.from_pretrained(model_path, local_files_only=True)
    config.vlm_config._attn_implementation = {"": "sdpa", "text_config": "sdpa", "vision_config": "sdpa"}
    config.vlm_config.text_config._attn_implementation = "sdpa"
    config.vlm_config.vision_config._attn_implementation = "sdpa"
    model = DM05ForConditionalGeneration.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    model.set_attention_implementation(
        llm_attn_implementation="eager",
        vision_attn_implementation="sdpa",
        action_attn_implementation="sdpa",
        bf16=True,
    )
    model.enable_gradient_checkpointing(
        vlm_gradient_checkpointing=True, ae_gradient_checkpointing=True
    )
    model.freeze_vlm_embedding()
    model = apply_lora_to_dm05_model(
        model, DM05LoraConfig(dump_trainable_path=None), base_model_name_or_path=str(model_path)
    ).cuda().train()
    pipeline = Pipeline([
        BuildAction(50, ActionMode.ABSOLUTE),
        LoadImages(["images_1", "images_2", "images_3"], image_dir="/"),
        PixelTransform(NoAugmentationPipeline()),
        Normalize(str(data_path / "norm_stats.json"), ["state", "action"], use_quantiles=True),
        ChatTokenization(processor, 256, 1024, ["Head", "Left wrist", "Right wrist"], add_state=True),
        PadAction(32),
    ])
    dataset = JsonlDataset(str(data_path), pipeline, "robotwin2_generalist", {
        "robot_type": RobotType.ALOHA_ROBOTWIN2.value,
        "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2],
    })
    collator = TrainingCollator(processor.tokenizer.pad_token_id, 1024)
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collator)))
    batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-4)
    result = model(**batch)
    result.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.save_pretrained(output)
    (output / "smoke.txt").write_text(f"loss={result.loss.item()}\nfm_loss={result.fm_loss.item()}\ngrad_norm={float(grad_norm)}\n")
    print((output / "smoke.txt").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.model, args.data, args.output)
