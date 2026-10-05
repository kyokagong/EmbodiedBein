FROM eb/rlinf:libero-pi05
COPY --from=lingbot / /opt/lingbot-vla2
COPY --from=utils3d / /opt/utils3d
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchdata==0.11.0 torchcodec==0.6.0 \
      transformers==4.57.3 tokenizers==0.22.2 datasets==3.6.0 huggingface-hub==0.34.3 \
      safetensors==0.5.3 qwen-vl-utils==0.0.11 peft==0.15.2 numpy==1.26.4 scipy==1.13.1 \
      pandas==2.3.1 pyarrow==21.0.0 pillow==11.3.0 opencv-python-headless==4.11.0.86 \
      av==15.0.0 imageio==2.34.2 einops==0.8.1 pyyaml==6.0.2 blobfile==3.2.0 \
      sentencepiece==0.2.0 protobuf==3.20.3 ninja==1.11.1.4 h5py==3.14.0 jsonlines==4.0.0 \
      zstandard==0.23.0 fsspec==2024.6.1 msgpack==1.1.1 omegaconf==2.3.0
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install --no-deps \
      -e /opt/lingbot-vla2/lingbotvla/models/vla/vision_models/lingbot-depth \
      -e /opt/lingbot-vla2/lingbotvla/models/vla/vision_models/MoGe \
      -e /opt/utils3d \
    && /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
      accelerate==1.7.0 matplotlib==3.10.6 trimesh==4.5.1 click==8.1.7
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
      nvidia-npp-cu12==12.4.1.87 \
    && echo /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/nvidia/npp/lib \
      > /etc/ld.so.conf.d/eb-nvidia-npp.conf \
    && ldconfig
# LingBot's legacy pin breaks Ray's generated dashboard protos; keep the
# base RLInf image's compatible protobuf runtime.
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install protobuf==6.33.6
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# The upstream v2 constructor hard-codes optional FlashAttention despite exposing
# eager as a supported config. Keep the official math path available on 4090 hosts
# without compiling an unpinned CUDA extension.
RUN sed -i 's/vlm_config\._attn_implementation = "flash_attention_2"/vlm_config._attn_implementation = self.config.vit_attn_implementation/; s/vlm_config\.text_config\._attn_implementation = "flash_attention_2"/vlm_config.text_config._attn_implementation = self.config.vit_attn_implementation/; s/self\.config\.qwen_expert_config\._attn_implementation = "flash_attention_2"/self.config.qwen_expert_config._attn_implementation = self.config.vit_attn_implementation/' /opt/lingbot-vla2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py
# Upstream's FSDP2 loader already supports sharded checkpoint materialization,
# but the LingBot-VLA-2 entrypoint does not pass the checkpoint path to it.
RUN python -c 'from pathlib import Path; p=Path("/opt/lingbot-vla2/tasks/vla/train_lingbotvla.py"); s=p.read_text(); old="    model = build_parallelize_model(\n        model,\n        enable_full_shard="; new="    model = build_parallelize_model(\n        model,\n        weights_path=args.model.model_path,\n        enable_full_shard="; assert s.count(old) == 1; p.write_text(s.replace(old, new))'
# The released V2 config points at a generic local tokenizer path, so upstream's
# path-name heuristic selects the Qwen2.5 patch for its Qwen3-VL backbone.
RUN python -c 'from pathlib import Path; p=Path("/opt/lingbot-vla2/lingbotvla/models/auto.py"); s=p.read_text(); old="    if \"qwen3\" in vlm_key and \"vl\" in vlm_key:"; new="    if (\"qwen3\" in vlm_key and \"vl\" in vlm_key) or config_cls.__class__.__name__ == \"LingbotVLAV2Config\":"; assert s.count(old) == 1; p.write_text(s.replace(old, new))'
COPY backend/eb/runners /workspace/eb_runner
ENV PYTHONPATH="/opt/lingbot-vla2:/opt/eb-openpi:/workspace/eb_runner:/workspace/rlinf"
ENTRYPOINT []
