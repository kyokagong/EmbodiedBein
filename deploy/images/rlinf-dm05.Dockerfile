FROM eb/rlinf:libero-pi05

COPY --from=opendm / /opt/opendm

# OpenDM is isolated because its Transformers stack is newer than OpenPI's.
# Keep the base CUDA/PyTorch runtime and install only DM0.5 runtime dependencies.
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
      transformers==5.3.0 accelerate==1.14.0 peft==0.19.1 tokenizers==0.22.1 \
      datasets==5.0.0 diffusers==0.38.0 timm==1.0.27 numpy==1.26.4 \
      einops einops-exts sentencepiece tiktoken decord av albumentations \
      megfile easydict orjson loguru numpydantic==1.1.0 tyro==1.0.13 \
    && /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install --no-deps -e /opt/opendm

COPY backend/eb/runners /workspace/eb_runner
ENV PYTHONPATH="/opt/opendm:/opt/eb-openpi:/workspace/eb_runner:/workspace/rlinf"
ENTRYPOINT []
