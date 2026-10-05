ARG BASE=rlinf-berkeley-openpi:local
FROM python:3.11-slim-bookworm AS python
FROM ${BASE} AS runtime
# The inherited venv pointed at a UV interpreter absent from the image.
COPY --from=python /usr/local /usr/local
RUN ln -sf /usr/local/bin/python3.11 /workspace/isaaclab_extension_template/.venv/bin/python
ENV PATH="/workspace/isaaclab_extension_template/.venv/bin:/usr/local/bin:${PATH}" \
    RAY_PORT=6390 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace/rlinf
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install --force-reinstall --no-deps hydra-core==1.3.2
# Local source context is read-only; no external repository is modified.
COPY --from=openpi /src/openpi /opt/eb-openpi/openpi
COPY --from=openpi /packages/openpi-client/src/openpi_client /opt/eb-openpi/openpi_client
ENV PYTHONPATH="/opt/eb-openpi:/workspace/eb_runner:/workspace/rlinf"
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
    'datasets==3.6.0' 'av==15.1.0' 'jsonlines==4.0.0' 'draccus==0.10.0' 'jax==0.5.3' 'flax==0.10.2' 'orbax-checkpoint==0.11.13' \
    'jaxtyping==0.2.36' 'numpy<2' 'beartype==0.19.0' 'chex==0.1.90' \
    'augmax==0.3.4' 'equinox==0.11.12' 'numpydantic==1.6.8' 'ml_collections==1.0.0' \
    'fsspec[gcs]' sentencepiece dm-tree treescope polars torchdata 'opencv-python-headless==4.11.0.86'
# Match OpenPI's pinned reader; RoboTwin's published training view is LeRobot v2.1.
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install --no-deps \
    https://github.com/huggingface/lerobot/archive/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5.zip
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install transformers==4.53.2 accelerate==1.10.1
COPY --from=openpi /src/openpi/models_pytorch/transformers_replace/ /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/transformers/
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install tqdm-loggable wandb==0.24.0 termcolor 'packaging>=24.2,<26' tensorboard
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install numpydantic==1.6.9 pydantic==2.11.5
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install 'ray[default]==2.47.0' nvitop icmplib lz4 zstandard sortedcontainers swanlab==0.6.11 peft==0.11.1 tyro==1.0.13
ENV OPENPI_DATA_HOME=/opt/eb-openpi-cache
RUN /usr/local/bin/python -c "import pathlib,urllib.request,hashlib; p=pathlib.Path('/opt/eb-openpi-cache/big_vision/paligemma_tokenizer.model'); p.parent.mkdir(parents=True,exist_ok=True); data=urllib.request.urlopen('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model',timeout=60).read(); p.write_bytes(data); p.with_suffix('.model.sha256').write_text(hashlib.sha256(data).hexdigest()+'\\n')"
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install rlinf-libero==0.1.3
# Use RLinf's pinned fork, including its lazy LeRobot import compatibility.
RUN /usr/local/bin/python -m pip install --no-deps --upgrade --target /opt/eb-openpi rlinf-openpi==0.1.1
RUN /usr/local/bin/python -c "import shutil; shutil.copytree('/opt/eb-openpi/openpi/models_pytorch/transformers_replace', '/workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/transformers', dirs_exist_ok=True)"
COPY backend/eb/runners /workspace/eb_runner
# LIBERO jobs execute the supplied Python command directly; no Isaac launcher.
ENTRYPOINT []
FROM runtime AS complete
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}
COPY deploy/images/libero_assets.py /tmp/libero_assets.py
RUN --mount=type=cache,target=/root/.cache/huggingface /workspace/isaaclab_extension_template/.venv/bin/python /tmp/libero_assets.py
