# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python-is-python3 build-essential git \
    libegl1 libgl1 libgles2 libglib2.0-0 libglvnd0 libosmesa6 libgomp1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/images/rlinf-pbfm-requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements.lock
COPY deploy/images/rlinf-pbfm-extras.lock /tmp/extras.lock
RUN python -m pip install --no-cache-dir --timeout 120 --retries 5 -r /tmp/extras.lock
COPY --from=pbfm pyproject.toml README.md README_zh-CN.md LICENSE .gitattributes .gitignore /opt/pbfm-upstream/
COPY --from=pbfm pbfm /opt/pbfm-upstream/pbfm
COPY --from=pbfm docs /opt/pbfm-upstream/docs
COPY --from=pbfm scripts /opt/pbfm-upstream/scripts
COPY --from=pbfm tests /opt/pbfm-upstream/tests
COPY --from=rlinf rlinf /opt/rlinf-upstream/rlinf
COPY deploy/patches/pbfm /opt/eb-patches/pbfm
COPY backend/eb /opt/eb-native/eb
# The host driver is injected by NVIDIA Container Toolkit. GLVND also needs its
# vendor manifest; the host may provide driver libraries without this JSON.
COPY deploy/images/pbfm-nvidia-egl.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY deploy/patches/mujoco-warp /opt/eb-patches/mujoco-warp
RUN python /opt/eb-patches/mujoco-warp/fix_sensor_axis.py > /opt/pbfm-mujoco-warp-patch.json \
    && python /opt/eb-patches/mujoco-warp/verify_sensor_axis.py \
    && python /opt/eb-patches/mujoco-warp/size_eager_convex_scratch.py > /opt/pbfm-convex-scratch-patch.json
ENV PYTHONPATH=/opt/eb-native:/opt/rlinf-upstream MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
RUN python -c 'from eb.runners.pbfm_source import prepare_pbfm_source; from pathlib import Path; prepare_pbfm_source(Path("/opt/pbfm-upstream"),Path("/opt/pbfm-build-probe"))' \
    && PYTHONPATH=/opt/pbfm-build-probe:/opt/rlinf-upstream:/opt/eb-native python -c \
       'import sys,importlib.metadata as m; from rlinf.scheduler import Worker,Cluster; from pbfm.train import build_ufo_mjlab_config; from pbfm.training.workspace import Workspace; from pbfm.motion.library.motion_lib_robot import MotionLibRobot; import pbfm.inference.export.depth_policy; from eb.runners.rlinf_pbfm_worker import PbfmTrainWorker; assert sys.version_info[:2]==(3,10); assert m.version("torch")=="2.7.1+cu128"; assert m.version("mjlab")=="1.4.0"; assert m.version("mujoco")=="3.8.0"' \
    && python -m pip freeze --all > /opt/pbfm-runtime-freeze.txt \
    && python -m pip check
WORKDIR /run/eb-output
ENTRYPOINT []
