FROM eb/rlinf:rpo-bfm-ufo-mjlab-v12
SHELL ["/bin/bash", "-ic"]
RUN --mount=type=bind,source=backend/eb,target=/tmp/eb-src,readonly \
    --mount=type=bind,source=deploy/patches/rlinf,target=/tmp/patches,readonly \
    cp -a /tmp/eb-src/runners/. /workspace/eb_runner/ && \
    cp /tmp/eb-src/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py && \
    /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patches/patch_rpo_ufo_mjlab_viability.py
