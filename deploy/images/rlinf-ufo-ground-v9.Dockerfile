FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1
RUN --mount=type=bind,source=backend/eb,target=/tmp/eb-src,readonly \
    --mount=type=bind,source=deploy,target=/tmp/deploy-src,readonly \
    cp -a /tmp/eb-src/runners/. /workspace/eb_runner/ && \
    cp /tmp/eb-src/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py && \
    cp /tmp/deploy-src/patches/rlinf/patch_rpo_ufo_ground.py /tmp/patch_rpo_ufo_ground.py && \
    /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_ground.py
