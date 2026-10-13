# Makefile checks the frozen F parent; all mutations are in this new image.
FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-scale6144-v6
# F has 124 layers; keep this derivative below Docker's 128-layer mount limit.
RUN --mount=type=bind,source=backend/eb,target=/tmp/eb-src,readonly \
    --mount=type=bind,source=deploy,target=/tmp/deploy-src,readonly \
    cp -a /tmp/eb-src/runners/. /workspace/eb_runner/ && \
    cp /tmp/eb-src/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py && \
    cp /tmp/deploy-src/images/code_snapshot_hash.py /tmp/code_snapshot_hash.py && \
    cp /tmp/deploy-src/patches/rlinf/patch_rpo_ufo_reachable.py /tmp/patch_rpo_ufo_reachable.py && \
    /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_reachable.py
