FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1
COPY backend/eb/runners /workspace/eb_runner
COPY deploy/patches/rlinf/patch_rpo_ufo_safe.py /tmp/
RUN /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_safe.py
