FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-stable-v3
COPY backend/eb/runners /workspace/eb_runner
COPY backend/eb/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py
COPY deploy/patches/rlinf/patch_rpo_ufo_slew.py /tmp/
RUN /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_slew.py
