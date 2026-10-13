FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-safe-v2
COPY backend/eb/runners /workspace/eb_runner
COPY deploy/patches/rlinf/patch_rpo_ufo_stable.py /tmp/
RUN PYTHONPATH=/workspace/eb_runner:/opt/ufo /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_stable.py
