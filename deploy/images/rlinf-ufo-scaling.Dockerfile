# Makefile verifies the immutable E producer parent.
FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-slew-velocity-v5
COPY backend/eb/runners /workspace/eb_runner
COPY backend/eb/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py
COPY backend/tests/runtime/rpo_ufo_tracking_diagnostic.py /workspace/tests/rpo_ufo_tracking_diagnostic.py
COPY deploy/patches/rlinf/patch_rpo_ufo_scaling.py /tmp/
RUN /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_scaling.py
