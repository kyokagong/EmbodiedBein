FROM eb/rlinf:rpo-ufo-eval-v1
COPY backend/eb/runners /workspace/eb_runner
COPY backend/tests/runtime/rpo_ufo_tracking_diagnostic.py /workspace/tests/rpo_ufo_tracking_diagnostic.py
