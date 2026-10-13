FROM eb/tools:rpo-g1-source-v1
COPY backend/eb/runners/g1_lafan_bridge.py backend/eb/runners/g1_lafan_export.py /workspace/eb_runner/
