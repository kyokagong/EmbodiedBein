FROM eb/rlinf:rpo-v2-bfm-reference-v5
COPY backend/eb/runners/g1_lafan_source.py /workspace/eb_runner/g1_lafan_source.py
COPY backend/eb/runners/g1_lafan_bridge.py /workspace/eb_runner/g1_lafan_bridge.py
