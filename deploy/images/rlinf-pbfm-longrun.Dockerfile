# Local images have an image ID, not a registry manifest digest; Makefile verifies
# this tag's exact8a1 image ID before building. The resulting parent layers are audited.
FROM eb/rlinf-pbfm:rpo-depth-v1
COPY backend/eb/pbfm_training.py backend/eb/pbfm_training_checkpoint.py backend/eb/pbfm_longrun.py /opt/eb-native/eb/
COPY backend/eb/runners/rlinf_pbfm_runtime.py backend/eb/runners/rlinf_pbfm_worker.py backend/eb/runners/pbfm_longrun.py /opt/eb-native/eb/runners/
RUN python -c 'import json; from pathlib import Path; from eb.pbfm_longrun import runtime_compatibility_receipt; Path("/opt/pbfm-long-runtime.json").write_text(json.dumps(runtime_compatibility_receipt(), sort_keys=True, indent=2))'
