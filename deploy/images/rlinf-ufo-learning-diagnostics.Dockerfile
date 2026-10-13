# Local image ID is checked before building; it is not a registry manifest digest.
FROM eb/rlinf:rpo-bfm-ufo-mjlab-viability-v13
COPY backend/eb/runners/rpo_ufo_learning_diagnostics.py /workspace/eb_runner/rpo_ufo_learning_diagnostics.py
RUN /workspace/isaaclab_extension_template/.venv/bin/python -c 'import hashlib,json; from pathlib import Path; p=Path("/workspace/eb_runner/rpo_ufo_learning_diagnostics.py"); Path("/opt/learning-diagnostics-source.json").write_text(json.dumps({"schema_version":"eb/rpo-learning-diagnostic-source/v1","baseline_image":"sha256:e23281518d388565f4b1162dc9d2f21947543b2fd0a014b0288365be2788cee2","diagnostic_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"baseline_training_semantics":"unchanged","optimizer_updates":0}))'
