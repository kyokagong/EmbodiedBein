# Derive a diagnostic image without changing the frozen model/physics modules.
FROM eb/rlinf:rpo-ufo-learning-diagnostics-v4
COPY backend/eb/runners/rpo_ufo_dynamics_diagnostics.py /workspace/eb_runner/rpo_ufo_dynamics_diagnostics.py
RUN /workspace/isaaclab_extension_template/.venv/bin/python -c 'import hashlib,json; from pathlib import Path; p=Path("/workspace/eb_runner/rpo_ufo_dynamics_diagnostics.py"); Path("/opt/dynamics-diagnostics-source.json").write_text(json.dumps({"schema_version":"eb/rpo-dynamics-diagnostic-source/v1","parent_image":"sha256:b4d976ea9163a6f57b033572cb3fb78225ec9f480e6b40e2afd3cae0ddf0e455","diagnostic_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"baseline_training_semantics":"unchanged","optimizer_updates":0}))'
