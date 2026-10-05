"""Patch the pinned RLinf source copy and check AMP hook syntax/placement."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RLINF = Path("/data/kk/RLinf")
TARGETS = (
    "rlinf/envs/sim/isaaclab/__init__.py",
    "rlinf/envs/sim/isaaclab/isaaclab_env.py",
    "rlinf/models/embodiment/mlp_policy/mlp_policy.py",
    "rlinf/models/embodiment/mlp_policy/__init__.py",
    "rlinf/workers/env/env_worker.py",
    "rlinf/data/schema/embodied_types.py",
    "examples/embodiment/train_embodied_agent.py",
)


def test_amp_patch_applies_after_rpo_patch_and_keeps_python_parseable(tmp_path):
    for relative in TARGETS:
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RLINF / relative, dest)
    for patch in ("patch_rpo_wbc.py", "patch_rpo_amp.py"):
        subprocess.run([sys.executable, str(ROOT / "deploy/patches/rlinf" / patch),
                        str(tmp_path)], check=True, capture_output=True, text=True)
    for relative in TARGETS:
        ast.parse((tmp_path / relative).read_text(), filename=relative)
    source = (tmp_path / "rlinf/workers/env/env_worker.py").read_text()
    assert '"amp_window": amp_info["amp_window"]' in source
    assert 'amp_valid' in source
    assert 'last_step_truncations = last_step_truncations & ~env_output.terminations[:, -1]' in source
    factory = (tmp_path / "rlinf/models/embodiment/mlp_policy/__init__.py").read_text()
    assert 'if cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof" else cfg.obs_dim' in factory
    schema = ast.parse((tmp_path / "rlinf/data/schema/embodied_types.py").read_text())
    env_output = next(node for node in schema.body if isinstance(node, ast.ClassDef) and node.name == "EnvOutput")
    prepare = next(node for node in env_output.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_observations")
    namespace = {"Any": Any}
    exec(compile(ast.Module(body=[prepare], type_ignores=[]), "<prepare_observations>", "exec"), namespace)
    assert namespace["prepare_observations"](None, {"states": 780, "critic_states": 1390})["critic_states"] == 1390


def test_rpo_image_applies_amp_patch_to_its_rlinf_copy():
    image = (ROOT / "deploy/images/rlinf-rpo.Dockerfile").read_text()
    assert "COPY deploy/patches/rlinf/patch_rpo_amp.py" in image
    assert "python3 /tmp/patch_rpo_amp.py /workspace/rlinf" in image
    assert "/workspace/eb_runner/rpo_amp_extension.py" in image
    assert "/workspace/eb_runner/rpo_amp_worker.py" in image
    assert "COPY backend/eb/rpo_wbc.py /workspace/eb_runner/rpo_wbc.py" in image
