"""Validate managed patch chain against the pinned read-only RLinf source."""

from pathlib import Path
import ast
import runpy
import shutil
import sys
import tempfile


def test_flat_patch_chain():
    repo = Path(__file__).resolve().parents[3]
    files = (
        "rlinf/envs/sim/isaaclab/__init__.py",
        "rlinf/models/embodiment/mlp_policy/mlp_policy.py",
        "rlinf/models/embodiment/mlp_policy/__init__.py",
        "rlinf/data/schema/embodied_types.py",
        "rlinf/envs/sim/isaaclab/isaaclab_env.py",
        "rlinf/workers/env/env_worker.py",
        "examples/embodiment/train_embodied_agent.py",
        "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
        "rlinf/algorithms/losses.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in files:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path("/data/kk/RLinf") / name, target)
        previous = sys.argv
        try:
            for patch in ("patch_rpo_wbc.py", "patch_rpo_amp.py", "patch_rpo_flat.py"):
                sys.argv = [patch, str(root)]
                runpy.run_path(str(repo / "deploy/patches/rlinf" / patch), run_name="__main__")
        finally:
            sys.argv = previous
        for name in files:
            ast.parse((root / name).read_text())
        worker = (root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py").read_text()
        assert 'loss_kwargs["rpo_flat_value_mse"] = True' in worker
        factory = (root / "rlinf/models/embodiment/mlp_policy/__init__.py").read_text()
        assert "from rpo_flat_policy import RPOFlatPolicy as MLPPolicy" in factory
        assert '"RPO-Flat-RLinf-v1": RPOFlatEnv' in (root / files[0]).read_text()


if __name__ == "__main__":
    test_flat_patch_chain()
    print("WBC → AMP → Flat patch chain passed against pinned external RLinf")
