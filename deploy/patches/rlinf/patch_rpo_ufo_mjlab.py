"""Image-local MJLab dispatch and measured dependency/source receipt."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def patch(root):
    from patch_rpo_ufo import replace
    replace(root, "rlinf/envs/__init__.py", '    ISAACLAB = "isaaclab"\n',
        '    ISAACLAB = "isaaclab"\n    MJLAB = "mjlab"\n')
    replace(root, "rlinf/envs/__init__.py", '    env_type = SupportedEnvType(env_type)\n',
        '    env_type = SupportedEnvType(env_type)\n'
        '    if env_type == SupportedEnvType.MJLAB:\n'
        '        from rpo_ufo_mjlab import RPOUFOMJLabEnv\n'
        '        return RPOUFOMJLabEnv\n')
    path = Path("/opt/ufo-source-receipt.json")
    receipt = json.loads(path.read_text())
    for key in ("ground_profile", "dynamics_profile", "ankle_control_profile", "contact_profile"):
        receipt.pop(key, None)
    receipt.update(scope="native UFO agent + platform RPO MJCF/MJLab bridge; tested Python3.11 import",
        simulator="mjlab", physics_profile="native_mjcf_v12", contact_profile="native_mjcf_v12",
        mjlab_patch_hash="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for package in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "gymnasium", "tensordict"):
        receipt["packages"][package] = importlib.metadata.version(package)
    path.write_text(json.dumps(receipt, indent=2))
    path = Path("/opt/ufo-auxiliary-source-files.json")
    hashes = json.loads(path.read_text())
    hashes["rpo_ufo_mjlab.py"] = ""
    hashes["rpo_ufo_mjlab_probe.py"] = ""
    for name in hashes:
        file = (Path("/workspace/tests/rpo_ufo_tracking_diagnostic.py") if name == "rpo_ufo_tracking_diagnostic.py"
                else Path("/workspace/eb_runner") / name)
        hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(hashes, indent=2))
    receipt["auxiliary_source_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt["action_control_source_sha256"] = "sha256:" + hashlib.sha256(
        Path("/workspace/eb_runner/rpo_bfm_codec.py").read_bytes()).hexdigest()
    Path("/opt/ufo-source-receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    patch(Path("/workspace/rlinf"))
