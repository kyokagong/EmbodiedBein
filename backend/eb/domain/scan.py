from __future__ import annotations

import re
from pathlib import Path

from eb.domain.urdf import parse_urdf_joint_tree

_GYM_ID = re.compile(r"""register\(\s*id\s*=\s*['"]([^'"]+)['"]""")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "logs"}


def scan_embodiment(root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        return {"status": "error", "error_message": f"目录不存在: {root}", "urdf_variants": [], "gym_ids": [], "rlinf_envs": []}
    variants: list[dict] = []
    meshes: list[str] = []
    gym_ids: list[str] = []
    rlinf_envs: list[str] = []
    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        suffix = path.suffix.lower()
        if suffix == ".urdf":
            text = path.read_text(errors="replace")
            joints = parse_urdf_joint_tree(text)
            movable = [j for j in joints if j.get("type") not in ("fixed", "")]
            stem = path.stem
            variants.append(
                {
                    "id": stem,
                    "path": rel,
                    "dof": len(movable) or len(joints),
                    "joints": joints,
                }
            )
        elif suffix in {".stl", ".obj", ".dae", ".glb"}:
            meshes.append(rel)
        elif suffix in {".py", ".pyc"} and suffix == ".py":
            try:
                gym_ids.extend(_GYM_ID.findall(path.read_text(errors="replace")))
            except OSError:
                pass
        elif suffix in {".yaml", ".yml"} and ("/env/" in f"/{rel}" or rel.startswith("env/")):
            rlinf_envs.append(rel)
    variants.sort(key=lambda v: v["path"])
    default_urdf = variants[0]["path"] if variants else None
    joint_tree = variants[0]["joints"] if variants else []
    return {
        "status": "ready",
        "error_message": None,
        "default_urdf": default_urdf,
        "urdf_variants": variants,
        "joint_tree": joint_tree,
        "gym_ids": sorted(set(gym_ids)),
        "rlinf_envs": sorted(set(rlinf_envs)),
        "meshes": meshes[:400],
        "interface_snapshot": None,
    }
