from __future__ import annotations

import re
from pathlib import PurePosixPath

from eb.domain.urdf import parse_urdf_joint_tree

_GYM_ID = re.compile(r"""register\(\s*id\s*=\s*['"]([^'"]+)['"]""")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "logs"}


def scan_embodiment(files: list[dict], texts: dict[str, str]) -> dict:
    variants: list[dict] = []
    meshes: list[str] = []
    gym_ids: list[str] = []
    rlinf_envs: list[str] = []
    for item in files:
        path = PurePosixPath(item["path"])
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if item.get("is_dir") or item.get("is_symlink"):
            continue
        rel = path.as_posix()
        suffix = path.suffix.lower()
        if suffix == ".urdf":
            text = texts[rel]
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
                gym_ids.extend(_GYM_ID.findall(texts[rel]))
            except KeyError:
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
