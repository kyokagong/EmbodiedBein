from __future__ import annotations

import re
from pathlib import Path

_JOINT_NAME = re.compile(r"<joint\b[^>]*\bname=['\"]([^'\"]+)['\"]", re.I)
_JOINT_BLOCK = re.compile(r"<joint\b[^>]*>.*?</joint>", re.I | re.S)
_ATTR = re.compile(r"\b(name|type)=['\"]([^'\"]+)['\"]", re.I)
_LINK = re.compile(r"<(parent|child)\b[^>]*\blink=['\"]([^'\"]+)['\"]", re.I)
_LIMIT = re.compile(
    r"<limit\b([^>]*)/?>",
    re.I,
)
_KV = re.compile(r"""(\w+)\s*=\s*['"]([^'"]+)['"]""")
_XYZ = re.compile(r"""<(origin|axis)\b([^>]*)/?>""", re.I)


def parse_urdf_joints(text: str) -> list[dict[str, str]]:
    """Extract joint name / type / parent / child without a URDF library."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    blocks = _JOINT_BLOCK.findall(text)
    chunks = blocks or [text]
    for chunk in chunks:
        attrs = {k.lower(): v for k, v in _ATTR.findall(chunk)}
        name = attrs.get("name")
        if not name:
            m = _JOINT_NAME.search(chunk)
            name = m.group(1) if m else None
        if not name or name in seen:
            continue
        seen.add(name)
        links = {k.lower(): v for k, v in _LINK.findall(chunk)}
        out.append(
            {
                "name": name,
                "type": attrs.get("type", ""),
                "parent": links.get("parent", ""),
                "child": links.get("child", ""),
            }
        )
    if not out:
        for name in _JOINT_NAME.findall(text):
            if name not in seen:
                seen.add(name)
                out.append({"name": name, "type": "", "parent": "", "child": ""})
    return out


def _floats(text: str) -> list[float]:
    out: list[float] = []
    for part in text.replace(",", " ").split():
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def parse_urdf_joint_tree(text: str) -> list[dict]:
    """Joint metadata including limits, for 3D sliders (rereview 01)."""
    joints: list[dict] = []
    for chunk in _JOINT_BLOCK.findall(text) or [text]:
        attrs = {k.lower(): v for k, v in _ATTR.findall(chunk)}
        name = attrs.get("name")
        if not name:
            m = _JOINT_NAME.search(chunk)
            name = m.group(1) if m else None
        if not name:
            continue
        links = {k.lower(): v for k, v in _LINK.findall(chunk)}
        limit = {"lower": -3.14, "upper": 3.14, "effort": 0.0, "velocity": 0.0}
        lm = _LIMIT.search(chunk)
        if lm:
            kvs = {k.lower(): v for k, v in _KV.findall(lm.group(1))}
            for key in ("lower", "upper", "effort", "velocity"):
                if key in kvs:
                    try:
                        limit[key] = float(kvs[key])
                    except ValueError:
                        pass
        origin = {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
        axis = [0.0, 0.0, 1.0]
        for tag, raw in _XYZ.findall(chunk):
            kvs = {k.lower(): v for k, v in _KV.findall(raw)}
            if tag.lower() == "origin":
                if "xyz" in kvs:
                    origin["xyz"] = _floats(kvs["xyz"])[:3] or origin["xyz"]
                if "rpy" in kvs:
                    origin["rpy"] = _floats(kvs["rpy"])[:3] or origin["rpy"]
            elif tag.lower() == "axis" and "xyz" in kvs:
                axis = _floats(kvs["xyz"])[:3] or axis
        joints.append(
            {
                "name": name,
                "type": attrs.get("type", ""),
                "parent": links.get("parent", ""),
                "child": links.get("child", ""),
                "origin": origin,
                "axis": axis,
                "limit": limit,
            }
        )
    return joints


def posix_norm(rel: str) -> str:
    parts: list[str] = []
    for p in rel.replace("\\", "/").split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    return "/".join(parts)


def iter_mesh_filenames(urdf_text: str) -> list[str]:
    return re.findall(r"""filename\s*=\s*['"]([^'"]+)['"]""", urdf_text, re.I)


def resolve_mesh_rel(urdf_rel: str, filename: str) -> str:
    """Map package:// or relative mesh href onto a path relative to the embodiment root."""
    raw = filename.strip()
    if raw.startswith("package://"):
        raw = raw[len("package://") :]
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    if raw.startswith("/"):
        return posix_norm(raw.lstrip("/"))
    return posix_norm(str(Path(urdf_rel).parent / raw))


def collect_urdf_joints(root: Path, rel_paths: list[str] | None = None) -> tuple[list[str], str | None]:
    """Return unique joint names and the first readable URDF relative path."""
    names: list[str] = []
    seen: set[str] = set()
    used: str | None = None
    candidates = list(rel_paths or [])
    if root.is_dir() and not candidates:
        candidates = [str(p.relative_to(root)) for p in root.rglob("*.urdf") if p.is_file()]
    for rel in candidates:
        dest = root / rel
        if not dest.is_file():
            continue
        text = dest.read_text(errors="replace")
        parsed = parse_urdf_joints(text)
        if not parsed:
            continue
        if used is None:
            used = rel
        for j in parsed:
            if j["name"] not in seen:
                seen.add(j["name"])
                names.append(j["name"])
    return names, used


def missing_joints(joint_order: list[str], asset_joints: list[str]) -> list[str]:
    have = set(asset_joints)
    return [j for j in joint_order if j not in have]


def next_clone_id(base: str, taken: set[str]) -> str:
    m = re.match(r"^(.*)_v(\d+)$", base)
    stem = m.group(1) if m else base
    n = int(m.group(2)) + 1 if m else 2
    while True:
        cand = f"{stem}_v{n}"
        if cand not in taken:
            return cand
        n += 1
