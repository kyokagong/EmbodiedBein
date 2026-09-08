from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eb.domain.scan import scan_embodiment
from eb.store.models import EmbodimentRow


def embodiment_root(row: EmbodimentRow) -> Path:
    raw = row.root_path or (row.source or {}).get("path") or ""
    return Path(str(raw))


def apply_scan(row: EmbodimentRow, root: Path | None = None) -> dict[str, Any]:
    dest = root or embodiment_root(row)
    result = scan_embodiment(dest)
    row.root_path = str(dest)
    row.default_urdf = result.get("default_urdf")
    row.urdf_variants = result.get("urdf_variants") or []
    row.joint_tree = result.get("joint_tree") or []
    row.gym_ids = result.get("gym_ids") or []
    row.interface_snapshot = result.get("interface_snapshot")
    row.error_message = result.get("error_message")
    row.scanned_at = datetime.now(timezone.utc)
    if result.get("status") == "error":
        row.status = "error"
    elif row.status in ("scanning", "error"):
        row.status = "ready"
    src = dict(row.source or {})
    src.setdefault("kind", src.get("kind") or "git_external")
    src["path"] = str(dest)
    row.source = src
    return result


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
