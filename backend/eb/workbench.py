from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import httpx

from eb.domain.scan import scan_embodiment, _SKIP_DIRS
from eb.store.models import EmbodimentRow


def embodiment_root(row: EmbodimentRow) -> Path:
    raw = row.root_path or (row.source or {}).get("path") or ""
    return Path(str(raw))


async def apply_scan(row: EmbodimentRow, root: Path | None = None) -> dict[str, Any]:
    from eb.store.ebd import EbdClient, storage_inventory, storage_root
    dest = root or embodiment_root(row)
    try:
        inventory = await storage_inventory(dest, exclude_dirs=tuple(sorted(_SKIP_DIRS)))
        texts = {}
        for item in inventory:
            if not item["is_dir"] and not item.get("is_symlink") and Path(item["path"]).suffix.lower() in {".urdf", ".py"}:
                texts[item["path"]] = await EbdClient().read_storage_text(storage_root(dest), item["path"], max_bytes=16 * 1024**2)
        result = scan_embodiment(inventory, texts)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        result = {"status": "error", "error_message": f"目录不存在: {dest}", "urdf_variants": [], "gym_ids": [], "rlinf_envs": []}
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


async def dir_size(path: Path) -> int:
    from eb.store.ebd import EbdClient, storage_root
    try:
        return (await EbdClient().inspect_storage(storage_root(path), "", mode="size"))["bytes"]
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return 0
        raise


async def disk_preflight(target_dir: str, needed_bytes: int = 0, archive: bool = False):
    from eb.domain.disk import disk_preflight as check
    from eb.store.ebd import EbdClient, storage_root
    facts = await EbdClient().inspect_storage(storage_root(Path(target_dir)), "", mode="disk")
    return check(facts, needed_bytes=needed_bytes, archive=archive)
