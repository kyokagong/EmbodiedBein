from __future__ import annotations

from dataclasses import dataclass

HEADROOM_BYTES = 10 * 1024 * 1024 * 1024
LOW_RATIO = 0.10
ARCHIVE_FACTOR = 2.5


@dataclass
class DiskCheck:
    ok: bool
    free_bytes: int
    total_bytes: int
    needed_bytes: int
    reason: str = ""
    warn: bool = False


def disk_preflight(facts: dict, needed_bytes: int = 0, archive: bool = False) -> DiskCheck:
    free = int(facts["free_bytes"])
    total = int(facts["total_bytes"])
    need = int(needed_bytes * (ARCHIVE_FACTOR if archive else 1))
    if free < need + HEADROOM_BYTES:
        return DiskCheck(False, free, total, need, reason="剩余磁盘不足（需目标大小 + 10GB 余量）")
    warn = total > 0 and (free / total) < LOW_RATIO
    reason = "磁盘剩余低于 10%" if warn else ""
    return DiskCheck(True, free, total, need, reason=reason, warn=warn)
