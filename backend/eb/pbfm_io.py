"""Bounded metadata and output integrity shared by Python 3.10 PBFM Jobs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def hash_value(value: str, *, prefixed: bool = False) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(("sha256:" if prefixed else "") + r"[a-f0-9]{64}", value)
    )


def relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("invalid relative artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {".", ".."} for p in value.split("/")) or path.as_posix() != value:
        raise ValueError("artifact path escapes frozen root")
    return value


def safe_file(root: Path, name: str, *, maximum: int | None = None) -> Path:
    path = root / relative(name)
    if (
        not path.is_file()
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.resolve().is_relative_to(root.resolve())
        or (maximum is not None and path.stat().st_size > maximum)
    ):
        raise ValueError(f"missing, linked or oversized PBFM input: {name}")
    return path


def read_json(path: Path, maximum: int = 16 * 1024**2) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("missing, linked or oversized PBFM metadata")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not finite(value):
        raise ValueError("PBFM metadata must be a finite object")
    return value


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("refusing to overwrite immutable PBFM output")
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_index(root: Path) -> list[dict]:
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and "/runtime/" not in str(path)
    ]


def verify_index(root: Path, files: list[dict], required: set[str]) -> dict[str, dict]:
    if not isinstance(files, list) or not files or len(files) > 10000:
        raise ValueError("PBFM receipt has an invalid file index")
    indexed = {}
    for row in files:
        name = relative(row.get("path"))
        if (
            name in indexed
            or type(row.get("bytes")) is not int
            or row["bytes"] < 1
            or not hash_value(row.get("sha256"))
        ):
            raise ValueError("PBFM output index requires unique complete files")
        path = safe_file(root, name)
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"PBFM output integrity differs: {name}")
        indexed[name] = row
    if not required <= indexed.keys():
        raise ValueError("PBFM output receipt is incomplete")
    return indexed
