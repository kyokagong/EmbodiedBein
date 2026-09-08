from __future__ import annotations

from pathlib import Path


class PathNotAllowed(ValueError):
    pass


def default_prefixes() -> list[str]:
    from eb.settings import get_settings

    raw = get_settings().allowed_path_prefixes
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return list(raw)


def validate_allowed_path(path: str, prefixes: list[str] | None = None) -> Path:
    text = (path or "").strip()
    if not text:
        raise PathNotAllowed("路径为空")
    prefixes = prefixes if prefixes is not None else default_prefixes()
    try:
        real = Path(text).expanduser().resolve()
    except OSError as e:
        raise PathNotAllowed(f"无法解析路径: {e}") from e
    allowed = [Path(p).expanduser().resolve() for p in prefixes]
    if not any(_is_relative_to(real, root) for root in allowed):
        raise PathNotAllowed(f"路径不在允许前缀内: {real}")
    return real


def is_under(root: Path, candidate: Path) -> bool:
    try:
        dest = candidate.resolve()
        dest.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
