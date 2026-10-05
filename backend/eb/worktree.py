from __future__ import annotations

import shutil
from pathlib import Path

from eb.settings import get_settings


def git_is_dirty(repo: Path) -> None:
    """Compatibility probe: unknown; the platform does not execute Git."""
    return None


def resolve_commit(repo: Path, commit: str | None = None) -> str:
    """Legacy callers must not mistake a content snapshot for a Git revision."""
    raise ValueError("Git revision resolution is unavailable; use an ebd content snapshot")


async def prepare_worktree(
    repo: Path, commit: str | None, run_id: str, slug: str = "bhl", *, ebd=None
) -> Path:
    """Compatibility name: ebd freezes current bytes; commit only accepts sha256:<hash>."""
    from eb.store.ebd import EbdClient

    result = await (ebd or EbdClient()).snapshot(str(repo), commit)
    return Path(result["path"])


def overlay_mounts(worktree: Path, run_root: Path, writable_paths: list[str]) -> list[tuple[str, str, str]]:
    """Return (src, dst, mode). repo -> ro; each writable path bind from run_root/<path> rw."""
    mounts: list[tuple[str, str, str]] = [(str(worktree), "/workspace/isaaclab_extension_template", "ro")]
    for rel in writable_paths:
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not Path(rel).parts:
            raise ValueError("writable overlay must be a relative path inside the run")
        host = run_root / rel
        if not host.resolve().is_relative_to(run_root.resolve()):
            raise ValueError("writable overlay escapes run root")
        host.mkdir(parents=True, exist_ok=True)
        mounts.append((str(host), f"/workspace/isaaclab_extension_template/{rel}", "rw"))
    return mounts


def prepare_run_dirs(run_id: str) -> Path:
    run_root = get_settings().data_root / "runs" / run_id
    for name in ("logs", "outputs", "configs", "logs/rsl_rl"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    return run_root


def seed_rsl_resume(run_root: Path, source_run_id: str) -> tuple[str, str]:
    """Copy the newest RSL-RL checkpoint from another run into this run's logs overlay.

    Returns (load_run directory name, checkpoint filename) for play.py / train.py.
    """
    src = get_settings().data_root / "runs" / source_run_id / "logs" / "rsl_rl"
    pts = list(p for p in src.rglob("model_*.pt") if p.is_file())
    if not pts:
        raise FileNotFoundError(f"no model_*.pt under {src}")

    def _ckpt_sort(p: Path) -> tuple[float, int]:
        n = -1
        stem = p.stem
        if stem.startswith("model_"):
            try:
                n = int(stem.split("_", 1)[1])
            except ValueError:
                n = -1
        return (p.stat().st_mtime, n)

    latest = max(pts, key=_ckpt_sort)
    load_run = latest.parent.name
    dest = run_root / "logs" / "rsl_rl" / latest.parent.parent.name / load_run
    dest.mkdir(parents=True, exist_ok=True)
    for item in latest.parent.iterdir():
        target = dest / item.name
        if item.is_file() and item.suffix in {".pt", ".yaml", ".pkl", ".json"}:
            shutil.copy2(item, target)
        elif item.is_dir() and item.name == "params":
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
    return load_run, latest.name
