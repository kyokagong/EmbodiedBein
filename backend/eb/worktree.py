from __future__ import annotations

from pathlib import Path

from eb.settings import get_settings
from eb.store.ebd import EbdClient, storage_inventory


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


async def seed_rsl_resume(run_root: Path, source_run_id: str) -> tuple[str, str]:
    """Choose RSL input by its existing mtime/step rule; ebd prepares the bytes."""
    src = get_settings().data_root / "runs" / source_run_id / "logs" / "rsl_rl"
    facts = await storage_inventory(src)
    files = [item for item in facts if not item["is_dir"] and not item["is_symlink"]]
    points = [item for item in files if Path(item["path"]).match("model_*.pt")]
    if not points:
        raise FileNotFoundError(f"no model_*.pt under {src}")

    def checkpoint_order(item):
        step = Path(item["path"]).stem.removeprefix("model_")
        return (item["mtime_ns"], int(step) if step.isdigit() else -1)

    latest = Path(max(points, key=checkpoint_order)["path"])
    load_run = latest.parent.name
    directory = latest.parent
    destination = Path("logs/rsl_rl") / latest.parent.parent.name / load_run
    members = []
    for item in files:
        path = Path(item["path"])
        if not path.is_relative_to(directory):
            continue
        relative = path.relative_to(directory)
        if ((len(relative.parts) == 1 and path.suffix in {".pt", ".yaml", ".pkl", ".json"})
                or len(relative.parts) > 1 and relative.parts[0] == "params"):
            members.append({"source_path": (Path("logs/rsl_rl") / path).as_posix(),
                            "target_path": (destination / relative).as_posix(),
                            "expected_bytes": item["bytes"], "expected_sha256": item.get("sha256")})
    await EbdClient().prepare_resume_files(source_run_id, run_root.name, members)
    return load_run, latest.name
