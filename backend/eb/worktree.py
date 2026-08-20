from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from eb.settings import get_settings


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _copy_populated_submodules(src: Path, dest: Path) -> None:
    """Copy already-checked-out submodule trees. Do not fetch GitHub over SSH."""
    try:
        listing = _git(["submodule", "status", "--recursive"], cwd=src)
    except subprocess.CalledProcessError:
        return
    for line in listing.stdout.splitlines():
        parts = line.strip().lstrip("-+ ").split()
        if len(parts) < 2:
            continue
        rel = parts[1]
        sdir = src / rel
        ddir = dest / rel
        if not sdir.is_dir():
            continue
        if ddir.exists():
            populated = any(p.name != ".git" for p in ddir.iterdir())
            if populated:
                continue
            shutil.rmtree(ddir)
        ddir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sdir, ddir, symlinks=True, ignore=shutil.ignore_patterns(".git"))


def prepare_worktree(repo: Path, commit: str | None, run_id: str) -> Path:
    """Snapshot repo under /var/lib/eb/worktrees/<run_id>.

    Uses `git clone` into the data root (not `git worktree add`) so the source
    repo — including `.git` — is never written. External trees are bind-mounted
    read-only.
    """
    root = get_settings().data_root / "worktrees" / run_id
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        return root
    _git(["clone", "--local", str(repo), str(root)])
    target = commit or "HEAD"
    _git(["checkout", "--detach", target], cwd=root)
    try:
        _git(["submodule", "update", "--init", "--recursive"], cwd=root)
    except subprocess.CalledProcessError:
        pass
    _copy_populated_submodules(repo, root)
    return root


def overlay_mounts(worktree: Path, run_root: Path, writable_paths: list[str]) -> list[tuple[str, str, str]]:
    """Return (src, dst, mode). repo -> ro; each writable path bind from run_root/<path> rw."""
    mounts: list[tuple[str, str, str]] = [(str(worktree), "/workspace/isaaclab_extension_template", "ro")]
    for rel in writable_paths:
        host = run_root / rel
        host.mkdir(parents=True, exist_ok=True)
        mounts.append((str(host), f"/workspace/isaaclab_extension_template/{rel}", "rw"))
    return mounts


def prepare_run_dirs(run_id: str) -> Path:
    run_root = get_settings().data_root / "runs" / run_id
    for name in ("logs", "outputs", "configs", "logs/rsl_rl"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    return run_root
