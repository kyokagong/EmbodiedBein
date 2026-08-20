import subprocess

from eb.worktree import overlay_mounts, prepare_worktree


def test_worktree_leaves_source_clean(tmp_path, monkeypatch):
    src = tmp_path / "repo"
    src.mkdir()
    subprocess.run(["git", "init"], cwd=src, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=src, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=src, check=True, capture_output=True)
    (src / "README").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=src, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=src, check=True, capture_output=True)

    data = tmp_path / "eb"
    monkeypatch.setenv("EB_DATA_ROOT", str(data))
    from eb.settings import get_settings

    get_settings.cache_clear()
    wt = prepare_worktree(src, commit=None, run_id="run_t")
    assert (src / ".git").exists()
    porcelain = subprocess.run(
        ["git", "-C", str(src), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert porcelain.strip() == ""
    assert wt.exists()
    assert (wt / "README").read_text() == "hi\n"
    (wt / "scratch").write_text("x")
    porcelain2 = subprocess.run(
        ["git", "-C", str(src), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert porcelain2.strip() == ""
    get_settings.cache_clear()


def test_seed_rsl_resume_copies_checkpoint(tmp_path, monkeypatch):
    from eb.worktree import seed_rsl_resume

    data = tmp_path / "eb"
    monkeypatch.setenv("EB_DATA_ROOT", str(data))
    from eb.settings import get_settings

    get_settings.cache_clear()
    src_dir = data / "runs" / "run_src" / "logs" / "rsl_rl" / "biped" / "2026-08-20_biped-smoke"
    src_dir.mkdir(parents=True)
    (src_dir / "model_0.pt").write_bytes(b"old")
    (src_dir / "model_4.pt").write_bytes(b"latest-ckpt")
    (src_dir / "params").mkdir()
    (src_dir / "params" / "env.yaml").write_text("ok\n")
    dest = data / "runs" / "run_export"
    dest.mkdir(parents=True)
    load_run, ckpt = seed_rsl_resume(dest, "run_src")
    assert load_run == "2026-08-20_biped-smoke"
    assert ckpt == "model_4.pt"
    copied = dest / "logs" / "rsl_rl" / "biped" / load_run / "model_4.pt"
    assert copied.read_bytes() == b"latest-ckpt"
    assert (dest / "logs" / "rsl_rl" / "biped" / load_run / "params" / "env.yaml").read_text() == "ok\n"
    get_settings.cache_clear()


def test_overlay_writable_binds(tmp_path):
    wt = tmp_path / "wt"
    run = tmp_path / "run"
    wt.mkdir()
    mounts = overlay_mounts(wt, run, ["logs", "outputs", "configs", "logs/rsl_rl"])
    modes = {m[2] for m in mounts}
    assert "ro" in modes and "rw" in modes
    assert (run / "logs" / "rsl_rl").is_dir()
