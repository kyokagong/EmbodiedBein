from eb.domain.disk import disk_preflight


def test_preflight_blocks_when_free_below_need_plus_headroom(monkeypatch):
    def fake_statvfs(_path: str):
        class S:
            f_bavail = 100
            f_blocks = 1000
            f_frsize = 1_000_000  # 100 MB free, 1 GB total

        return S()

    monkeypatch.setattr("os.statvfs", fake_statvfs)
    result = disk_preflight("/var/lib/eb/datasets", needed_bytes=50_000_000, archive=False)
    assert result.ok is False
    assert "不足" in result.reason


def test_preflight_ok_when_plenty(monkeypatch):
    def fake_statvfs(_path: str):
        class S:
            f_bavail = 80_000
            f_blocks = 100_000
            f_frsize = 1_000_000  # 80 GB free

        return S()

    monkeypatch.setattr("os.statvfs", fake_statvfs)
    result = disk_preflight("/var/lib/eb/datasets", needed_bytes=1_000_000_000, archive=False)
    assert result.ok is True
    assert result.free_bytes > 0
