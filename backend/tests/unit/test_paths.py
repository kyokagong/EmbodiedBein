from pathlib import Path

import pytest

from eb.paths import PathNotAllowed, validate_allowed_path


def test_accepts_data_kk_prefix(tmp_path, monkeypatch):
    target = tmp_path / "Berkeley-Humanoid-Lite"
    target.mkdir()
    real = validate_allowed_path(str(target), prefixes=[str(tmp_path)])
    assert real == target.resolve()


def test_rejects_etc():
    with pytest.raises(PathNotAllowed):
        validate_allowed_path("/etc/passwd", prefixes=["/data/kk", "/var/lib/eb"])


def test_rejects_traversal_outside_prefix(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    sneaky = allowed / ".." / "secret"
    with pytest.raises(PathNotAllowed):
        validate_allowed_path(str(sneaky), prefixes=[str(allowed)])


def test_rejects_blank():
    with pytest.raises(PathNotAllowed):
        validate_allowed_path("  ", prefixes=["/data/kk"])
