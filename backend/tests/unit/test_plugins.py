from eb.adapters import KIND_ADAPTER
from eb.domain.job import JobKind
from eb.pluginhost import PluginHost


def test_plugins_load_and_unload(monkeypatch):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "plugins"
    sys.path.insert(0, str(root))
    from eb_plugin_isaaclab import setup

    saved = dict(KIND_ADAPTER)
    host = PluginHost()
    host.load("isaaclab", setup)
    assert "isaaclab" in host.plugins
    assert JobKind.TRAIN_RL in KIND_ADAPTER
    host.unload("isaaclab")
    assert "isaaclab" not in host.plugins
    assert JobKind.TRAIN_RL not in KIND_ADAPTER
    KIND_ADAPTER.update(saved)
