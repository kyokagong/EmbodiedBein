from eb.pluginhost import PluginHost


def test_plugins_load_and_unload():
    host = PluginHost()

    def setup(ctx):
        ctx.provide("x", 1)

    host.load("demo", setup)
    assert "demo" in host.plugins
    host.unload("demo")
    assert "demo" not in host.plugins
