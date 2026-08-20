from eb.pluginhost import Context, PluginManifest

MANIFEST = PluginManifest(
    name="rlinf",
    version="0.1.0",
    api_version="1",
    requires=["scheduler", "capacity"],
    contributes=["adapters", "estimators", "tools"],
)


def setup(ctx: Context) -> None:
    ctx.provide("adapter.rlinf", "eb.adapters.rlinf.RlinfAdapter")
