from eb.pluginhost import Context, PluginManifest

MANIFEST = PluginManifest(
    name="isaaclab",
    version="0.1.0",
    api_version="1",
    contributes=["adapters", "estimators", "knowledge"],
)


def setup(ctx: Context) -> None:
    ctx.provide("adapter.isaaclab", "eb.adapters.isaaclab_rslrl.IsaacLabRslRlAdapter")
