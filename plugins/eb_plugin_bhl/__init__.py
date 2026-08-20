from eb.pluginhost import Context, PluginManifest

MANIFEST = PluginManifest(
    name="bhl",
    version="0.1.0",
    api_version="1",
    contributes=["validators", "ui"],
)


def setup(ctx: Context) -> None:
    ctx.provide("contracts", "contracts/")
