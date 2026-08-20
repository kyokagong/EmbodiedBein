from eb.pluginhost import Context, PluginManifest

MANIFEST = PluginManifest(
    name="isaaclab",
    version="0.1.0",
    api_version="1",
    contributes=["adapters", "estimators", "knowledge"],
)


def setup(ctx: Context) -> None:
    from eb.adapters import KIND_ADAPTER
    from eb.adapters.isaaclab_rslrl import IsaacLabRslRlAdapter
    from eb.domain.job import JobKind

    ad = IsaacLabRslRlAdapter()
    KIND_ADAPTER[JobKind.TRAIN_RL] = ad
    KIND_ADAPTER[JobKind.EVAL_SIM] = ad
    ctx.provide("adapter.isaaclab", ad)

    def dispose() -> None:
        KIND_ADAPTER.pop(JobKind.TRAIN_RL, None)
        KIND_ADAPTER.pop(JobKind.EVAL_SIM, None)

    ctx.on_dispose(dispose)
