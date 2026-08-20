from eb.adapters.base import Adapter, RenderedJob


class DeployAdapter(Adapter):
    name = "deploy"

    def render(self, spec, run_id: str) -> RenderedJob:
        return RenderedJob(
            image=spec.image or "eb/tools:latest",
            command=["python", "-m", "eb_tools.bundle", "--run-id", run_id],
            env={},
            writable_paths=["outputs"],
        )
