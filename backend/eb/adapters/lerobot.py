from eb.adapters.base import Adapter, RenderedJob


class LerobotAdapter(Adapter):
    name = "lerobot"

    def render(self, spec, run_id: str) -> RenderedJob:
        return RenderedJob(
            image=spec.image or "eb/tools:latest",
            command=["python", "-m", "lerobot.convert", "--run-id", run_id],
            env={},
            writable_paths=["outputs"],
        )
