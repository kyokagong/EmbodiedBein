from __future__ import annotations

from dataclasses import dataclass, field

from eb.domain.job import JobKind, JobSpec


@dataclass
class RenderedJob:
    image: str
    command: list[str]
    env: dict[str, str]
    writable_paths: list[str]
    mounts: list[tuple[str, str, str]] = field(default_factory=list)
    shm_gb: int = 2
    devices: list[str] = field(default_factory=list)
    gpus: int = 0
    workdir: str = "/workspace"
    outputs: list[str] = field(default_factory=lambda: ["outputs"])
    metrics_glob: str = "logs/**/events.out.tfevents.*"


class Adapter:
    kind: JobKind
    name: str

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        raise NotImplementedError


def _image(spec: JobSpec, default: str) -> str:
    return spec.image or default
