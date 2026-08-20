from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class JobKind(StrEnum):
    DATA_SCAN = "data.scan"
    DATA_COMPILE = "data.compile"
    DATA_COLLECT = "data.collect"
    DATA_CONVERT = "data.convert"
    DATA_CANONICALIZE = "data.canonicalize"
    DATA_SLICE = "data.slice"
    DATA_RETARGET = "data.retarget"
    TRAIN_RL = "train.rl"
    TRAIN_SFT = "train.sft"
    TRAIN_RLFT = "train.rlft"
    EVAL_SIM = "eval.sim"
    EVAL_SIM2SIM = "eval.sim2sim"
    EXPORT_POLICY = "export.policy"
    BUNDLE_BUILD = "bundle.build"
    SERVE_POLICY = "serve.policy"
    REPORT_RENDER = "report.render"
    ASSET_VALIDATE = "asset.validate"


CONTAINERLESS_KINDS = {JobKind.DATA_SCAN, JobKind.DATA_SLICE}


class CodeRef(BaseModel):
    repo: str
    commit: str | None = None
    branch: str | None = None


class ResourceSpec(BaseModel):
    gpus: int = 0
    gpu_mode: str = "exclusive"
    cpus: int = 4
    memory_gb: int = 8
    shm_gb: int = 2
    devices: list[str] = Field(default_factory=list)


class PolicySpec(BaseModel):
    priority: int = 50
    preemptible: bool = False
    timeout_h: float = 24
    max_retries: int = 1
    retry_on: list[str] = Field(default_factory=lambda: ["infra_failed", "oom"])


class JobInputs(BaseModel):
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    base_model: str | None = None
    resume_from: str | None = None


class JobSpec(BaseModel):
    api_version: str = "eb/v1"
    kind: JobKind
    contract_id: str
    name: str = ""
    experiment: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    code_ref: CodeRef | None = None
    image: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    overrides: list[str] = Field(default_factory=list)
    inputs: JobInputs = Field(default_factory=JobInputs)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    policy: PolicySpec = Field(default_factory=PolicySpec)
    outputs: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    gates: list[str] = Field(default_factory=list)
    fanout: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _name_default(self) -> JobSpec:
        if not self.name:
            self.name = f"{self.kind}-{self.contract_id}"
        return self

    @property
    def task(self) -> str | None:
        t = self.params.get("task")
        return str(t) if t is not None else None

    def canonical_hash(self) -> str:
        from eb.ids import sha256_text

        payload = self.model_dump(mode="json", exclude={"name", "labels", "experiment"})
        return sha256_text(repr(sorted(payload.items())))
