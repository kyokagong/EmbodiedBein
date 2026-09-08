from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from eb.domain.result import CheckResult, Violation
from eb.ids import sha256_text


class ObsTerm(BaseModel):
    term: str
    dim: int
    source: str = ""
    noise: str | None = None
    ref: str | None = None


class Modality(BaseModel):
    name: str
    shape: list[int]
    dtype: str = "uint8"


class Observation(BaseModel):
    dim: int
    history_length: int = 1
    concat: bool = True
    layout: list[ObsTerm] = Field(default_factory=list)
    modalities: list[Modality] = Field(default_factory=list)


class ActionTerm(BaseModel):
    name: str
    type: str
    dim: int
    scale: float | None = None
    indices: list[int] = Field(default_factory=list)


class Action(BaseModel):
    dim: int
    type: str | None = None
    scale: float | None = None
    offset: str | None = None
    absolute: bool = False
    terms: list[ActionTerm] = Field(default_factory=list)


class Control(BaseModel):
    policy_hz: int
    control_hz: int = 200
    sim_dt: float = 0.005
    decimation: int = 8
    max_policy_latency_ms: float = 12
    command_dim: int | None = None


class SimBindings(BaseModel):
    isaac_task: str | None = None
    experiment_name: str | None = None
    mjcf: str | None = None
    usd: str | None = None


class RealBindings(BaseModel):
    udp: dict[str, int] = Field(default_factory=dict)
    motor_bus: str | None = None
    firmware_n_joints: int | None = None


class EmbodimentEnv(BaseModel):
    id: str
    embodiment_id: str
    robot_family: str
    morphology: str
    interface_version: int = 1
    status: str = "draft"
    dof: int
    joint_order: list[str]
    joint_groups: dict[str, Any] = Field(default_factory=dict)
    observation: Observation
    action: Action
    control: Control
    sim_bindings: SimBindings = Field(default_factory=SimBindings)
    real_bindings: RealBindings = Field(default_factory=RealBindings)
    gains: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_ok(self) -> EmbodimentEnv:
        return self

    def spec_hash(self) -> str:
        return sha256_text(self.model_dump_json())

    def self_assertions(self) -> list[Violation]:
        v: list[Violation] = []
        layout_sum = sum(t.dim for t in self.observation.layout) * self.observation.history_length
        if self.observation.concat and layout_sum != self.observation.dim:
            v.append(
                Violation(
                    path="observation.dim",
                    expected=layout_sum,
                    actual=self.observation.dim,
                    hint="observation.dim must equal sum(layout.dim) * history_length",
                )
            )
        if self.action.terms:
            term_sum = sum(t.dim for t in self.action.terms)
            if term_sum != self.action.dim:
                v.append(
                    Violation(
                        path="action.dim",
                        expected=term_sum,
                        actual=self.action.dim,
                        hint="action.dim must equal sum of action terms",
                    )
                )
        elif self.action.dim != self.dof:
            v.append(
                Violation(
                    path="action.dim",
                    expected=self.dof,
                    actual=self.action.dim,
                    hint="single-term action.dim must equal dof",
                )
            )
        expected_hz = round(1 / (self.control.sim_dt * self.control.decimation))
        if expected_hz != self.control.policy_hz:
            v.append(
                Violation(
                    path="control.policy_hz",
                    expected=expected_hz,
                    actual=self.control.policy_hz,
                    hint="1/(sim_dt*decimation) must equal policy_hz",
                )
            )
        if len(self.joint_order) != self.dof:
            v.append(
                Violation(
                    path="joint_order",
                    expected=self.dof,
                    actual=len(self.joint_order),
                    hint="len(joint_order) must equal dof",
                )
            )
        return v

    def validate_self(self) -> CheckResult:
        vs = self.self_assertions()
        return CheckResult(passed=not vs, violations=vs)


def load_env(path: Path) -> EmbodimentEnv:
    data = yaml.safe_load(path.read_text())
    return EmbodimentEnv.model_validate(data)


def load_env_dir(directory: Path) -> list[EmbodimentEnv]:
    if not directory.exists():
        return []
    out: list[EmbodimentEnv] = []
    for p in sorted(directory.glob("ee_*.yaml")):
        out.append(load_env(p))
    return out
