from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    username: Mapped[str] = mapped_column(Text, unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    actor_type: Mapped[str] = mapped_column(Text, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    principal: Mapped[str] = mapped_column(Text)
    token_hash: Mapped[str] = mapped_column(Text, unique=True)
    actor_type: Mapped[str] = mapped_column(Text, default="user")
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbodimentContractRow(Base):
    __tablename__ = "embodiment_contracts"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    spec_hash: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="draft")
    obs_dim: Mapped[int | None] = mapped_column(Integer)
    action_dim: Mapped[int | None] = mapped_column(Integer)
    policy_hz: Mapped[int | None] = mapped_column(Integer)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    contract_id: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    experiment_id: Mapped[str | None] = mapped_column(Text, ForeignKey("experiments.id"))
    contract_id: Mapped[str] = mapped_column(Text, ForeignKey("embodiment_contracts.id"))
    kind: Mapped[str] = mapped_column(Text)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    spec_hash: Mapped[str] = mapped_column(Text)
    idem_key: Mapped[str | None] = mapped_column(Text, unique=True)
    status: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=50)
    version: Mapped[int] = mapped_column(Integer, default=1)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[str] = mapped_column(Text)
    gpu_seconds: Mapped[int] = mapped_column(BigInteger, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_jobs_sched", "status", "priority", "created_at"),)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    job_id: Mapped[str] = mapped_column(Text, ForeignKey("jobs.id"))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(Text)
    abort_reason: Mapped[str | None] = mapped_column(Text)
    container_id: Mapped[str | None] = mapped_column(Text)
    gpu_ids: Mapped[list[Any] | None] = mapped_column(JSONB)
    metrics_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    provenance: Mapped[str] = mapped_column(Text, default="full")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    sha256: Mapped[str] = mapped_column(Text, index=True)
    kind: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    storage: Mapped[str] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("runs.id"))
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    refcount: Mapped[int] = mapped_column(Integer, default=0)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    discovered_at: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    format: Mapped[str] = mapped_column(Text, default="unknown")
    license: Mapped[str | None] = mapped_column(Text)
    contract_id: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(Text, ForeignKey("datasets.id"))
    version: Mapped[str] = mapped_column(Text)
    compatible_contracts: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    funnel: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    content_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)


class ModelFamily(Base):
    __tablename__ = "model_families"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    family: Mapped[str] = mapped_column(Text, default="mlp")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    family_id: Mapped[str] = mapped_column(Text, ForeignKey("model_families.id"))
    name: Mapped[str] = mapped_column(Text)
    contract_id: Mapped[str | None] = mapped_column(Text)
    io_signature: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    mem_profile: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    card: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    tags: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    provenance: Mapped[str] = mapped_column(Text, default="full")
    artifact_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LineageEdge(Base):
    __tablename__ = "lineage_edges"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    src_type: Mapped[str] = mapped_column(Text)
    src_id: Mapped[str] = mapped_column(Text)
    dst_type: Mapped[str] = mapped_column(Text)
    dst_id: Mapped[str] = mapped_column(Text)
    relation: Mapped[str] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("src_type", "src_id", "dst_type", "dst_id", "relation"),
        Index("ix_lineage_src", "src_type", "src_id"),
        Index("ix_lineage_dst", "dst_type", "dst_id"),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    principal: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    trace_id: Mapped[str | None] = mapped_column(Text)
    approval_id: Mapped[str | None] = mapped_column(Text)


class EvalSuite(Base):
    __tablename__ = "eval_suites"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    contract_id: Mapped[str] = mapped_column(Text)
    seeds: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    frozen: Mapped[bool] = mapped_column(Boolean, default=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class EvalReport(Base):
    __tablename__ = "eval_reports"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    suite_id: Mapped[str] = mapped_column(Text, ForeignKey("eval_suites.id"))
    model_version_id: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    gate_results: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Bundle(Base):
    __tablename__ = "bundles"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_version_id: Mapped[str | None] = mapped_column(Text)
    contract_id: Mapped[str] = mapped_column(Text)
    stage: Mapped[str] = mapped_column(Text, default="S0")
    signed: Mapped[bool] = mapped_column(Boolean, default=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Robot(Base):
    __tablename__ = "robots"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    contract_id: Mapped[str] = mapped_column(Text)
    calibration: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Share(Base):
    __tablename__ = "shares"
    token: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentSession(Base):
    __tablename__ = "agent_sessions"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, default="active")
    autonomy: Mapped[str] = mapped_column(Text, default="suggest")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentStep(Base):
    __tablename__ = "agent_steps"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, ForeignKey("agent_sessions.id"))
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    tool: Mapped[str | None] = mapped_column(Text)
    guardrail_result: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Pipeline(Base):
    __tablename__ = "pipelines"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FailureKnowledge(Base):
    __tablename__ = "failure_knowledge"
    signature: Mapped[str] = mapped_column(Text, primary_key=True)
    cause: Mapped[str] = mapped_column(Text)
    fix: Mapped[str] = mapped_column(Text)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
