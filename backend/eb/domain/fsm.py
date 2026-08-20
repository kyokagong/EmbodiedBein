from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    DRAFT = "draft"
    BLOCKED = "blocked"
    QUEUED = "queued"
    ADMITTED = "admitted"
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INFRA_FAILED = "infra_failed"
    PREEMPTED = "preempted"


class RunStatus(StrEnum):
    PULLING_IMAGE = "pulling_image"
    PREPARING_FS = "preparing_fs"
    CREATED = "created"
    RUNNING = "running"
    COLLECTING_OUT = "collecting_out"
    GATING = "gating"
    FINISHED = "finished"
    ABORTED = "aborted"


JOB_TERMINAL = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.TIMEOUT,
    JobStatus.CANCELLED,
    JobStatus.INFRA_FAILED,
}

# from -> allowed to
JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.DRAFT: {JobStatus.BLOCKED, JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.BLOCKED: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.QUEUED: {JobStatus.ADMITTED, JobStatus.CANCELLED, JobStatus.BLOCKED},
    JobStatus.ADMITTED: {JobStatus.PREPARING, JobStatus.CANCELLED, JobStatus.INFRA_FAILED},
    JobStatus.PREPARING: {JobStatus.RUNNING, JobStatus.INFRA_FAILED, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
        JobStatus.CANCELLED,
        JobStatus.INFRA_FAILED,
        JobStatus.PREEMPTED,
    },
    JobStatus.PREEMPTED: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.INFRA_FAILED: {JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.FAILED: set(),
    JobStatus.SUCCEEDED: set(),
    JobStatus.TIMEOUT: set(),
    JobStatus.CANCELLED: set(),
}


class IllegalTransition(ValueError):
    pass


def transition(current: JobStatus, target: JobStatus) -> JobStatus:
    if current in JOB_TERMINAL and target not in JOB_TRANSITIONS.get(current, set()):
        raise IllegalTransition(f"{current} -> {target} is not allowed")
    allowed = JOB_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise IllegalTransition(f"{current} -> {target} is not allowed")
    return target
