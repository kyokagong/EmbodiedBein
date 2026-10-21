"""Two fixed, operator-only GPU fault diagnostics; never training or model evidence.

Keep imports usable by the Python 3.10 diagnostic image. Control-plane imports
are local, so the image needs neither the backend nor its dependency set.
"""
# ruff: noqa: UP017

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from eb.ufo_native import ENV_ID, plan_hash, read_json, safe_file

OPERATION = "ufo_failure_probe_v1"
SCHEMA = "eb/ufo-failure-probe-plan/v1"
RECEIPT_SCHEMA = "eb/ufo-failure-probe-receipt/v1"
ARTIFACT_KIND = "ufo_failure_probe_diagnostic"
IMAGE = "eb/ufo-native:failure-probe-v4"
BASE_IMAGE = "sha256:7a0babe5e5b8cff1b63bedee6c1746cb4710710a4305ec93667d8a52df58cb3e"
RLINF_HASH = "sha256:3cf50ddf948ab993361316647b338ba991e404c5c0c72e1c591782730d7ef676"
CASES = ("cuda_oom", "nccl_collective_timeout")
RESOURCES = dict(gpus=2, gpu_mode="exclusive", cpus=8, memory_gb=8, shm_gb=2, devices=[])
FILES = {
    "gpu-baseline.json",
    "identities.json",
    "rank-0.json",
    "rank-1.json",
    "fault-evidence.json",
    "fault-console.txt",
    "driver-cleanup.json",
}


def fixed_plan(case, image_digest, source_host_path):
    if (
        case not in CASES
        or not isinstance(image_digest, str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", image_digest)
    ):
        raise ValueError("Only the two fixed fault cases and a frozen diagnostic image are supported")
    if (
        not isinstance(source_host_path, str)
        or not source_host_path.startswith("/")
        or "\x00" in source_host_path
    ):
        raise ValueError("Frozen RLinf source is required")
    plan = dict(
        schema_version=SCHEMA,
        operation=OPERATION,
        case=case,
        world_size=2,
        image_digest=image_digest,
        base_image_digest=BASE_IMAGE,
        source_ref=dict(
            path="/sources/rlinf",
            manifest_path="/sources/rlinf-manifest.json",
            snapshot_hash=RLINF_HASH,
            host_path=source_host_path,
        ),
        job_timeout_seconds=120,
        watchdog_seconds=105,
        driver_timeout_seconds=90,
        collective_timeout_seconds=10,
    )
    plan["hash"] = plan_hash(plan)
    return plan


def validate_plan(plan):
    try:
        expected = fixed_plan(plan["case"], plan["image_digest"], plan["source_ref"]["host_path"])
        if "run_id" in plan:
            if not re.fullmatch(r"run_[a-zA-Z0-9_-]+", plan["run_id"]):
                raise ValueError("Invalid diagnostic Run ID")
            expected["run_id"] = plan["run_id"]
            expected["hash"] = plan_hash(expected)
        if plan != expected:
            raise ValueError("Fault plan differs from the fixed execution contract")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete fault plan") from exc
    return plan


def spec_for(plan):
    from eb.domain.job import JobSpec

    validate_plan(plan)
    return JobSpec(
        kind="asset.validate",
        embodiment_env_id=ENV_ID,
        name="UFO GPU failure diagnostic: " + plan["case"],
        image=plan["image_digest"],
        code_refs={
            "stack": {
                "kind": "content_snapshot",
                "path": plan["source_ref"]["host_path"],
                "commit": RLINF_HASH,
            }
        },
        params={"operation": OPERATION, "failure_probe_plan": plan},
        resources=RESOURCES,
        runtime={"network": "none", "hostname": "localhost"},
        policy={"timeout_h": 120 / 3600, "max_retries": 0, "retry_on": []},
    )


async def build_plan(case):
    from eb.settings import get_settings
    from eb.store.ebd import EbdClient

    if case not in CASES:
        raise ValueError("Only cuda_oom and nccl_collective_timeout are supported")
    root = get_settings().data_root / "snapshots" / RLINF_HASH.removeprefix("sha256:")
    if (
        not (root / "code").is_dir()
        or hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != RLINF_HASH[7:]
    ):
        raise ValueError("The approved frozen RLinf source is unavailable")
    image = await EbdClient().image_info(IMAGE)
    if image.get("ok") is not True:
        raise ValueError("Build the separate approved failure-probe image first")
    return fixed_plan(case, image.get("digest", ""), str(root / "code"))


async def validate_submission(session, spec):
    supplied = spec.params.get("failure_probe_plan") or {}
    current = await build_plan(supplied.get("case"))
    if supplied != current or spec.model_dump(mode="json") != spec_for(current).model_dump(mode="json"):
        raise ValueError("Fault diagnostic spec, image, source or fixed resource limits differ")


def render(spec, run_id):
    from eb.adapters.base import RenderedJob
    from eb.settings import get_settings

    plan = spec.params["failure_probe_plan"]
    if spec.model_dump(mode="json") != spec_for(plan).model_dump(mode="json"):
        raise ValueError("Unsupported fault diagnostic command or execution limits")
    root = get_settings().data_root / "runs" / run_id
    runtime_plan = {**plan, "run_id": run_id}
    runtime_plan["hash"] = plan_hash(runtime_plan)
    path = root / "failure-probe-plan.json"
    if run_id not in {"run_preview", "run_validate"}:
        encoded = json.dumps(runtime_plan, sort_keys=True, allow_nan=False)
        if path.is_symlink() or path.exists() and path.read_text() != encoded:
            raise ValueError("Frozen fault diagnostic Run plan changed")
        path.write_text(encoded)
    source = Path(plan["source_ref"]["host_path"])
    return RenderedJob(
        image=spec.image,
        command=[
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "100s",
            "python",
            "-m",
            "eb.runners.ufo_failure_probe",
            "--plan",
            "/failure-probe-plan.json",
        ],
        env={"PYTHONPATH": "/opt/eb-native", "PYTHONDONTWRITEBYTECODE": "1"},
        mounts=[
            (str(source), "/sources/rlinf", "ro"),
            (str(source.parent / "manifest.json"), "/sources/rlinf-manifest.json", "ro"),
            (str(root), "/run/eb-output", "rw"),
            (str(path), "/failure-probe-plan.json", "ro"),
        ],
        writable_paths=[str(root)],
        workdir="/run/eb-output",
        gpus=2,
        shm_gb=2,
    )


def fault_evidence(case, identities, states, native_text):
    if (
        len(identities) != 2
        or len(states) != 2
        or {r.get("rank") for r in identities} != {0, 1}
        or {r.get("rank") for r in states} != {0, 1}
        or len({r.get("gpu_uuid") for r in identities}) != 2
        or len({r.get("pid") for r in identities}) != 2
    ):
        raise ValueError("Fault requires two distinct, isolated real GPU ranks")
    if any(
        type(r.get("pid")) is not int
        or r["pid"] <= 0
        or type(r.get("total_bytes")) is not int
        or r["total_bytes"] <= 0
        or not isinstance(r.get("gpu_uuid"), str)
        or not r["gpu_uuid"]
        for r in identities
    ):
        raise ValueError("Fault hardware identity is missing")
    identities = sorted(identities, key=lambda r: r["rank"])
    states = sorted(states, key=lambda r: r["rank"])
    if any(
        any(s.get(k) != r.get(k) for k in ("rank", "pid", "gpu_uuid", "total_bytes"))
        for s, r in zip(states, identities)
    ):
        raise ValueError("Fault rank evidence changed identity")
    if case == "cuda_oom":
        state = states[0]
        if (
            state.get("exception_type") != "OutOfMemoryError"
            or type(state.get("requested_bytes")) is not int
            or state["requested_bytes"] != state["total_bytes"] + 1024**3
        ):
            raise ValueError("A real over-physical CUDA allocation OOM was not observed")
    elif case == "nccl_collective_timeout":
        native_timeout = any(
            re.search(r"WorkNCCL[^\n]*OpType=ALLREDUCE[^\n]*Timeout\(ms\)=10000", line)
            and ("collective operation timeout" in line.lower() or "before timing out" in line.lower())
            for line in native_text.splitlines()
        )
        if (
            not all(s.get("warmup_passed") is True for s in states)
            or states[0].get("collective_entered") is not True
            or states[1].get("skipping_collective") is not True
            or not native_timeout
        ):
            raise ValueError("Native NCCL ALLREDUCE timeout after a working peer collective was not observed")
    else:
        raise ValueError("Unsupported fault case")
    return {
        "case": case,
        "expected_failure_observed": True,
        "actual_optimizer_updates": 0,
        "host_or_cgroup_oom_claimed": False,
    }


async def verify_release(client, run, receipt):
    """Conservative post-exit proof using existing ebd APIs, without new Rust protocol."""
    runtime = await client.get_run(run.id)
    ps = await client.ps()
    doctor = await client.doctor()
    now = datetime.now(timezone.utc)
    started = run.started_at
    if started is None:
        raise ValueError("Diagnostic Run has no start clock")
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed = (now - started).total_seconds()
    assigned = set(run.gpu_ids or [])
    baseline = {r["index"]: r for r in receipt["gpu_baseline"]}
    devices = {r["index"]: r for r in doctor.get("gpus", []) if r.get("index") in assigned}
    if (
        ps.get("error")
        or doctor.get("error")
        or not 0 <= elapsed <= 120
        or len(assigned) != 2
        or set(baseline) != assigned
        or set(devices) != assigned
        or runtime.get("status") != "exited"
        or runtime.get("exit_code") != 0
        or runtime.get("oom_killed") is not False
        or runtime.get("image_digest") != receipt.get("image_digest")
        or any(r.get("run_id") == run.id and r.get("status") == "running" for r in ps.get("items", []))
        or any(
            devices[i].get("run_id") is not None
            or devices[i].get("memory_free_mb", -1) < baseline[i]["memory_free_mb"] - 64
            for i in assigned
        )
    ):
        raise ValueError("Container, GPU lease and memory release were not confirmed within 120 seconds")
    return {
        "verified": True,
        "observed_at": now.isoformat(),
        "elapsed_seconds": elapsed,
        "container": runtime,
        "gpus": list(devices.values()),
        "scope": "recorded actor processes exited (absent or zombie); container exited; "
        "GPU lease cleared and free memory restored",
        "host_gpu_process_list_queried": False,
    }


def read_receipt(outputs, plan, run_id):
    validate_plan(plan)
    expected = {**plan, "run_id": run_id}
    expected["hash"] = plan_hash(expected)
    receipt = read_json(safe_file(outputs, "failure-probe-receipt.json"))
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("operation") != OPERATION
        or receipt.get("run_id") != run_id
        or receipt.get("case") != plan["case"]
        or receipt.get("plan_hash") != expected["hash"]
        or receipt.get("image_digest") != plan["image_digest"]
        or receipt.get("base_image_digest") != BASE_IMAGE
        or receipt.get("diagnostic_only") is not True
        or receipt.get("model_published") is not False
        or receipt.get("actual_optimizer_updates") != 0
    ):
        raise ValueError("Fault receipt is not bound to its fixed diagnostic Run")
    files = receipt.get("files", [])
    if len(files) != len(FILES) or {r.get("path") for r in files} != FILES:
        raise ValueError("Fault diagnostic evidence is incomplete")
    for row in files:
        path = safe_file(outputs, row["path"])
        if (
            type(row.get("bytes")) is not int
            or not 0 < row["bytes"] <= 2 * 1024**2
            or path.stat().st_size != row["bytes"]
            or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))
        ):
            raise ValueError("Fault diagnostic file index is invalid")
    ranks = read_json(safe_file(outputs, "identities.json"))["ranks"]
    states = [read_json(safe_file(outputs, f"rank-{i}.json")) for i in range(2)]
    evidence = fault_evidence(
        plan["case"], ranks, states, safe_file(outputs, "fault-console.txt").read_text()
    )
    cleanup = read_json(safe_file(outputs, "driver-cleanup.json"))
    # Existing v3 successful Artifacts retain their original receipt/hash/scope.
    # New observations distinguish zombie entries from genuinely missing PIDs.
    if "observation_version" in cleanup:
        actors = cleanup.get("actors", [])
        ids = {rank["pid"] for rank in ranks}
        if (
            cleanup["observation_version"] != 2
            or len(actors) != len(ranks)
            or {a.get("pid") for a in actors} != ids
            or any(
                a.get("final_state") not in {"gone", "Z"}
                or a.get("identity_verified") is not True
                or not a.get("observations")
                or a["observations"][-1].get("state") != a["final_state"]
                or any(
                    a.get(key) != next(r for r in ranks if r["pid"] == a["pid"]).get(key)
                    for key in ("proc_start_ticks", "pid_namespace")
                )
                for a in actors
            )
            or cleanup.get("live_pids") != []
            or cleanup.get("unverified_pids") != []
            or cleanup.get("gone_pids") != [a["pid"] for a in actors if a["final_state"] == "gone"]
            or cleanup.get("zombie_pids") != [a["pid"] for a in actors if a["final_state"] == "Z"]
            or not isinstance(cleanup.get("forced_pids"), list)
            or not set(cleanup["forced_pids"]) <= ids
        ):
            raise ValueError("Fault diagnostic actor exit observations are incomplete or inconsistent")
    baseline = read_json(safe_file(outputs, "gpu-baseline.json"))["gpus"]
    if (
        read_json(safe_file(outputs, "fault-evidence.json")) != evidence
        or cleanup.get("complete") is not True
        or cleanup.get("remaining_actor_pids") != []
        or cleanup.get("errors") != []
        or baseline != receipt.get("gpu_baseline")
        or len(baseline) != 2
        or {r.get("gpu_uuid") for r in baseline} != {r["gpu_uuid"] for r in ranks}
    ):
        raise ValueError("Fault diagnostic cleanup or identity evidence is incomplete")
    return receipt


async def collect(session, job, run, outputs):
    from sqlalchemy import select

    from eb.ids import new_id
    from eb.orchestrator.ufo_native import verify_index
    from eb.store.ebd import EbdClient
    from eb.store.models import Artifact, LineageEdge

    plan = job.spec["params"]["failure_probe_plan"]
    receipt = read_receipt(outputs, plan, run.id)
    await verify_index(outputs, receipt["files"])
    artifact = await session.scalar(
        select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == ARTIFACT_KIND)
    )
    if artifact:
        meta = artifact.meta or {}
        if (
            artifact.sha256 != plan_hash({"receipt": receipt, "release": meta.get("release")})
            or meta.get("release", {}).get("verified") is not True
            or meta.get("release_hash") != plan_hash(meta["release"])
        ):
            raise ValueError("Immutable fault diagnostic Artifact changed")
        release = artifact.meta["release"]
    else:
        release = await verify_release(EbdClient(), run, receipt)
        artifact = Artifact(
            id=new_id("art"),
            kind=ARTIFACT_KIND,
            sha256=plan_hash({"receipt": receipt, "release": release}),
            size_bytes=sum(r["bytes"] for r in receipt["files"]),
            storage=str(outputs),
            run_id=run.id,
            pinned=True,
            meta={
                "receipt": receipt,
                "release": release,
                "release_hash": plan_hash(release),
                "diagnostic_only": True,
                "actual_optimizer_updates": 0,
            },
        )
        session.add(artifact)
        await session.flush()
        session.add(
            LineageEdge(
                src_type="run",
                src_id=run.id,
                dst_type="artifact",
                dst_id=artifact.id,
                relation="produced",
                run_id=run.id,
            )
        )
    run.metrics_summary = {
        **(run.metrics_summary or {}),
        "failure_probe": {
            "artifact_id": artifact.id,
            "case": plan["case"],
            "expected_failure_observed": True,
            "release_verified": True,
            "diagnostic_only": True,
            "actual_optimizer_updates": 0,
        },
    }


def api_router():
    """Local imports keep the contract available in the small diagnostic image."""
    from fastapi import APIRouter, Header, HTTPException, Request
    from pydantic import BaseModel, ConfigDict, Field
    from sqlalchemy import func, select

    from eb.api.deps import PrincipalDep, SessionDep
    from eb.domain.job import JobSpec
    from eb.store.models import Artifact, Job, Run

    router = APIRouter(prefix="/ufo-native/failure-probes")

    class ProbeIn(BaseModel):
        model_config = ConfigDict(extra="forbid")
        preview_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")

    def operator(principal):
        if principal.actor_type == "agent":
            raise HTTPException(403, "GPU fault diagnostics require an operator")

    @router.post("/{case}/preview")
    async def preview(case: str, body: ProbeIn, session: SessionDep, principal: PrincipalDep):
        operator(principal)
        try:
            plan = await build_plan(case)
            spec = spec_for(plan)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "plan": plan,
            "preview_hash": plan_hash(spec.model_dump(mode="json")),
            "resources": spec.resources.model_dump(),
            "policy": spec.policy.model_dump(),
            "world_size": 2,
            "actual_optimizer_updates": 0,
            "diagnostic_only": True,
            "qualification": "Expected GPU fault and bounded cleanup; no model or training baseline.",
        }

    @router.post("/{case}")
    async def submit(
        case: str,
        body: ProbeIn,
        session: SessionDep,
        principal: PrincipalDep,
        idem: str = Header(..., alias="Idempotency-Key"),
    ):
        operator(principal)
        if not 1 <= len(idem) <= 200:
            raise HTTPException(400, "Idempotency-Key must have 1..200 characters")
        try:
            spec = spec_for(await build_plan(case))
            expected = plan_hash(spec.model_dump(mode="json"))
            if body.preview_hash != expected:
                raise HTTPException(
                    409, "Fault diagnostic preview changed; review the image and fixed plan again"
                )
            key = "ufo-failure:" + hashlib.sha256((principal.id + ":" + idem).encode()).hexdigest()
            lock = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
            await session.execute(select(func.pg_advisory_xact_lock(lock)))
            previous = await session.scalar(select(Job).where(Job.idem_key == key))
            if previous:
                if (
                    previous.created_by != principal.id
                    or plan_hash(JobSpec.model_validate(previous.spec).model_dump(mode="json")) != expected
                ):
                    raise HTTPException(409, "Idempotency-Key binds a different fault diagnostic")
                return {"id": previous.id, "status": previous.status, "reused": True}
            from eb.orchestrator.service import Orchestrator

            job = await Orchestrator().submit(session, spec, principal.id, key)
            return {"id": job.id, "status": job.status, "reused": False}
        except (ValueError, OSError, PermissionError) as exc:
            raise HTTPException(400, str(exc)) from exc

    async def resolved(run_id, session):
        from eb.settings import get_settings

        run = await session.get(Run, run_id)
        job = await session.get(Job, run.job_id) if run else None
        artifact = await session.scalar(
            select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == ARTIFACT_KIND)
        )
        if (
            not artifact
            or not job
            or job.status != "succeeded"
            or job.spec.get("params", {}).get("operation") != OPERATION
        ):
            raise HTTPException(404, "No validated failure diagnostic Artifact")
        outputs = get_settings().data_root / "runs" / run_id / "outputs"
        try:
            receipt = read_receipt(outputs, job.spec["params"]["failure_probe_plan"], run_id)
            release = (artifact.meta or {}).get("release")
            if (
                not isinstance(release, dict)
                or release.get("verified") is not True
                or plan_hash({"receipt": receipt, "release": release}) != artifact.sha256
            ):
                raise ValueError("Diagnostic receipt or release proof changed")
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return outputs, receipt, artifact

    @router.get("/runs/{run_id}")
    async def diagnostic(run_id: str, session: SessionDep, principal: PrincipalDep):
        _, receipt, artifact = await resolved(run_id, session)
        return {
            "artifact_id": artifact.id,
            "sha256": artifact.sha256,
            "receipt": receipt,
            "release": artifact.meta["release"],
        }

    @router.get("/runs/{run_id}/files/{path:path}")
    async def output_file(
        run_id: str, path: str, request: Request, session: SessionDep, principal: PrincipalDep
    ):
        from eb.api.routes_ufo_native import stream_output
        from eb.ufo_native import sha256_file

        outputs, receipt, _ = await resolved(run_id, session)
        try:
            if path != "failure-probe-receipt.json":
                row = next((r for r in receipt["files"] if r["path"] == path), None)
                if not row or sha256_file(safe_file(outputs, path)) != row["sha256"]:
                    raise ValueError("Only unchanged indexed diagnostic files are downloadable")
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return await stream_output(request, run_id, path)

    return router
