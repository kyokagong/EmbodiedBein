import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from eb import ufo_failure_probe as probe
from eb.ufo_native import plan_hash

UTC = timezone.utc


def plan(case="cuda_oom"):
    return probe.fixed_plan(case, "sha256:" + "d" * 64, "/frozen/rlinf")


def test_nvml_uuid_matches_torch_uuid_without_losing_original_identity(monkeypatch):
    from eb.runners import ufo_failure_probe as entry

    native = "GPU-1b9c7093-3204-5084-3854-7ed540d2b1c5"
    monkeypatch.setattr(
        entry.subprocess, "run", Mock(return_value=NS(stdout=f"2, {native}, 24564, 24080, 1\n"))
    )
    rows = entry.gpu_baseline()
    assert rows == [
        {
            "index": 2,
            "gpu_uuid": native.removeprefix("GPU-"),
            "nvml_uuid": native,
            "memory_total_mb": 24564,
            "memory_free_mb": 24080,
            "memory_used_mb": 1,
        }
    ]


@pytest.mark.parametrize("case", probe.CASES)
def test_fixed_probe_uses_only_two_exclusive_gpus_and_no_training_inputs(case, monkeypatch, tmp_path):
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    spec = probe.spec_for(plan(case))
    assert spec.kind == "asset.validate" and spec.params["operation"] == probe.OPERATION
    assert spec.resources.model_dump() == probe.RESOURCES
    assert spec.policy.timeout_h == 120 / 3600 and spec.policy.max_retries == 0
    assert spec.inputs.model_dump() == {"datasets": [], "base_model": None, "resume_from": None}
    from eb.adapters import get_adapter
    from eb.capacity import estimate_job

    rendered = get_adapter(spec).render(spec, "run_preview")
    assert rendered.gpus == 2 and rendered.image == spec.image
    assert rendered.command[:5] == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "100s", "python"]
    assert {dst for _, dst, _ in rendered.mounts} == {
        "/sources/rlinf",
        "/sources/rlinf-manifest.json",
        "/run/eb-output",
        "/failure-probe-plan.json",
    }
    assert [dst for _, dst, mode in rendered.mounts if mode == "rw"] == ["/run/eb-output"]
    assert estimate_job(spec).per_gpu_peak_gb <= 2.1


@pytest.mark.parametrize(
    "key,value",
    [("world_size", 4), ("case", "python"), ("watchdog_seconds", 999), ("script", "arbitrary.py")],
)
def test_plan_rejects_any_unfrozen_execution_option(key, value):
    value_plan = plan()
    value_plan[key] = value
    value_plan["hash"] = plan_hash(value_plan)
    with pytest.raises(ValueError):
        probe.validate_plan(value_plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["gpu", "timeout", "mount", "input", "image"])
async def test_raw_job_submission_must_match_the_fixed_probe_spec(monkeypatch, change):
    value = plan()
    monkeypatch.setattr(probe, "build_plan", AsyncMock(return_value=value))
    spec = probe.spec_for(value)
    if change == "gpu":
        spec.resources.gpus = 4
    elif change == "timeout":
        spec.policy.timeout_h = 1
    elif change == "mount":
        from eb.domain.job import GitCodeRef

        spec.code_refs.embodiment = GitCodeRef(path="/arbitrary")
    elif change == "input":
        spec.inputs.datasets = [{"path": "/training-data"}]
    else:
        spec.image = probe.BASE_IMAGE
    with pytest.raises(ValueError):
        await probe.validate_submission(NS(), spec)


def test_expected_fault_evidence_requires_native_exception_and_both_ranks():
    ranks = [{"rank": i, "pid": 100 + i, "gpu_uuid": str(i), "total_bytes": 24 * 1024**3} for i in range(2)]
    states = [{**ranks[0], "exception_type": "OutOfMemoryError", "requested_bytes": 25 * 1024**3}, ranks[1]]
    assert probe.fault_evidence("cuda_oom", ranks, states, "")["expected_failure_observed"] is True
    states[0]["exception_type"] = "RuntimeError"
    with pytest.raises(ValueError):
        probe.fault_evidence("cuda_oom", ranks, states, "CUDA out of memory")
    with pytest.raises(ValueError):
        probe.fault_evidence("nccl_collective_timeout", ranks, ranks, "WorkNCCL ALLREDUCE Timeout")
    states = [{**rank, "warmup_passed": True} for rank in ranks]
    states[0]["collective_entered"] = True
    states[1]["skipping_collective"] = True
    with pytest.raises(ValueError):
        probe.fault_evidence("nccl_collective_timeout", ranks, states, "Python timeout")
    with pytest.raises(ValueError):
        probe.fault_evidence(
            "nccl_collective_timeout",
            ranks,
            states,
            "WorkNCCL(SeqNum=2, OpType=ALLREDUCE, Timeout(ms)=10000)",
        )
    evidence = probe.fault_evidence(
        "nccl_collective_timeout",
        ranks,
        states,
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=2, OpType=ALLREDUCE, Timeout(ms)=10000)",
    )
    assert evidence["expected_failure_observed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["late", "running", "gpu_busy", "memory", "missing"])
async def test_external_release_must_be_confirmed_before_diagnostic_publication(monkeypatch, failure):
    run = NS(id="run_probe", gpu_ids=[0, 1], started_at=datetime.now(UTC) - timedelta(seconds=30))
    receipt = {"gpu_baseline": [{"index": i, "memory_free_mb": 24000} for i in (0, 1)]}
    runtime = {"status": "exited", "exit_code": 0, "oom_killed": False}
    doctor = {"gpus": [{"index": i, "memory_free_mb": 24000, "run_id": None} for i in (0, 1)]}
    if failure == "late":
        run.started_at -= timedelta(seconds=121)
    elif failure == "running":
        runtime["status"] = "running"
    elif failure == "gpu_busy":
        doctor["gpus"][0]["run_id"] = run.id
    elif failure == "memory":
        doctor["gpus"][0]["memory_free_mb"] = 23000
    else:
        doctor["gpus"] = []
    client = NS(
        get_run=AsyncMock(return_value=runtime),
        doctor=AsyncMock(return_value=doctor),
        ps=AsyncMock(return_value={"items": []}),
    )
    with pytest.raises(ValueError):
        await probe.verify_release(client, run, receipt)


def test_probe_module_does_not_enter_frozen_training_image_copy_patterns():
    from fnmatch import fnmatch

    assert not fnmatch("ufo_failure_probe.py", "rlinf_ufo_*.py")
    assert not fnmatch("ufo_failure_probe.py", "ufo_native_*.py")
    dockerfile = (Path(__file__).parents[3] / "deploy/images/ufo-failure-probe.Dockerfile").read_text()
    assert "FROM eb/ufo-native:rpo-train-v3" in dockerfile
    assert "FROM sha256:" not in dockerfile


def endpoints():
    return {r.name: r.endpoint for r in probe.api_router().routes}


@pytest.mark.asyncio
async def test_api_preview_is_operator_only_and_rejects_extra_configuration(monkeypatch):
    from fastapi import HTTPException
    from pydantic import ValidationError

    from eb.api.deps import Principal

    routes = endpoints()
    body_class = routes["preview"].__annotations__["body"]
    with pytest.raises(ValidationError):
        body_class(script="anything.py")
    with pytest.raises(HTTPException) as exc:
        await routes["preview"]("cuda_oom", body_class(), None, Principal("bot", "agent"))
    assert exc.value.status_code == 403
    monkeypatch.setattr(probe, "build_plan", AsyncMock(return_value=plan()))
    response = await routes["preview"]("cuda_oom", body_class(), None, Principal("test", "user"))
    assert response["actual_optimizer_updates"] == 0 and response["world_size"] == 2
    assert response["policy"]["timeout_h"] * 3600 == 120


@pytest.mark.asyncio
async def test_api_stale_preview_and_idempotency_bind_case_image_and_actor(monkeypatch):
    from fastapi import HTTPException

    from eb.api.deps import Principal

    routes = endpoints()
    body_class = routes["submit"].__annotations__["body"]
    value, principal = plan(), Principal("test", "user")
    monkeypatch.setattr(probe, "build_plan", AsyncMock(return_value=value))
    session = NS(execute=AsyncMock(), scalar=AsyncMock())
    with pytest.raises(HTTPException) as exc:
        await routes["submit"](
            "cuda_oom", body_class(preview_hash="sha256:" + "0" * 64), session, principal, "key"
        )
    assert exc.value.status_code == 409
    session.execute.assert_not_awaited()
    spec = probe.spec_for(value)
    session.scalar.return_value = NS(
        id="job_old", status="cancelled", created_by=principal.id, spec=spec.model_dump(mode="json")
    )
    body = body_class(preview_hash=plan_hash(spec.model_dump(mode="json")))
    assert await routes["submit"]("cuda_oom", body, session, principal, "key") == {
        "id": "job_old",
        "status": "cancelled",
        "reused": True,
    }
    session.scalar.return_value.spec = probe.spec_for(plan("nccl_collective_timeout")).model_dump(mode="json")
    with pytest.raises(HTTPException) as exc:
        await routes["submit"]("cuda_oom", body, session, principal, "key")
    assert exc.value.status_code == 409


@pytest.fixture
def driver(tmp_path, monkeypatch):
    from eb.runners import ufo_failure_probe as entry

    original = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGUSR1)}
    value = plan()
    value["run_id"] = "run_failure"
    value["hash"] = plan_hash(value)
    state = NS(entry=entry, plan=value, root=tmp_path, kill=Mock(), shutdown=Mock(), stop=Mock())
    ranks = [dict(rank=i, pid=900000 + i, gpu_uuid=str(i), total_bytes=24 * 1024**3) for i in range(2)]
    state.ranks = ranks
    baseline = [
        dict(index=i, gpu_uuid=str(i), memory_total_mb=24564, memory_free_mb=24080, memory_used_mb=1)
        for i in range(2)
    ]
    state.baseline = baseline
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "prepare", lambda *_: None)
    monkeypatch.setattr(entry, "gpu_baseline", lambda: baseline)
    monkeypatch.setattr(entry, "native_console", lambda *_: "CUDA OOM diagnostics\n")
    for rank in ranks:
        entry.write(tmp_path / f"outputs/rank-{rank['rank']}.json", rank)

    def execute():
        row = {**ranks[0], "exception_type": "OutOfMemoryError", "requested_bytes": 25 * 1024**3}
        entry.write(tmp_path / "outputs/rank-0.json", row)
        raise RuntimeError("a worker failed")

    state.execute = execute
    group = NS(
        worker_info_list=[NS(worker=i) for i in (0, 1)],
        launch=Mock(),
        describe=lambda: NS(wait=lambda: ranks),
        execute=lambda: NS(wait=lambda: state.execute()),
    )
    cluster = NS(_distributed_log_collector=NS(stop=state.stop))
    monkeypatch.setattr(entry, "worker_class", lambda: NS(create_group=lambda _: group))
    monkeypatch.setitem(
        sys.modules,
        "ray",
        NS(
            kill=state.kill,
            shutdown=state.shutdown,
            nodes=lambda: [{"Alive": True}],
            _private=NS(
                worker=NS(
                    _global_node=NS(get_session_dir_path=lambda: str(tmp_path / "runtime/ray/session_test"))
                )
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.scheduler",
        NS(Cluster=lambda **_: cluster, PackedPlacementStrategy=lambda *_: None),
    )
    yield state
    signal.alarm(0)
    for sig, handler in original.items():
        signal.signal(sig, handler)


def test_harness_publishes_only_after_native_fault_and_every_actor_cleanup(driver):
    receipt = driver.entry.run(driver.plan)
    assert receipt["actual_optimizer_updates"] == 0 and receipt["model_published"] is False
    assert driver.kill.call_count == 2
    driver.stop.assert_called_once()
    driver.shutdown.assert_called_once()
    assert signal.alarm(0) == 0
    assert probe.read_receipt(driver.root / "outputs", plan(), "run_failure") == receipt


@pytest.mark.parametrize("used", [257, 1024])
def test_harness_refuses_actual_used_gpu_memory_before_injecting_fault(driver, used):
    driver.baseline[0]["memory_used_mb"] = used
    with pytest.raises(RuntimeError, match="Both assigned GPUs must be idle"):
        driver.entry.run(driver.plan)
    assert not (driver.root / "outputs/fault-evidence.json").exists()


@pytest.mark.parametrize("fault", ["success", "alarm", "other_error", "cleanup"])
def test_harness_never_publishes_unexpected_fault_or_incomplete_cleanup(driver, fault):
    if fault == "success":
        driver.execute = lambda: None
    elif fault == "alarm":
        driver.execute = lambda: os.kill(os.getpid(), signal.SIGALRM)
    elif fault == "other_error":

        def other_error():
            raise RuntimeError("not a CUDA OOM")

        driver.execute = other_error
    else:
        driver.stop.side_effect = OSError("collector failed")
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGUSR1)}
    with pytest.raises((RuntimeError, ValueError)):
        driver.entry.run(driver.plan)
    assert driver.kill.call_count == 2
    driver.shutdown.assert_called_once()
    assert not (driver.root / "outputs/failure-probe-receipt.json").exists()
    assert all(signal.getsignal(sig) == handler for sig, handler in old.items())


@pytest.mark.asyncio
async def test_collector_publishes_diagnostic_only_once_after_external_release(driver, monkeypatch):
    from eb.store.models import Artifact

    receipt = driver.entry.run(driver.plan)
    session = NS(scalar=AsyncMock(return_value=None), add=Mock(), flush=AsyncMock())
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", AsyncMock())
    release = {"verified": True, "scope": "test simulated ebd exit"}
    verifier = AsyncMock(return_value=release)
    monkeypatch.setattr(probe, "verify_release", verifier)
    run = NS(id="run_failure", metrics_summary={})
    job = NS(spec=probe.spec_for(plan()).model_dump(mode="json"))
    await probe.collect(session, job, run, driver.root / "outputs")
    added = [call.args[0] for call in session.add.call_args_list]
    assert len(added) == 2 and isinstance(added[0], Artifact)
    assert added[0].kind == probe.ARTIFACT_KIND
    assert added[0].sha256 == plan_hash({"receipt": receipt, "release": release})
    assert run.metrics_summary["failure_probe"]["actual_optimizer_updates"] == 0
    session.scalar.return_value = added[0]
    await probe.collect(session, job, run, driver.root / "outputs")
    verifier.assert_awaited_once()  # Recollection doesn't reinterpret later GPU occupants.
    assert session.add.call_count == 2


@pytest.mark.asyncio
async def test_collector_release_failure_does_not_publish_any_artifact_or_model(driver, monkeypatch):
    driver.entry.run(driver.plan)
    session = NS(scalar=AsyncMock(return_value=None), add=Mock(), flush=AsyncMock())
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", AsyncMock())
    monkeypatch.setattr(probe, "verify_release", AsyncMock(side_effect=ValueError("GPU not released")))
    with pytest.raises(ValueError, match="GPU not released"):
        await probe.collect(
            session,
            NS(spec=probe.spec_for(plan()).model_dump(mode="json")),
            NS(id="run_failure"),
            driver.root / "outputs",
        )
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_external_release_positive_proof_binds_image_and_no_ebd_errors():
    run = NS(id="run_probe", gpu_ids=[2, 3], started_at=datetime.now(UTC) - timedelta(seconds=45))
    receipt = {
        "image_digest": plan()["image_digest"],
        "gpu_baseline": [{"index": i, "memory_free_mb": 24000} for i in (2, 3)],
    }
    runtime = {
        "status": "exited",
        "exit_code": 0,
        "oom_killed": False,
        "image_digest": receipt["image_digest"],
    }
    doctor = {"gpus": [{"index": i, "memory_free_mb": 23980, "run_id": None} for i in (2, 3)]}
    client = NS(
        get_run=AsyncMock(return_value=runtime),
        doctor=AsyncMock(return_value=doctor),
        ps=AsyncMock(return_value={"items": []}),
    )
    assert (await probe.verify_release(client, run, receipt))["verified"] is True
    client.ps.return_value = {"items": [], "error": "ebd unavailable"}
    with pytest.raises(ValueError):
        await probe.verify_release(client, run, receipt)
    client.ps.return_value = {"items": []}
    runtime["image_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        await probe.verify_release(client, run, receipt)


@pytest.mark.parametrize(
    "damage", ["file_missing", "fault_type", "cleanup", "index_extra", "source_identity"]
)
def test_receipt_rejects_incomplete_or_misbound_evidence(driver, damage):
    driver.entry.run(driver.plan)
    outputs = driver.root / "outputs"
    if damage == "file_missing":
        (outputs / "fault-console.txt").unlink()
    elif damage == "fault_type":
        value = json.loads((outputs / "rank-0.json").read_text())
        value["exception_type"] = "RuntimeError"
        driver.entry.write(outputs / "rank-0.json", value)
    elif damage == "cleanup":
        driver.entry.write(
            outputs / "driver-cleanup.json", {"complete": False, "remaining_actor_pids": [100], "errors": []}
        )
    else:
        value = json.loads((outputs / "failure-probe-receipt.json").read_text())
        if damage == "index_extra":
            value["files"].append({"path": "arbitrary.py"})
        else:
            value["gpu_baseline"][0]["gpu_uuid"] = "unassigned GPU"
        driver.entry.write(outputs / "failure-probe-receipt.json", value)
    with pytest.raises((ValueError, OSError)):
        probe.read_receipt(outputs, plan(), "run_failure")


@pytest.mark.parametrize("exception_name", ["OutOfMemoryError", "RuntimeError"])
def test_worker_makes_single_overphysical_request_and_preserves_actual_exception(
    tmp_path, monkeypatch, exception_name
):
    from eb.runners import ufo_failure_probe as entry

    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "rlinf.scheduler", NS(Worker=object))
    worker = entry.worker_class()("cuda_oom")
    worker.state = {"rank": 0, "pid": 999999, "gpu_uuid": "test", "total_bytes": 24 * 1024**3}
    monkeypatch.setattr(worker, "describe", lambda: worker.state)
    failure_class = type(exception_name, (RuntimeError,), {})
    allocate = Mock(side_effect=failure_class("original native exception"))
    fake_torch = NS(
        empty=allocate,
        uint8="uint8",
        cuda=NS(set_device=Mock(), max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", NS())
    # Python resolves a.b through both sys.modules and the parent module.
    fake_torch.distributed = sys.modules["torch.distributed"]
    with pytest.raises(failure_class, match="original native exception"):
        worker.execute()
    allocate.assert_called_once_with(25 * 1024**3, dtype="uint8", device="cuda:0")
    state = json.loads((tmp_path / "outputs/rank-0.json").read_text())
    assert state["exception_type"] == exception_name


@pytest.mark.asyncio
async def test_raw_orchestrator_agent_cannot_reach_fault_runtime(monkeypatch):
    from eb.orchestrator.service import Orchestrator

    monkeypatch.setattr("eb.orchestrator.deletion.guard_storage_inputs", AsyncMock())
    with pytest.raises(PermissionError, match="operator"):
        await Orchestrator().submit(NS(), probe.spec_for(plan()), "agent:test", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("free_count", [1, 2])
async def test_probe_waits_for_two_platform_admitted_gpus(monkeypatch, free_count):
    from eb.scheduler.loop import Scheduler

    spec = probe.spec_for(plan())
    job = NS(
        spec=spec.model_dump(mode="json"), kind="asset.validate", embodiment_env_id=spec.embodiment_env_id
    )
    session = NS(scalars=AsyncMock(side_effect=[NS(all=lambda: []), NS(all=lambda: [job])]))
    admission = AsyncMock()
    scheduler = Scheduler(NS(admit_and_start=admission))
    scheduler.reap_running = AsyncMock()
    scheduler.advance_pipelines = AsyncMock()
    scheduler.ebd = NS(
        doctor=AsyncMock(
            return_value={
                "memory_available_mb": 64 * 1024,
                "gpus": [
                    {"index": i, "memory_free_mb": 24000, "run_id": None if i < free_count else "other_run"}
                    for i in range(4)
                ],
            }
        )
    )
    await scheduler.tick(session)
    if free_count == 2:
        admission.assert_awaited_once_with(session, job, [0, 1])
    else:
        admission.assert_not_awaited()


def test_openapi_exposes_case_only_requests_and_no_general_executor():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(probe.api_router())
    document = app.openapi()
    assert set(document["paths"]) == {
        "/ufo-native/failure-probes/{case}/preview",
        "/ufo-native/failure-probes/{case}",
        "/ufo-native/failure-probes/runs/{run_id}",
        "/ufo-native/failure-probes/runs/{run_id}/files/{path}",
    }
    schema = document["components"]["schemas"]["ProbeIn"]
    assert set(schema["properties"]) == {"preview_hash"} and schema["additionalProperties"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["fault-console.txt", "../../private", "model.safetensors"])
async def test_download_only_exposes_indexed_hash_bound_fault_files(driver, monkeypatch, path):
    from fastapi import HTTPException

    from eb.store.models import Run

    receipt = driver.entry.run(driver.plan)
    release = {"verified": True}
    artifact = NS(
        id="art_fault", sha256=plan_hash({"receipt": receipt, "release": release}), meta={"release": release}
    )
    run = NS(job_id="job_fault")
    job = NS(status="succeeded", spec=probe.spec_for(plan()).model_dump(mode="json"))
    session = NS(
        get=AsyncMock(side_effect=lambda cls, key: run if cls is Run else job),
        scalar=AsyncMock(return_value=artifact),
    )
    # Existing driver fixture output is relocated only by the settings root mapping.
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=driver.root))
    monkeypatch.setattr(probe, "read_receipt", lambda *_: receipt)
    monkeypatch.setattr(probe, "safe_file", lambda root, name: driver.root / "outputs" / name)
    stream = AsyncMock(return_value="response")
    monkeypatch.setattr("eb.api.routes_ufo_native.stream_output", stream)
    endpoint = endpoints()["output_file"]
    if path == "fault-console.txt":
        assert await endpoint("run_failure", path, None, session, None) == "response"
    else:
        with pytest.raises(HTTPException) as exc:
            await endpoint("run_failure", path, None, session, None)
        assert exc.value.status_code == 404
        stream.assert_not_awaited()


@pytest.fixture
def real_actor_child():
    """Own a real child in this test container; always reap it after the check."""
    from eb.runners import ufo_failure_probe as entry

    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        while True:
            signal.pause()
    state = entry.process_state(pid)
    assert state["state"] not in {"gone", "unknown", "Z"}
    identity = {
        "rank": 0,
        "pid": pid,
        "proc_start_ticks": state["start_ticks"],
        "pid_namespace": state["pid_namespace"],
    }
    try:
        yield entry, identity
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)


def test_proc_cleanup_accepts_a_real_zombie_without_signalling_or_claiming_pid_gone(
    real_actor_child, monkeypatch
):
    entry, identity = real_actor_child
    os.kill(identity["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 2
    while entry.process_state(identity["pid"])["state"] != "Z" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert Path(f"/proc/{identity['pid']}").exists()  # This was the real false-failure boundary.
    signal_pidfd = Mock(wraps=signal.pidfd_send_signal)
    monkeypatch.setattr(signal, "pidfd_send_signal", signal_pidfd)
    result = entry.cleanup_actor_processes([identity])
    assert result["complete"] is True
    assert result["zombie_pids"] == [identity["pid"]] and result["gone_pids"] == []
    assert result["forced_pids"] == result["live_pids"] == result["unverified_pids"] == []
    assert result["actors"][0]["observations"][-1]["state"] == "Z"
    signal_pidfd.assert_not_called()


def test_proc_cleanup_pidfd_kills_only_the_recorded_live_child_and_observes_its_exit(real_actor_child):
    entry, identity = real_actor_child
    started = time.monotonic()
    result = entry.cleanup_actor_processes([identity])
    assert time.monotonic() - started < 5.5
    assert result["complete"] is True and result["forced_pids"] == [identity["pid"]]
    assert result["zombie_pids"] == [identity["pid"]]
    assert result["live_pids"] == result["unverified_pids"] == []
    assert result["actors"][0]["observations"][-1]["state"] == "Z"


@pytest.mark.parametrize("mismatch", ["proc_start_ticks", "pid_namespace"])
def test_proc_cleanup_refuses_pid_reuse_or_another_namespace_without_sending_signal(
    real_actor_child, monkeypatch, mismatch
):
    entry, identity = real_actor_child
    identity[mismatch] = identity[mismatch] + 1 if mismatch == "proc_start_ticks" else "pid:[different]"
    signal_pidfd = Mock(wraps=signal.pidfd_send_signal)
    monkeypatch.setattr(signal, "pidfd_send_signal", signal_pidfd)
    result = entry.cleanup_actor_processes([identity])
    assert result["complete"] is False and result["unverified_pids"] == [identity["pid"]]
    assert result["forced_pids"] == []
    assert entry.process_state(identity["pid"])["state"] not in {"gone", "Z"}
    signal_pidfd.assert_not_called()


def test_proc_cleanup_signal_failure_keeps_live_child_and_reports_failure(real_actor_child, monkeypatch):
    entry, identity = real_actor_child
    monkeypatch.setattr(signal, "pidfd_send_signal", Mock(side_effect=PermissionError("denied pidfd signal")))
    result = entry.cleanup_actor_processes([identity])
    assert result["complete"] is False and result["live_pids"] == [identity["pid"]]
    assert result["forced_pids"] == [] and "denied pidfd signal" in result["errors"][0]["error"]


def test_proc_cleanup_bounds_wait_to_five_seconds_if_recorded_live_child_does_not_exit(
    real_actor_child, monkeypatch
):
    entry, identity = real_actor_child
    monkeypatch.setattr(signal, "pidfd_send_signal", Mock())  # Acknowledged signal with no observed exit.
    started = time.monotonic()
    result = entry.cleanup_actor_processes([identity])
    assert 5 <= time.monotonic() - started < 6
    assert result["complete"] is False and result["live_pids"] == [identity["pid"]]
    assert result["forced_pids"] == [identity["pid"]]


def test_actor_cleanup_failure_never_publishes_a_diagnostic_receipt(driver, monkeypatch):
    monkeypatch.setattr(
        driver.entry,
        "cleanup_actor_processes",
        lambda _: {
            "complete": False,
            "actors": [],
            "forced_pids": [],
            "zombie_pids": [],
            "gone_pids": [],
            "live_pids": [900001],
            "unverified_pids": [],
            "remaining_actor_pids": [900001],
            "errors": [],
        },
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        driver.entry.run(driver.plan)
    assert not (driver.root / "outputs/failure-probe-receipt.json").exists()


@pytest.mark.asyncio
async def test_existing_v3_artifact_get_preserves_original_cleanup_scope_and_hash(driver, monkeypatch):
    import hashlib

    from eb.store.models import Run

    receipt = driver.entry.run(driver.plan)
    outputs = driver.root / "outputs"
    old_cleanup = {"complete": True, "errors": [], "remaining_actor_pids": [], "elapsed_seconds": 27.21}
    driver.entry.write(outputs / "driver-cleanup.json", old_cleanup)
    row = next(row for row in receipt["files"] if row["path"] == "driver-cleanup.json")
    raw = (outputs / "driver-cleanup.json").read_bytes()
    row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    driver.entry.write(outputs / "failure-probe-receipt.json", receipt)
    release = {
        "verified": True,
        "scope": "actor PIDs gone inside task; container exited; GPU lease cleared and free memory restored",
    }
    artifact = NS(
        id="art_v3", sha256=plan_hash({"receipt": receipt, "release": release}), meta={"release": release}
    )
    original_hash = artifact.sha256
    run = NS(job_id="job_v3")
    job = NS(status="succeeded", spec=probe.spec_for(plan()).model_dump(mode="json"))
    session = NS(
        get=AsyncMock(side_effect=lambda cls, key: run if cls is Run else job),
        scalar=AsyncMock(return_value=artifact),
        add=Mock(),
        commit=AsyncMock(),
    )
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=driver.root))
    reader = probe.read_receipt
    monkeypatch.setattr(probe, "read_receipt", lambda *_: reader(outputs, plan(), "run_failure"))
    result = await endpoints()["diagnostic"]("run_failure", session, None)
    assert result["sha256"] == original_hash and result["release"] == release
    assert artifact.sha256 == original_hash
    assert json.loads((outputs / "driver-cleanup.json").read_text()) == old_cleanup
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


@pytest.mark.parametrize("field", ["live_pids", "unverified_pids", "zombie_pids", "gone_pids"])
def test_receipt_rejects_contradictory_actor_exit_classification(driver, field):
    import hashlib

    driver.entry.run(driver.plan)
    path = driver.root / "outputs/driver-cleanup.json"
    value = json.loads(path.read_text())
    value[field] = [900001] if field != "gone_pids" else []
    driver.entry.write(path, value)
    receipt_path = driver.root / "outputs/failure-probe-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    row = next(row for row in receipt["files"] if row["path"] == path.name)
    row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    driver.entry.write(receipt_path, receipt)
    with pytest.raises(ValueError, match="observations"):
        probe.read_receipt(driver.root / "outputs", plan(), "run_failure")
