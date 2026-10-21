"""Actual driver signals and a CPU c10d timeout, without GPU training."""

import json
import os
import signal
import sys
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver(tmp_path, monkeypatch):
    from eb.runners import rlinf_ufo_entry as entry
    from eb.runners.rlinf_ufo_runtime import write_json

    original_signals = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGUSR1)}
    state = SimpleNamespace(entry=entry, root=tmp_path, stage=None)
    state.plan = {"run_id": "run_boundaries", "world_size": 2,
                  "communication_timeout_seconds": 1, "segment_timeout_seconds": 2}
    state.on_segment = lambda: [{"loader_verified": True} for _ in range(2)]
    for name in ("validate_plan", "verify_source_refs", "configure_environment",
                 "verify_training_data", "prepare_segment"):
        monkeypatch.setattr(entry, name, lambda *_: None)
    monkeypatch.setattr(entry, "parse_native_args", lambda *_: {})
    monkeypatch.setattr(entry, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(entry, "NATIVE_ROOT", tmp_path / "outputs/native")
    state.kill, state.shutdown = Mock(), Mock()
    ray = SimpleNamespace(
        _private=SimpleNamespace(worker=SimpleNamespace(_global_node=SimpleNamespace(
            get_session_dir_path=lambda: str(tmp_path / "runtime/ray/session_test")))),
        nodes=lambda: [{"Alive": True, "NodeID": "local", "NodeManagerAddress": "127.0.0.1"}],
        kill=state.kill, shutdown=state.shutdown)
    monkeypatch.setitem(sys.modules, "ray", ray)

    def block_at(stage):
        if state.stage == stage:
            time.sleep(2)  # The production one-second SIGALRM must interrupt this.

    def identities():
        block_at("describe")
        return [{"rank": rank, "gpu_uuid": str(rank)} for rank in range(2)]

    def segment():
        block_at("segment")
        return state.on_segment()

    state.group = SimpleNamespace(
        worker_info_list=[SimpleNamespace(worker=rank) for rank in range(2)],
        launch=Mock(side_effect=lambda **_: block_at("launch")),
        describe_runtime=lambda: SimpleNamespace(wait=identities),
        run_segment=Mock(return_value=SimpleNamespace(wait=segment)))
    state.cluster = SimpleNamespace(_distributed_log_collector=None)

    def cluster(**_):
        block_at("cluster")
        return state.cluster

    cluster.NAMESPACE = "boundary-test"
    monkeypatch.setitem(sys.modules, "rlinf.scheduler", SimpleNamespace(
        Cluster=cluster, PackedPlacementStrategy=lambda *_: None))
    monkeypatch.setitem(sys.modules, "eb.runners.rlinf_ufo_worker", SimpleNamespace(
        UfoTrainWorker=SimpleNamespace(create_group=lambda _: state.group)))

    def checkpoint(*_, **__):
        write_json(tmp_path / "outputs/receipt.json", {"status": "complete"})
        return {"status": "complete"}

    state.publish = Mock(side_effect=checkpoint)
    monkeypatch.setitem(sys.modules, "eb.ufo_training_checkpoint", SimpleNamespace(
        write_native_checkpoint_receipt=state.publish))
    monkeypatch.setitem(sys.modules, "eb.runners.ufo_training_analysis", SimpleNamespace(
        build_training_evidence=lambda *_: block_at("analysis")))
    yield state
    # Restore even when the test exposes a broken production cleanup path.
    signal.alarm(0)
    for number, handler in original_signals.items():
        signal.signal(number, handler)


def assert_failed_cleanup(driver, *, actors=True):
    assert driver.kill.call_count == (2 if actors else 0)
    if actors:
        assert [call.args[0] for call in driver.kill.call_args_list] == [0, 1]
    driver.shutdown.assert_called_once()
    assert not (driver.root / "outputs/receipt.json").exists()
    assert not (driver.root / "outputs/probe-receipt.json").exists()
    assert signal.alarm(0) == 0


@pytest.mark.parametrize("stage", ["cluster", "launch", "describe"])
def test_real_startup_alarm_bounds_owned_ray_and_worker_cleanup(driver, stage):
    driver.stage = stage
    before = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=f"signal {signal.SIGALRM}"):
        driver.entry.run(driver.plan)
    assert time.monotonic() - started < 2
    driver.publish.assert_not_called()
    assert_failed_cleanup(driver, actors=stage != "cluster")
    assert signal.getsignal(signal.SIGALRM) == before


@pytest.mark.parametrize("stage", ["segment", "analysis"])
def test_real_segment_alarm_records_failure_and_never_leaves_success_receipt(driver, monkeypatch, stage):
    driver.stage = stage
    path = driver.root / "plan.json"
    path.write_text(json.dumps(driver.plan))
    monkeypatch.setattr(sys, "argv", ["rlinf_ufo_entry", "--plan", str(path)])
    with pytest.raises(RuntimeError, match=f"signal {signal.SIGALRM}"):
        driver.entry.main()
    failure = json.loads((driver.root / "outputs/runtime-failure.json").read_text())
    assert failure["status"] == "failed" and failure["run_id"] == driver.plan["run_id"]
    assert failure["error_type"] == "RuntimeError"
    assert driver.publish.call_count == (1 if stage == "analysis" else 0)
    assert_failed_cleanup(driver)


def test_worker_failure_signal_cleans_every_owned_actor_without_publication(driver):
    previous = signal.getsignal(signal.SIGUSR1)
    driver.on_segment = lambda: os.kill(os.getpid(), signal.SIGUSR1)
    with pytest.raises(RuntimeError, match=f"signal {signal.SIGUSR1}"):
        driver.entry.run(driver.plan)
    driver.publish.assert_not_called()
    assert_failed_cleanup(driver)
    assert signal.getsignal(signal.SIGUSR1) == previous


def test_elapsed_startup_budget_cleans_actors_before_segment_rpc(driver, monkeypatch):
    clock = iter((100.0, 103.0))
    monkeypatch.setattr(driver.entry, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(TimeoutError, match="consumed during worker startup"):
        driver.entry.run(driver.plan)
    driver.group.run_segment.assert_not_called()
    driver.publish.assert_not_called()
    assert_failed_cleanup(driver)


def test_real_cpu_process_group_timeout_propagates_without_publishing(driver, monkeypatch):
    dist = pytest.importorskip("torch.distributed")
    assert not dist.is_initialized()
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    driver.plan.update(communication_timeout_seconds=10, segment_timeout_seconds=10)

    def missing_peer():
        # Actual c10d FileStore/Gloo rendezvous timeout; this is not an NCCL hardware test.
        dist.init_process_group("gloo", init_method=(driver.root / "rendezvous").as_uri(),
                                rank=0, world_size=2, timeout=timedelta(seconds=1))

    driver.on_segment = missing_peer
    try:
        with pytest.raises(RuntimeError, match="Wait timeout"):
            driver.entry.run(driver.plan)
        driver.publish.assert_not_called()
        assert_failed_cleanup(driver)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_log_collector_cleanup_failure_cannot_skip_ray_shutdown(driver):
    previous = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGUSR1)}
    driver.cluster._distributed_log_collector = SimpleNamespace(
        stop=Mock(side_effect=OSError("collector stop failed")))

    def rank_failure():
        raise RuntimeError("rank failed before publication")

    driver.on_segment = rank_failure
    with pytest.raises(RuntimeError, match="rank failed before publication"):
        driver.entry.run(driver.plan)
    driver.publish.assert_not_called()
    assert_failed_cleanup(driver)
    assert all(signal.getsignal(number) == handler for number, handler in previous.items())
    failure = json.loads((driver.root / "outputs/runtime-cleanup-failure.json").read_text())
    assert failure["primary_error"]["error"] == "rank failed before publication"
    assert failure["cleanup_errors"][0]["error"] == "collector stop failed"


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("failed_stage", ["collector", "ray_shutdown", "actor"])
def test_successful_work_with_failed_cleanup_has_no_collectible_receipt(driver, failed_stage, probe):
    previous = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGUSR1)}
    if failed_stage == "collector":
        driver.cluster._distributed_log_collector = SimpleNamespace(
            stop=Mock(side_effect=OSError("collector stop failed")))
    elif failed_stage == "ray_shutdown":
        driver.shutdown.side_effect = OSError("Ray shutdown failed")
    else:
        driver.kill.side_effect = [OSError("first actor kill failed"), None]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        driver.entry.run(driver.plan, probe=probe)
    assert_failed_cleanup(driver)
    assert all(signal.getsignal(number) == handler for number, handler in previous.items())
    failure = json.loads((driver.root / "outputs/runtime-cleanup-failure.json").read_text())
    assert failure["status"] == "failed" and failure["primary_error"] is None
    assert len(failure["cleanup_errors"]) == 1


def test_training_failure_survives_ray_shutdown_failure_and_restores_handlers(driver):
    previous = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGUSR1)}
    driver.shutdown.side_effect = OSError("Ray shutdown failed")

    def rank_failure():
        raise RuntimeError("original rank failure")

    driver.on_segment = rank_failure
    with pytest.raises(RuntimeError, match="original rank failure"):
        driver.entry.run(driver.plan)
    assert_failed_cleanup(driver)
    assert all(signal.getsignal(number) == handler for number, handler in previous.items())
    failure = json.loads((driver.root / "outputs/runtime-cleanup-failure.json").read_text())
    assert failure["primary_error"]["error"] == "original rank failure"
    assert failure["cleanup_errors"][0]["error"] == "Ray shutdown failed"
