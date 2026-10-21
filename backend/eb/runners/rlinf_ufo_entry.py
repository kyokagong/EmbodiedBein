"""Start a task-local RLinf WorkerGroup and run one native UFO segment."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from eb.runners.rlinf_ufo_runtime import (
    NATIVE_ROOT,
    OUTPUT_ROOT,
    canonical_plan_hash,
    configure_environment,
    parse_native_args,
    prepare_segment,
    validate_plan,
    verify_source_refs,
    verify_training_data,
    write_json,
)


def run(plan: dict, *, probe: bool = False, probe_fail_rank: int | None = None) -> dict:
    started = time.monotonic()
    validate_plan(plan)
    if probe_fail_rank is not None and (not probe or not 0 <= probe_fail_rank < plan["world_size"]):
        raise ValueError("Failure injection requires a valid diagnostic probe rank")
    verify_source_refs(plan)
    configure_environment(plan)
    if not probe:
        verify_training_data(plan)
    args = parse_native_args(plan)
    if not probe:
        prepare_segment(plan, args)
    write_json(OUTPUT_ROOT / "outputs/resolved-native-args.json", args)
    import ray
    from rlinf.scheduler import Cluster, PackedPlacementStrategy

    from eb.runners.rlinf_ufo_worker import UfoTrainWorker

    def failed(signum, frame):
        raise RuntimeError(f"RLinf worker failure or segment timeout (signal {signum})")

    previous_alarm = signal.signal(signal.SIGALRM, failed)
    previous_failure = signal.getsignal(signal.SIGUSR1)
    group = cluster = None
    signal.alarm(
        min(plan.get("communication_timeout_seconds", 300), plan.get("segment_timeout_seconds", 21600))
    )
    try:
        cluster = Cluster(num_nodes=1, distributed_log_dir=str(OUTPUT_ROOT / "outputs/rlinf-logs"))
        signal.signal(signal.SIGUSR1, failed)
        session = Path(ray._private.worker._global_node.get_session_dir_path()).resolve()
        if not session.is_relative_to((OUTPUT_ROOT / "runtime/ray").resolve()):
            raise RuntimeError("RLinf connected to a Ray instance outside this Run")
        nodes = [node for node in ray.nodes() if node["Alive"]]
        if len(nodes) != 1:
            raise RuntimeError("A UFO task must have exactly one local Ray node")
        group = UfoTrainWorker.create_group(plan)
        group.launch(
            cluster=cluster,
            placement_strategy=PackedPlacementStrategy(0, plan["world_size"] - 1),
            name="ufo_" + plan["run_id"],
            isolate_gpu=True,
            catch_system_failure=False,
        )
        identities = group.describe_runtime().wait()
        if sorted(item["rank"] for item in identities) != list(range(plan["world_size"])):
            raise RuntimeError("WorkerGroup rank coverage mismatch")
        if len({item["gpu_uuid"] for item in identities}) != plan["world_size"]:
            raise RuntimeError("WorkerGroup allocated duplicate physical GPUs")
        write_json(
            OUTPUT_ROOT / "outputs/runtime/cluster.json",
            {
                "session_dir": str(session),
                "namespace": Cluster.NAMESPACE,
                "nodes": [
                    {"NodeID": n["NodeID"], "NodeManagerAddress": n["NodeManagerAddress"]} for n in nodes
                ],
                "plan_hash": canonical_plan_hash(plan),
                "ranks": identities,
            },
        )
        remaining = int(plan.get("segment_timeout_seconds", 21600) - (time.monotonic() - started))
        if remaining <= 0:
            raise TimeoutError("Segment budget was consumed during worker startup")
        signal.alarm(remaining)
        results = group.run_segment(args, probe=probe, probe_fail_rank=probe_fail_rank).wait()
        if probe:
            receipt = {
                "schema_version": "eb/ufo-native-runtime-probe/v1",
                "ranks": results,
                "plan_hash": canonical_plan_hash(plan),
                "status": "complete",
                "training_data_qualified": False,
            }
            write_json(OUTPUT_ROOT / "outputs/probe-receipt.json", receipt)
            return receipt
        from eb.ufo_training_checkpoint import write_native_checkpoint_receipt

        receipt = write_native_checkpoint_receipt(
            NATIVE_ROOT,
            plan,
            plan["run_id"],
            rank_results=results,
            loader_verified=all(r["loader_verified"] for r in results),
        )
        try:
            from eb.runners.ufo_training_analysis import build_training_evidence

            build_training_evidence(OUTPUT_ROOT, plan)
        except BaseException:
            (OUTPUT_ROOT / "outputs/receipt.json").unlink(missing_ok=True)
            raise
        return receipt
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []

        def cleanup(stage, operation):
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(
                    {"stage": stage, "error_type": type(error).__name__, "error": str(error)[:4000]}
                )

        try:
            # WorkerGroup._close waits on RPCs; kill every owned actor directly.
            if group is not None:
                for index, info in enumerate(group.worker_info_list):
                    cleanup(f"actor_{index}", lambda: ray.kill(info.worker, no_restart=True))
            if cluster is not None and cluster._distributed_log_collector is not None:
                cleanup("collector.stop", cluster._distributed_log_collector.stop)
        finally:
            try:
                cleanup("ray.shutdown", ray.shutdown)
            finally:
                cleanup("alarm.cancel", lambda: signal.alarm(0))
                cleanup("handler.SIGALRM", lambda: signal.signal(signal.SIGALRM, previous_alarm))
                cleanup("handler.SIGUSR1", lambda: signal.signal(signal.SIGUSR1, previous_failure))
        if cleanup_errors:
            for name in ("receipt.json", "probe-receipt.json"):
                cleanup(f"remove.{name}", lambda: (OUTPUT_ROOT / "outputs" / name).unlink(missing_ok=True))
            diagnostic = {
                "status": "failed",
                "run_id": plan.get("run_id"),
                "primary_error": None if primary_error is None else {
                    "error_type": type(primary_error).__name__, "error": str(primary_error)[:4000]
                },
                "cleanup_errors": cleanup_errors,
            }
            cleanup("diagnostic.write", lambda: write_json(
                OUTPUT_ROOT / "outputs/runtime-cleanup-failure.json", diagnostic
            ))
            try:
                print(json.dumps(diagnostic), file=sys.stderr, flush=True)
            except Exception:
                pass
            if primary_error is None:
                raise RuntimeError("RLinf cleanup failed; see outputs/runtime-cleanup-failure.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="A0 only: validate placement, NCCL and reset/step without training",
    )
    parser.add_argument(
        "--probe-fail-rank", type=int, help="A0 only: fail one diagnostic rank before PG setup"
    )
    args = parser.parse_args()
    if args.probe_fail_rank is not None and not args.probe:
        parser.error("--probe-fail-rank requires --probe")
    plan = json.loads(args.plan.read_text())
    try:
        receipt = run(plan, probe=args.probe, probe_fail_rank=args.probe_fail_rank)
        print(json.dumps({"status": receipt.get("status", "complete"), "run_id": plan["run_id"]}), flush=True)
    except BaseException as error:
        write_json(
            OUTPUT_ROOT / "outputs/runtime-failure.json",
            {
                "status": "failed",
                "run_id": plan.get("run_id"),
                "error_type": type(error).__name__,
                "error": str(error)[:4000],
            },
        )
        raise


if __name__ == "__main__":
    main()
