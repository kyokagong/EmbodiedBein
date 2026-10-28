"""Task-local RLinf driver; no P-BFM launcher or child training process."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from eb.runners.pbfm_source import file_sha256
from eb.runners.rlinf_pbfm_runtime import (
    NATIVE_ROOT,
    OUTPUT_ROOT,
    STAGING_ROOT,
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
        raise ValueError("Failure injection requires a diagnostic probe rank")
    verify_source_refs(plan)
    configure_environment(plan)
    verify_training_data(plan)
    args = parse_native_args(plan)
    if not probe:
        prepare_segment(plan, args)
    write_json(OUTPUT_ROOT / "outputs/resolved-native-args.json", args)
    import ray
    from rlinf.scheduler import Cluster, PackedPlacementStrategy

    from eb.runners.rlinf_pbfm_worker import PbfmTrainWorker

    def failed(signum, frame):
        raise RuntimeError(f"P-BFM RLinf rank failure/timeout/cancel (signal {signum})")

    previous = {
        number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM)
    }
    group = cluster = None
    results = None
    cleanup_errors = []
    for number in previous:
        signal.signal(number, failed)
    signal.alarm(
        min(plan.get("communication_timeout_seconds", 300), plan.get("segment_timeout_seconds", 21600))
    )
    try:
        cluster = Cluster(num_nodes=1, distributed_log_dir=str(OUTPUT_ROOT / "outputs/rlinf-logs"))
        signal.signal(signal.SIGUSR1, failed)
        session = Path(ray._private.worker._global_node.get_session_dir_path()).resolve()
        if not session.is_relative_to((OUTPUT_ROOT / "runtime/ray").resolve()):
            raise RuntimeError("P-BFM connected to Ray outside this Run")
        nodes = [node for node in ray.nodes() if node["Alive"]]
        if len(nodes) != 1:
            raise RuntimeError("P-BFM must use one task-local Ray node")
        group = PbfmTrainWorker.create_group(plan)
        group.launch(
            cluster=cluster,
            placement_strategy=PackedPlacementStrategy(0, plan["world_size"] - 1),
            name="pbfm_" + plan["run_id"],
            isolate_gpu=True,
            catch_system_failure=False,
        )
        identities = group.describe_runtime().wait()
        if (
            sorted(item["rank"] for item in identities) != list(range(plan["world_size"]))
            or len({item["gpu_uuid"] for item in identities}) != plan["world_size"]
        ):
            raise RuntimeError("P-BFM worker ranks/physical GPUs are missing or duplicated")
        write_json(
            OUTPUT_ROOT / "outputs/runtime/cluster.json",
            {
                "session_dir": str(session),
                "namespace": Cluster.NAMESPACE,
                "ranks": identities,
                "plan_hash": canonical_plan_hash(plan),
            },
        )
        remaining = int(plan.get("segment_timeout_seconds", 21600) - (time.monotonic() - started))
        if remaining <= 0:
            raise TimeoutError("P-BFM startup consumed its segment budget")
        signal.alarm(remaining)
        results = group.run_segment(args, probe=probe, probe_fail_rank=probe_fail_rank).wait()
    finally:

        def cleanup(stage, operation):
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(
                    {"stage": stage, "error_type": type(error).__name__, "error": str(error)[:4000]}
                )

        if group is not None:
            for index, info in enumerate(group.worker_info_list):
                cleanup(f"actor_{index}", lambda: ray.kill(info.worker, no_restart=True))
        if cluster is not None and cluster._distributed_log_collector is not None:
            cleanup("collector.stop", cluster._distributed_log_collector.stop)
        cleanup("ray.shutdown", ray.shutdown)
        cleanup("alarm.cancel", lambda: signal.alarm(0))
        for number, handler in previous.items():
            cleanup("signal.restore", lambda number=number, handler=handler: signal.signal(number, handler))
        if cleanup_errors:
            write_json(
                OUTPUT_ROOT / "outputs/runtime-cleanup-failure.json",
                {"status": "failed", "errors": cleanup_errors},
            )
            if sys.exc_info()[1] is None:
                raise RuntimeError("P-BFM cleanup failed; completion was not published")
    if probe:
        from eb.runners.pbfm_depth_probe import RECEIPT_SCHEMA, required_probe_names, validate_depth_report

        files = []
        for rank in range(plan["world_size"]):
            paths = [OUTPUT_ROOT / f"outputs/runtime/rank-{rank}.json"]
            folder = OUTPUT_ROOT / f"outputs/runtime/probe-rank-{rank}"
            report = validate_depth_report(json.loads((folder / "depth-validation.json").read_text()))
            result = next(row for row in results if row["rank"] == rank)
            if report.get("rank") != rank or result.get("depth_validation") != report:
                raise ValueError("P-BFM probe geometry differs from its rank result")
            if not required_probe_names() <= {path.name for path in folder.iterdir() if path.is_file()}:
                raise ValueError("P-BFM probe lacks complete raw ray, pose and depth comparison evidence")
            paths += sorted(folder.iterdir())
            for path in paths:
                if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("P-BFM probe media/evidence file is missing")
                files.append(
                    {
                        "path": str(path.relative_to(OUTPUT_ROOT / "outputs")),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "run_id": plan["run_id"],
            "status": "complete",
            "ranks": results,
            "world_size": plan["world_size"],
            "plan_hash": canonical_plan_hash(plan),
            "training_completed": False,
            "files": files,
        }
        write_json(OUTPUT_ROOT / "outputs/probe-receipt.json", receipt)
        return receipt
    from eb.pbfm_training_checkpoint import publish_pbfm_checkpoint

    return publish_pbfm_checkpoint(STAGING_ROOT, NATIVE_ROOT, plan, plan["run_id"], rank_results=results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-fail-rank", type=int)
    args = parser.parse_args()
    if args.probe_fail_rank is not None and not args.probe:
        parser.error("--probe-fail-rank requires --probe")
    plan = json.loads(args.plan.read_text())
    try:
        receipt = run(plan, probe=args.probe, probe_fail_rank=args.probe_fail_rank)
        print(json.dumps({"status": receipt["status"], "run_id": plan["run_id"]}), flush=True)
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
