"""Task-only, fixed CUDA OOM / missing-peer NCCL timeout diagnostics."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from eb.ufo_failure_probe import FILES, OPERATION, RECEIPT_SCHEMA, fault_evidence, validate_plan

ROOT = Path("/run/eb-output")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    temporary.replace(path)


def process_state(pid):
    """Read only a recorded PID in this task's /proc namespace."""

    def read_stat():
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "start_ticks": int(fields[19])}

    try:
        state = read_stat()
        try:
            state["pid_namespace"] = os.readlink(f"/proc/{pid}/ns/pid")
        except FileNotFoundError:
            state = read_stat()  # Exit may race the stat read; do not call a surviving zombie "gone".
            if state["state"] != "Z":
                return {
                    "pid": pid,
                    "state": "unknown",
                    "observed_state": state["state"],
                    "error": "Actor PID namespace link is unavailable",
                }
            state["pid_namespace"] = None  # Zombie namespace links can disappear before /proc/PID.
        return state
    except FileNotFoundError:
        return {"pid": pid, "state": "gone"}
    except (OSError, ValueError, IndexError) as exc:
        return {"pid": pid, "state": "unknown", "error": str(exc)[:1000]}


def cleanup_actor_processes(identities):
    """Observe exits and, when necessary, signal only an identity-bound local actor.

    A zombie has exited and owns no CUDA descriptors. It may still have a /proc
    entry until its parent reaps it; that is recorded separately from a gone PID.
    pidfds prevent PID reuse between the identity check and the forced signal.
    """
    started = time.monotonic()
    namespace = os.readlink("/proc/self/ns/pid")
    actors, forced, errors = [], [], []

    def observe(actor):
        snapshot = process_state(actor["pid"])
        if not actor["observations"] or any(
            actor["observations"][-1].get(key) != snapshot.get(key)
            for key in ("state", "start_ticks", "pid_namespace", "error")
        ):
            actor["observations"].append({**snapshot, "elapsed_seconds": time.monotonic() - started})
        actor["final_state"] = snapshot["state"]
        actor["identity_verified"] = snapshot["state"] == "gone" or (
            actor["pid"] > 1
            and actor["pid"] != os.getpid()
            and actor["pid_namespace"] == namespace
            and snapshot.get("start_ticks") == actor["proc_start_ticks"]
            and snapshot.get("start_ticks") is not None
            and (snapshot.get("pid_namespace") == namespace or snapshot["state"] == "Z")
            and snapshot["state"] != "unknown"
        )

    for identity in identities:
        pid = identity["pid"]
        if type(pid) is not int or pid <= 1 or pid == os.getpid() or any(a["pid"] == pid for a in actors):
            raise ValueError("Only distinct recorded actor PIDs may be inspected or terminated")
        actor = {
            "rank": identity["rank"],
            "pid": pid,
            "proc_start_ticks": identity.get("proc_start_ticks"),
            "pid_namespace": identity.get("pid_namespace"),
            "observations": [],
        }
        observe(actor)
        actors.append(actor)

    def live_verified():
        return [a for a in actors if a["final_state"] not in {"gone", "Z"} and a["identity_verified"]]

    # Allow asynchronous Ray termination one second, inside the original five-second budget.
    while live_verified() and time.monotonic() < started + 1:
        time.sleep(0.05)
        for actor in actors:
            observe(actor)
    for actor in live_verified():
        descriptor = None
        try:
            descriptor = os.pidfd_open(actor["pid"])
            observe(actor)
            if not actor["identity_verified"]:
                raise ValueError("Recorded actor identity changed before forced termination")
            if actor["final_state"] not in {"gone", "Z"}:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                forced.append(actor["pid"])
        except ProcessLookupError:
            pass  # Exit raced the pidfd open/signal; the final observation must still confirm it.
        except (OSError, ValueError, AttributeError) as exc:
            errors.append({"stage": "actor.pidfd_sigkill", "pid": actor["pid"], "error": str(exc)[:1000]})
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    errors.append(
                        {"stage": "actor.pidfd_close", "pid": actor["pid"], "error": str(exc)[:1000]}
                    )
    for actor in actors:
        observe(actor)
    while live_verified() and not errors and time.monotonic() < started + 5:
        time.sleep(0.05)
        for actor in actors:
            observe(actor)
    gone = [a["pid"] for a in actors if a["final_state"] == "gone"]
    zombies = [a["pid"] for a in actors if a["final_state"] == "Z"]
    live = [a["pid"] for a in actors if a["final_state"] not in {"gone", "Z", "unknown"}]
    unverified = [a["pid"] for a in actors if not a["identity_verified"]]
    return {
        "observation_version": 2,
        "complete": not errors and not live and not unverified,
        "actors": actors,
        "forced_pids": forced,
        "zombie_pids": zombies,
        "gone_pids": gone,
        "live_pids": live,
        "unverified_pids": unverified,
        "remaining_actor_pids": sorted(set(live + unverified)),
        "errors": errors,
    }


def prepare(plan):
    validate_plan(plan)
    ref = plan["source_ref"]
    source, manifest = Path(ref["path"]), Path(ref["manifest_path"])
    if "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest() != ref["snapshot_hash"]:
        raise ValueError("RLinf manifest identity mismatch")
    for row in json.loads(manifest.read_text())["entries"]:
        relative = Path(row["path"])
        path = source / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError("Unsafe frozen RLinf source path")
        if row["sha256"] is None:
            if not path.is_dir():
                raise ValueError("Missing frozen RLinf directory")
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"].removeprefix("sha256:"):
                raise ValueError("Frozen RLinf source bytes differ")
    ray_root = ROOT / "runtime/ray"
    if (
        os.environ.get("RAY_ADDRESS")
        or os.environ.get("RLINF_EXT_MODULE")
        or os.environ.get("RAY_TMPDIR") not in (None, "", str(ray_root))
        or (ray_root / "ray/ray_current_cluster").exists()
        or "rlinf" in sys.modules
    ):
        raise ValueError("External Ray, extensions or previously imported RLinf are forbidden")
    env = {
        "RAY_TMPDIR": str(ray_root),
        "TMPDIR": str(ROOT / "runtime/tmp"),
        "XDG_CACHE_HOME": str(ROOT / "runtime/cache"),
        "RLINF_CODE_WORKING_DIR": "0",
        "RLINF_CATCH_FAILURE": "0",
        "RLINF_TIMEOUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(256 * 1024**2),
        "PYTHONPATH": "/sources/rlinf:/opt/eb-native",
    }
    for value in env.values():
        if value.startswith(str(ROOT)):
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    sys.path.insert(0, str(source))


def gpu_baseline():
    # Only this task process invokes the fixed read-only hardware query.
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, total, free, used = [part.strip() for part in line.split(",")]
        rows.append(
            dict(
                index=int(index),
                gpu_uuid=uuid.removeprefix("GPU-"),
                nvml_uuid=uuid,
                memory_total_mb=int(total),
                memory_free_mb=int(free),
                memory_used_mb=int(used),
            )
        )
    return rows


def worker_class():
    from rlinf.scheduler import Worker

    class FailureWorker(Worker):
        def __init__(self, case):
            super().__init__()
            self.case = case

        def describe(self):
            import rlinf
            import torch

            if (
                int(os.environ["WORLD_SIZE"]) != 2
                or int(os.environ["LOCAL_RANK"]) != 0
                or torch.cuda.device_count() != 1
                or not Path(rlinf.__file__).resolve().is_relative_to(Path("/sources/rlinf"))
            ):
                raise RuntimeError("Expected two isolated frozen-RLinf GPU workers")
            prop = torch.cuda.get_device_properties(0)
            process = process_state(os.getpid())
            self.state = dict(
                rank=int(os.environ["RANK"]),
                pid=os.getpid(),
                proc_start_ticks=process["start_ticks"],
                pid_namespace=process["pid_namespace"],
                world_size=2,
                gpu_uuid=str(prop.uuid),
                total_bytes=prop.total_memory,
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                torch_version=torch.__version__,
                nccl_version=torch.cuda.nccl.version(),
            )
            self.save()
            return self.state

        def save(self):
            import torch

            self.state.update(
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                last_observed_monotonic_seconds=time.monotonic(),
            )
            write(ROOT / f"outputs/rank-{self.state['rank']}.json", self.state)

        def execute(self):
            import torch
            import torch.distributed as dist

            self.describe()
            rank = self.state["rank"]
            torch.cuda.set_device(0)
            try:
                if self.case == "cuda_oom":
                    if rank == 0:
                        self.state["requested_bytes"] = self.state["total_bytes"] + 1024**3
                        self.save()
                        # One over-physical request fails immediately; no gradual memory pressure.
                        torch.empty(self.state["requested_bytes"], dtype=torch.uint8, device="cuda:0")
                        raise RuntimeError("The over-physical allocation unexpectedly succeeded")
                    time.sleep(80)
                    raise RuntimeError("The OOM peer survived instead of being terminated")
                if dist.is_initialized():
                    raise RuntimeError("A default process group already exists")
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=rank,
                    world_size=2,
                    timeout=timedelta(seconds=10),
                    device_id=torch.device("cuda:0"),
                )
                value = torch.tensor([rank + 1.0], device="cuda:0")
                dist.all_reduce(value)
                torch.cuda.synchronize()
                if value.item() != 3:
                    raise RuntimeError("NCCL warmup returned the wrong result")
                self.state["warmup_passed"] = True
                self.state["collective_entered" if rank == 0 else "skipping_collective"] = True
                self.save()
                if rank == 0:
                    dist.all_reduce(value)
                    torch.cuda.synchronize()
                    raise RuntimeError("The missing-peer collective unexpectedly succeeded")
                time.sleep(80)
                raise RuntimeError("The missing-peer survivor was not terminated")
            except BaseException as exc:
                self.state.update(
                    exception_type=type(exc).__name__,
                    exception=str(exc)[:16000],
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                self.save()
                raise
            # Deliberately no post-fault NCCL barrier/destroy: driver terminates all actors.

    return FailureWorker


def native_console(session):
    chunks = []
    paths = list((session / "logs").glob("*.err")) if session else []
    paths += sorted((ROOT / "outputs/rlinf-logs").rglob("*.log"))
    for path in paths:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 256 * 1024))
            chunks.append(stream.read().decode(errors="replace"))
    for rank in range(2):
        path = ROOT / f"outputs/rank-{rank}.json"
        if path.exists():
            chunks.append(json.loads(path.read_text()).get("exception", ""))
    return "\n".join(chunks)[-1900 * 1024 :] or "No native stderr was emitted.\n"


def run(plan):
    started = time.monotonic()
    prepare(plan)
    baseline = gpu_baseline()  # Before any driver/worker CUDA context.
    import ray
    from rlinf.scheduler import Cluster, PackedPlacementStrategy

    def failed(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError("Diagnostic driver exceeded its fixed 90-second bound")
        raise RuntimeError("RLinf reported a failed diagnostic worker")

    previous_alarm, previous_failure = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGUSR1)
    group = cluster = session = None
    identities, failure, errors = [], None, []
    signal.signal(signal.SIGALRM, failed)
    signal.alarm(90)
    try:
        cluster = Cluster(num_nodes=1, distributed_log_dir=str(ROOT / "outputs/rlinf-logs"))
        signal.signal(signal.SIGUSR1, failed)
        session = Path(ray._private.worker._global_node.get_session_dir_path()).resolve()
        if (
            not session.is_relative_to((ROOT / "runtime/ray").resolve())
            or len([n for n in ray.nodes() if n["Alive"]]) != 1
        ):
            raise RuntimeError("Diagnostic connected to an external Ray cluster")
        group = worker_class().create_group(plan["case"])
        group.launch(
            cluster=cluster,
            placement_strategy=PackedPlacementStrategy(0, 1),
            name="fault_" + plan["run_id"],
            isolate_gpu=True,
            catch_system_failure=False,
        )
        identities = group.describe().wait()
        selected = [r for r in baseline if r["gpu_uuid"] in {i["gpu_uuid"] for i in identities}]
        if (
            len(selected) != 2
            or len({i["gpu_uuid"] for i in identities}) != 2
            or any(r["memory_used_mb"] > 256 for r in selected)
        ):
            raise RuntimeError("Both assigned GPUs must be idle before the diagnostic starts")
        baseline = selected
        write(ROOT / "outputs/gpu-baseline.json", {"gpus": baseline})
        write(ROOT / "outputs/identities.json", {"ranks": identities})
        try:
            group.execute().wait()
        except Exception as exc:
            failure = exc
        if failure is None or isinstance(failure, TimeoutError):
            raise RuntimeError("Expected native fault was absent or diagnostic timed out")
    finally:
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)

        def clean(stage, fn):
            try:
                fn()
            except BaseException as exc:
                errors.append({"stage": stage, "error": str(exc)[:4000]})

        try:
            if group is not None:
                for info in group.worker_info_list:
                    clean("ray.kill", lambda: ray.kill(info.worker, no_restart=True))
            if cluster is not None and cluster._distributed_log_collector is not None:
                clean("collector.stop", cluster._distributed_log_collector.stop)
        finally:
            try:
                clean("ray.shutdown", ray.shutdown)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGUSR1, previous_failure)
        try:
            cleanup = cleanup_actor_processes(identities)
            errors.extend(cleanup["errors"])
        except BaseException as exc:
            errors.append({"stage": "actor.process_cleanup", "error": str(exc)[:4000]})
            cleanup = {
                "complete": False,
                "actors": [],
                "forced_pids": [],
                "zombie_pids": [],
                "gone_pids": [],
                "live_pids": [],
                "unverified_pids": [r["pid"] for r in identities],
                "remaining_actor_pids": [r["pid"] for r in identities],
            }
        cleanup.update(
            complete=cleanup["complete"] and not errors,
            errors=errors,
            elapsed_seconds=time.monotonic() - started,
        )
        write(
            ROOT / "outputs/driver-cleanup.json",
            cleanup,
        )
    if not cleanup["complete"]:
        raise RuntimeError("Diagnostic cleanup failed")
    console = native_console(session)
    (ROOT / "outputs/fault-console.txt").write_text(console)
    states = [json.loads((ROOT / f"outputs/rank-{rank}.json").read_text()) for rank in range(2)]
    evidence = fault_evidence(plan["case"], identities, states, console)
    write(ROOT / "outputs/fault-evidence.json", evidence)
    files = []
    for name in sorted(FILES):
        path = ROOT / "outputs" / name
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    receipt = dict(
        schema_version=RECEIPT_SCHEMA,
        operation=OPERATION,
        run_id=plan["run_id"],
        case=plan["case"],
        status="complete",
        plan_hash=plan["hash"],
        image_digest=plan["image_digest"],
        base_image_digest=plan["base_image_digest"],
        actual_optimizer_updates=0,
        diagnostic_only=True,
        model_published=False,
        gpu_baseline=baseline,
        files=files,
    )
    write(ROOT / "outputs/failure-probe-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(json.loads(args.plan.read_text()))
        print(json.dumps({"status": receipt["status"], "case": receipt["case"]}), flush=True)
    except BaseException as exc:
        (ROOT / "outputs/failure-probe-receipt.json").unlink(missing_ok=True)
        write(
            ROOT / "outputs/driver-failure.json", {"error_type": type(exc).__name__, "error": str(exc)[:4000]}
        )
        raise


if __name__ == "__main__":
    main()
