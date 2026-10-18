"""Bounded deployment-policy session. Runs only inside an ebd task container."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import numpy as np

from eb.runners.ufo_native_package import receipt, verified_manifest
from eb.runners.ufo_native_sim import (
    SUBSTEP_COLUMNS,
    MjlabBackend,
    MujocoBackend,
    package_policy,
    probe_video,
)
from eb.ufo_native import TASKS, read_json, sha256_file

TASK_IDS = tuple(task["id"] for task in TASKS)


def atomic_json(path, value):
    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(encoded + "\n")
    temporary.replace(path)


def optional_json(path):
    try:
        return read_json(path, limit=16 * 1024)
    except FileNotFoundError:
        return None
    except ValueError:
        if not path.exists():
            return None
        raise


def lease_expired(lease, *, now, ready_at):
    updated = (lease or {}).get("updated_at")
    if type(updated) not in (int, float) or not math.isfinite(updated) or updated > now + 1:
        updated = ready_at
    return now - max(ready_at, updated) >= 15


class LiveController:
    def __init__(self, task, seed):
        if task not in TASK_IDS:
            raise ValueError("Task has no frozen embedding")
        self.task, self.seed = task, seed
        self.status, self.step, self.episode = "running", 0, 0
        self.ack, self.termination = None, None
        self.seen = {}

    def apply(self, command, backend, policy):
        cid, action = command.get("id"), command.get("action")
        allowed = {"id", "action", "received_at"} | ({"task"} if action == "task" else set())
        if (
            not isinstance(cid, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid)
            or action not in ("task", "pause", "resume", "reset", "stop")
            or not set(command) <= allowed
            or (action == "task" and command.get("task") not in TASK_IDS)
        ):
            raise ValueError("Invalid live control command")
        identity = (action, command.get("task"))
        if cid in self.seen:
            if self.seen[cid] != identity:
                raise ValueError("Live command ID was reused with another action")
            return False
        if len(self.seen) >= 1024:
            raise ValueError("Live command budget exhausted")
        if action == "task":
            self.task = command["task"]
        elif action == "pause" and self.status == "running":
            self.status = "paused"
        elif action == "resume" and self.status == "paused":
            self.status = "running"
        elif action == "reset":
            backend.reset(self.seed)
            policy.codec.reset()
            self.episode += 1
            self.status = "paused"
        elif action == "stop":
            self.termination = "operator_stop"
        self.seen[cid] = identity
        self.ack = dict(id=cid, action=action, task=self.task, applied_step=self.step, episode=self.episode)
        return True


class ChunkedTrace:
    def __init__(self, output, chunk_steps=500):
        self.output, self.chunk_steps = output, chunk_steps
        self.rows, self.files = [], []
        (output / "traces").mkdir(parents=True, exist_ok=True)

    def append(self, row):
        if not all(np.isfinite(np.asarray(v)).all() for v in row.values()):
            raise ValueError("Non-finite live trajectory")
        self.rows.append(row)
        if len(self.rows) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        name = f"traces/chunk-{len(self.files):04d}.npz"
        path = self.output / name
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        temporary.replace(path)
        self.files.append(name)
        self.rows.clear()
        atomic_json(self.output / "trace-index.json", {"chunks": self.files, "complete": False})


def _numbers(node, name, default, count):
    result = [float(v) for v in node.get(name, default).split()]
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise ValueError("Invalid frozen scene transform")
    return result


def scene_metadata(payload, manifest):
    root = ET.parse(payload / "assets/rpo.xml").getroot()
    if root.findall(".//include"):
        raise ValueError("Live scene cannot resolve external includes")
    assets = {node.get("name"): node for node in root.findall("./asset/mesh")}
    files = {item["path"]: item for item in manifest["files"]}
    bodies, meshes = [], []
    for body in root.findall("./worldbody//body"):
        name = body.get("name")
        if not name or name in bodies:
            raise ValueError("Live scene requires unique named bodies")
        index = len(bodies)
        bodies.append(name)
        for geom in body.findall("./geom"):
            if geom.get("type") != "mesh" or geom.get("mesh") not in assets:
                raise ValueError("Live scene supports frozen mesh geometry only")
            if any(
                k in geom.attrib
                for k in ("euler", "axisangle", "xyaxes", "zaxis", "fromto", "class", "material")
            ):
                raise ValueError("Unsupported frozen mesh transform/material")
            asset = assets[geom.get("mesh")]
            filename = asset.get("file", "")
            if PurePosixPath(filename).name != filename or not filename.lower().endswith(".stl"):
                raise ValueError("Unsafe frozen mesh asset")
            relative = "assets/meshes/" + filename
            if relative not in files:
                raise ValueError("Live mesh is absent from frozen manifest")
            quat = _numbers(geom, "quat", "1 0 0 0", 4)
            if abs(sum(v * v for v in quat) - 1) > 1e-5:
                raise ValueError("Invalid frozen mesh quaternion")
            meshes.append(
                dict(
                    body=index,
                    asset=relative,
                    position=_numbers(geom, "pos", "0 0 0", 3),
                    quaternion=quat,
                    scale=_numbers(asset, "scale", "1 1 1", 3),
                    rgba=_numbers(geom, "rgba", ".5 .5 .5 1", 4),
                    sha256=files[relative]["sha256"],
                )
            )
    if not bodies or not meshes:
        raise ValueError("Live scene contains no mesh bodies")
    return {"bodies": bodies, "meshes": meshes, "floor": {"size": 200, "height": 0}}


class BodyPoses:
    """Kinematics of actual integrated qpos, without advancing or modifying physics."""

    def __init__(self, backend, body_names):
        import mujoco

        self.mj = mujoco
        self.model = backend.model if isinstance(backend, MujocoBackend) else backend.renderer.model
        self.data = mujoco.MjData(self.model)
        self.ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
        if min(self.ids) <= 0:
            raise ValueError("Frozen scene body is absent from physics")

    def read(self, qpos):
        self.data.qpos[:] = qpos
        self.mj.mj_kinematics(self.model, self.data)
        return np.concatenate((self.data.xpos[self.ids], self.data.xquat[self.ids]), axis=1).tolist()


def _fingerprint(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _file_record(output, name):
    path = output / name
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finalize_media(output, trace, initial_qpos, backend, publish):
    import matplotlib
    from ufo.inference.runtime import MujocoQposRenderer, video_writer

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = []
    for name in trace.files:
        with np.load(output / name, allow_pickle=False) as data:
            arrays.append({k: data[k].copy() for k in data.files})
    combined = {k: np.concatenate([row[k] for row in arrays]) for k in arrays[0]} if arrays else {}
    qpos = combined.get("qpos", np.asarray([initial_qpos]))
    (output / "videos").mkdir(exist_ok=True)
    own_renderer = isinstance(backend, MujocoBackend)
    render_size = 320 if own_renderer else 480
    renderer = (
        MujocoQposRenderer(Path("/input/assets/rpo.xml"), render_size=render_size)
        if own_renderer
        else backend.renderer
    )
    try:
        with video_writer(output / "videos/session.mp4", shape=(render_size, render_size), fps=5) as writer:
            for i in range(0, len(qpos), 10):
                writer.add_image(renderer.render_qpos(qpos[i]))
                if i % 50 == 0:
                    publish("finalizing")
    finally:
        if own_renderer:
            renderer.close()
    media = probe_video(output / "videos/session.mp4")
    media.update(width=render_size, height=render_size)
    if media["frames"] != math.ceil(len(qpos) / 10) or media["fps"] != 5:
        raise ValueError("Live recording coverage is incomplete")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if combined:
        x = combined["sim_seconds"]
        axes[0].plot(x, combined["velocity"][:, 0], label="actual forward m/s")
        axes[0].step(x, [TASKS[int(i)]["speed"] for i in combined["task"]], label="target m/s")
        axes[1].plot(x, combined["height"], label="height m")
        axes[1].plot(x, combined["tilt"], label="tilt rad")
        axes[2].plot(x, combined["latency_ms"], label="inference ms")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
    else:
        axes[0].text(0.2, 0.5, "Session ended before the first policy step")
    axes[2].set_xlabel("simulated seconds; pauses excluded, resets recorded in events.jsonl")
    fig.suptitle("Interactive deployment diagnostics; task qualification not assessed")
    fig.tight_layout()
    fig.savefig(output / "timeseries.svg")
    plt.close(fig)
    return media


def run_live(output, plan):
    from eb.ufo_live import validate_live_plan

    validate_live_plan(plan)
    payload, source, controls = Path("/input"), Path("/native-source"), Path("/control")
    manifest = verified_manifest(payload, plan["source_artifact_sha256"])
    verified_manifest(source, plan["native_source_sha256"])
    policy = package_policy(payload)
    scene = scene_metadata(payload, manifest)
    if len(scene["bodies"]) != 24 or not set(TASK_IDS) <= set(policy.prompts):
        raise ValueError("Native live scene/prompts differ from deployed RPO contract")
    identity = dict(
        run_id=os.environ["EB_RUN_ID"],
        model_id=plan["model_id"],
        payload_sha256=plan["source_artifact_sha256"],
    )
    atomic_json(
        output / "live/scene.json",
        {
            "schema_version": "eb/ufo-live-scene/v1",
            **identity,
            **scene,
            "asset_base_url": f"/api/v1/ufo-native/live/models/{plan['model_id']}/assets/",
        },
    )
    ctl = LiveController(plan["initial_task"], plan["seed"])
    trace = ChunkedTrace(output)
    backend = None
    stream, stream_stats = None, None
    events = (output / "events.jsonl").open("w", buffering=1)
    seq, last_publish, ready_at, ready_monotonic = 0, 0, None, None
    active_wall, latency, state, poses = 0.0, 0.0, None, None
    limits = np.zeros(3, dtype=np.int64)

    def event(kind, **fields):
        events.write(
            json.dumps(
                {
                    "event": kind,
                    "updated_at": time.time(),
                    "step": ctl.step,
                    "episode": ctl.episode,
                    "task": ctl.task,
                    **fields,
                },
                allow_nan=False,
            )
            + "\n"
        )
        events.flush()

    def publish(status=None, error=None):
        nonlocal seq, last_publish
        seq += 1
        last_publish = time.monotonic()
        wall = max(0.0, last_publish - ready_monotonic) if ready_monotonic else 0.0
        value = {
            "schema_version": "eb/ufo-live-state/v1",
            **identity,
            "plan_hash": plan["hash"],
            "seq": seq,
            "status": status or ctl.status,
            "task": ctl.task,
            "episode": ctl.episode,
            "step": ctl.step,
            "sim_seconds": ctl.step * 0.02,
            "wall_seconds": wall,
            "updated_at": time.time(),
            "realtime_factor": ctl.step * 0.02 / max(0.001, active_wall),
            "body_poses": poses.read(state["qpos"]) if poses and state else [],
            "root_position": state["qpos"][:3].tolist() if state else [0, 0, 0.75],
            "velocity": np.asarray(state["velocity"]).tolist() if state else [0, 0],
            "height": float(state["height"]) if state else 0.75,
            "tilt": float(state["tilt"]) if state else 0,
            "limits": dict(zip(("joint", "velocity", "torque"), map(int, limits))),
            "inference_ms": latency,
            "ack": ctl.ack,
        }
        if error:
            value["error"] = error
        atomic_json(output / "live/state.json", value)

    try:
        publish("initializing")
        backend = (
            MjlabBackend(payload, source, seconds=plan["duration_seconds"] + 1)
            if plan["backend"] == "mjlab"
            else MujocoBackend(payload, source)
        )
        backend.reset(plan["seed"])
        policy.codec.reset()
        state = backend.state()
        initial_qpos = state["qpos"].copy()
        poses = BodyPoses(backend, scene["bodies"])
        if plan["suite_spec"].get("stream"):
            from eb.runners.ufo_native_stream import NativeStream
            stream = NativeStream(payload, output, controls)
            stream.offer(state["qpos"], step=ctl.step, episode=ctl.episode)
            stream.start()
        ready_at, ready_monotonic = time.time(), time.monotonic()
        event("ready", plan_hash=plan["hash"], payload_sha256=plan["source_artifact_sha256"])
        publish()
        while not ctl.termination:
            tick = time.monotonic()
            if stream:
                stream.check()
            if tick - ready_monotonic >= plan["duration_seconds"]:
                ctl.termination = "duration_limit"
                break
            if lease_expired(optional_json(controls / "lease.json"), now=time.time(), ready_at=ready_at):
                ctl.termination = "lease_expired"
                break
            command = optional_json(controls / "command.json")
            if command:
                before_qpos = _fingerprint(state["qpos"])
                before_history = _fingerprint(
                    np.concatenate([v.ravel() for v in policy.codec.history.values()])
                )
                before_task = ctl.task
                if ctl.apply(command, backend, policy):
                    state = backend.state()
                    event(
                        "command_applied",
                        **ctl.ack,
                        previous_task=before_task,
                        received_at=command.get("received_at"),
                        qpos_before_sha256=before_qpos,
                        qpos_after_sha256=_fingerprint(state["qpos"]),
                        history_before_sha256=before_history,
                        history_after_sha256=_fingerprint(
                            np.concatenate([v.ravel() for v in policy.codec.history.values()])
                        ),
                    )
                    if command["action"] in ("reset", "pause", "stop"):
                        trace.flush()
                    publish()
            if ctl.termination:
                break
            if ctl.status == "running":
                start = time.perf_counter()
                raw, target = policy.act(state["q"], state["dq"], state["gravity"], state["omega"], ctl.task)
                latency = (time.perf_counter() - start) * 1000
                if isinstance(backend, MjlabBackend):
                    backend.check_observation(policy.last_observation)
                backend.step(raw, target)
                state = backend.state()
                ctl.step += 1
                samples = np.asarray(backend.substeps)
                if samples.shape != (8, 6):
                    raise ValueError("Live physical substep evidence count differs")
                limits += (samples[:, :3] > 0.001).sum(axis=0)
                trace.append(
                    {
                        **{k: state[k] for k in ("qpos", "q", "dq", "torque", "height", "tilt", "velocity")},
                        "raw_action": raw.copy(),
                        "target": target.copy(),
                        "step": ctl.step,
                        "episode": ctl.episode,
                        "task": TASK_IDS.index(ctl.task),
                        "sim_seconds": ctl.step * 0.02,
                        "wall_seconds": tick - ready_monotonic,
                        "latency_ms": latency,
                        "substep_limits": samples.copy(),
                    }
                )
                if state["height"] < 0.5 or state["tilt"] > 0.8:
                    ctl.status = "fallen"
                    trace.flush()
                    event("fall")
                    publish()
                elapsed = time.monotonic() - tick
                remaining = max(0, 0.02 - elapsed)
                time.sleep(remaining)
                active_wall += time.monotonic() - tick
            else:
                time.sleep(0.02)
            if stream:
                stream.offer(state["qpos"], step=ctl.step, episode=ctl.episode)
            if time.monotonic() - last_publish >= 0.05:
                publish()
        trace.flush()
        if stream:
            stream.close()
            stream.check()
            stream_stats = stream.stats()
        event("live_ended", reason=ctl.termination)
        publish("finalizing")
        media = finalize_media(output, trace, initial_qpos, backend, publish)
        event("recording_complete", frames=media["frames"])
        atomic_json(output / "trace-index.json", {"chunks": trace.files, "complete": True, "steps": ctl.step})
        files = [
            _file_record(output, name)
            for name in [
                *trace.files,
                "trace-index.json",
                "events.jsonl",
                "timeseries.svg",
                "videos/session.mp4",
            ]
        ]
        receipt(
            output,
            plan,
            model_id=plan["model_id"],
            payload_sha256=plan["source_artifact_sha256"],
            environment_id=plan["environment_id"],
            environment_hash=plan["environment_hash"],
            diagnostic_only=True,
            complete=True,
            steps=ctl.step,
            episodes=ctl.episode + 1,
            task_order=list(TASK_IDS),
            command_count=len(ctl.seen),
            termination=ctl.termination,
            recording={"path": "videos/session.mp4", **media},
            files=files,
            limits=dict(zip(("joint", "velocity", "torque"), map(int, limits))),
            substep_limit_columns=list(SUBSTEP_COLUMNS),
            codec_max_error=getattr(backend, "codec_error", None),
            trace_chunks=trace.files,
            wall_seconds=time.monotonic() - ready_monotonic,
            live_stream=stream_stats,
        )
        publish("finished")
    except Exception as exc:
        trace.flush()
        event("error", reason=str(exc))
        # Keep the last usable pose if an engine emitted NaNs; never publish them.
        if state and not all(np.isfinite(np.asarray(v)).all() for v in state.values()):
            state = None
        publish("error", str(exc))
        raise
    finally:
        events.close()
        if stream:
            stream.close()
        if backend:
            backend.close()
