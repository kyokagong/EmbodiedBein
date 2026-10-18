"""Latest-only native GPU rendering, isolated from the policy/physics thread."""

from __future__ import annotations

import io
import json
import math
import struct
import threading
import time
from collections import deque

import numpy as np


def encode_frame(rgb, metadata):
    from PIL import Image

    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Native stream requires 640x480 RGB8")
    start = time.perf_counter()
    image = io.BytesIO()
    Image.fromarray(rgb).save(image, format="JPEG", quality=75)
    meta = {**metadata, "width": 640, "height": 480,
            "encode_ms": (time.perf_counter() - start) * 1000}
    header = json.dumps(meta, allow_nan=False, separators=(",", ":")).encode()
    packet = struct.pack("!I", len(header)) + header + image.getvalue()
    if len(header) > 4096 or len(packet) > 2 * 1024**2:
        raise ValueError("Native stream frame exceeds protocol budget")
    return packet


def read_camera(path):
    default = dict(seq=0, azimuth=135.0, elevation=-18.0, distance=3.0)
    try:
        if path.stat().st_size > 4096:
            raise ValueError("Camera command exceeds budget")
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return default
    if not isinstance(value, dict) or not {"seq", "azimuth", "elevation", "distance"} <= value.keys():
        raise ValueError("Invalid camera fields")
    if type(value["seq"]) is not int or not 0 <= value["seq"] <= 2**31 - 1:
        raise ValueError("Invalid camera sequence")
    for key, low, high in (("azimuth", -360, 360), ("elevation", -85, -5), ("distance", 1, 8)):
        number = value[key]
        if type(number) not in (float, int) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError("Invalid camera range")
    return {key: value[key] for key in default}


class NativeRenderer:
    def __init__(self, payload):
        import mujoco

        self.mj = mujoco
        spec = mujoco.MjSpec.from_file(str(payload / "assets/rpo.xml"))
        spec.worldbody.add_geom(name="stream_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[100, 100, .02], material="matplane", contype=0, conaffinity=0)
        spec.worldbody.add_light(pos=[0, -3, 4], dir=[.2, .5, -1], ambient=[.35, .35, .35])
        self.model = spec.compile()
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 640)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 480)
        self.model.vis.quality.offsamples = 4
        self.model.vis.headlight.ambient[:] = [.4, .4, .4]
        self.model.vis.headlight.diffuse[:] = [.7, .7, .7]
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        from OpenGL import GL
        self.device = {"vendor": GL.glGetString(GL.GL_VENDOR).decode(),
                       "renderer": GL.glGetString(GL.GL_RENDERER).decode()}
        if "NVIDIA" not in self.device["vendor"]:
            self.renderer.close()
            raise RuntimeError("Native stream requires the admitted NVIDIA EGL device")
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, qpos, camera):
        self.data.qpos[:] = qpos
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        self.mj.mj_camlight(self.model, self.data)
        self.camera.lookat[:] = [*qpos[:2], max(.65, qpos[2])]
        for name in ("azimuth", "elevation", "distance"):
            setattr(self.camera, name, camera[name])
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


class NativeStream:
    def __init__(self, payload, output, controls, *, renderer_factory=NativeRenderer):
        self.payload, self.output, self.controls = payload, output, controls
        self.renderer_factory = renderer_factory
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.latest, self.error = None, None
        self.count, self.bytes = 0, 0
        self.device = {}
        self.samples = deque(maxlen=900)
        self.started_at = None
        self.thread = threading.Thread(target=self._run, name="native-gpu-render", daemon=True)

    def start(self):
        self.started_at = time.monotonic()
        self.thread.start()

    def offer(self, qpos, *, step, episode):
        if not np.isfinite(qpos).all():
            raise ValueError("Non-finite render snapshot")
        # Copy before publishing: renderer never reads a live engine buffer.
        with self.lock:
            self.latest = (np.asarray(qpos).copy(), dict(step=step, episode=episode, updated_at=time.time()))

    def check(self):
        if self.error:
            raise RuntimeError(f"Native GPU stream failed: {self.error}") from self.error

    def _run(self):
        renderer = None
        try:
            renderer = self.renderer_factory(self.payload)
            self.device = getattr(renderer, "device", {})
            path = self.output / "live/frame.bin"
            path.parent.mkdir(exist_ok=True, parents=True)
            while not self.stopped.is_set():
                tick = time.monotonic()
                with self.lock:
                    snapshot = self.latest
                if snapshot is not None:
                    qpos, metadata = snapshot
                    camera = read_camera(self.controls / "camera.json")
                    start = time.perf_counter()
                    rgb = renderer.render(qpos, camera)
                    render_ms = (time.perf_counter() - start) * 1000
                    packet = encode_frame(rgb, {**metadata, "seq": self.count + 1,
                                               "rendered_at": time.time(), "camera_seq": camera["seq"],
                                               "render_ms": render_ms})
                    temporary = path.with_name(".frame.bin.tmp")
                    temporary.write_bytes(packet)
                    temporary.replace(path)
                    self.count += 1
                    self.bytes += len(packet)
                    self.samples.append(render_ms)
                self.stopped.wait(max(0, 1 / 30 - (time.monotonic() - tick)))
        except Exception as exc:
            self.error = exc
        finally:
            if renderer:
                try:
                    renderer.close()
                except Exception as exc:
                    if self.error is None:
                        self.error = exc

    def close(self):
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("Native renderer failed to stop within 3 seconds")

    def stats(self):
        elapsed = max(.001, time.monotonic() - (self.started_at or time.monotonic()))
        return {"transport": "websocket-jpeg", "frames": self.count, "bytes": self.bytes,
                "wall_seconds": elapsed, "fps": self.count / elapsed,
                "device": self.device,
                "render_ms_p95": float(np.percentile(self.samples, 95)) if self.samples else 0}
