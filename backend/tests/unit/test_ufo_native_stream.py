import json
import struct
import time

import numpy as np
import pytest


def test_new_stream_profile_reserves_gpu_without_changing_v1():
    from eb.api.routes_ufo_native import spec_for
    from eb.ufo_live import environment_spec, live_suite_spec, validate_live_plan
    from eb.ufo_native import plan_hash, validate_job
    from test_ufo_live import live_plan

    plan = live_plan()
    assert environment_spec("mujoco")["id"].endswith("_v2")
    assert live_suite_spec("mujoco")["stream"]["fps"] == 30
    assert spec_for(plan).resources.gpus == 1
    assert validate_job(spec_for(plan)) == plan
    old = {**plan, "environment_id": environment_spec("mujoco", version=1)["id"],
           "environment_hash": environment_spec("mujoco", version=1)["content_hash"],
           "suite_id": environment_spec("mujoco", version=1)["suite_id"],
           "suite_spec": live_suite_spec("mujoco", version=1)}
    old["hash"] = plan_hash(old)
    assert "stream" not in old["suite_spec"]
    assert validate_live_plan(old) == old
    assert spec_for(old).resources.gpus == 0
    assert validate_job(spec_for(old)) == old


def test_frame_packet_is_binary_jpeg_bounded_and_metadata_finite():
    from eb.runners.ufo_native_stream import encode_frame

    packet = encode_frame(np.zeros((480, 640, 3), dtype=np.uint8), {"seq": 1, "updated_at": 123.0})
    size = struct.unpack("!I", packet[:4])[0]
    meta = json.loads(packet[4:4 + size])
    assert meta["seq"] == 1 and meta["width"] == 640 and meta["height"] == 480
    assert meta["encode_ms"] >= 0 and packet[4 + size:6 + size] == b"\xff\xd8"
    assert len(packet) < 2 * 1024**2
    with pytest.raises(ValueError):
        encode_frame(np.zeros((480, 640, 3), dtype=np.uint8), {"seq": 2, "updated_at": float("nan")})


def test_render_owns_copied_latest_snapshot_and_camera_does_not_touch_physics(tmp_path):
    from eb.runners.ufo_native_stream import NativeStream

    seen, rendered = [], []

    class FakeRenderer:
        def __init__(self, payload):
            pass

        def render(self, qpos, camera):
            seen.append((qpos.copy(), camera.copy()))
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def close(self):
            rendered.append("closed")

    control = tmp_path / "control"
    control.mkdir()
    stream = NativeStream(tmp_path, tmp_path, control, renderer_factory=FakeRenderer)
    qpos = np.arange(30, dtype=float)
    stream.offer(qpos, step=123, episode=0)
    qpos[:] = -1
    stream.start()
    deadline = time.monotonic() + 2
    path = tmp_path / "live/frame.bin"
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.005)
    assert path.exists()
    assert np.array_equal(seen[0][0], np.arange(30))
    (control / "camera.json").write_text(json.dumps({"seq": 1, "azimuth": 50, "elevation": -20, "distance": 2}))
    while not any(c[1]["seq"] == 1 for c in seen) and time.monotonic() < deadline:
        time.sleep(.005)
    stream.close()
    assert any(c[1]["seq"] == 1 for c in seen)
    assert len(seen) <= 65 and rendered == ["closed"]
    assert all(np.array_equal(row[0], np.arange(30)) for row in seen)
    assert stream.stats()["frames"] == len(seen)


def test_renderer_failure_propagates_instead_of_silent_black_screen(tmp_path):
    from eb.runners.ufo_native_stream import NativeStream

    def broken(payload):
        raise RuntimeError("EGL failed")

    stream = NativeStream(tmp_path, tmp_path, tmp_path, renderer_factory=broken)
    stream.start()
    stream.thread.join(1)
    with pytest.raises(RuntimeError, match="EGL failed"):
        stream.check()
    stream.close()


def test_renderer_teardown_failure_cannot_publish_complete_receipt(tmp_path):
    from eb.runners.ufo_native_stream import NativeStream

    class Renderer:
        def close(self):
            raise RuntimeError("EGL teardown failed")

    stream = NativeStream(tmp_path, tmp_path, tmp_path, renderer_factory=lambda payload: Renderer())
    stream.start()
    stream.close()
    with pytest.raises(RuntimeError, match="EGL teardown failed"):
        stream.check()


@pytest.mark.parametrize("patch", [{"distance": .1}, {"elevation": float("nan")},
                                   {"seq": True}, {"azimuth": 999}, {"distance": "3"}])
def test_renderer_rejects_out_of_range_camera_defensively(tmp_path, patch):
    from eb.runners.ufo_native_stream import read_camera

    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"seq": 1, "azimuth": 135, "elevation": -18, "distance": 3, **patch}))
    with pytest.raises(ValueError):
        read_camera(path)
