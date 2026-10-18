import json
from types import SimpleNamespace

import numpy as np
import pytest

from eb.runners import ufo_native_live as live


def controller():
    calls = []
    backend = SimpleNamespace(reset=lambda seed: calls.append(("physics_reset", seed)))
    policy = SimpleNamespace(codec=SimpleNamespace(reset=lambda: calls.append(("history_reset",))))
    return live.LiveController("move-ego-0-0", 71000), backend, policy, calls


def test_task_switch_preserves_physics_history_and_applies_once():
    ctl, backend, policy, calls = controller()
    ctl.step = 123
    command = {"id": "first", "action": "task", "task": "move-ego-0-0.3"}
    assert ctl.apply(command, backend, policy)
    assert ctl.task == "move-ego-0-0.3" and ctl.step == 123 and ctl.episode == 0
    assert ctl.ack == dict(id="first", action="task", task=ctl.task, applied_step=123, episode=0)
    assert calls == []
    assert not ctl.apply(command, backend, policy)
    ctl.apply(dict(id="second", action="task", task="move-ego-0-0.7"), backend, policy)
    assert not ctl.apply(command, backend, policy)
    assert ctl.task == "move-ego-0-0.7"


def test_pause_fall_requires_reset_and_reset_clears_only_episode_history():
    ctl, backend, policy, calls = controller()
    ctl.step = 200
    ctl.apply(dict(id="p", action="pause"), backend, policy)
    assert ctl.status == "paused"
    ctl.apply(dict(id="r", action="resume"), backend, policy)
    assert ctl.status == "running"
    ctl.status = "fallen"
    ctl.apply(dict(id="r2", action="resume"), backend, policy)
    assert ctl.status == "fallen"
    ctl.apply(dict(id="reset", action="reset"), backend, policy)
    assert ctl.status == "paused" and ctl.step == 200 and ctl.episode == 1
    assert calls == [("physics_reset", 71000), ("history_reset",)]
    assert ctl.ack["applied_step"] == 200


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "x", "action": "task", "task": "move-ego-0-9"},
        {"id": "x", "action": "shell"},
        {"id": "x", "action": "reset", "task": "move-ego-0-0"},
        {"id": "../x", "action": "stop"},
    ],
)
def test_worker_defensively_rejects_invalid_commands(bad):
    ctl, backend, policy, calls = controller()
    with pytest.raises(ValueError):
        ctl.apply(bad, backend, policy)
    assert calls == []


def test_lease_has_bounded_initial_grace_and_no_future_extension():
    assert not live.lease_expired(None, now=110, ready_at=100)
    assert live.lease_expired(None, now=115, ready_at=100)
    assert not live.lease_expired({"updated_at": 114}, now=120, ready_at=100)
    assert live.lease_expired({"updated_at": 104}, now=120, ready_at=100)
    assert live.lease_expired({"updated_at": 999}, now=120, ready_at=100)


def test_trace_chunks_preserve_task_episode_and_do_not_require_pickle(tmp_path):
    trace = live.ChunkedTrace(tmp_path, chunk_steps=2)
    for step in range(3):
        trace.append({"qpos": np.arange(30), "step": step + 1, "task": step % 2, "episode": step // 2})
    trace.flush()
    assert len(trace.files) == 2
    rows = [np.load(tmp_path / item, allow_pickle=False) for item in trace.files]
    assert np.concatenate([r["step"] for r in rows]).tolist() == [1, 2, 3]
    assert np.concatenate([r["task"] for r in rows]).tolist() == [0, 1, 0]
    assert np.concatenate([r["episode"] for r in rows]).tolist() == [0, 0, 1]
    assert all(r["qpos"].shape[1] == 30 for r in rows)


def test_scene_uses_original_mesh_local_pose_and_hash(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "rpo.xml").write_text("""<mujoco><asset><mesh name="m" file="base.STL" scale="1 2 3"/></asset>
    <worldbody><body name="base"><geom type="mesh" mesh="m"
      pos="1 0 0" quat="1 0 0 0" rgba=".5 .6 .7 1"/></body></worldbody></mujoco>""")
    manifest = {"files": [{"path": "assets/meshes/base.STL", "sha256": "a" * 64}]}
    scene = live.scene_metadata(tmp_path, manifest)
    assert scene["bodies"] == ["base"]
    assert scene["meshes"][0] == {
        "body": 0,
        "asset": "assets/meshes/base.STL",
        "position": [1, 0, 0],
        "quaternion": [1, 0, 0, 0],
        "scale": [1, 2, 3],
        "rgba": [0.5, 0.6, 0.7, 1],
        "sha256": "a" * 64,
    }
    (assets / "rpo.xml").write_text(
        '<mujoco><worldbody><body name="base"><geom type="sphere"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="mesh"):
        live.scene_metadata(tmp_path, manifest)


def test_atomic_snapshot_does_not_publish_nan_or_replace_good_state(tmp_path):
    path = tmp_path / "state.json"
    live.atomic_json(path, {"seq": 1})
    with pytest.raises(ValueError):
        live.atomic_json(path, {"seq": 2, "height": float("nan")})
    assert json.loads(path.read_text()) == {"seq": 1}
