"""CPU check: a dead downward sensor still keeps its original failure evidence."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, "/opt/pbfm-build-probe")

from pbfm.perception.instinct_direct_depth import depth_config_for_profile  # noqa: E402

from eb.runners import rlinf_pbfm_worker as worker  # noqa: E402


def main():
    depth = torch.full((2, 8, 18, 32), 255, dtype=torch.uint8)
    distances = torch.full((2, 36 * 64), -1.0)
    qpos = np.arange(60, dtype=np.float32).reshape(2, 30)
    closed = []
    origins, directions = torch.zeros(2, 36 * 64, 3), torch.zeros(2, 36 * 64, 3)
    origins[..., 2], directions[..., 2] = 1.0, -1.0
    sensor = NS(
        data=NS(distances=distances, pos_w=None, quat_w=torch.ones(2, 4)),
        cfg=NS(include_geom_groups=(0, 2, 5), exclude_parent_body=False, max_distance=4.0),
        _cached_world_origins=origins, _cached_world_rays=directions,
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 .1" group="5"/></worldbody></mujoco>'
    )
    core = NS(
        mjlab_env=NS(
            scene=NS(sensors={"g1_direct_depth": sensor}, spec=None), cfg=NS(seed=43),
            sim=NS(mj_model=model, data=NS(qpos=torch.empty(2, 0))),
        ),
        direct_depth_config=depth_config_for_profile("optical_z_maxcrop18_reset_v2"),
        _direct_depth_runtime=NS(delay_frames=torch.tensor([0, 1])),
        robot_root_states=torch.arange(26, dtype=torch.float32).reshape(2, 13),
        base_quat=torch.ones(2, 4), projected_gravity=torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
        motion_ids=torch.tensor([4, 18]), motion_start_times=torch.tensor([0.2, 0.3]),
    )
    obs = {"depth_image": depth}
    env = NS(
        base_env=core, reset=lambda **kwargs: (obs, {}),
        step=lambda *args, **kwargs: (obs, 0, torch.tensor([False, False]), torch.tensor([False, False]), {}),
        _get_qpos_qvel=lambda **kwargs: (qpos, np.zeros((2, 29), dtype=np.float32)),
        close=lambda: closed.append("env"),
    )
    cfg = NS(seed=43, online_parallel_envs=2048,
             env=NS(seed=43, build=lambda **kwargs: (env, {}), mjcf_path="/unused.xml"))

    class Renderer:
        def __init__(self, *args, **kwargs):
            pass

        def render_qpos(self, value):
            np.testing.assert_array_equal(value, qpos[0])
            return np.full((4, 4, 3), 100, dtype=np.uint8)

        def close(self):
            closed.append("renderer")

    zeros = torch.zeros
    with tempfile.TemporaryDirectory(prefix="eb-probe-evidence-") as temporary:
        output = Path(temporary)
        with (
            patch.object(worker, "OUTPUT_ROOT", output),
            patch.object(worker, "verify_environment", return_value={}),
            patch.object(worker, "simulation_execution_evidence", return_value={}),
            patch("pbfm.inference.runtime.MujocoQposRenderer", Renderer),
            patch("pbfm.perception.instinct_direct_depth.raycast_ranges_to_image_plane",
                  return_value=torch.full((2, 36, 64), 2.5)),
            patch.object(torch, "zeros", side_effect=lambda *args, **kwargs: zeros(*args, device="cpu")),
        ):
            try:
                worker.PbfmTrainWorker._probe(NS(), cfg, {"rank": 1, "world_size": 4})
            except ValueError as error:
                assert "ray" in str(error)
            else:
                raise AssertionError("all-miss sensor facing the ground must fail independent geometry")
        folder = output / "outputs/runtime/probe-rank-1"
        np.testing.assert_array_equal(np.load(folder / "network-depth.npy"), depth.numpy())
        np.testing.assert_array_equal(np.load(folder / "reset-network-depth.npy"), depth.numpy())
        np.testing.assert_array_equal(np.load(folder / "sensor-ranges.npy"), distances.numpy())
        np.testing.assert_array_equal(np.load(folder / "qpos.npy"), qpos)
        np.testing.assert_array_equal(np.load(folder / "metric-depth.npy"), np.full((2, 36, 64), 2.5))
        assert (folder / "network-depth.png").is_file() and (folder / "robot-terrain.png").is_file()
        stats = json.loads((folder / "probe-stats.json").read_text())
        assert stats["seed"]["cfg_seed"] == stats["seed"]["environment_seed"] == 43
        assert stats["seed"]["worker_explicit_seeded"] is False
        assert stats["post_step"]["network_depth"]["unique_count"] == 1
        assert stats["post_step"]["sensor_ranges"]["negative_count"] == 2 * 36 * 64
        assert stats["terminated"] == stats["truncated"] == [False, False]
        assert stats["rgb"]["verified"] is True
        assert not (folder / "camera-pos-w.npy").exists()
        for path in folder.glob("*.npy"):
            np.load(path, allow_pickle=False)
    assert closed == ["renderer", "env"]
    print("BAD_SENSOR_FAILURE_RETAINS_RAW_RESET_STEP_SENSOR_POSE_RGB_STATS", flush=True)


if __name__ == "__main__":
    main()
