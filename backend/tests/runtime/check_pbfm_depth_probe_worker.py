"""CPU boundary check: legal random all-far plus real geometric control passes.

The env/renderer are small doubles. Rays intersect a real MuJoCo plane and the
37-frame depth runtime is native; this does not claim a GPU probe passed.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, "/opt/pbfm-build-probe")

from pbfm.perception.instinct_direct_depth import DirectDepthRuntime, depth_config_for_profile  # noqa: E402

from eb.runners import rlinf_pbfm_worker as worker  # noqa: E402


def main():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 .1" group="5"/></worldbody></mujoco>'
    )
    depth_cfg = depth_config_for_profile("optical_z_maxcrop18_reset_v2")
    runtime = DirectDepthRuntime(2, "cpu", depth_cfg, enable_noise=False)
    origins = torch.zeros(2, 36 * 64, 3)
    origins[..., 2] = 0.95
    directions = torch.zeros_like(origins)
    directions[..., 2] = 1
    sensor = NS(
        data=NS(distances=torch.full((2, 36 * 64), -1.0)),
        cfg=NS(include_geom_groups=(0, 2, 5), exclude_parent_body=False, max_distance=4.0),
        _cached_world_origins=origins, _cached_world_rays=directions,
    )
    core = NS(
        device="cpu", direct_depth_config=depth_cfg, _direct_depth_runtime=runtime,
        env_origins=torch.zeros(2, 3), default_dof_pos=torch.zeros(2, 23),
        mjlab_env=NS(scene=NS(sensors={"g1_direct_depth": sensor}, spec=None), cfg=NS(seed=43),
                     sim=NS(mj_model=model, data=NS(qpos=torch.empty(2, 0)))),
    )
    resets, closed = [], []

    def reset(**kwargs):
        target = kwargs.get("target_states")
        resets.append(target is not None)
        if target is not None:
            np.testing.assert_allclose(target["root_states"][:, 2], 0.75)
            np.testing.assert_array_equal(target["root_states"][:, 3:7], [[0, 0, 0, 1], [0, 0, 0, 1]])
            rows, cols = torch.meshgrid(torch.arange(36), torch.arange(64), indexing="ij")
            directions[..., 0] = (0.8 + rows * 0.01).reshape(1, -1)
            directions[..., 1] = (cols * 0.001).reshape(1, -1)
            directions[..., 2] = (-0.8 + rows * 0.009).reshape(1, -1)
            directions.div_(torch.linalg.vector_norm(directions, dim=-1, keepdim=True))
            sensor.data.distances = -origins[..., 2] / directions[..., 2]
        runtime.reset_from_sensor(sensor, torch.tensor([0, 1]))
        return {"depth_image": runtime.observation()}, {}

    def step(*args, **kwargs):
        runtime.append_from_sensor(sensor)
        return (
            {"depth_image": runtime.observation()}, 0, torch.tensor([False, False]),
            torch.tensor([False, False]), {},
        )

    qpos = np.zeros((2, 30), dtype=np.float32)
    env = NS(base_env=core, reset=reset, step=step,
             _get_qpos_qvel=lambda **kwargs: (qpos, np.zeros((2, 29), dtype=np.float32)),
             close=lambda: closed.append("env"))
    cfg = NS(seed=43, online_parallel_envs=2048,
             env=NS(seed=43, build=lambda **kwargs: (env, {}), mjcf_path="/unused.xml"))

    class Renderer:
        def __init__(self, *args, **kwargs):
            pass

        def render_qpos(self, value):
            return np.full((4, 4, 3), 100, dtype=np.uint8)

        def close(self):
            closed.append("renderer")

    zeros = torch.zeros
    with tempfile.TemporaryDirectory(prefix="eb-probe-control-") as temporary:
        with (
            patch.object(worker, "OUTPUT_ROOT", Path(temporary)),
            patch.object(worker, "verify_environment", return_value={}),
            patch.object(worker, "simulation_execution_evidence", return_value={}),
            patch("pbfm.inference.runtime.MujocoQposRenderer", Renderer),
            patch("pbfm.inference.runtime._assign_training_tile"),
            patch.object(torch, "zeros", side_effect=lambda *args, **kwargs: zeros(*args, device="cpu")),
        ):
            result = worker.PbfmTrainWorker._probe(NS(), cfg, {"rank": 1, "world_size": 4})
        assert result["depth_verified"] is True and result["training_completed"] is False
        phases = result["depth_validation"]["phases"]
        assert phases["random_reset"]["lanes"][0]["witness_count"] == 0
        assert phases["positive_control"]["lanes"][0]["witness_count"] == 16
        folder = Path(temporary) / "outputs/runtime/probe-rank-1"
        assert np.load(folder / "network-depth.npy", allow_pickle=False).min() == 255
        assert np.unique(np.load(folder / "positive-network-depth.npy", allow_pickle=False)).size > 1
        for path in folder.glob("*.npy"):
            np.load(path, allow_pickle=False)
    assert resets == [False, True] and closed == ["renderer", "env"]
    print("RANDOM_ALL_FAR_REQUIRES_SAME_FRAME_GEOMETRY_AND_NATIVE_POSITIVE_CONTROL", flush=True)


if __name__ == "__main__":
    main()
