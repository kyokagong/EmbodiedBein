"""CPU regression for the exporter's real single-environment successor boundary.

Run in the PBFM task image with the repository mounted read-only at /workspace.
The small actual ONNX graph and native action methods isolate this boundary;
this is not a full trained-policy export or a simulation-quality validation.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from eb.runners import pbfm_eval, pbfm_export
from eb.runners.pbfm_codec import PolicyCodec
from eb.runners.pbfm_source import prepare_pbfm_source


def main():
    private = Path("/tmp/pbfm-export-successor-source")
    prepare_pbfm_source(Path("/opt/pbfm-upstream"), private)
    sys.path.insert(0, str(private))
    from pbfm.envs.mjlab.core import HumanoidVerseMjlabCore
    from pbfm.envs.mjlab.vector_env import HumanoidVerseMjlabVectorEnv

    order = np.roll(np.arange(23)[::-1], 5).copy()
    defaults = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    offsets = np.linspace(0.01, 0.12, 23, dtype=np.float32)
    scales = np.linspace(0.3, 0.7, 23, dtype=np.float32)
    bias = torch.linspace(-0.015, 0.025, 23).reshape(1, 23)
    codec_config = {
        "history": {key: 4 for key in ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")},
        "obs_scales": {
            key: 1.0
            for key in ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity", "history_actor")
        },
        "default_dof_pos": defaults.tolist(),
        "action_target_scale": scales.tolist(),
        "action_clip": 5.0,
        "normalize_action_from": 1.0,
        "normalize_action_to": 5.0,
    }
    core = object.__new__(HumanoidVerseMjlabCore)
    core.device = "cpu"
    core.config = SimpleNamespace(
        robot=SimpleNamespace(
            control=SimpleNamespace(
                normalize_action=True,
                normalize_action_from=1.0,
                normalize_action_to=5.0,
                action_clip_value=5.0,
            )
        )
    )
    core.dof_names = [f"joint_{index}" for index in range(23)]
    core._action_term_dof_indices = torch.tensor(order)
    core.default_dof_pos_offset = torch.tensor(offsets).reshape(1, 23)
    core.action_target_scale = torch.tensor(scales).reshape(1, 23)
    core.actions = torch.zeros(1, 23)
    core.applied_actions = torch.zeros(1, 23)
    core.dof_pos = torch.tensor(defaults + offsets).reshape(1, 23)
    core.dof_vel = torch.zeros(1, 23)
    core.base_ang_vel = torch.zeros(1, 3)
    core.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]])
    core.robot_root_states = torch.zeros(1, 13)
    core.robot_root_states[0, 6] = 1.0
    core.torques = torch.zeros(1, 23)
    core._ctrl_delay_enabled = True
    core._ctrl_delay_max_step = 1
    core._ctrl_delay_steps = torch.ones(1, dtype=torch.long)
    core._ctrl_delay_env_indices = torch.zeros(1, dtype=torch.long)
    core._ctrl_delay_buffer = torch.zeros(2, 1, 23)
    term = SimpleNamespace(
        _processed_actions=torch.zeros(1, 23),
        _entity=SimpleNamespace(data=SimpleNamespace(encoder_bias=bias)),
        target_ids=torch.tensor(order),
        target_names=[core.dof_names[index] for index in order],
    )
    core.mjlab_env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    native_codec = PolicyCodec(codec_config)
    successors, targets, requested = [], [], []

    def observation():
        scalar = native_codec.observations(
            core.dof_pos,
            core.dof_vel,
            core.projected_gravity,
            core.base_ang_vel,
            core.actions,
            default_offset=core.default_dof_pos_offset.numpy(),
        )
        return {
            **{key: torch.from_numpy(value) for key, value in scalar.items()},
            "depth_image": torch.full((1, 8, 18, 32), 128, dtype=torch.uint8),
        }

    def physics_step(action):
        requested.append(action.numpy().copy())
        core.actions[:] = core._normalized_action(action)
        core.applied_actions[:] = core._apply_ctrl_delay(core.actions)
        term._processed_actions[:] = (
            torch.tensor(defaults[order]).reshape(1, 23)
            + core._mjlab_action_input(core.applied_actions) * core.action_target_scale[:, order]
        )
        core.dof_pos[:, order] = term._processed_actions - bias[:, order]

    core.step = physics_step
    wrapped = SimpleNamespace(_env=core, close=lambda: None)
    wrapped._get_qpos_qvel = lambda to_numpy=False: HumanoidVerseMjlabVectorEnv._get_qpos_qvel(
        wrapped, to_numpy
    )
    wrapped.reset = lambda **kwargs: (observation(), {})

    def vector_step(action, **kwargs):
        core.step(action)
        return observation(), 0, np.asarray([False]), np.asarray([False]), {}

    wrapped.step = vector_step
    original_advance = pbfm_eval.advance_with_successor

    def capture_successor(*args):
        result, successor = original_advance(*args)
        successors.append(successor)
        return result, successor

    class CheckedCodec(PolicyCodec):
        def action_target(self, *args, **kwargs):
            clipped, target = super().action_target(*args, **kwargs)
            if kwargs.get("delayed_command") is not None:
                targets.append(target)
            return clipped, target

    def export_policy(policy, destination, **kwargs):
        graph = helper.make_graph(
            [helper.make_node("Gather", ["z", "indices"], ["action"], axis=1)],
            "successor-boundary",
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 256])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 23])],
            [numpy_helper.from_array(np.arange(23, dtype=np.int64), "indices")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        onnx.checker.check_model(model)
        onnx.save(model, destination)
        return {
            "format": "direct_depth_full_v1",
            "depth_shape": [8, 18, 32],
            "z_dim": 256,
            "output_action_dim": 23,
            "input_shapes": {"state": [380]},
        }

    class TraceComplete(Exception):
        pass

    z = np.zeros((1000, 256), dtype=np.float32)
    z[:, :23] = np.linspace(-1.4, 1.4, 23) + np.arange(1000)[:, None] * 0.01
    policy = SimpleNamespace(act=lambda obs, latent, **kwargs: latent[:, :23])
    with tempfile.TemporaryDirectory() as temporary:
        with (
            patch.object(pbfm_export, "validate_action_plan"),
            patch.object(
                pbfm_export,
                "prepare_runtime",
                return_value=(policy, None, SimpleNamespace(build=lambda **kwargs: [wrapped]), None, {}),
            ),
            patch.object(pbfm_export, "tracking_prompt", return_value=({"z": z}, {})),
            patch.object(pbfm_export, "write_inference_package", return_value=codec_config),
            patch.object(pbfm_export, "commanded_observation", side_effect=lambda obs, *args: obs),
            patch.object(pbfm_export, "paired_closed_loop", side_effect=TraceComplete),
            patch.object(pbfm_eval, "advance_with_successor", side_effect=capture_successor),
            patch("eb.runners.pbfm_codec.PolicyCodec", CheckedCodec),
            patch("pbfm.inference.export.depth_policy.export_direct_depth_policy", side_effect=export_policy),
        ):
            try:
                pbfm_export.run(
                    {
                        "motion_id": 0,
                        "model_version_id": "cpu-fixture",
                        "model_hash": "a" * 64,
                        "prompt_manifest_sha256": "b" * 64,
                        "prompt_file_sha256": "c" * 64,
                    },
                    Path(temporary),
                )
            except TraceComplete:
                pass
    assert len(targets) == len(successors) == len(requested) == 32
    joints = "".join(
        f'<body><joint name="joint_{index}"/><geom type="sphere" size=".01"/></body>' for index in range(23)
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".01"/>'
        f"{joints}</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    assert model.nq == 30
    previous = np.zeros((1, 23), dtype=np.float32)
    for action, successor, target in zip(requested, successors, targets):
        assert successor["applied_actions"].shape == successor["default_offset"].shape == (23,)
        assert target.shape == (1, 23)
        np.testing.assert_allclose(successor["applied_actions"], previous[0], atol=1e-6)
        expected = defaults + offsets + previous * scales
        np.testing.assert_allclose(target, expected, atol=1e-6)
        np.testing.assert_allclose(
            target[:, order] - bias.numpy()[:, order], successor["pd_target"], atol=1e-6
        )
        data.qpos[7 + order] = (target[:, order] - bias.numpy()[:, order])[0]
        np.testing.assert_allclose(data.qpos[7:], successor["qpos"][7:], atol=1e-6)
        previous = np.clip(action * 5.0, -5.0, 5.0)
    print(
        json.dumps(
            {
                "successor_vector_shape": [23],
                "pd_target_batch_shape": [1, 23],
                "native_delay_steps": 1,
                "real_onnx_steps": 32,
                "joint_reorder_offset_bias_verified": True,
                "mujoco_qpos_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
