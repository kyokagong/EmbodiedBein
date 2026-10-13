import pytest

torch = pytest.importorskip("torch")


def test_slew_uses_effective_decoder_gain_and_preserves_requested_actions():
    from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q, SLEW_CONTROL
    from eb.runners.rpo_bfm_codec import action_anchor, limit_action_slew, policy_to_target

    requested = torch.ones(2, 23)
    original = requested.clone()
    q = torch.tensor(DEFAULT_Q).repeat(2, 1)
    anchor = action_anchor(q)
    applied = limit_action_slew(requested, anchor)
    torch.testing.assert_close(requested, original)
    target_delta = policy_to_target(applied) - policy_to_target(anchor)
    cap = 0.5 * torch.tensor(SLEW_CONTROL["joint_velocity_limits_rad_s"]) * 0.02
    assert (target_delta.abs() <= cap + 1e-6).all()
    # A gain-nine joint also advances by its physical target velocity cap.
    index = ACTION_GAIN.index(9)
    torch.testing.assert_close(target_delta[:, index], cap[index].expand(2))
    outside = q.clone()
    outside[:, index] += 4
    projected = action_anchor(outside)
    assert projected.abs().max() <= 1
    assert (policy_to_target(projected)[:, index] - outside[:, index]).abs().min() > 0


def test_slew_rejects_invalid_controller_state():
    from eb.runners.rpo_bfm_codec import limit_action_slew

    with pytest.raises(ValueError, match="controller"):
        limit_action_slew(torch.zeros(2, 23), torch.full((2, 23), 2.0))


def test_partial_reset_anchor_keeps_other_environment_controller_state(monkeypatch):
    import sys
    from types import SimpleNamespace

    from eb import rpo_bfm
    from eb.runners import rpo_bfm_extension
    from eb.runners.rpo_bfm_codec import action_anchor

    monkeypatch.setitem(sys.modules, "robolab.tasks.direct.base.base_env", SimpleNamespace(BaseEnv=object))
    monkeypatch.setitem(sys.modules, "rpo_bfm", rpo_bfm)
    cls = rpo_bfm_extension.__getattr__("RPOBFMIsaacEnv")
    env = object.__new__(cls)
    env.bfm_control_profile = "v1"
    q = torch.tensor(rpo_bfm.DEFAULT_Q).repeat(2, 1) + 0.1
    targets = []
    env.robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos=q),
        set_joint_position_target=lambda value, **kw: targets.append((value.clone(), kw)),
    )
    env.bfm_last_action = torch.full((2, 23), 0.7)
    env.bfm_previous_action = env.bfm_last_action.clone()
    env._reset_slew_anchor(torch.tensor([0]))
    torch.testing.assert_close(env.bfm_last_action[0], action_anchor(q[0]))
    assert (env.bfm_last_action[1] == 0.7).all() and (env.bfm_previous_action[1] == 0.7).all()
    assert targets[0][1]["env_ids"].tolist() == [0]


def test_evaluation_requires_exported_controller_binding(monkeypatch):
    from eb.rpo_bfm import SLEW_CONTROL
    from eb.runners.rpo_bfm_eval import configure_action_control

    execution = {"recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4"}
    with pytest.raises(ValueError, match="action control"):
        configure_action_control({"execution": execution})
    configure_action_control({"execution": execution, "action_control": SLEW_CONTROL})
    import os

    assert os.environ["EB_UFO_LEARNING_PROFILE"] == "physical_limits_slew_v4"
    configure_action_control({})
    assert "EB_UFO_LEARNING_PROFILE" not in os.environ
