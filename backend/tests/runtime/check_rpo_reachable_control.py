"""CPU-only behavior check, run in the native producer image with NVIDIA disabled."""
import torch
from eb.rpo_bfm import DEFAULT_Q, JOINT_POSITION_MIN, JOINT_POSITION_MAX, SLEW_CONTROL, CODEC_HASH, CODEC_V2_HASH
from eb.runners.rpo_bfm_codec import action_anchor, policy_to_target, limit_action_slew, encode_state


def test_reachable_control():
    lo, hi = torch.tensor(JOINT_POSITION_MIN), torch.tensor(JOINT_POSITION_MAX)
    assert CODEC_HASH != CODEC_V2_HASH
    torch.testing.assert_close(policy_to_target(-torch.ones(23), control_profile="reachable_v2"), lo)
    torch.testing.assert_close(policy_to_target(torch.ones(23), control_profile="reachable_v2"), hi)
    q = torch.tensor(DEFAULT_Q)
    q[17] = -.6
    anchor = action_anchor(q, control_profile="reachable_v2")
    torch.testing.assert_close(policy_to_target(anchor, control_profile="reachable_v2"), q)
    assert policy_to_target(action_anchor(q))[17] > 0
    outside = q.clone(); outside[17] = -2
    assert abs(policy_to_target(action_anchor(outside, control_profile="reachable_v2"), control_profile="reachable_v2")[17].item() + .6) < 1e-6
    for requested in (-torch.ones(23), torch.ones(23)):
        applied = limit_action_slew(requested, anchor, control_profile="reachable_v2")
        delta = (policy_to_target(applied, control_profile="reachable_v2") - q).abs()
        assert (delta <= .5 * .02 * torch.tensor(SLEW_CONTROL["joint_velocity_limits_rad_s"]) + 1e-6).all()
    state = encode_state(q, torch.zeros(23), torch.zeros(3), torch.zeros(3))
    assert abs(state[17].item() - (-.6 - .78)) < 1e-6


if __name__ == "__main__":
    test_reachable_control()
    print("reachable control checks passed")


def test_soft_velocity_v7_keeps_v5_threshold():
    from eb.runners.rpo_ufo_isaac import raw_aux_rewards, VELOCITY_AUX_NAMES
    q = torch.zeros(1, 23)
    quat = torch.zeros(1, 24, 4); quat[..., 0] = 1
    data = dict(q=q, dq=torch.full_like(q, .75), velocity_limits=torch.ones(23),
        torques=q, joint_limits=torch.tensor([[-1., 1.]]*23), torque_limits=torch.ones(23),
        body_quat_w=quat, body_lin_vel_w=torch.zeros(1, 24, 3))
    contact = dict(forces=torch.zeros(1, 24, 3), body_names=list(range(24)))
    mapping = dict(feet=[1, 2], contact_feet=[1, 2], ankles=[0, 1], undesired_contacts=[3])
    v5 = raw_aux_rewards(data, q, q, contact, mapping, aux_names=VELOCITY_AUX_NAMES, auxiliary_profile='soft_velocity_v5')
    v7 = raw_aux_rewards(data, q, q, contact, mapping, aux_names=VELOCITY_AUX_NAMES, auxiliary_profile='soft_velocity_v7')
    index = VELOCITY_AUX_NAMES.index('limits_dof_vel')
    assert v5[0, index] == 0
    torch.testing.assert_close(v7[0, index], torch.tensor(23 * .05**2))


def test_native_step_receives_reachable_physical_targets(monkeypatch):
    import sys
    from types import SimpleNamespace
    from eb import rpo_bfm
    from eb.runners import rpo_ufo_isaac
    from rpo_flat_extension import FlatResetCompat
    class Native:
        def step(self, action):
            return torch.tensor(DEFAULT_Q) + .25 * action
    class Bfm(FlatResetCompat, Native): pass
    monkeypatch.setitem(sys.modules, 'rpo_bfm', rpo_bfm)
    monkeypatch.setitem(sys.modules, 'rpo_bfm_extension', SimpleNamespace(RPOBFMIsaacEnv=Bfm))
    env = object.__new__(rpo_ufo_isaac.__getattr__('RPOUFOIsaacPhysics'))
    env.num_envs, env.device = 1, 'cpu'
    env.bfm_control_profile, env.bfm_slew, env.ufo_safe = 'reachable_v2', True, True
    q = torch.tensor(DEFAULT_Q).reshape(1, 23); q[0, 17] = -.6
    env.bfm_last_action = action_anchor(q, control_profile='reachable_v2')
    env.bfm_history = SimpleNamespace(append=lambda *args: None)
    env.bfm_step_id = torch.zeros(1, 1, dtype=torch.long)
    env.ufo_cache = rpo_ufo_isaac.ObservationCache()
    env._get_observations = lambda: {'states': torch.zeros(1, 52)}
    requested = torch.ones(1, 23)
    expected = policy_to_target(limit_action_slew(requested, env.bfm_last_action, control_profile='reachable_v2'), control_profile='reachable_v2')
    torch.testing.assert_close(env.step(requested), expected)
    torch.testing.assert_close(env.ufo_requested_action, requested)
    assert env.bfm_last_action[0, 17] < 0
