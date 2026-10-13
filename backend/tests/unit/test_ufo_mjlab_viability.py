"""Dynamic RSI boundaries are independent of relaxed training termination."""
import numpy as np
import pytest


def test_rsi_lookahead_rejects_outward_momentum_without_editing_reference():
    from eb.runners.rpo_ufo_contact import joint_lookahead_mask
    from eb.rpo_bfm import JOINT_POSITION_MAX, JOINT_POSITION_MIN
    q = np.zeros((4, 23)); dq = q.copy()
    q[:, 1] = [0.99, 0.99, 0.98, 0.9998]
    dq[:, 1] = [1., -1., 0., -1.]
    saved = q.copy(), dq.copy()
    assert joint_lookahead_mask(q, dq, control_dt=.02, margin=.001).tolist() == [False, True, True, False]
    np.testing.assert_array_equal(q, saved[0]); np.testing.assert_array_equal(dq, saved[1])
    q[0, 0] = JOINT_POSITION_MIN[0] + .01; dq[0, 0] = -1.
    q[0, 1] = 0.; dq[0, 1] = 0.
    assert not joint_lookahead_mask(q, dq, control_dt=.02, margin=.001)[0]
    with pytest.raises(ValueError):
        joint_lookahead_mask(q, np.full_like(dq, np.nan), control_dt=.02, margin=.001)


def test_training_position_buffer_does_not_change_strict_shadow_or_torque():
    torch = pytest.importorskip('torch')
    from eb.runners.rpo_ufo_isaac import PhysicsInterval
    limits = {'joint_pos': torch.tensor([[-1., .2]] * 23),
              'joint_vel': torch.ones(23), 'applied_torque': torch.full((23,), 27.)}
    value = PhysicsInterval(3, limits, device='cpu', position_tolerance=.02)
    q = torch.zeros(3,23); q[0,0] = .210; q[1,0] = -1.021
    torque = torch.zeros_like(q); torque[2,0] = 27.01
    for i in range(8):
        value.sample(i,q,torch.zeros_like(q),torque,torch.zeros_like(q),
                     torch.zeros(3,2),torch.zeros(3,1,3))
    assert value.strict_flags[:,0].tolist() == [True,True,False]
    assert value.training_flags.tolist() == [[False,False,False],[True,False,False],[False,False,True]]
    value.finish(torch.tensor([1,1,1]),torch.zeros(3,dtype=torch.bool))
    assert value.first_violation.tolist() == [1,1,1]


def test_viability_recipe_is_100_then_500_and_keeps_v12_frozen():
    from eb.domain.training_recipe import (UFO_MJLAB_VIABILITY_RECIPE, UFO_MJLAB_RECIPE,
        validate_update_target, recipe_resources)
    from eb.domain.rlinf_form import get_template
    from eb.rpo_bfm import action_control_for_execution
    for start,target in [(0,100),(100,600)]:
        validate_update_target(target,start=start,recipe_id=UFO_MJLAB_VIABILITY_RECIPE)
    for start,target in [(0,16),(0,500),(24,524),(600,1100)]:
        with pytest.raises(ValueError):
            validate_update_target(target,start=start,recipe_id=UFO_MJLAB_VIABILITY_RECIPE)
    old = action_control_for_execution({'recipe_id': UFO_MJLAB_RECIPE})['physics']
    new = action_control_for_execution({'recipe_id': UFO_MJLAB_VIABILITY_RECIPE})['physics']
    assert 'rsi_joint_lookahead_s' not in old
    assert new['rsi_joint_lookahead_s'] == .02 and new['training_position_tolerance_rad'] == .02
    assert recipe_resources(UFO_MJLAB_VIABILITY_RECIPE)['gpus'] == 4
    assert '+actor.model.physics_profile=native_mjcf_viability_v13' in get_template(UFO_MJLAB_VIABILITY_RECIPE).default_overrides
