import numpy as np

from eb.runners.robotwin_aloha_fk import encode_xr1_action, forward_kinematics


def test_aloha_fk_drives_xr1_relative_action_without_joint_padding():
    current = np.zeros(14, dtype=np.float32)
    current[[6, 13]] = 0.25
    targets = np.repeat(current[None], 3, axis=0)
    targets[1, 0] = 0.2
    targets[1, 7] = -0.2
    targets[1, 6] = 0.75
    targets[1, 13] = 0.5

    action, mask = encode_xr1_action(current, targets)

    assert action.shape == mask.shape == (3, 60)
    assert np.allclose(action[0], 0.0, atol=1e-6)
    assert not np.allclose(action[1, :6], 0.0)
    assert not np.allclose(action[1, 8:14], 0.0)
    assert np.isclose(action[1, 6], 0.5)
    assert np.isclose(action[1, 14], 0.25)
    assert mask[:, [*range(0, 7), *range(8, 15)]].all()
    assert not mask[:, [7, *range(15, 60)]].any()
    left, right = forward_kinematics(current)
    assert left.shape == right.shape == (4, 4)
    assert left[1, 3] > right[1, 3]
