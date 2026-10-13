import numpy as np
import pytest


def test_ufo_bridge_preserves_fps_names_and_converts_root_axis_angle():
    from eb.runners.g1_lafan_bridge import joint_trajectory

    names = [f"joint_{i}" for i in range(29)]
    pose = np.zeros((12, 30, 3))
    pose[:, 0, 2] = np.pi/2
    clip = {"fps":60, "dof_pos":np.zeros((12, 29)),
            "root_trans_offset":np.tile([1., 2., .8], (12, 1)), "pose_aa":pose}
    result = joint_trajectory(clip, names, limit_frames=10)
    assert result["joint_q"].shape == (10, 36)
    assert np.allclose(result["joint_q"][0, :7], [1., 2., .8, 0., 0., 2**-.5, 2**-.5])
    assert result["fps"] == 60 and result["root_quat_format"] == "xyzw"
    assert result["dof_names"].tolist() == names
    for wrong in ({"dof_pos":np.zeros((12, 23))}, {"fps":0},
                  {"dof_names":names[::-1]}, {"root_trans_offset":np.full((12, 3), np.nan)}):
        with pytest.raises(ValueError):
            joint_trajectory({**clip, **wrong}, names)
    del clip["dof_pos"]
    with pytest.raises(ValueError, match="dof_pos"):
        joint_trajectory(clip, names)


def test_bridge_rejects_finite_values_that_overflow_output_or_rotation_norm():
    from eb.runners.g1_lafan_bridge import joint_trajectory

    names = [f"joint_{i}" for i in range(29)]
    clip = {"fps":50, "dof_pos":np.full((10, 29), 1e40),
            "root_trans_offset":np.zeros((10, 3)), "pose_aa":np.zeros((10, 30, 3))}
    with pytest.raises(ValueError):
        joint_trajectory(clip, names)

    clip.update(dof_pos=np.zeros((10, 29)), pose_aa=np.full((10, 30, 3), 1e200))
    with pytest.raises(ValueError):
        joint_trajectory(clip, names)


def test_actual_ufo_dof_alias_requires_axis_order_consistency():
    from eb.runners.g1_lafan_bridge import verified_ufo_clip

    names = [f"joint_{i}" for i in range(29)]
    axes = np.tile([0., 1., 0.], (29, 1))
    q = np.full((10, 29), .2)
    pose = np.zeros((10, 30, 3))
    pose[:, 1:] = q[..., None]*axes
    clip = {"fps":30, "dof":q, "pose_aa":pose, "root_trans_offset":np.zeros((10, 3))}
    result = verified_ufo_clip(clip, names, axes)
    assert np.allclose(result["joint_q"][:, 7:], q)
    assert result["fps"] == 30
    pose[:, 2, 0] = .1
    with pytest.raises(ValueError, match="axis"):
        verified_ufo_clip(clip, names, axes)
