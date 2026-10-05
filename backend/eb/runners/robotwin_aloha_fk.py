"""Fixed RoboTwin ALOHA/AgileX FK and Xiaomi XR-1 action codec."""
import math

import numpy as np


_ORIGINS = (
    ((0.0, 0.0, 0.058), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.025013, 0.00060169, 0.042), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.26396, 0.0044548, 0.0), (-3.1416, 0.0, -0.015928), (0.0, 1.0, 0.0)),
    ((0.246, -0.00025, -0.06), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0015, -0.0855), (0.0, 0.0, -0.015928), (0.0, 0.0, 1.0)),
    ((0.03095, 0.0, 0.0855), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
_BASES = (((0.2305, 0.297, 0.782), (0.0, 0.0, 0.02)),
          ((0.2315, -0.3063, 0.781), (0.0, 0.0, 0.01)))
_TCP = (0.08457, 0.0, -0.000103515)


def _rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                     (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                     (-sp, cp * sr, cp * cr)), dtype=np.float64)


def _transform(xyz, rotation):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3], value[:3, 3] = rotation, xyz
    return value


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    cross = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]),
                      (-axis[1], axis[0], 0.0)))
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _arm_fk(joints, side):
    xyz, rpy = _BASES[side]
    pose = _transform(xyz, _rpy(rpy))
    for angle, (origin, origin_rpy, axis) in zip(joints, _ORIGINS):
        pose = pose @ _transform(origin, _rpy(origin_rpy)) @ _transform((0, 0, 0), _axis_angle(axis, float(angle)))
    return pose @ _transform(_TCP, np.eye(3))


def forward_kinematics(joint14):
    joint14 = np.asarray(joint14, dtype=np.float64)
    if joint14.shape != (14,):
        raise ValueError(f"joint14 expected shape (14,), got {joint14.shape}")
    return _arm_fk(joint14[:6], 0), _arm_fk(joint14[7:13], 1)


def _rotation_vector(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cosine)
    if angle < 1e-7:
        return np.zeros(3, dtype=np.float32)
    axis = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1])) / (2.0 * math.sin(angle))
    return (axis * angle).astype(np.float32)


def encode_xr1_action(current_joint14, target_joint14):
    current = np.asarray(current_joint14, dtype=np.float32)
    targets = np.asarray(target_joint14, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 14:
        raise ValueError(f"target joint actions expected (N, 14), got {targets.shape}")
    action = np.zeros((len(targets), 60), dtype=np.float32)
    mask = np.zeros_like(action, dtype=bool)
    current_poses = forward_kinematics(current)
    for index, target in enumerate(targets):
        for side, (source, destination, offset, grip) in enumerate(
            zip(current_poses, forward_kinematics(target), (0, 8), (6, 13))
        ):
            action[index, offset:offset + 3] = source[:3, :3].T @ (destination[:3, 3] - source[:3, 3])
            action[index, offset + 3:offset + 6] = _rotation_vector(source[:3, :3].T @ destination[:3, :3])
            action[index, offset + 6] = target[grip] - current[grip]
            mask[index, offset:offset + 7] = True
    return action, mask
