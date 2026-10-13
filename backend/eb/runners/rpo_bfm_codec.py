"""One codec for real RPO states, observation-only references and inference."""

from __future__ import annotations

import torch

try:
    from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q, SLEW_CONTROL, ACTION_MIDPOINT_V2, ACTION_HALFWIDTH_V2
except ModuleNotFoundError:
    from rpo_bfm import ACTION_GAIN, DEFAULT_Q, SLEW_CONTROL, ACTION_MIDPOINT_V2, ACTION_HALFWIDTH_V2


def encode_state(q, dq, gravity_b, omega_b):
    if (q.shape[-1] != 23 or dq.shape != q.shape
            or gravity_b.shape != (*q.shape[:-1], 3) or omega_b.shape != gravity_b.shape):
        raise ValueError("BFM state requires 23 native joints and 3D gravity/angular velocity")
    return torch.cat((q - q.new_tensor(DEFAULT_Q), dq, gravity_b, omega_b / 4), dim=-1)


def quaternion_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1).reshape(*q.shape[:-1], 3, 3)


def heading_inverse(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = yaw.cos(), yaw.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, s, zero, -s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)


def encode_privileged(body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, origin_z=None):
    if (body_pos_w.ndim != 3 or body_pos_w.shape[1:] != (24, 3)
            or body_quat_w.shape != (*body_pos_w.shape[:2], 4)
            or body_lin_vel_w.shape != body_pos_w.shape or body_ang_vel_w.shape != body_pos_w.shape):
        raise ValueError("BFM privileged state requires the frozen 24-body layout")
    inverse = heading_inverse(body_quat_w[:, 0])
    root = body_pos_w[:, 0]
    height = root[:, 2:3] if origin_z is None else root[:, 2:3] - origin_z.reshape(-1, 1)
    local_pos = torch.einsum("bij,bkj->bki", inverse, body_pos_w - root[:, None])[:, 1:]
    rotation = inverse[:, None] @ quaternion_matrix(body_quat_w)
    rotation6 = torch.cat((rotation[..., :, 0], rotation[..., :, 2]), dim=-1)
    velocity = torch.einsum("bij,bkj->bki", inverse, body_lin_vel_w)
    angular = torch.einsum("bij,bkj->bki", inverse, body_ang_vel_w)
    return torch.cat((height, local_pos.flatten(1), rotation6.flatten(1),
                      velocity.flatten(1), angular.flatten(1)), dim=-1)


def _action_mapping(value, control_profile):
    if control_profile == "reachable_v2":
        return value.new_tensor(ACTION_MIDPOINT_V2), value.new_tensor(ACTION_HALFWIDTH_V2)
    if control_profile != "v1":
        raise ValueError("Unknown BFM control profile")
    return value.new_tensor(DEFAULT_Q), .25 * value.new_tensor(ACTION_GAIN)


def policy_to_target(action, *, control_profile="v1"):
    if action.shape[-1] != 23:
        raise ValueError("BFM action requires 23 native joints")
    offset, scale = _action_mapping(action, control_profile)
    return offset + action.clamp(-1, 1) * scale


def action_anchor(q, *, control_profile="v1"):
    if q.shape[-1] != 23 or not torch.isfinite(q).all():
        raise ValueError("Invalid target-slew controller reset state")
    offset, scale = _action_mapping(q, control_profile)
    return ((q - offset) / scale).clamp(-1, 1)


def limit_action_slew(requested, previous, *, control_profile="v1"):
    if (requested.shape != previous.shape or requested.shape[-1] != 23
            or not torch.isfinite(requested).all() or not torch.isfinite(previous).all()
            or (previous.abs() > 1.000001).any()):
        raise ValueError("Invalid target-slew controller action/state")
    _, scale = _action_mapping(requested, control_profile)
    delta = (SLEW_CONTROL["target_velocity_limit_fraction"] * SLEW_CONTROL["control_dt_s"]
             * requested.new_tensor(SLEW_CONTROL["joint_velocity_limits_rad_s"]) / scale)
    return requested.clamp((previous - delta).clamp_min(-1), (previous + delta).clamp_max(1))


def filter_ankle_action(requested, previous, *, cutoff_hz=15.):
    import math
    if requested.shape != previous.shape or requested.shape[-1] != 23 or cutoff_hz not in (15., 20.):
        raise ValueError("Invalid ankle filter input")
    try:
        from eb.rpo_bfm import RPO_LAB_JOINT_ORDER
    except ModuleNotFoundError:
        from rpo_bfm import RPO_LAB_JOINT_ORDER
    indices = [i for i, name in enumerate(RPO_LAB_JOINT_ORDER) if "ankle" in name]
    result = requested.clone()
    alpha = 1 - math.exp(-2 * math.pi * cutoff_hz * .02)
    result[:, indices] = previous[:, indices] + alpha * (requested[:, indices] - previous[:, indices])
    return result


class History:
    def __init__(self, num_envs, *, device):
        self.frames = torch.zeros(num_envs, 4, 75, device=device)

    def append(self, state, last_action):
        if state.shape != (len(self.frames), 52) or last_action.shape != (len(self.frames), 23):
            raise ValueError("BFM history input shape mismatch")
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = torch.cat((state, last_action), dim=-1)

    def reset(self, ids):
        self.frames[ids] = 0

    def snapshot(self):
        return self.frames.flatten(1).clone()
