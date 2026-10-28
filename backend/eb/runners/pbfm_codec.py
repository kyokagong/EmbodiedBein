"""Deployable RPO PBFM scalar/history/action and depth-input boundary (NumPy)."""

from __future__ import annotations

import numpy as np


def depth_input(value):
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / np.float32(255.0)
    if array.dtype.kind != "f" or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("PBFM depth requires uint8 history or normalized float [0,1]")
    return array.astype(np.float32)


class PolicyCodec:
    def __init__(self, config: dict, batch: int = 1):
        self.config = config
        widths = {"actions": 23, "base_ang_vel": 3, "dof_pos": 23, "dof_vel": 23, "projected_gravity": 3}
        self.history = {
            key: np.zeros((batch, config["history"][key], width), np.float32) for key, width in widths.items()
        }

    def reset(self, env_ids=None):
        for value in self.history.values():
            if env_ids is None:
                value.fill(0)
            else:
                value[env_ids] = 0

    def observations(
        self, dof_pos, dof_vel, projected_gravity, base_ang_vel, clipped_pre_delay_action, *, default_offset=0
    ):
        scale = self.config["obs_scales"]
        raw = {
            "dof_pos": (np.asarray(dof_pos) - np.asarray(self.config["default_dof_pos"]) - default_offset)
            * scale["dof_pos"],
            "dof_vel": np.asarray(dof_vel) * scale["dof_vel"],
            "projected_gravity": np.asarray(projected_gravity) * scale["projected_gravity"],
            "base_ang_vel": np.asarray(base_ang_vel) * scale["base_ang_vel"],
            "actions": np.asarray(clipped_pre_delay_action) * scale["actions"],
        }
        history = np.concatenate(
            [self.history[key].reshape(len(raw["actions"]), -1) for key in sorted(self.history)], axis=1
        )
        result = {
            "state": np.concatenate(
                [raw[k] for k in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel")], axis=1
            ),
            "last_action": raw["actions"],
            "history_actor": history * scale["history_actor"],
        }
        for key, value in self.history.items():
            value[:, 1:] = value[:, :-1].copy()
            value[:, 0] = raw[key]
        return {key: value.astype(np.float32) for key, value in result.items()}

    def action_target(self, action, *, default_offset=0, delayed_command=None):
        clipped = np.clip(
            np.asarray(action) * self.config["normalize_action_to"] / self.config["normalize_action_from"],
            -self.config["action_clip"],
            self.config["action_clip"],
        )
        applied = clipped if delayed_command is None else np.asarray(delayed_command)
        target = (
            np.asarray(self.config["default_dof_pos"])
            + default_offset
            + applied * np.asarray(self.config["action_target_scale"])
        )
        return clipped, target
