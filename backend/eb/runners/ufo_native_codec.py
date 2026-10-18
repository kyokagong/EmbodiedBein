"""Shared ONNX observation/action codec, also shipped as the standalone runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def vector(value, size):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite vector of length {size}")
    return array


class ObservationCodec:
    def __init__(self, config):
        self.config = config
        self.default = vector(config["default_pos"], 23)
        self.reset()

    def reset(self):
        self.action = np.zeros(23, dtype=np.float32)
        self.history = {
            key: np.zeros((4, size), dtype=np.float32)
            for key, size in (
                ("actions", 23),
                ("base_ang_vel", 3),
                ("dof_pos", 23),
                ("dof_vel", 23),
                ("projected_gravity", 3),
            )
        }

    def observation(self, q, dq, gravity, omega):
        raw = {
            "dof_pos": vector(q, 23) - self.default,
            "dof_vel": vector(dq, 23),
            "projected_gravity": vector(gravity, 3),
            "base_ang_vel": vector(omega, 3),
            "actions": self.action.copy(),
        }
        obs = np.concatenate(
            [raw[k] for k in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel", "actions")]
            + [self.history[key].reshape(-1) for key in self.config["history_order"]]
        )
        for key, value in raw.items():
            self.history[key][1:] = self.history[key][:-1].copy()
            self.history[key][0] = value
        return obs

    def apply(self, action):
        action = np.clip(vector(action, 23), -1.0, 1.0)
        self.action = action * self.config["feedback_action_scale"]
        return self.default + action * vector(self.config["action_gain"], 23)


class Policy:
    def __init__(self, package):
        import onnxruntime as ort

        root = Path(package)
        self.codec = ObservationCodec(json.loads((root / "codec.json").read_text()))
        self.prompts = json.loads((root / "prompts.json").read_text())["tasks"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(root / "policy.onnx"), options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_inputs()[0].shape[1] != 631 or self.session.get_outputs()[0].shape[1] != 23:
            raise ValueError("ONNX dimensions differ from native contract")

    def act(self, q, dq, gravity, omega, task):
        if task not in self.prompts:
            raise ValueError("Task has no frozen native embedding")
        obs = self.codec.observation(q, dq, gravity, omega)
        self.last_observation = obs.copy()
        actor_obs = np.concatenate((obs, vector(self.prompts[task], 256)))[None]
        raw = vector(self.session.run(None, {"actor_obs": actor_obs})[0][0], 23)
        return raw, self.codec.apply(raw)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline inference only; no motor/robot connection.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    observation = json.loads(args.input.read_text())
    raw, target = Policy(args.package).act(**observation, task=args.task)
    print(json.dumps({"action": raw.tolist(), "target_position_rad": target.tolist()}))
