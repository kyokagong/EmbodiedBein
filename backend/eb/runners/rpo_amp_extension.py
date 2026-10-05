"""RPO locomotion AMP task, sharing the existing WBC physics bridge."""

from __future__ import annotations

try:
    from .rpo_wbc_extension import RPOWBCEnv, _runtime_classes
except ImportError:  # Image copies runners as top-level modules.
    from rpo_wbc_extension import RPOWBCEnv, _runtime_classes


class AMPFrameHistory:
    def __init__(self, num_envs: int, device):
        import torch

        self.frames = torch.zeros((num_envs, 3, 74), device=device, dtype=torch.float32)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset(self, env_ids, initial_frame):
        if len(env_ids) == 0:
            return
        self.frames[env_ids] = 0
        self.frames[env_ids, 2] = initial_frame
        self.count[env_ids] = 1

    def advance(self, frame):
        self.frames[:, :2] = self.frames[:, 1:].clone()
        self.frames[:, 2] = frame
        self.count.clamp_max_(2).add_(1)
        return self.frames.clone(), self.count >= 3


class RPOAMPEnv(RPOWBCEnv):
    """The parent bridge handles only RLinf's Isaac app and observation routing."""


def __getattr__(name):
    if name not in {"RPOAMPEnvCfg", "RPOAMPIsaacEnv"}:
        raise AttributeError(name)
    cfg, env = _runtime_classes(basic=True, amp=True)
    return cfg if name.endswith("EnvCfg") else env
