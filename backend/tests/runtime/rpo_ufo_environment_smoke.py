"""Fixed two-environment Isaac probe; execute only through the registered Job operation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def probe(output: Path):
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from bfm_ufo_agent import AUX_NAMES, OBS_DIMS
        from rpo_bfm_extension import RPOBFMEnvCfg
        from rpo_ufo_isaac import GYM_ID, register_task
        from rpo_ufo_motion import digest, load_robot_state

        root = Path(os.environ["EB_UFO_MOTION_ROOT"])
        _, expert = load_robot_state(root, asset_sha256=os.environ["EB_UFO_ASSET_SHA256"], seed=42)
        if any(not windows for windows in expert.windows[250].values()):
            raise ValueError("Both expert sources require train-only 250-frame windows")
        register_task()
        cfg = RPOBFMEnvCfg()
        cfg.seed, cfg.scene.num_envs = 42, 2
        env = gym.make(GYM_ID, cfg=cfg, render_mode=None).unwrapped
        obs, _ = env.reset()
        for native, width in OBS_DIMS.items():
            key = "states" if native == "state" else native
            assert obs[key].shape == (2, width) and torch.isfinite(obs[key]).all()
        for key, value in obs.items():
            torch.testing.assert_close(value, env._get_observations()[key], atol=0, rtol=0)
        before = {key: value[1].clone() for key, value in obs.items()}
        env.reset(env_ids=torch.tensor([0], device=env.device))
        partial = env._get_observations()
        for key, value in before.items():
            torch.testing.assert_close(value, partial[key][1], atol=0, rtol=0)
        assert partial["step_id"][0] == 0
        assert partial["episode_id"][0] == obs["episode_id"][0] + 1
        env.episode_length_buf[0] = env.episode_length
        terminal, _, terminated, truncated, _ = env.step(torch.zeros(2, 23, device=env.device))
        assert bool(truncated[0])
        assert terminal["step_id"][0] == 1
        assert terminal["ufo_aux_rewards"].shape == (2, len(AUX_NAMES))
        assert torch.isfinite(terminal["ufo_aux_rewards"]).all()
        captured = {key: value.clone() for key, value in terminal.items()}
        env.reset(env_ids=torch.tensor([0], device=env.device))
        assert captured["step_id"][0] == 1 and env._get_observations()["step_id"][0] == 0
        receipt = {
            "schema_version": "eb/rpo-ufo-environment-probe/v1",
            "status": "passed",
            "backend": "isaaclab",
            "gym_id": GYM_ID,
            "envs": 2,
            "seed": 42,
            "asset_sha256": os.environ["EB_UFO_ASSET_SHA256"],
            "manifest_sha256": digest(root / "manifest.json"),
            "aux_names": list(AUX_NAMES),
            "aux_units": "raw unweighted per-control-step",
            "checks": [
                "observation_idempotent",
                "reference_partial_reset",
                "timeout_terminal",
                "raw_aux_eight",
            ],
            "terminated": terminated.cpu().tolist(),
            "truncated": truncated.cpu().tolist(),
            "qualification": "not_requested",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    probe(parser.parse_args().output)
