from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_play(path: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Parse Isaac play.py / eval_*.json into success_rate and per-seed vector."""
    if isinstance(path, dict):
        data = path
        source = "dict"
    else:
        data = json.loads(Path(path).read_text())
        source = str(path)
    metrics = data.get("metrics") or {}
    if "success_rate" in data:
        rate = float(data["success_rate"])
    elif "success_rate" in metrics:
        rate = float(metrics["success_rate"])
    else:
        # time_out fraction ≈ survived to horizon; other Episode_Termination/* are failures
        timeout = float(metrics.get("Episode_Termination/time_out") or 0.0)
        fails = sum(
            float(v)
            for k, v in metrics.items()
            if k.startswith("Episode_Termination/") and k != "Episode_Termination/time_out"
        )
        rate = timeout if timeout or fails else 0.0
        if timeout or fails:
            rate = max(0.0, min(1.0, timeout / max(timeout + fails, 1e-9)))
    seeds = data.get("seeds") or data.get("per_seed")
    if isinstance(seeds, dict):
        per_seed = [float(seeds[k]) for k in sorted(seeds, key=lambda x: str(x))]
    elif isinstance(seeds, list) and seeds and isinstance(seeds[0], (int, float)):
        per_seed = [float(rate)] * len(seeds) if all(isinstance(s, int) for s in seeds) else [float(x) for x in seeds]
    elif isinstance(seeds, list) and seeds and isinstance(seeds[0], dict):
        per_seed = [float(x.get("success_rate", rate)) for x in seeds]
    else:
        seed = data.get("seed")
        per_seed = [float(rate)] if seed is None else [float(rate)]
    mean_reward = metrics.get("Episode_Reward/track_lin_vel_xy_exp") or data.get("mean_reward")
    return {
        "success_rate": float(rate),
        "per_seed": per_seed,
        "mean_reward": float(mean_reward) if mean_reward is not None else None,
        "episodes": data.get("episodes"),
        "num_envs": data.get("num_envs"),
        "metrics": metrics,
        "source": source,
    }
