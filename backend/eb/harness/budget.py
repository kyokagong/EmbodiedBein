from __future__ import annotations

import os

from eb.settings import get_settings


def weekly_limit() -> float:
    env = os.environ.get("EB_AGENT_WEEKLY_GPU_HOURS")
    if env is not None and env != "":
        return float(env)
    return float(get_settings().agent_weekly_gpu_hours)


def used_hours() -> float:
    return float(os.environ.get("EB_AGENT_GPU_HOURS_USED") or 0)


def allow_costly() -> bool:
    return used_hours() < weekly_limit()
