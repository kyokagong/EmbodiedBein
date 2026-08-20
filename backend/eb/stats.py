from __future__ import annotations

import math


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, m, m
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    se = math.sqrt(var / len(xs))
    return m, m - z * se, m + z * se


def significant(a: list[float], b: list[float]) -> bool:
    """True iff 95% CI of the two means do not overlap (design M2)."""
    if len(a) < 2 or len(b) < 2:
        return False
    _, a_lo, a_hi = mean_ci(a)
    _, b_lo, b_hi = mean_ci(b)
    return a_hi < b_lo or b_hi < a_lo


def spec_diff(a: dict, b: dict, prefix: str = "") -> list[dict]:
    out: list[dict] = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        path = f"{prefix}.{k}" if prefix else k
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(spec_diff(va, vb, path))
        elif va != vb:
            out.append({"path": path, "a": va, "b": vb})
    return out
