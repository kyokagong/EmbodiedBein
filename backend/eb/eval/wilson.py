"""Wilson score interval for binomial proportion confidence estimation.

Design reference: docs/design/05-evaluation-delivery.md
Wilson score interval provides accurate coverage for small sample sizes
and extreme proportions (near 0 or 1), avoiding the zero-width artifact of
normal approximation.
"""

from __future__ import annotations
import math
from typing import TypedDict


class ConfidenceInterval(TypedDict):
    point: float
    lower: float
    upper: float
    n: int
    confidence: float
    label: str


def report_interval(metrics: dict) -> ConfidenceInterval | None:
    """Use reported trial counts only; an aggregate rate does not imply n=64."""
    n = metrics.get("n_trials", metrics.get("n"))
    rate = metrics.get("success_rate")
    if (type(n) is not int or n <= 0 or type(rate) not in (int, float)
            or not math.isfinite(rate) or not 0 <= rate <= 1):
        return None
    return wilson_ci(round(rate * n), n)


def wilson_ci(k: int | float, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Compute Wilson score confidence interval for k successes in n trials.

    Args:
        k: Number of successes (0 <= k <= n)
        n: Number of trials (n > 0)
        confidence: Desired confidence level (default 0.95)

    Returns:
        ConfidenceInterval with point, lower, upper, n, confidence, and formatted label.
    """
    if n <= 0:
        return {
            "point": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n": 0,
            "confidence": confidence,
            "label": "0.0% [0.0%, 0.0%] (n=0)",
        }

    k = max(0, min(float(k), float(n)))
    p_hat = k / n

    # Standard normal quantile z for two-sided interval
    # For 95%, alpha = 0.05, z = 1.95996
    alpha = 1.0 - confidence
    # Approximate normal inverse CDF
    # For common confidence levels:
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        # Standard approximation via erfcinv
        z = math.sqrt(2.0) * _erfcinv(alpha)

    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    point_pct = round(p_hat * 100, 1)
    lower_pct = round(lower * 100, 1)
    upper_pct = round(upper * 100, 1)
    label = f"{point_pct}% [{lower_pct}%, {upper_pct}%] (n={n})"

    return {
        "point": round(p_hat, 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "n": n,
        "confidence": confidence,
        "label": label,
    }


def is_statistically_significant(ci_a: ConfidenceInterval, ci_b: ConfidenceInterval) -> bool:
    """Determine if difference between two confidence intervals is statistically significant.

    If intervals do not overlap, difference is significant at the given confidence level.
    """
    return ci_a["lower"] > ci_b["upper"] or ci_b["lower"] > ci_a["upper"]


def _erfcinv(x: float) -> float:
    """Approximation of inverse complementary error function."""
    if x >= 2.0:
        return -100.0
    if x <= 0.0:
        return 100.0
    w = -math.log(x * (2.0 - x)) if x < 1.0 else -math.log((2.0 - x) * x)
    if w < 5.0:
        w = w - 2.500000
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * w
        p = -3.5233877e-06 + p * w
        p = -4.39150654e-06 + p * w
        p = 0.00021858087 + p * w
        p = -0.00125372503 + p * w
        p = -0.00417768164 + p * w
        p = 0.246640727 + p * w
        p = 1.50140941 + p * w
    else:
        w = math.sqrt(w) - 3.000000
        p = -0.000200214257
        p = 0.000100950558 + p * w
        p = 0.00134934322 + p * w
        p = -0.00367342985 + p * w
        p = 0.00573950773 + p * w
        p = -0.0076224613 + p * w
        p = 0.00943887047 + p * w
        p = 1.00167406 + p * w
        p = 2.83297682 + p * w
    return p if x < 1.0 else -p
