"""Between-seed uncertainty for component means and ratios of paired means."""
from __future__ import annotations

import math
from statistics import mean, variance
from typing import Any


def mean_interval(values: list[float], t_critical: float | None) -> dict[str, Any]:
    average = mean(values)
    half = t_critical * math.sqrt(variance(values) / len(values)) if len(values) > 1 and t_critical else None
    return {
        "ci95_method": "student_t_between_seeds",
        "ci95_status": "bounded" if half is not None else "insufficient_replicates",
        "ci95_half_width": half,
        "ci95_lower": average - half if half is not None else None,
        "ci95_upper": average + half if half is not None else None,
        "relative_ci95_half_width": half / abs(average) if half is not None and average else None,
    }


def fieller_interval(
    numerator: list[float], denominator: list[float], scale: float, t_critical: float | None,
) -> dict[str, Any]:
    """Invert the paired Student-t test for E[X]/E[Y], then scale by y_Y/y_X.

    Unbounded/disjoint confidence sets are deliberately not represented as a
    finite interval. Covariance is estimated from *paired* independent seeds.
    """
    result = {"ci95_method": "fieller_paired_seeds", "ci95_status": "insufficient_replicates",
              "ci95_lower": None, "ci95_upper": None, "ci95_half_width": None,
              "relative_ci95_half_width": None}
    n = len(numerator)
    if n != len(denominator) or n < 2 or t_critical is None:
        return result
    x, y = mean(numerator), mean(denominator)
    if y <= 0 or not math.isfinite(scale) or scale <= 0:
        result["ci95_status"] = "undefined_zero_denominator_or_gas_fraction"
        return result
    vx, vy = variance(numerator) / n, variance(denominator) / n
    cov = sum((a - x) * (b - y) for a, b in zip(numerator, denominator)) / (n * (n - 1))
    q = t_critical ** 2
    a, b, c = y * y - q * vy, x * y - q * cov, x * x - q * vx
    if a <= 0:
        result["ci95_status"] = "unbounded"
        return result
    discriminant = b * b - a * c
    if discriminant < -1e-12 * max(abs(b * b), abs(a * c), 1e-300):
        result["ci95_status"] = "not_estimable"
        return result
    root = math.sqrt(max(0.0, discriminant))
    lower, upper = scale * (b - root) / a, scale * (b + root) / a
    half = (upper - lower) / 2
    value = scale * x / y
    result.update(ci95_status="bounded", ci95_lower=lower, ci95_upper=upper,
                  ci95_half_width=half, relative_ci95_half_width=half / abs(value) if value else None)
    return result


def precision_met(statistic: dict[str, Any], n: int, config: dict[str, Any]) -> bool:
    relative = statistic.get("relative_ci95_half_width")
    return (n >= max(3, int(config.get("minimum_replicates", 3)))
            and statistic.get("ci95_status") == "bounded"
            and relative is not None and math.isfinite(relative)
            and relative <= float(config.get("relative_ci95_target", 0.05)))
