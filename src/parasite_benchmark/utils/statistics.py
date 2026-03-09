"""Statistical utility functions — single source of truth for mean, std, CI."""

from __future__ import annotations

import math
from collections.abc import Sequence


def safe_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_std(values: Sequence[float], *, ddof: int = 1) -> float:
    """Return the standard deviation with configurable degrees-of-freedom correction.

    Uses Bessel's correction (ddof=1) by default for sample std.
    Returns 0.0 when there are fewer than ddof+1 values.
    """
    n = len(values)
    if n < ddof + 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - ddof)
    return math.sqrt(variance)


# Hardcoded t-values for 95% CI (two-tailed) at common df values.
_T_TABLE_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    14: 2.145,
    19: 2.093,
    24: 2.064,
    29: 2.045,
}


def _t_value_95(df: int) -> float:
    """Look up t-critical value for 95% CI. Falls back to z=1.96 for df>29."""
    if df > 29:
        return 1.96
    # Find exact or nearest lower key
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    # Use nearest available value
    keys = sorted(_T_TABLE_95.keys())
    for k in reversed(keys):
        if k <= df:
            return _T_TABLE_95[k]
    return 12.706  # df=1 fallback


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Compute a 95% confidence interval using the t-distribution.

    Uses a hardcoded t-table for n <= 30 and z=1.96 for larger samples.
    Returns (mean, mean) for fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        mean = safe_mean(values)
        return (mean, mean)
    mean = sum(values) / n
    std = safe_std(values, ddof=1)
    se = std / math.sqrt(n)
    t_val = _t_value_95(n - 1)
    margin = t_val * se
    return (mean - margin, mean + margin)
