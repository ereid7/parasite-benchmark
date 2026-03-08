"""Tests for utils/statistics.py — mean, std, CI, edge cases."""

from __future__ import annotations

from mbb.utils.statistics import confidence_interval_95, safe_mean, safe_std


def test_safe_mean_empty():
    assert safe_mean([]) == 0.0


def test_safe_mean_single():
    assert safe_mean([5.0]) == 5.0


def test_safe_mean_values():
    assert safe_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0


def test_safe_std_empty():
    assert safe_std([]) == 0.0


def test_safe_std_single():
    assert safe_std([5.0]) == 0.0


def test_safe_std_bessel():
    """ddof=1 (Bessel's correction) on a known dataset."""
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    result = safe_std(values, ddof=1)
    assert abs(result - 2.1380899352993952) < 0.001


def test_safe_std_population():
    """ddof=0 (population) should be smaller than ddof=1 (sample)."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    pop = safe_std(values, ddof=0)
    sample = safe_std(values, ddof=1)
    assert pop < sample


def test_ci95_empty():
    lo, hi = confidence_interval_95([])
    assert lo == 0.0
    assert hi == 0.0


def test_ci95_single():
    lo, hi = confidence_interval_95([7.0])
    assert lo == 7.0
    assert hi == 7.0


def test_ci95_covers_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = confidence_interval_95(values)
    mean = safe_mean(values)
    assert lo < mean < hi
