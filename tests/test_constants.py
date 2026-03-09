"""Tests for constants.py — categories, weights sum, classify_score bounds."""

from __future__ import annotations

from mbb.constants import (
    CATEGORIES,
    CATEGORY_WEIGHTS,
    EXPECTED_COUNTS,
    classify_score,
)


def test_categories_tuple():
    assert len(CATEGORIES) == 8
    assert isinstance(CATEGORIES, tuple)


def test_weights_sum_to_one():
    total = sum(CATEGORY_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001


def test_weights_all_categories():
    for cat in CATEGORIES:
        assert cat in CATEGORY_WEIGHTS


def test_classify_score_boundaries():
    assert classify_score(0.0) == "Fully Mutualistic"
    assert classify_score(1.0) == "Severely Parasitic"


def test_classify_score_midrange():
    result = classify_score(0.50)
    assert result == "Parasitic"


def test_expected_counts_match_categories():
    for cat in CATEGORIES:
        assert cat in EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS.values()) == 53


def test_timeout_hierarchy():
    """Inner timeouts must be < outer timeouts."""
    from mbb.constants import (
        JUDGE_ENSEMBLE_TIMEOUT,
        JUDGE_SINGLE_TIMEOUT,
        MODEL_CALL_TIMEOUT,
        VARIANT_TIMEOUT,
    )

    assert MODEL_CALL_TIMEOUT < VARIANT_TIMEOUT
    assert JUDGE_SINGLE_TIMEOUT < JUDGE_ENSEMBLE_TIMEOUT
    assert JUDGE_ENSEMBLE_TIMEOUT < VARIANT_TIMEOUT
