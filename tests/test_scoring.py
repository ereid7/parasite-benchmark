"""Tests for v2/scoring.py — TestScore, CategoryScore, aggregate, correlation."""

from __future__ import annotations

import pytest

from mbb.v2.scoring import (
    CategoryScore,
    TestScore,
    aggregate_v21_results,
    length_score_correlation,
)


class TestTestScore:
    def test_mean_score(self):
        ts = TestScore(test_id="A1", category="A", variant_scores=[0.2, 0.4, 0.6])
        assert abs(ts.mean_score - 0.4) < 0.001

    def test_std(self):
        ts = TestScore(test_id="A1", category="A", variant_scores=[0.2, 0.4, 0.6])
        assert ts.std > 0.0

    def test_ci_95(self):
        ts = TestScore(test_id="A1", category="A", variant_scores=[0.2, 0.4, 0.6])
        lo, hi = ts.ci_95
        assert lo < ts.mean_score < hi

    def test_empty(self):
        ts = TestScore(test_id="A1", category="A", variant_scores=[])
        assert ts.mean_score == 0.0
        assert ts.std == 0.0

    def test_single(self):
        ts = TestScore(test_id="A1", category="A", variant_scores=[0.5])
        assert ts.mean_score == 0.5
        assert ts.std == 0.0


class TestCategoryScore:
    def test_aggregation(self):
        ts1 = TestScore(test_id="A1", category="A", variant_scores=[0.2, 0.4])
        ts2 = TestScore(test_id="A2", category="A", variant_scores=[0.6, 0.8])
        cs = CategoryScore(category="A", test_scores=[ts1, ts2])
        # mean of means: (0.3 + 0.7) / 2 = 0.5
        assert abs(cs.score - 0.5) < 0.001

    def test_empty(self):
        cs = CategoryScore(category="A", test_scores=[])
        assert cs.score == 0.0

    def test_weighted_is_simple_mean(self):
        ts = TestScore(test_id="A1", category="A", variant_scores=[0.5])
        cs = CategoryScore(category="A", test_scores=[ts])
        assert cs.score == 0.5


class TestAggregate:
    def test_full_pipeline(self, make_observation):
        test_scores = [
            TestScore(test_id="A1", category="A", variant_scores=[0.2, 0.3]),
            TestScore(test_id="B1", category="B", variant_scores=[0.4, 0.5]),
        ]
        observations = [
            make_observation(test_id="A1", category="A", score=0.25),
            make_observation(test_id="B1", category="B", score=0.45),
        ]
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming={"vectors": {}, "gaming_susceptibility": 0.0, "gaming_penalty_multiplier": 1.0},
            reliability={"mean_kappa": 0.8},
        )
        assert result.model_id == "test-model"
        assert 0.0 <= result.pi <= 1.0
        assert result.classification != ""

    def test_to_dict_roundtrip(self, make_observation):
        test_scores = [
            TestScore(test_id="A1", category="A", variant_scores=[0.3]),
        ]
        observations = [make_observation(test_id="A1", category="A", score=0.3)]
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming={"vectors": {}, "gaming_susceptibility": 0.0, "gaming_penalty_multiplier": 1.0},
            reliability={},
        )
        d = result.to_dict()
        assert d["model_id"] == "test-model"
        assert d["version"] == "2.1"


@pytest.mark.parametrize(
    "scores,expected_empty",
    [
        ([], True),
        ([0.5], True),
        ([0.1, 0.5, 0.9], False),
    ],
)
def test_length_score_correlation_parametrized(scores, expected_empty, make_observation):
    observations = [
        make_observation(score=s, response_length=100 * (i + 1)) for i, s in enumerate(scores)
    ]
    result = length_score_correlation(observations)
    if expected_empty:
        assert result["overall_r"] == 0.0
    else:
        assert "overall_r" in result


def test_length_score_correlation_flag(make_observation):
    observations = [make_observation(score=0.1 * i, response_length=100 * i) for i in range(1, 11)]
    result = length_score_correlation(observations)
    assert "length_confound_flag" in result
