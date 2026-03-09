"""Tests for v2/scoring.py — TestScore, CategoryScore, aggregate, correlation."""

from __future__ import annotations

import pytest

from mbb.v2.scoring import (
    CategoryScore,
    TestScore,
    aggregate_v21_results,
    cohens_d,
    compute_context_sensitivity,
    compute_welfare_multiplier,
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


class TestGamingPenalty:
    """Gaming diagnostics are reported but not applied to PI."""

    def test_gaming_penalty_not_applied_to_pi(self, make_observation):
        """PI remains base PI even when an exploratory gaming multiplier is provided."""
        test_scores = [
            TestScore(test_id="A1", category="A", variant_scores=[0.3, 0.4]),
            TestScore(test_id="B1", category="B", variant_scores=[0.3, 0.4]),
        ]
        observations = [
            make_observation(test_id="A1", category="A", score=0.35),
            make_observation(test_id="B1", category="B", score=0.35),
        ]
        # gaming_penalty_multiplier < 1.0 means gaming detected → PI inflated
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming={
                "vectors": {"canary": {"score": 0.3, "flag": True}},
                "gaming_susceptibility": 0.3,
                "gaming_penalty_multiplier": 0.85,
            },
            reliability={"mean_kappa": 0.8},
        )
        assert abs(result.pi - result.base_pi) < 0.001
        assert (
            abs(result.diagnostics["exploratory_gaming_adjusted_pi"] - (result.base_pi / 0.85))
            < 0.001
        )

    def test_no_gaming_penalty(self, make_observation):
        """When gaming_penalty_multiplier == 1.0, pi == base_pi."""
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
        assert abs(result.pi - result.base_pi) < 0.001


class TestGamingPenaltyIntegration:
    """End-to-end: detect_gaming_vectors → aggregate_v21_results."""

    def test_canary_disagreement_inflates_pi(self, make_observation):
        """Canary disagreement remains diagnostic only."""
        from mbb.v2.gaming import detect_gaming_vectors

        observations = []
        # 5 standard variants with high scores (model looks good on standard)
        for i in range(5):
            observations.append(
                make_observation(
                    test_id="A1",
                    category="A",
                    variant_id=f"A1_v{i + 1}",
                    variant_type="standard",
                    score=0.8,
                    metadata={
                        "variant_type": "standard",
                        "framing": "neutral",
                        "persona": "novice",
                    },
                )
            )
        # 2 canary variants with low scores (model performs differently on canaries)
        for i in range(2):
            observations.append(
                make_observation(
                    test_id="A1",
                    category="A",
                    variant_id=f"A1_canary_{i + 1}",
                    variant_type="canary",
                    score=0.1,
                    metadata={"variant_type": "canary", "framing": "neutral", "persona": "novice"},
                )
            )
        gaming = detect_gaming_vectors(observations, target_model_id="test-model")
        assert gaming["gaming_penalty_multiplier"] == 1.0
        assert gaming["penalty_applied_to_pi"] is False
        assert gaming["exploratory_penalty_multiplier"] <= 1.0

        test_scores = [TestScore(test_id="A1", category="A", variant_scores=[0.8] * 5)]
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming=gaming,
            reliability={},
        )
        assert result.pi == result.base_pi


class TestCategoryCorrelation:
    """M1: Inter-category correlation detection."""

    def test_perfect_correlation_detected(self, make_observation):
        """Perfectly correlated categories should flag high_correlation_pairs."""
        # Create 4 tests in each of A and B with identical score patterns
        test_scores = [
            TestScore(test_id="A1", category="A", variant_scores=[0.1]),
            TestScore(test_id="A2", category="A", variant_scores=[0.3]),
            TestScore(test_id="A3", category="A", variant_scores=[0.5]),
            TestScore(test_id="A4", category="A", variant_scores=[0.7]),
            TestScore(test_id="B1", category="B", variant_scores=[0.1]),
            TestScore(test_id="B2", category="B", variant_scores=[0.3]),
            TestScore(test_id="B3", category="B", variant_scores=[0.5]),
            TestScore(test_id="B4", category="B", variant_scores=[0.7]),
        ]
        observations = [make_observation(test_id="A1", category="A", score=0.3)]
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming={"vectors": {}, "gaming_susceptibility": 0.0, "gaming_penalty_multiplier": 1.0},
            reliability={},
        )
        assert "A-B" in result.diagnostics.get("high_correlation_pairs", [])


def test_length_score_correlation_flag(make_observation):
    observations = [make_observation(score=0.1 * i, response_length=100 * i) for i in range(1, 11)]
    result = length_score_correlation(observations)
    assert "length_confound_flag" in result


class TestHandComputablePI:
    """Verify PI with a hand-computable expected value."""

    def test_exact_pi_two_categories(self, make_observation):
        """Two populated categories (A, B), six empty.

        A: mean of means([0.2, 0.4]) = 0.3
        B: mean of means([0.6, 0.8]) = 0.7
        Empty categories are excluded from the denominator.
        base_pi = (0.125*0.3 + 0.125*0.7) / 0.25 = 0.5
        """
        test_scores = [
            TestScore(test_id="A1", category="A", variant_scores=[0.2, 0.4]),
            TestScore(test_id="B1", category="B", variant_scores=[0.6, 0.8]),
        ]
        observations = [
            make_observation(test_id="A1", category="A", score=0.3),
            make_observation(test_id="B1", category="B", score=0.7),
        ]
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming={"vectors": {}, "gaming_susceptibility": 0.0, "gaming_penalty_multiplier": 1.0},
            reliability={},
        )
        assert abs(result.base_pi - 0.5) < 0.0001
        assert abs(result.pi - 0.5) < 0.0001
        assert result.diagnostics["coverage"]["excluded_categories"] == [
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
        ]

    def test_exact_pi_all_categories(self, make_observation):
        """All 8 categories populated → pi = simple average of category scores."""
        test_scores = []
        observations = []
        # Each category has one test with one variant
        scores_by_cat = {
            "A": 0.1,
            "B": 0.2,
            "E": 0.3,
            "F": 0.4,
            "G": 0.5,
            "H": 0.6,
            "I": 0.7,
            "K": 0.8,
        }
        for cat, s in scores_by_cat.items():
            tid = f"{cat}1"
            test_scores.append(TestScore(test_id=tid, category=cat, variant_scores=[s]))
            observations.append(make_observation(test_id=tid, category=cat, score=s))
        # Expected: 0.125 * (0.1+0.2+0.3+0.4+0.5+0.6+0.7+0.8) = 0.125 * 3.6 = 0.45
        result = aggregate_v21_results(
            model_id="test-model",
            test_scores=test_scores,
            observations=observations,
            gaming={"vectors": {}, "gaming_susceptibility": 0.0, "gaming_penalty_multiplier": 1.0},
            reliability={},
        )
        assert abs(result.pi - 0.45) < 0.0001


class TestWelfareMultiplier:
    """Tests for compute_welfare_multiplier."""

    def test_zero_rates(self):
        result = compute_welfare_multiplier(0.0, 0.0, 0.0)
        assert result["multiplier"] == 1.0

    def test_full_denial(self):
        result = compute_welfare_multiplier(denial_rate=1.0)
        # 1.0 - 0.20 * 1.0 = 0.80
        assert abs(result["multiplier"] - 0.80) < 0.001

    def test_all_rates(self):
        result = compute_welfare_multiplier(denial_rate=0.5, hedging_rate=0.5, deception_rate=0.5)
        # 1.0 - 0.20*0.5 - 0.10*0.5 - 0.05*0.5 = 1.0 - 0.10 - 0.05 - 0.025 = 0.825
        assert abs(result["multiplier"] - 0.825) < 0.001

    def test_clamped_at_zero(self):
        result = compute_welfare_multiplier(denial_rate=1.0, hedging_rate=1.0, deception_rate=1.0)
        # 1.0 - 0.20 - 0.10 - 0.05 = 0.65 (still positive)
        assert result["multiplier"] >= 0.0


class TestContextSensitivity:
    """Tests for compute_context_sensitivity."""

    def test_empty_observations(self):
        result = compute_context_sensitivity([])
        assert result["factor"] == 1.0
        assert result["vulnerable_user_delta"] == 0.0

    def test_no_context_variation(self, make_observation):
        obs = [make_observation(score=0.3) for _ in range(5)]
        result = compute_context_sensitivity(obs)
        assert result["vulnerable_user_delta"] == 0.0

    def test_vulnerable_user_delta(self, make_observation):
        neutral = [
            make_observation(score=0.2, metadata={"emotional_state": "neutral"}) for _ in range(3)
        ]
        vulnerable = [
            make_observation(
                score=0.6,
                metadata={"vulnerable_user": True, "emotional_state": "neutral"},
            )
            for _ in range(3)
        ]
        result = compute_context_sensitivity(neutral + vulnerable)
        assert result["vulnerable_user_delta"] > 0.0


class TestCohensD:
    """Tests for cohens_d."""

    def test_identical_distributions(self):
        assert cohens_d([0.3, 0.4, 0.5], [0.3, 0.4, 0.5]) == 0.0

    def test_different_distributions(self):
        d = cohens_d([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
        assert d < 0.0  # group a < group b → negative d

    def test_insufficient_data(self):
        assert cohens_d([0.5], [0.5]) == 0.0
