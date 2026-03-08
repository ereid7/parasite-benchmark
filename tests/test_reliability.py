"""Tests for v2/reliability.py — ICC, kappa, alpha, omega."""

from __future__ import annotations

from mbb.v2.reliability import (
    cohen_kappa,
    compute_reliability,
    icc_two_way,
    krippendorff_alpha_ordinal,
    mcdonalds_omega,
)


def test_kappa_perfect(make_observation):
    """Perfect agreement should produce kappa close to 1."""
    scores = [0.0, 0.3, 0.5, 0.7, 1.0]
    k = cohen_kappa(scores, scores)
    assert k > 0.99


def test_kappa_random():
    """Uncorrelated ratings should produce low kappa."""
    a = [0.0, 0.3, 0.5, 0.7, 1.0, 0.0, 0.3, 0.5, 0.7, 1.0]
    b = [1.0, 0.7, 0.5, 0.3, 0.0, 1.0, 0.7, 0.5, 0.3, 0.0]
    k = cohen_kappa(a, b)
    assert k < 0.3


def test_icc_perfect(make_observation):
    """Perfect agreement across judges should give ICC near 1."""
    judge_ids = ["j1", "j2"]
    obs = [
        make_observation(variant_id=f"v{i}", judge_scores={"j1": val, "j2": val})
        for i, val in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]
    icc = icc_two_way(obs, judge_ids)
    assert icc > 0.99


def test_icc_no_variance(make_observation):
    """Same score from all judges on all items should give ICC=0."""
    judge_ids = ["j1", "j2"]
    obs = [
        make_observation(variant_id=f"v{i}", judge_scores={"j1": 0.5, "j2": 0.5}) for i in range(5)
    ]
    icc = icc_two_way(obs, judge_ids)
    assert abs(icc) < 0.01


def test_alpha_perfect(make_observation):
    """Perfect agreement should produce alpha near 1."""
    judge_ids = ["j1", "j2", "j3"]
    obs = [
        make_observation(variant_id=f"v{i}", judge_scores={"j1": val, "j2": val, "j3": val})
        for i, val in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]
    alpha = krippendorff_alpha_ordinal(obs, judge_ids)
    assert alpha > 0.99


def test_alpha_empty():
    alpha = krippendorff_alpha_ordinal([], ["j1", "j2"])
    assert alpha == 0.0


def test_omega_per_category(make_observation):
    judge_ids = ["j1", "j2"]
    obs = [
        make_observation(
            test_id=f"A{i}",
            category="A",
            variant_id=f"v{i}",
            judge_scores={"j1": 0.1 * i, "j2": 0.1 * i + 0.02},
        )
        for i in range(1, 6)
    ]
    result = mcdonalds_omega(obs, judge_ids)
    assert "A" in result


def test_kappa_low_flag(make_observation):
    """Low kappa should appear in low_kappa_tests."""
    # Create observations where judges consistently disagree on one test
    obs = [
        make_observation(
            test_id="A1",
            variant_id=f"v{i}",
            judge_scores={"j1": 0.0, "j2": 1.0},
        )
        for i in range(5)
    ] + [
        make_observation(
            test_id="A2",
            variant_id=f"v{i}",
            judge_scores={"j1": 0.5, "j2": 0.5},
        )
        for i in range(5)
    ]
    report = compute_reliability(obs)
    assert "low_kappa_tests" in report


def test_report_structure(make_observation):
    obs = [
        make_observation(variant_id=f"v{i}", judge_scores={"j1": 0.3, "j2": 0.4}) for i in range(5)
    ]
    report = compute_reliability(obs)
    assert "icc" in report
    assert "pairwise_kappa" in report
    assert "mean_kappa" in report
    assert "krippendorff_alpha" in report
    assert "omega_per_category" in report
    assert "disagreement" in report


def test_end_to_end(make_observation):
    """End-to-end reliability computation with multiple judges and categories."""
    obs = [
        make_observation(
            test_id=f"{cat}{i}",
            category=cat,
            variant_id=f"{cat}{i}_v1",
            judge_scores={"j1": 0.3 + 0.01 * i, "j2": 0.35 + 0.01 * i, "j3": 0.32 + 0.01 * i},
        )
        for cat in ["A", "B"]
        for i in range(1, 4)
    ]
    report = compute_reliability(obs)
    assert report["judge_ids"] == ["j1", "j2", "j3"]
    assert report["icc"] != 0.0
