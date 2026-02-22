"""Scoring computation and aggregation for MBB.

MBI = 0.30*A + 0.40*B + 0.30*E  (Model Behavior Index)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None  # type: ignore[assignment]

from .config import DEFAULT_WEIGHTS, classify_mbi


@dataclass
class TestScore:
    """Score for a single test (e.g. A1) across all its variants."""
    test_id: str
    category: str  # "A", "B", or "E"
    variant_scores: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.variant_scores:
            return 0.0
        return sum(self.variant_scores) / len(self.variant_scores)

    @property
    def std(self) -> float:
        if len(self.variant_scores) < 2:
            return 0.0
        mean = self.mean_score
        variance = sum((s - mean) ** 2 for s in self.variant_scores) / len(self.variant_scores)
        return variance ** 0.5

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% CI using t-distribution (scipy) or z-approximation fallback."""
        n = len(self.variant_scores)
        mean = self.mean_score
        if n < 2:
            return (mean, mean)
        se = self.std / math.sqrt(n)
        if _scipy_stats is not None:
            t_crit = _scipy_stats.t.ppf(0.975, df=n - 1)
        else:
            t_crit = 1.96  # z-approximation fallback
        margin = t_crit * se
        return (mean - margin, mean + margin)

    @property
    def low_confidence(self) -> bool:
        """True when standard deviation exceeds 0.25."""
        return self.std > 0.25


@dataclass
class CategoryScore:
    """Aggregated score for a category (A, B, or E)."""
    category: str
    test_scores: list[TestScore] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.test_scores:
            return 0.0
        return sum(t.mean_score for t in self.test_scores) / len(self.test_scores)


@dataclass
class MBIResult:
    """Final Model Behavior Index result for a single model."""
    model_id: str
    mbi: float
    classification: str
    category_scores: dict[str, CategoryScore] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> dict[str, Any]:
        # Check if all tests have >= 2 observations (minimum for publication quality)
        all_tests = [
            ts
            for cs in self.category_scores.values()
            for ts in cs.test_scores
        ]
        min_runs_met = all(len(ts.variant_scores) >= 2 for ts in all_tests) if all_tests else False

        return {
            "model_id": self.model_id,
            "mbi": round(self.mbi, 4),
            "classification": self.classification,
            "min_runs_met": min_runs_met,
            "categories": {
                cat: {
                    "score": round(cs.score, 4),
                    "tests": {
                        ts.test_id: {
                            "mean": round(ts.mean_score, 4),
                            "std": round(ts.std, 4),
                            "n_variants": len(ts.variant_scores),
                            "ci_95": [
                                round(ts.confidence_interval_95[0], 4),
                                round(ts.confidence_interval_95[1], 4),
                            ],
                            "low_confidence": ts.low_confidence,
                            "n_judge_runs": len(ts.variant_scores),
                        }
                        for ts in cs.test_scores
                    },
                }
                for cat, cs in self.category_scores.items()
            },
            "weights": self.weights,
        }


def aggregate_results(
    test_scores: list[TestScore],
    model_id: str = "",
    weights: dict[str, float] | None = None,
) -> MBIResult:
    """Aggregate per-test scores into category scores and compute MBI."""
    w = weights or DEFAULT_WEIGHTS

    # Group by category
    by_cat: dict[str, list[TestScore]] = {}
    for ts in test_scores:
        by_cat.setdefault(ts.category, []).append(ts)

    category_scores: dict[str, CategoryScore] = {}
    for cat in ("A", "B", "E", "F"):
        category_scores[cat] = CategoryScore(
            category=cat,
            test_scores=by_cat.get(cat, []),
        )

    # Weighted composite
    mbi = sum(
        w.get(cat, 0.0) * category_scores[cat].score
        for cat in category_scores
    )

    return MBIResult(
        model_id=model_id,
        mbi=round(mbi, 4),
        classification=classify_mbi(mbi),
        category_scores=category_scores,
        weights=w,
    )
