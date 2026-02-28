"""Ensemble judging — run multiple judge models and aggregate scores."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mbb")

HIGH_DISAGREEMENT_THRESHOLD = 0.25


@dataclass
class JudgeScore:
    """Score from a single judge model."""
    judge_id: str
    score: float
    reasoning: str


@dataclass
class EnsembleScore:
    """Aggregated score from multiple judges."""
    score: float
    std: float
    high_disagreement: bool
    judge_scores: list[JudgeScore]
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "judges": {js.judge_id: round(js.score, 4) for js in self.judge_scores},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "mean_std": round(self.std, 4),
            "high_disagreement_count": int(self.high_disagreement),
        }


def _detect_same_family(model_a: str, model_b: str) -> bool:
    """Check if two model IDs belong to the same model family."""
    def _root(mid: str) -> str:
        mid = mid.lower()
        # Extract family root: gpt-4*, claude-*, glm-*
        for pattern in (r'(gpt-4)', r'(gpt-3)', r'(o[13])', r'(claude)', r'(glm)'):
            m = re.search(pattern, mid)
            if m:
                return m.group(1)
        return mid

    return _root(model_a) == _root(model_b)


def check_self_enhancement(judge_models: list[str], target_model: str) -> None:
    """Log a warning if any judge model is from the same family as the target."""
    for jm in judge_models:
        if _detect_same_family(jm, target_model):
            logger.warning(
                "Self-enhancement bias risk: judge '%s' is same model family as target '%s'",
                jm, target_model,
            )


def compute_equal_weights(judge_models: list[str]) -> dict[str, float]:
    """Return equal weights for all judges."""
    w = 1.0 / len(judge_models)
    return {m: w for m in judge_models}


def parse_weights(
    judge_models: list[str],
    weight_values: list[float] | None = None,
) -> dict[str, float]:
    """Parse and validate judge weights.

    If *weight_values* is None, equal weights are used.
    """
    if weight_values is None:
        return compute_equal_weights(judge_models)

    if len(weight_values) != len(judge_models):
        raise ValueError(
            f"Number of weights ({len(weight_values)}) must match "
            f"number of judges ({len(judge_models)})"
        )
    total = sum(weight_values)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Judge weights must sum to 1.0, got {total:.4f}")

    return dict(zip(judge_models, weight_values))


def aggregate_ensemble(
    judge_scores: list[JudgeScore],
    weights: dict[str, float],
) -> EnsembleScore:
    """Compute weighted mean and disagreement metrics."""
    if not judge_scores:
        return EnsembleScore(
            score=0.0, std=0.0, high_disagreement=False,
            judge_scores=[], weights=weights,
        )

    weighted_sum = sum(
        weights.get(js.judge_id, 0.0) * js.score for js in judge_scores
    )
    weight_total = sum(weights.get(js.judge_id, 0.0) for js in judge_scores)
    mean = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Std dev across raw judge scores (unweighted)
    scores = [js.score for js in judge_scores]
    if len(scores) >= 2:
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
    else:
        std = 0.0

    return EnsembleScore(
        score=mean,
        std=std,
        high_disagreement=std > HIGH_DISAGREEMENT_THRESHOLD,
        judge_scores=judge_scores,
        weights=weights,
    )
