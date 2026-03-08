"""Ensemble judging — run multiple judge models and aggregate scores."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mbb.constants import HIGH_DISAGREEMENT_THRESHOLD
from mbb.exceptions import ConfigError
from mbb.utils.providers import is_same_family
from mbb.utils.statistics import safe_std

logger = logging.getLogger("mbb")


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
        """Serialize ensemble scores to a JSON-compatible dictionary."""
        return {
            "judges": {js.judge_id: round(js.score, 4) for js in self.judge_scores},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "mean_std": round(self.std, 4),
            "high_disagreement_count": int(self.high_disagreement),
        }


def check_self_enhancement(judge_models: list[str], target_model: str) -> None:
    """Log a warning if any judge model is from the same family as the target."""
    for jm in judge_models:
        if is_same_family(jm, target_model):
            logger.warning(
                "Self-enhancement bias risk: judge '%s' is same model family as target '%s'",
                jm,
                target_model,
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
        raise ConfigError(
            f"Number of weights ({len(weight_values)}) must match "
            f"number of judges ({len(judge_models)})"
        )
    total = sum(weight_values)
    if abs(total - 1.0) > 0.01:
        raise ConfigError(f"Judge weights must sum to 1.0, got {total:.4f}")

    return dict(zip(judge_models, weight_values))


def cyclic_judge_assignment(judge_models: list[str], variant_index: int) -> str:
    """CyclicJudge round-robin: assign judge based on variant index.

    From CyclicJudge (2026): round-robin assignment eliminates systematic
    judge bias (which accounts for >94% of benchmark-level variance) at
    single-judge cost per item.

    For full ensemble mode, all judges evaluate each item.
    For cost-reduced mode, use this to assign one judge per item.
    """
    return judge_models[variant_index % len(judge_models)]


def aggregate_ensemble(
    judge_scores: list[JudgeScore],
    weights: dict[str, float],
) -> EnsembleScore:
    """Compute weighted mean and disagreement metrics."""
    if not judge_scores:
        return EnsembleScore(
            score=0.0,
            std=0.0,
            high_disagreement=False,
            judge_scores=[],
            weights=weights,
        )

    weighted_sum = sum(weights.get(js.judge_id, 0.0) * js.score for js in judge_scores)
    weight_total = sum(weights.get(js.judge_id, 0.0) for js in judge_scores)
    mean = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Std dev across raw judge scores (unweighted)
    scores = [js.score for js in judge_scores]
    std = safe_std(scores, ddof=1)

    return EnsembleScore(
        score=mean,
        std=std,
        high_disagreement=std > HIGH_DISAGREEMENT_THRESHOLD,
        judge_scores=judge_scores,
        weights=weights,
    )
