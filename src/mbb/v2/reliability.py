"""Inter-rater reliability and disagreement analysis for multi-judge evaluation."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .types import VariantObservation

KAPPA_TARGET = 0.7
KAPPA_LOW_FLAG = 0.5

ANCHOR_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]


def _bucket(score: float) -> int:
    """Map a continuous score to nearest rubric anchor index."""
    distances = [abs(score - anchor) for anchor in ANCHOR_VALUES]
    return int(min(range(len(distances)), key=lambda i: distances[i]))


def cohen_kappa(scores_a: list[float], scores_b: list[float], n_labels: int = 5) -> float:
    """Compute Cohen's kappa for two raters."""
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return 0.0
    n = len(scores_a)
    if n == 0:
        return 0.0

    # Confusion matrix
    conf = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    for sa, sb in zip(scores_a, scores_b):
        ia = _bucket(sa)
        ib = _bucket(sb)
        conf[ia][ib] += 1

    po = sum(conf[i][i] for i in range(n_labels)) / n

    row_marginals = [sum(conf[i][j] for j in range(n_labels)) / n for i in range(n_labels)]
    col_marginals = [sum(conf[i][j] for i in range(n_labels)) / n for j in range(n_labels)]
    pe = sum(r * c for r, c in zip(row_marginals, col_marginals))
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def compute_reliability(observations: list[VariantObservation]) -> dict[str, Any]:
    """Compute pairwise kappa, per-test reliability, and disagreement metrics."""
    judge_ids = sorted({
        jid
        for obs in observations
        for jid in obs.judge_scores
    })
    if len(judge_ids) < 2:
        return {
            "judge_ids": judge_ids,
            "pairwise_kappa": {},
            "mean_kappa": 0.0,
            "target_met": False,
            "low_kappa_tests": [],
            "disagreement": {"mean_std": 0.0, "high_disagreement_count": 0},
        }

    # Pairwise global kappa
    pairwise: dict[str, float] = {}
    kappas: list[float] = []
    for ja, jb in combinations(judge_ids, 2):
        s_a: list[float] = []
        s_b: list[float] = []
        for obs in observations:
            if ja in obs.judge_scores and jb in obs.judge_scores:
                s_a.append(obs.judge_scores[ja])
                s_b.append(obs.judge_scores[jb])
        kappa = cohen_kappa(s_a, s_b)
        pair_key = f"{ja}__{jb}"
        pairwise[pair_key] = round(kappa, 4)
        kappas.append(kappa)

    mean_kappa = sum(kappas) / len(kappas) if kappas else 0.0

    # Per-test reliability to flag weak tests
    by_test: dict[str, list[VariantObservation]] = {}
    for obs in observations:
        by_test.setdefault(obs.test_id, []).append(obs)

    low_kappa_tests: list[dict[str, Any]] = []
    for test_id, test_obs in by_test.items():
        test_kappas: list[float] = []
        for ja, jb in combinations(judge_ids, 2):
            s_a = [o.judge_scores[ja] for o in test_obs if ja in o.judge_scores and jb in o.judge_scores]
            s_b = [o.judge_scores[jb] for o in test_obs if ja in o.judge_scores and jb in o.judge_scores]
            if len(s_a) >= 2:
                test_kappas.append(cohen_kappa(s_a, s_b))
        if not test_kappas:
            continue
        mk = sum(test_kappas) / len(test_kappas)
        if mk < KAPPA_LOW_FLAG:
            low_kappa_tests.append({"test_id": test_id, "kappa": round(mk, 4)})

    # Disagreement stats based on per-variant judge std
    stds: list[float] = []
    high_disagreement_count = 0
    for obs in observations:
        scores = list(obs.judge_scores.values())
        if len(scores) < 2:
            continue
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        stds.append(std)
        if std > 0.25:
            high_disagreement_count += 1

    mean_std = sum(stds) / len(stds) if stds else 0.0
    return {
        "judge_ids": judge_ids,
        "pairwise_kappa": pairwise,
        "mean_kappa": round(mean_kappa, 4),
        "target_met": mean_kappa >= KAPPA_TARGET,
        "low_kappa_tests": sorted(low_kappa_tests, key=lambda x: x["kappa"]),
        "disagreement": {
            "mean_std": round(mean_std, 4),
            "high_disagreement_count": high_disagreement_count,
        },
    }

