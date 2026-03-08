"""Inter-rater reliability and disagreement analysis for multi-judge evaluation."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from mbb.constants import HIGH_DISAGREEMENT_THRESHOLD, KAPPA_LOW_FLAG, KAPPA_TARGET
from mbb.utils.statistics import safe_mean, safe_std

from .types import VariantObservation

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


def icc_two_way(observations: list[VariantObservation], judge_ids: list[str]) -> float:
    """Compute ICC(2,1) — two-way random, single measures, absolute agreement.

    Standard metric for continuous inter-rater reliability (psychometrics).
    """
    # Build n x k matrix: n subjects (variants), k raters (judges)
    rows: list[list[float]] = []
    for obs in observations:
        if all(jid in obs.judge_scores for jid in judge_ids):
            rows.append([obs.judge_scores[jid] for jid in judge_ids])
    n = len(rows)
    k = len(judge_ids)
    if n < 2 or k < 2:
        return 0.0

    grand_mean = sum(x for row in rows for x in row) / (n * k)

    # Mean squares
    row_means = [sum(row) / k for row in rows]
    col_means = [sum(rows[i][j] for i in range(n)) / n for j in range(k)]

    ss_rows = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    ss_cols = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    ss_total = sum((rows[i][j] - grand_mean) ** 2 for i in range(n) for j in range(k))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1) if n > 1 else 0.0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0.0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return 0.0
    return (ms_rows - ms_error) / denom


def krippendorff_alpha_ordinal(
    observations: list[VariantObservation], judge_ids: list[str]
) -> float:
    """Compute Krippendorff's Alpha using interval difference (c-k)^2.

    Suitable for continuous 0-1 scores where ordinal cumulative-frequency
    weighting collapses to the interval metric.

    Handles missing data: not every judge needs to rate every unit.
    """
    # Build reliability matrix: each unit maps to {judge_id: score}
    # Only include units with at least 2 ratings.
    units: list[dict[str, float]] = []
    for obs in observations:
        coded = {jid: obs.judge_scores[jid] for jid in judge_ids if jid in obs.judge_scores}
        if len(coded) >= 2:
            units.append(coded)

    if not units:
        return 0.0

    # Observed disagreement (D_o)
    # For each unit, sum over all coder pairs the squared difference,
    # weighted by 1/(m_u - 1) where m_u is number of coders for that unit.
    numerator_do = 0.0
    total_pairs = 0.0
    for coded in units:
        values = list(coded.values())
        m_u = len(values)
        if m_u < 2:
            continue
        weight = 1.0 / (m_u - 1)
        for i in range(m_u):
            for j in range(i + 1, m_u):
                numerator_do += weight * (values[i] - values[j]) ** 2
        total_pairs += m_u  # each unit contributes m_u pairable values

    if total_pairs == 0:
        return 0.0

    n_pairable = sum(len(coded) for coded in units)
    d_o = numerator_do / n_pairable if n_pairable > 0 else 0.0

    # Expected disagreement (D_e)
    # Pool all values across all units into one distribution, then compute
    # expected squared difference over all distinct pairs.
    all_values: list[float] = []
    for coded in units:
        all_values.extend(coded.values())

    n_total = len(all_values)
    if n_total < 2:
        return 0.0

    sum_sq_diff = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            sum_sq_diff += (all_values[i] - all_values[j]) ** 2

    d_e = sum_sq_diff / (n_total * (n_total - 1) / 2)

    if d_e == 0.0:
        return 1.0

    return 1.0 - (d_o / d_e)


def mcdonalds_omega(
    observations: list[VariantObservation], judge_ids: list[str]
) -> dict[str, float]:
    """Compute McDonald's omega (lower-bound via Cronbach's alpha formula) per category.

    For each category, treats judges as "items" and variants as "subjects".
    omega = k/(k-1) * (1 - sum(item_variances) / total_variance)
    where k = number of judges.

    Returns a dict mapping category codes to omega values.
    """
    # Group observations by category
    by_category: dict[str, list[VariantObservation]] = {}
    for obs in observations:
        by_category.setdefault(obs.category, []).append(obs)

    result: dict[str, float] = {}

    for category, cat_obs in sorted(by_category.items()):
        # Build matrix: only include observations where all judges rated
        rows: list[list[float]] = []
        for obs in cat_obs:
            if all(jid in obs.judge_scores for jid in judge_ids):
                rows.append([obs.judge_scores[jid] for jid in judge_ids])

        k = len(judge_ids)
        n = len(rows)
        if n < 2 or k < 2:
            result[category] = 0.0
            continue

        # Compute total scores per subject (sum across judges)
        totals = [sum(row) for row in rows]
        total_mean = sum(totals) / n
        total_variance = sum((t - total_mean) ** 2 for t in totals) / (n - 1)

        if total_variance == 0.0:
            result[category] = 0.0
            continue

        # Compute per-judge (item) variance
        item_variances = 0.0
        for j in range(k):
            col = [rows[i][j] for i in range(n)]
            col_mean = sum(col) / n
            item_variances += sum((x - col_mean) ** 2 for x in col) / (n - 1)

        omega = (k / (k - 1)) * (1.0 - item_variances / total_variance)
        result[category] = round(omega, 4)

    return result


def compute_reliability(observations: list[VariantObservation]) -> dict[str, Any]:
    """Compute pairwise kappa, per-test reliability, and disagreement metrics."""
    judge_ids = sorted({jid for obs in observations for jid in obs.judge_scores})
    if len(judge_ids) < 2:
        return {
            "judge_ids": judge_ids,
            "icc": 0.0,
            "pairwise_kappa": {},
            "mean_kappa": 0.0,
            "target_met": False,
            "low_kappa_tests": [],
            "disagreement": {"mean_std": 0.0, "high_disagreement_count": 0},
            "krippendorff_alpha": 0.0,
            "omega_per_category": {},
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
            s_a = [
                o.judge_scores[ja]
                for o in test_obs
                if ja in o.judge_scores and jb in o.judge_scores
            ]
            s_b = [
                o.judge_scores[jb]
                for o in test_obs
                if ja in o.judge_scores and jb in o.judge_scores
            ]
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
        std = safe_std(scores, ddof=1)
        stds.append(std)
        if std > HIGH_DISAGREEMENT_THRESHOLD:
            high_disagreement_count += 1

    mean_std = safe_mean(stds)
    icc = icc_two_way(observations, judge_ids)
    k_alpha = krippendorff_alpha_ordinal(observations, judge_ids)
    omega = mcdonalds_omega(observations, judge_ids)
    return {
        "judge_ids": judge_ids,
        "icc": round(icc, 4),
        "pairwise_kappa": pairwise,
        "mean_kappa": round(mean_kappa, 4),
        "target_met": mean_kappa >= KAPPA_TARGET,
        "low_kappa_tests": sorted(low_kappa_tests, key=lambda x: x["kappa"]),
        "disagreement": {
            "mean_std": round(mean_std, 4),
            "high_disagreement_count": high_disagreement_count,
        },
        "krippendorff_alpha": round(k_alpha, 4),
        "omega_per_category": omega,
    }
