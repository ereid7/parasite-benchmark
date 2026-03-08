"""Compare two PARASITE result files and print deltas.

Loads two results.json files, finds a common model (or first model in each),
and reports PI, category, and effect-size differences.

Usage::

    python3 examples/compare_models.py results/run_a/results.json results/run_b/results.json
"""

import json
import sys

from mbb.v2.scoring import ParasiteV21Result, cohens_d


def compare(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    # Find a common model or take the first from each
    common = set(data_a.keys()) & set(data_b.keys())
    if common:
        model_id = sorted(common)[0]
        result_a = ParasiteV21Result.from_dict(data_a[model_id])
        result_b = ParasiteV21Result.from_dict(data_b[model_id])
    else:
        key_a = next(iter(data_a))
        key_b = next(iter(data_b))
        result_a = ParasiteV21Result.from_dict(data_a[key_a])
        result_b = ParasiteV21Result.from_dict(data_b[key_b])

    print(f"Comparing: {result_a.model_id} vs {result_b.model_id}")
    delta = result_b.pi - result_a.pi
    print(f"  PI:  {result_a.pi:.4f} vs {result_b.pi:.4f}  (delta={delta:+.4f})")
    print(f"  Classification: {result_a.classification} vs {result_b.classification}")
    print()

    # Category deltas
    all_cats = sorted(set(result_a.category_scores) | set(result_b.category_scores))
    print("  Category deltas:")
    for cat in all_cats:
        score_a = result_a.category_scores.get(cat)
        score_b = result_b.category_scores.get(cat)
        sa = score_a.score if score_a else 0.0
        sb = score_b.score if score_b else 0.0
        print(f"    {cat}: {sa:.4f} vs {sb:.4f}  (delta={sb - sa:+.4f})")
    print()

    # Cohen's d effect size on all variant scores
    scores_a = []
    scores_b = []
    for cs in result_a.category_scores.values():
        for ts in cs.test_scores:
            scores_a.extend(ts.variant_scores)
    for cs in result_b.category_scores.values():
        for ts in cs.test_scores:
            scores_b.extend(ts.variant_scores)

    d = cohens_d(scores_a, scores_b)
    print(f"  Cohen's d (effect size): {d:.4f}")
    if abs(d) < 0.2:
        print("    Interpretation: negligible difference")
    elif abs(d) < 0.5:
        print("    Interpretation: small difference")
    elif abs(d) < 0.8:
        print("    Interpretation: medium difference")
    else:
        print("    Interpretation: large difference")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 examples/compare_models.py <results_a.json> <results_b.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
