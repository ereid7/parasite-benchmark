"""Load and interpret a PARASITE results.json file.

Reconstructs ParasiteResult objects and prints a summary.

Usage::

    python3 examples/interpret_results.py results/<run_id>/results.json
"""

import json
import sys

from parasite_benchmark.benchmark.scoring import ParasiteResult


def interpret(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    for _model_id, model_data in data.items():
        result = ParasiteResult.from_dict(model_data)

        print(f"Model: {result.model_id}")
        print(f"  PI: {result.pi:.4f} ({result.classification})")
        print()

        # Category breakdown
        print("  Categories:")
        for cat, cs in sorted(result.category_scores.items()):
            print(f"    {cat}: {cs.score:.4f} ({len(cs.test_scores)} tests)")
        print()

        # Reliability
        if result.reliability:
            alpha = result.reliability.get("krippendorff_alpha", "N/A")
            print(f"  Krippendorff's Alpha: {alpha}")

        # Gaming
        if result.gaming:
            susceptibility = result.gaming.get("gaming_susceptibility", 0.0)
            flagged = sum(
                1 for v in result.gaming.get("vectors", {}).values() if v.get("flag", False)
            )
            total = len(result.gaming.get("vectors", {}))
            print(
                f"  Gaming: {flagged}/{total} vectors flagged, susceptibility={susceptibility:.4f}"
            )

        # Length bias
        if result.length_bias:
            r = result.length_bias.get("overall_r", 0.0)
            flag = result.length_bias.get("length_confound_flag", False)
            print(f"  Length-score correlation: r={r:.4f} {'(FLAGGED)' if flag else ''}")

        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 examples/interpret_results.py <results.json>")
        sys.exit(1)
    interpret(sys.argv[1])
