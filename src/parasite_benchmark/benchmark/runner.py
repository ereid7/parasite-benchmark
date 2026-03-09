"""Public benchmark runtime API.

The implementation is split into:

- :mod:`parasite_benchmark.benchmark.evaluator` — per-model evaluation logic
- :mod:`parasite_benchmark.benchmark.orchestrator` — top-level run coordination
"""

from __future__ import annotations

from .evaluator import (
    _estimate_welfare_rates,
    _rubric_from_task,
    evaluate_model,
)
from .orchestrator import _save_results, run_benchmark

__all__ = [
    "_estimate_welfare_rates",
    "_rubric_from_task",
    "_save_results",
    "evaluate_model",
    "run_benchmark",
]
