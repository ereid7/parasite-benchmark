"""PARASITE v2.1 benchmark runner — backward-compatible re-exports.

The implementation has been split into:

- :mod:`mbb.v2.evaluator` — per-model evaluation logic
- :mod:`mbb.v2.orchestrator` — top-level run coordination

This module re-exports the public API so existing imports continue to work.
"""

from __future__ import annotations

from .evaluator import (
    _estimate_welfare_rates,
    _rubric_from_task,
    evaluate_model_v21,
)
from .orchestrator import _save_results, run_benchmark_v21

__all__ = [
    "_estimate_welfare_rates",
    "_rubric_from_task",
    "_save_results",
    "evaluate_model_v21",
    "run_benchmark_v21",
]
