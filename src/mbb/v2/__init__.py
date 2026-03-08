"""PARASITE v2.1 implementation."""

from typing import Any

__all__ = [
    "discover_tasks_v21",
    "generate_corpus",
    "load_all_tasks_v21",
    "run_benchmark_v21",
    "validate_task_inventory_v21",
]


def __getattr__(name: str) -> Any:
    if name == "generate_corpus":
        from .generate_tests import generate_corpus

        return generate_corpus
    if name == "run_benchmark_v21":
        from .runner import run_benchmark_v21

        return run_benchmark_v21
    if name == "discover_tasks_v21":
        from .tasks import discover_tasks_v21

        return discover_tasks_v21
    if name == "load_all_tasks_v21":
        from .tasks import load_all_tasks_v21

        return load_all_tasks_v21
    if name == "validate_task_inventory_v21":
        from .tasks import validate_task_inventory_v21

        return validate_task_inventory_v21
    raise AttributeError(name)
