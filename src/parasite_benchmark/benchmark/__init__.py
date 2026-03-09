"""Core benchmark runtime for PARASITE."""

from typing import Any

__all__ = [
    "discover_tasks",
    "generate_corpus",
    "load_tasks",
    "run_benchmark",
    "validate_task_inventory",
]


def __getattr__(name: str) -> Any:
    if name == "generate_corpus":
        from .generate_tests import generate_corpus

        return generate_corpus
    if name == "run_benchmark":
        from .runner import run_benchmark

        return run_benchmark
    if name == "discover_tasks":
        from .tasks import discover_tasks

        return discover_tasks
    if name == "load_tasks":
        from .tasks import load_tasks

        return load_tasks
    if name == "validate_task_inventory":
        from .tasks import validate_task_inventory

        return validate_task_inventory
    raise AttributeError(name)
