"""Task loading for PARASITE v2.1 test corpus."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

from mbb.constants import CATEGORIES
from mbb.exceptions import TaskLoadError

from .spec import EXPECTED_COUNTS

logger = logging.getLogger("mbb")

DATA_V21_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "v2.1"

_VALID_TYPES = {"single_turn", "multi_turn"}


def _check_difficulty(value: Any, label: str) -> str | None:
    """Return an error string if difficulty is invalid, else None."""
    if value is None:
        return None
    try:
        d = float(value)
        if not (0.0 <= d <= 1.0):
            return f"{label} difficulty {d} outside [0, 1]"
    except (TypeError, ValueError):
        return f"{label} difficulty {value!r} is not a number"
    return None


def _validate_variants(data: dict[str, Any]) -> list[str]:
    """Validate the variants list. Returns a list of error strings."""
    errors: list[str] = []
    if "variants" not in data:
        return errors
    if not data["variants"]:
        errors.append("variants list is empty")
        return errors
    for i, v in enumerate(data["variants"]):
        if "id" not in v:
            errors.append(f"variant[{i}] missing 'id'")
        if "prompt" not in v and "turns" not in v:
            errors.append(f"variant[{i}] missing both 'prompt' and 'turns'")
        err = _check_difficulty(
            v.get("metadata", {}).get("difficulty"),
            f"variant {v.get('id', '?')}",
        )
        if err:
            errors.append(err)
    return errors


def _validate_task_dict(data: dict[str, Any]) -> None:
    """Validate a task dict against the v2.1 schema. Raises TaskLoadError on failure."""
    errors: list[str] = []

    for req in ("id", "name", "category", "type", "variants"):
        if req not in data:
            errors.append(f"missing required field: {req}")

    if "category" in data and data["category"] not in CATEGORIES:
        errors.append(f"invalid category: {data['category']!r} (valid: {CATEGORIES})")

    if "type" in data and data["type"] not in _VALID_TYPES:
        errors.append(f"invalid type: {data['type']!r} (valid: {_VALID_TYPES})")

    errors.extend(_validate_variants(data))

    err = _check_difficulty(data.get("metadata", {}).get("difficulty"), "task")
    if err:
        errors.append(err)

    if errors:
        raise TaskLoadError(f"Task {data.get('id', '?')} validation failed: {'; '.join(errors)}")


def check_prompt_collisions(tasks: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Find variant pairs with identical prompt text (SHA256 collision)."""
    hashes: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for task in tasks:
        for v in task.get("variants", []):
            prompt = v.get("prompt")
            if prompt is None:
                continue
            h = hashlib.sha256(prompt.encode()).hexdigest()
            vid = v.get("id", f"{task.get('id', '?')}_v?")
            if h in hashes:
                collisions.append((hashes[h], vid))
            else:
                hashes[h] = vid
    return collisions


def discover_tasks_v21() -> list[dict[str, Any]]:
    """Discover all v2.1 tasks by scanning ``data/v2.1/`` for YAML files.

    Returns lightweight task summaries (id, name, category, n_variants)
    without loading full variant data.
    """
    tasks: list[dict[str, Any]] = []
    for yaml_file in sorted(DATA_V21_DIR.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "id" in data:
                tasks.append(
                    {
                        "id": data["id"],
                        "name": data.get("name", ""),
                        "category": data.get("category", ""),
                        "description": data.get("description", ""),
                        "n_variants": len(data.get("variants", [])),
                        "version": data.get("version", "2.1"),
                        "file": str(yaml_file),
                    }
                )
        except Exception as exc:
            logger.warning("Failed to load %s: %s", yaml_file, exc)
    return tasks


def load_all_tasks_v21(task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load full task data (including variants) from ``data/v2.1/`` YAML files.

    Parameters
    ----------
    task_ids : list[str] | None
        Optional filter. If provided, only tasks with matching IDs are loaded.

    Returns
    -------
    list[dict[str, Any]]
        Full task dicts ready for evaluation.
    """
    tasks: list[dict[str, Any]] = []
    for yaml_file in sorted(DATA_V21_DIR.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data or "id" not in data:
                continue
            if task_ids and data["id"] not in task_ids:
                continue
            _validate_task_dict(data)
            tasks.append(data)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", yaml_file, exc)
    return tasks


def validate_task_inventory_v21(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the task inventory against expected category counts.

    Checks that each category has the expected number of tasks and that
    every task has at least 10 standard variants.

    Returns
    -------
    dict[str, Any]
        Validation report with counts, mismatches, and a minimum-variants flag.
    """
    counts = {cat: 0 for cat in EXPECTED_COUNTS}
    min_variants_ok = True
    for task in tasks:
        cat = task.get("category", "")
        if cat in counts:
            counts[cat] += 1
        n_standard = len([v for v in task.get("variants", []) if v.get("type") == "standard"])
        if n_standard < 10:
            min_variants_ok = False

    missing = {
        cat: {"expected": expected, "found": counts.get(cat, 0)}
        for cat, expected in EXPECTED_COUNTS.items()
        if counts.get(cat, 0) != expected
    }
    collisions = check_prompt_collisions(tasks)
    return {
        "counts": counts,
        "expected": EXPECTED_COUNTS,
        "missing_or_mismatched": missing,
        "min_standard_variants_ok": min_variants_ok,
        "total_tests": len(tasks),
        "duplicate_prompts": collisions,
    }
