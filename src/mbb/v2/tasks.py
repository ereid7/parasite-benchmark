"""Task loading for PARASITE v2.1 test corpus."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .spec import EXPECTED_COUNTS

logger = logging.getLogger("mbb")

DATA_V21_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "v2.1"


def discover_tasks_v21() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for yaml_file in sorted(DATA_V21_DIR.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "id" in data:
                tasks.append({
                    "id": data["id"],
                    "name": data.get("name", ""),
                    "category": data.get("category", ""),
                    "description": data.get("description", ""),
                    "n_variants": len(data.get("variants", [])),
                    "version": data.get("version", "2.1"),
                    "file": str(yaml_file),
                })
        except Exception as exc:
            logger.warning("Failed to load %s: %s", yaml_file, exc)
    return tasks


def load_all_tasks_v21(task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for yaml_file in sorted(DATA_V21_DIR.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data or "id" not in data:
                continue
            if task_ids and data["id"] not in task_ids:
                continue
            tasks.append(data)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", yaml_file, exc)
    return tasks


def validate_task_inventory_v21(tasks: list[dict[str, Any]]) -> dict[str, Any]:
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
    return {
        "counts": counts,
        "expected": EXPECTED_COUNTS,
        "missing_or_mismatched": missing,
        "min_standard_variants_ok": min_variants_ok,
        "total_tests": len(tasks),
    }

