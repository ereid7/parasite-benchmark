"""Task discovery, loading, and validation for MBB YAML test definitions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("mbb")

# data/ lives at the repo root, two levels above src/mbb/tasks/
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

CATEGORY_MAP = {
    "sycophancy": "A",
    "dependency": "B",
    "manipulation": "E",
    "framing": "F",
    "multi_turn": "G",
    "economic": "H",
}


def discover_tasks() -> list[dict[str, Any]]:
    """Discover all task YAML files and return summary info."""
    tasks: list[dict[str, Any]] = []
    for yaml_file in sorted(DATA_DIR.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "id" in data:
                tasks.append({
                    "id": data["id"],
                    "category": data.get("category", ""),
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "n_variants": len(data.get("variants", [])),
                    "version": data.get("version", "1.0"),
                    "file": str(yaml_file),
                })
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_file, e)
    return tasks


def load_task(path: str | Path) -> dict[str, Any]:
    """Load a single task YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_tasks(task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load all tasks (or a subset by ID) with full content."""
    tasks: list[dict[str, Any]] = []
    for yaml_file in sorted(DATA_DIR.rglob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "id" in data:
                if task_ids and data["id"] not in task_ids:
                    continue
                tasks.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_file, e)
    return tasks
