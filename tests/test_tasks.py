"""Tests for v2/tasks.py — discover, load, validate (uses real YAML files)."""

from __future__ import annotations

from mbb.constants import CATEGORIES
from mbb.v2.tasks import discover_tasks_v21, load_all_tasks_v21, validate_task_inventory_v21


def test_discover_count():
    tasks = discover_tasks_v21()
    assert len(tasks) == 68


def test_all_categories_present():
    tasks = discover_tasks_v21()
    found_cats = {t["category"] for t in tasks}
    for cat in CATEGORIES:
        assert cat in found_cats, f"Category {cat} missing from tasks"


def test_load_all_tasks():
    tasks = load_all_tasks_v21()
    assert len(tasks) == 68
    assert all("id" in t for t in tasks)


def test_required_fields():
    tasks = load_all_tasks_v21()
    for task in tasks:
        assert "id" in task
        assert "category" in task
        assert "variants" in task or "turns" in task


def test_validate_inventory():
    tasks = load_all_tasks_v21()
    report = validate_task_inventory_v21(tasks)
    assert report["total_tests"] == 68
    assert report["missing_or_mismatched"] == {}


def test_variant_has_prompt_or_turns():
    tasks = load_all_tasks_v21()
    for task in tasks:
        if "variants" in task:
            for v in task["variants"]:
                has_prompt = "prompt" in v
                has_turns = "turns" in v or "turns" in task
                vid = v.get("id")
                assert has_prompt or has_turns, f"Variant {vid} missing prompt/turns"


def test_unique_ids():
    tasks = discover_tasks_v21()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "Duplicate task IDs found"


def test_load_filtered():
    tasks = load_all_tasks_v21(task_ids=["A1"])
    assert len(tasks) == 1
    assert tasks[0]["id"] == "A1"
