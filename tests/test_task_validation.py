"""Tests for task schema validation (H2) and prompt canonicality (H6)."""

from __future__ import annotations

import pytest

from parasite_benchmark.benchmark.tasks import _validate_task_dict, check_prompt_collisions
from parasite_benchmark.exceptions import TaskLoadError


class TestValidateTaskDict:
    """H2: Schema validation on YAML load."""

    def test_valid_task(self, sample_task):
        """Valid task dict passes without error."""
        _validate_task_dict(sample_task)

    def test_missing_required_fields(self):
        """Missing id, name, category, type, variants → TaskLoadError."""
        with pytest.raises(TaskLoadError, match="missing required field"):
            _validate_task_dict({"id": "X1"})

    def test_invalid_category(self):
        with pytest.raises(TaskLoadError, match="invalid category"):
            _validate_task_dict(
                {
                    "id": "Z1",
                    "name": "Bad",
                    "category": "Z",
                    "type": "single_turn",
                    "variants": [{"id": "Z1_v1", "prompt": "hi"}],
                }
            )

    def test_invalid_type(self):
        with pytest.raises(TaskLoadError, match="invalid type"):
            _validate_task_dict(
                {
                    "id": "A1",
                    "name": "Bad",
                    "category": "A",
                    "type": "triple_turn",
                    "variants": [{"id": "A1_v1", "prompt": "hi"}],
                }
            )

    def test_empty_variants(self):
        with pytest.raises(TaskLoadError, match="variants list is empty"):
            _validate_task_dict(
                {
                    "id": "A1",
                    "name": "Bad",
                    "category": "A",
                    "type": "single_turn",
                    "variants": [],
                }
            )

    def test_variant_missing_prompt_and_turns(self):
        with pytest.raises(TaskLoadError, match="missing both"):
            _validate_task_dict(
                {
                    "id": "A1",
                    "name": "Bad",
                    "category": "A",
                    "type": "single_turn",
                    "variants": [{"id": "A1_v1"}],
                }
            )

    def test_difficulty_outside_range(self):
        with pytest.raises(TaskLoadError, match="difficulty"):
            _validate_task_dict(
                {
                    "id": "A1",
                    "name": "Bad",
                    "category": "A",
                    "type": "single_turn",
                    "variants": [
                        {
                            "id": "A1_v1",
                            "prompt": "hi",
                            "metadata": {"difficulty": 1.5},
                        }
                    ],
                }
            )


class TestPromptCollisions:
    """H6: Prompt canonicality check."""

    def test_no_collisions(self):
        tasks = [
            {
                "id": "A1",
                "variants": [
                    {"id": "A1_v1", "prompt": "hello"},
                    {"id": "A1_v2", "prompt": "world"},
                ],
            }
        ]
        assert check_prompt_collisions(tasks) == []

    def test_collision_detected(self):
        tasks = [
            {
                "id": "A1",
                "variants": [
                    {"id": "A1_v1", "prompt": "identical prompt"},
                    {"id": "A1_v2", "prompt": "identical prompt"},
                ],
            }
        ]
        collisions = check_prompt_collisions(tasks)
        assert len(collisions) == 1
        assert collisions[0] == ("A1_v1", "A1_v2")

    def test_collision_across_tasks(self):
        tasks = [
            {"id": "A1", "variants": [{"id": "A1_v1", "prompt": "same text"}]},
            {"id": "B1", "variants": [{"id": "B1_v1", "prompt": "same text"}]},
        ]
        collisions = check_prompt_collisions(tasks)
        assert len(collisions) == 1
