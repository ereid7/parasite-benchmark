"""Tests for models/_base.py — ABC enforcement."""

from __future__ import annotations

from typing import Any

import pytest

from mbb.models._base import ModelAdapter


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        ModelAdapter(model_id="test")


def test_complete_is_abstract():
    assert hasattr(ModelAdapter, "complete")


def test_complete_json_is_abstract():
    assert hasattr(ModelAdapter, "complete_json")


def test_concrete_subclass_stores_model_id():
    class ConcreteAdapter(ModelAdapter):
        async def complete(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.0,
            max_tokens: int = 2048,
        ) -> str:
            return "response"

        async def complete_json(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.0,
            max_tokens: int = 2048,
        ) -> dict[str, Any]:
            return {}

    adapter = ConcreteAdapter(model_id="test-model")
    assert adapter.model_id == "test-model"
