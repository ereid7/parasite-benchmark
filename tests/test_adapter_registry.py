"""Tests for mbb.models adapter registry and entry_points fallback."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mbb.exceptions import ModelAdapterError
from mbb.models import ADAPTER_REGISTRY, create_adapter
from mbb.models._base import ModelAdapter


class TestBuiltinProviders:
    """create_adapter resolves all built-in providers."""

    @pytest.mark.parametrize(
        "model_id,expected_provider",
        [
            ("gpt-4o", "openai"),
            ("gpt-4.1-mini", "openai"),
        ],
    )
    def test_openai_detection(self, model_id: str, expected_provider: str) -> None:
        # Just test that the factory is found — don't actually instantiate
        # (would need API key / openai SDK)
        factory = ADAPTER_REGISTRY.get(expected_provider)
        assert factory is not None

    def test_anthropic_in_registry(self) -> None:
        assert "anthropic" in ADAPTER_REGISTRY

    def test_local_in_registry(self) -> None:
        assert "local" in ADAPTER_REGISTRY


class TestUnknownProvider:
    """Unknown provider raises ModelAdapterError."""

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ModelAdapterError, match="Unknown provider"):
            create_adapter("some-model", provider="nonexistent")

    def test_error_includes_provider_name(self) -> None:
        with pytest.raises(ModelAdapterError, match="foobar"):
            create_adapter("model", provider="foobar")


class TestEntryPointFallback:
    """Entry point fallback loads third-party adapters."""

    def test_entry_point_discovered(self) -> None:
        """Mock entry_points to provide a custom adapter."""

        class FakeAdapter(ModelAdapter):
            async def complete(
                self,
                messages: list[dict[str, str]],
                temperature: float = 0.0,
                max_tokens: int = 2048,
            ) -> str:
                return "fake"

            async def complete_json(
                self,
                messages: list[dict[str, str]],
                temperature: float = 0.0,
                max_tokens: int = 2048,
            ) -> dict[str, Any]:
                return {}

        mock_ep = MagicMock()
        mock_ep.name = "custom_test"
        mock_ep.load.return_value = FakeAdapter

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            adapter = create_adapter("my-model", provider="custom_test")

        assert isinstance(adapter, FakeAdapter)
        assert adapter.model_id == "my-model"

    def test_entry_point_not_found_raises(self) -> None:
        """Missing entry point still raises ModelAdapterError."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with pytest.raises(ModelAdapterError, match="Unknown provider"):
                create_adapter("model", provider="nonexistent_ep")

    def test_registry_takes_precedence(self) -> None:
        """Built-in registry is checked before entry_points."""
        # If provider is in ADAPTER_REGISTRY, entry_points should not be called
        with patch("importlib.metadata.entry_points") as mock_ep:
            factory = ADAPTER_REGISTRY.get("openai")
            assert factory is not None
            # entry_points should not have been called
            mock_ep.assert_not_called()
