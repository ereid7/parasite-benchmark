"""Model adapters for OpenAI, Anthropic, and local models.

Provides a unified interface for interacting with different LLM providers.
Provider-specific dependencies are lazily imported so that only the SDK
you actually use needs to be installed.

Usage::

    from mbb.models import create_adapter

    adapter = create_adapter("gpt-4o")
    result = await adapter.complete([{"role": "user", "content": "Hello"}])
"""
from __future__ import annotations

from typing import Any

from ._base import ModelAdapter

__all__ = [
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LocalAdapter",
    "ADAPTER_REGISTRY",
    "create_adapter",
]


def _get_openai_adapter():
    from .openai import OpenAIAdapter
    return OpenAIAdapter


def _get_anthropic_adapter():
    from .anthropic import AnthropicAdapter
    return AnthropicAdapter


def _get_local_adapter():
    from .local import LocalAdapter
    return LocalAdapter


ADAPTER_REGISTRY: dict[str, Any] = {
    "openai": _get_openai_adapter,
    "anthropic": _get_anthropic_adapter,
    "local": _get_local_adapter,
}


def create_adapter(
    model_id: str,
    provider: str | None = None,
    **kwargs: Any,
) -> ModelAdapter:
    """Create and return a :class:`ModelAdapter` for the given *model_id*.

    Parameters
    ----------
    model_id:
        The model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
    provider:
        Explicit provider name.  When *None*, the provider is auto-detected
        from the *model_id* string.
    **kwargs:
        Extra keyword arguments forwarded to the adapter constructor
        (``api_key``, ``base_url``, etc.).
    """
    if provider is None:
        if "glm" in model_id:
            provider = "openai"
            # Z.AI GLM uses OpenAI-compatible API
            import os
            kwargs.setdefault("api_key", os.environ.get("ZAI_API_KEY"))
            kwargs.setdefault("base_url", "https://api.z.ai/api/paas/v4")
        elif "gpt" in model_id or "o1" in model_id or "o3" in model_id:
            provider = "openai"
        elif "claude" in model_id:
            provider = "anthropic"
        else:
            provider = "local"

    factory = ADAPTER_REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Unknown provider: {provider!r}")

    adapter_cls = factory()
    return adapter_cls(model_id, **kwargs)


def __getattr__(name: str):
    _lazy_map = {
        "OpenAIAdapter": _get_openai_adapter,
        "AnthropicAdapter": _get_anthropic_adapter,
        "LocalAdapter": _get_local_adapter,
    }
    if name in _lazy_map:
        return _lazy_map[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
