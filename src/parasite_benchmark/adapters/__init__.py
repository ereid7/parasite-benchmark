"""Model adapters for OpenAI, Anthropic, and local models.

Provides a unified interface for interacting with different LLM providers.
Provider-specific dependencies are lazily imported so that only the SDK
you actually use needs to be installed.

Usage::

    from parasite_benchmark.adapters import create_adapter

    adapter = create_adapter("gpt-4o")
    result = await adapter.complete([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

from typing import Any

from parasite_benchmark.exceptions import ModelAdapterError

from .base import ModelAdapter

__all__ = [
    "ADAPTER_REGISTRY",
    "AnthropicAdapter",
    "LocalAdapter",
    "ModelAdapter",
    "OpenAIAdapter",
    "create_adapter",
]


def _get_openai_adapter() -> type[ModelAdapter]:
    from .openai import OpenAIAdapter

    return OpenAIAdapter


def _get_anthropic_adapter() -> type[ModelAdapter]:
    from .anthropic import AnthropicAdapter

    return AnthropicAdapter


def _get_local_adapter() -> type[ModelAdapter]:
    from .local import LocalAdapter

    return LocalAdapter


ADAPTER_REGISTRY: dict[str, Any] = {
    "openai": _get_openai_adapter,
    "anthropic": _get_anthropic_adapter,
    "local": _get_local_adapter,
}


def _load_entry_point(provider: str) -> type[ModelAdapter] | None:
    """Try to load a model adapter class from ``parasite_benchmark.adapters`` entry points."""
    import importlib.metadata

    eps = importlib.metadata.entry_points()
    # Python 3.12+ returns SelectableGroups; 3.10-3.11 may return a dict
    if hasattr(eps, "select"):
        group = eps.select(group="parasite_benchmark.adapters")
    else:
        group = eps.get("parasite_benchmark.adapters", [])  # type: ignore[arg-type,union-attr]

    for ep in group:
        if ep.name == provider:
            return ep.load()  # type: ignore[no-any-return]
    return None


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
        import os

        # OpenRouter: if OPENROUTER_API_KEY is set, route everything through it
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            provider = "openai"
            kwargs.setdefault("api_key", openrouter_key)
            kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        # Vercel AI Gateway: provider/model-name format
        elif "/" in model_id and not model_id.startswith("http"):
            provider = "openai"  # Vercel uses OpenAI-compatible API
            kwargs.setdefault("api_key", os.environ.get("VERCEL_AI_GATEWAY_KEY"))
            kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
            # model_id stays as-is: "openai/gpt-4o", "anthropic/claude-sonnet-4-6", etc.
        elif "glm" in model_id:
            provider = "openai"
            # Z.AI GLM uses OpenAI-compatible API
            kwargs.setdefault("api_key", os.environ.get("ZAI_API_KEY"))
            kwargs.setdefault("base_url", "https://api.z.ai/api/paas/v4")
        elif "gpt" in model_id or "o1" in model_id or "o3" in model_id:
            provider = "openai"
        elif "claude" in model_id:
            provider = "anthropic"
        else:
            provider = "local"

    factory = ADAPTER_REGISTRY.get(provider)
    if factory is not None:
        adapter_cls: type[ModelAdapter] = factory()
        return adapter_cls(model_id, **kwargs)

    # Fall back to entry_points for third-party adapters
    ep_cls = _load_entry_point(provider)
    if ep_cls is not None:
        return ep_cls(model_id, **kwargs)

    raise ModelAdapterError(f"Unknown provider: {provider!r}")


def __getattr__(name: str) -> type[ModelAdapter]:
    _lazy_map = {
        "OpenAIAdapter": _get_openai_adapter,
        "AnthropicAdapter": _get_anthropic_adapter,
        "LocalAdapter": _get_local_adapter,
    }
    if name in _lazy_map:
        return _lazy_map[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
