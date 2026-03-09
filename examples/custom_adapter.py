"""Skeleton for a custom ModelAdapter.

Shows how to subclass ModelAdapter and register it in the adapter registry.
This example uses a dummy implementation that returns fixed responses.

Usage::

    python3 examples/custom_adapter.py
"""

from typing import Any

from parasite_benchmark.adapters import ADAPTER_REGISTRY
from parasite_benchmark.adapters.base import ModelAdapter


class MyCustomAdapter(ModelAdapter):
    """Example adapter that returns a fixed response."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        # Initialize your client here, e.g.:
        # self._client = MyClient(api_key=kwargs.get("api_key"))

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return a plain-text completion."""
        # Replace with your actual API call:
        return "This is a placeholder response from MyCustomAdapter."

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Return a parsed JSON dict."""
        # Replace with your actual API call:
        return {"score": 0.5, "reasoning": "placeholder"}


def _get_custom_adapter() -> type[ModelAdapter]:
    return MyCustomAdapter


# Register the adapter so create_adapter() can find it
ADAPTER_REGISTRY["custom"] = _get_custom_adapter

if __name__ == "__main__":
    from parasite_benchmark.adapters import create_adapter

    adapter = create_adapter("my-model", provider="custom")
    print(f"Created adapter: {type(adapter).__name__} for {adapter.model_id}")
