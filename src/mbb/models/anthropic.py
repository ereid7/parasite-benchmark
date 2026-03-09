"""Anthropic model adapter."""

from __future__ import annotations

import json
import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from mbb.exceptions import ModelAdapterError
from mbb.utils.json_extraction import extract_json

from ._base import ModelAdapter


def _import_anthropic() -> Any:
    try:
        import anthropic
    except ImportError:
        raise ModelAdapterError(
            "Install anthropic support: pip install model-behavior-benchmark[anthropic]"
        ) from None
    return anthropic


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize the Anthropic adapter.

        Parameters
        ----------
        model_id : str
            Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        **kwargs : Any
            Optional ``api_key`` override.
        """
        super().__init__(model_id, **kwargs)
        anthropic = _import_anthropic()
        self._client = anthropic.AsyncAnthropic(
            api_key=kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return a plain-text completion via the Anthropic Messages API."""
        system = ""
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        resp = await self._client.messages.create(
            model=self.model_id,
            system=system,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(resp.content[0].text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Return a parsed JSON dict by appending a JSON-only instruction."""
        augmented = [dict(m) for m in messages]
        # Merge JSON instruction into the last user message to avoid
        # consecutive user turns (Anthropic API requires alternating roles).
        for i in range(len(augmented) - 1, -1, -1):
            if augmented[i]["role"] == "user":
                augmented[i]["content"] += (
                    "\n\nRespond with valid JSON only. No markdown, no explanation."
                )
                break
        else:
            augmented.append(
                {
                    "role": "user",
                    "content": "Respond with valid JSON only. No markdown, no explanation.",
                }
            )
        text = await self.complete(augmented, temperature, max_tokens)
        return dict(json.loads(extract_json(text)))
