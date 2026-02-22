"""Anthropic model adapter."""
from __future__ import annotations

import json
import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ._base import ModelAdapter


def _import_anthropic():
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Install anthropic support: pip install model-behavior-benchmark[anthropic]"
        ) from None
    return anthropic


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
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
        return resp.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        augmented = list(messages)
        augmented.append({
            "role": "user",
            "content": "Respond with valid JSON only. No markdown, no explanation.",
        })
        text = await self.complete(augmented, temperature, max_tokens)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())
