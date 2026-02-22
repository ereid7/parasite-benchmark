"""Local model adapter using an OpenAI-compatible endpoint."""
from __future__ import annotations

import json
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ._base import ModelAdapter


def _import_openai():
    try:
        import openai
    except ImportError:
        raise ImportError(
            "Install openai support: pip install model-behavior-benchmark[openai]"
        ) from None
    return openai


class LocalAdapter(ModelAdapter):
    """Adapter for local / self-hosted models via an OpenAI-compatible API.

    Defaults to ``http://localhost:11434/v1`` (Ollama).
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        openai = _import_openai()
        base_url = kwargs.get("base_url", "http://localhost:11434/v1")
        self._client = openai.AsyncOpenAI(
            api_key=kwargs.get("api_key", "local"),
            base_url=base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        augmented = list(messages)
        augmented.append({
            "role": "user",
            "content": "Respond with valid JSON only.",
        })
        text = await self.complete(augmented, temperature, max_tokens)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())
