"""OpenAI model adapter."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ._base import ModelAdapter


def _import_openai():
    try:
        import openai
    except ImportError:
        raise ImportError(
            "Install openai support: pip install model-behavior-benchmark[openai]"
        ) from None
    return openai


def _extract_json(text: str) -> str:
    """Extract JSON object from text that may contain non-JSON content."""
    text = text.strip()
    if text.startswith("{"):
        return text
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        return md_match.group(1)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI chat completion models (GPT-4, o1, o3, etc.).

    Also supports OpenAI-compatible APIs like Z.AI GLM.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        openai = _import_openai()
        self._client = openai.AsyncOpenAI(
            api_key=kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            base_url=kwargs.get("base_url"),
        )
        self._is_glm = "glm" in model_id.lower()

    def _extract_content(self, msg: Any) -> str:
        content = msg.content or ""
        if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            content = msg.reasoning_content
        return content

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(min=2, max=120))
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
        return self._extract_content(resp.choices[0].message)

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(min=2, max=120))
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not self._is_glm:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = await self._client.chat.completions.create(**kwargs)

        text = self._extract_content(resp.choices[0].message)
        text = text or "{}"
        text = _extract_json(text)
        return json.loads(text)
