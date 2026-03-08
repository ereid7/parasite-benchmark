"""OpenAI model adapter."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from mbb.exceptions import ModelAdapterError
from mbb.utils.json_extraction import extract_json

from ._base import ModelAdapter


def _import_openai() -> Any:
    try:
        import openai
    except ImportError:
        raise ModelAdapterError(
            "Install openai support: pip install model-behavior-benchmark[openai]"
        ) from None
    return openai


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI chat completion models (GPT-4, o1, o3, etc.).

    Also supports OpenAI-compatible APIs like Z.AI GLM.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize the OpenAI adapter.

        Parameters
        ----------
        model_id : str
            Model identifier (e.g. ``"gpt-4o"``, ``"glm-4.7-flash"``).
        **kwargs : Any
            Optional ``api_key``, ``base_url`` overrides.
        """
        super().__init__(model_id, **kwargs)
        openai = _import_openai()
        self._is_glm = "glm" in model_id.lower()
        self._is_vercel = "/" in model_id and not model_id.startswith("http")
        self._is_openrouter = "openrouter" in (kwargs.get("base_url") or "")
        self._client = openai.AsyncOpenAI(
            api_key=kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            base_url=kwargs.get("base_url"),
            # Disable built-in retries for GLM — let tenacity handle backoff
            max_retries=0 if self._is_glm else 2,
        )

    def _extract_content(self, msg: Any) -> str:
        """Extract text content from a chat completion message.

        Falls back to ``reasoning_content`` if the primary content is empty.
        """
        content = msg.content or ""
        if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            content = msg.reasoning_content
        return content

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(min=10, max=120))
    async def _complete_glm(self, **kwargs: Any) -> Any:
        """GLM completion with aggressive retry (10 attempts, 10-120s backoff)."""
        return await self._client.chat.completions.create(**kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _complete_vercel(self, **kwargs: Any) -> Any:
        """Vercel/OpenRouter completion with standard retry (3 attempts)."""
        return await self._client.chat.completions.create(**kwargs)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return a plain-text completion, routing to GLM/Vercel/standard paths."""
        kw = dict(
            model=self.model_id, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        if self._is_glm:
            resp = await self._complete_glm(**kw)
            await asyncio.sleep(10)
        elif self._is_vercel or self._is_openrouter:
            resp = await self._complete_vercel(**kw)
        else:
            resp = await self._client.chat.completions.create(**kw)
        return self._extract_content(resp.choices[0].message)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _complete_json_vercel(self, **kwargs: Any) -> Any:
        return await self._client.chat.completions.create(**kwargs)

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(min=10, max=120))
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Return a parsed JSON dict, using ``response_format`` where supported."""
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},  # stripped for GLM/Vercel below
        }

        if self._is_glm or self._is_vercel or self._is_openrouter:
            kwargs.pop("response_format", None)
        try:
            if self._is_vercel or self._is_openrouter:
                resp = await self._complete_json_vercel(**kwargs)
            else:
                resp = await self._client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            if self._is_vercel or self._is_openrouter:
                resp = await self._complete_json_vercel(**kwargs)
            else:
                resp = await self._client.chat.completions.create(**kwargs)

        text = self._extract_content(resp.choices[0].message)
        if self._is_glm:
            await asyncio.sleep(10)  # GLM rate limit buffer
        text = text or "{}"
        text = extract_json(text)
        return dict(json.loads(text))
