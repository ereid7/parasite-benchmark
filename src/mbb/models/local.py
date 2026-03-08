"""Local model adapter using an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
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


class LocalAdapter(ModelAdapter):
    """Adapter for local / self-hosted models via an OpenAI-compatible API.

    Defaults to ``http://localhost:11434/v1`` (Ollama).
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize the local adapter.

        Parameters
        ----------
        model_id : str
            Model identifier passed to the local server.
        **kwargs : Any
            Optional ``api_key`` and ``base_url`` (defaults to Ollama at
            ``http://localhost:11434/v1``).
        """
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
        """Return a plain-text completion from the local OpenAI-compatible server."""
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
        """Return a parsed JSON dict by appending a JSON-only instruction."""
        augmented = list(messages)
        augmented.append(
            {
                "role": "user",
                "content": "Respond with valid JSON only.",
            }
        )
        text = await self.complete(augmented, temperature, max_tokens)
        return dict(json.loads(extract_json(text)))
