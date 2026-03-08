"""JSON extraction from LLM responses — handles raw JSON, markdown blocks, and embedded JSON."""

from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Extract a JSON object string from text that may contain non-JSON content.

    Handles:
    - Raw JSON (text starts with ``{``)
    - Markdown code blocks (````json ... ``` ``)
    - JSON embedded in prose
    """
    text = text.strip()
    if text.startswith("{"):
        return text
    # Markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        return md_match.group(1)
    # Embedded JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text
