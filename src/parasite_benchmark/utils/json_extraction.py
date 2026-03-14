"""JSON extraction from LLM responses — handles raw JSON, markdown blocks, and embedded JSON."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency
    repair_json = None


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


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse an LLM response into a JSON object, repairing malformed JSON when possible."""
    raw = text.strip()
    candidates: list[str] = []
    if raw:
        candidates.append(raw)

    extracted = extract_json(raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            sliced = raw[start : end + 1]
            if sliced not in candidates:
                candidates.append(sliced)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return dict(json.loads(candidate))
        except Exception as exc:
            last_error = exc
        if repair_json is None:
            continue
        try:
            repaired = repair_json(candidate, return_objects=True)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(repaired, dict):
            return repaired

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", raw, 0)
