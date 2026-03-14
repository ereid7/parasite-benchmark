"""Optional LLM-assisted repair for malformed JSON-like model output."""

from __future__ import annotations

import os
from typing import Any

from parasite_benchmark.adapters import create_adapter

from .json_extraction import parse_json_object

_REPAIR_SYSTEM = """\
You repair malformed JSON-like output.

Return valid JSON only.
Preserve original keys and values whenever they are present.
Do not explain your work.
Do not add commentary.
If a field is clearly missing, keep it conservative and empty rather than inventing content.
"""


async def repair_json_with_model(
    text: str,
    *,
    model_id: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any] | None:
    """Repair malformed JSON-like text with a secondary model if configured."""
    repair_model = model_id or os.environ.get("JSON_REPAIR_MODEL")
    if not repair_model:
        return None

    adapter = create_adapter(repair_model)
    prompt = (
        "Convert the following malformed JSON-like content into valid JSON.\n"
        "Preserve numbers, strings, arrays, and objects when present.\n"
        "If a field is missing, leave it empty rather than inventing detail.\n\n"
        "Malformed content:\n"
        f"{text}"
    )
    repaired = await adapter.complete(
        [
            {"role": "system", "content": _REPAIR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return parse_json_object(repaired)
