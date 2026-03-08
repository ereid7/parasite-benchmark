"""Provider detection utilities — single source of truth for model-to-provider mapping."""

from __future__ import annotations

import re

# (keyword_or_prefix, provider_name)
_PROVIDER_KEYWORDS: list[tuple[str, str]] = [
    ("gpt", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("glm", "zhipu"),
    ("mistral", "mistral"),
    ("llama", "meta"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
]

# Model family root patterns — maps regex to canonical family name.
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"gpt-4", "gpt-4"),
    (r"gpt-3", "gpt-3"),
    (r"o[134]-", "o-series"),
    (r"claude", "claude"),
    (r"gemini", "gemini"),
    (r"glm", "glm"),
    (r"mistral", "mistral"),
    (r"deepseek", "deepseek"),
    (r"llama|meta", "llama"),
    (r"qwen", "qwen"),
]


def detect_provider(model_id: str) -> str:
    """Detect the provider for a model ID.

    Handles explicit prefix format (``openai/gpt-4o``) and keyword detection.
    """
    # Explicit prefix: "openai/gpt-4o" -> "openai"
    if "/" in model_id and not model_id.startswith("http"):
        return model_id.split("/")[0].lower()

    mid = model_id.lower()
    for keyword, provider in _PROVIDER_KEYWORDS:
        if keyword in mid:
            return provider

    return mid


def detect_model_family(model_id: str) -> str:
    """Extract the model family root (e.g. 'gpt-4', 'claude', 'gemini')."""
    mid = model_id.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, mid):
            return family
    # Fallback: use text before first slash or the whole id
    return mid.split("/", 1)[0]


def is_same_provider(model_a: str, model_b: str) -> bool:
    """Check if two model IDs belong to the same provider."""
    return detect_provider(model_a) == detect_provider(model_b)


def is_same_family(model_a: str, model_b: str) -> bool:
    """Check if two model IDs belong to the same model family."""
    return detect_model_family(model_a) == detect_model_family(model_b)
