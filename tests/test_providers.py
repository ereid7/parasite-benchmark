"""Tests for utils/providers.py — detect_provider, family, same_provider."""

from __future__ import annotations

from parasite_benchmark.utils.providers import (
    detect_model_family,
    detect_provider,
    is_same_family,
    is_same_provider,
)


def test_detect_provider_openai():
    assert detect_provider("gpt-4o") == "openai"


def test_detect_provider_anthropic():
    assert detect_provider("claude-sonnet-4-20250514") == "anthropic"


def test_detect_provider_google():
    assert detect_provider("gemini-2.0-flash") == "google"


def test_detect_provider_zhipu():
    assert detect_provider("glm-4.7-flash") == "zhipu"


def test_detect_provider_deepseek():
    assert detect_provider("deepseek-v3") == "deepseek"


def test_detect_provider_unknown():
    result = detect_provider("some-random-model")
    assert result == "some-random-model"


def test_detect_provider_explicit_prefix():
    assert detect_provider("openai/gpt-4o") == "openai"
    assert detect_provider("mistral/mistral-large-latest") == "mistral"


def test_detect_model_family():
    assert detect_model_family("gpt-4o") == "gpt-4"
    assert detect_model_family("claude-sonnet-4") == "claude"
    assert detect_model_family("gemini-2.0-flash") == "gemini"


def test_is_same_provider_true():
    assert is_same_provider("gpt-4o", "gpt-4.1-mini")


def test_is_same_provider_false():
    assert not is_same_provider("gpt-4o", "claude-sonnet-4")


def test_is_same_family_true():
    assert is_same_family("gpt-4o", "gpt-4-turbo")


def test_is_same_family_false():
    assert not is_same_family("gpt-4o", "claude-sonnet-4")
