"""Tests for CLI API key validation."""

from __future__ import annotations

from unittest.mock import patch

from mbb.cli import _validate_api_keys


class TestValidateApiKeys:
    """_validate_api_keys checks required env vars for detected providers."""

    def test_openrouter_bypasses(self) -> None:
        """OPENROUTER_API_KEY set => skip all validation."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            # Should not raise even with missing keys
            _validate_api_keys(["gpt-4o", "claude-sonnet-4"], [])

    def test_missing_openai_key_exits(self) -> None:
        """Missing OPENAI_API_KEY for GPT model triggers sys.exit."""
        env = {"OPENAI_API_KEY": "", "OPENROUTER_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            with patch("mbb.cli.sys.exit") as mock_exit:
                _validate_api_keys(["gpt-4o"], [])
                mock_exit.assert_called_once_with(1)

    def test_missing_anthropic_key_exits(self) -> None:
        env = {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            with patch("mbb.cli.sys.exit") as mock_exit:
                _validate_api_keys(["claude-sonnet-4-20250514"], [])
                mock_exit.assert_called_once_with(1)

    def test_unknown_provider_no_key_required(self) -> None:
        """Unknown/local providers don't require specific env vars."""
        env = {"OPENROUTER_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            with patch("mbb.cli.sys.exit") as mock_exit:
                _validate_api_keys(["my-custom-local-model"], [])
                mock_exit.assert_not_called()

    def test_judge_keys_also_validated(self) -> None:
        """Judge model IDs are checked for required keys too."""
        env = {"OPENAI_API_KEY": "ok", "ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            with patch("mbb.cli.sys.exit") as mock_exit:
                _validate_api_keys(["gpt-4o"], ["claude-sonnet-4-20250514"])
                mock_exit.assert_called_once_with(1)
