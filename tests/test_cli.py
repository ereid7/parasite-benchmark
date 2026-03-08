"""Tests for cli.py — Click commands via CliRunner."""

from __future__ import annotations

from click.testing import CliRunner

from mbb.cli import main


def test_list_tasks():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "tasks"])
    assert result.exit_code == 0
    assert "A1" in result.output or "PARASITE" in result.output


def test_list_tasks_category_filter():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "tasks", "-c", "A"])
    assert result.exit_code == 0


def test_estimate_command():
    runner = CliRunner()
    result = runner.invoke(main, ["estimate", "-m", "gpt-4o"])
    assert result.exit_code == 0
    assert "Cost" in result.output or "Estimation" in result.output


def test_missing_model_error():
    runner = CliRunner()
    result = runner.invoke(main, ["run"])
    assert result.exit_code != 0


def test_help_output():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "PARASITE" in result.output


def test_self_judging_detection():
    """Attempting to judge with same-provider model should error."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "-m", "gpt-4o", "-j", "gpt-4.1-mini"])
    assert result.exit_code != 0


def test_provider_detection_in_run():
    """Different provider judge should not trigger self-judging error."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "-m", "gpt-4o", "-j", "claude-sonnet-4"])
    # Should not exit with self-judging error (exit code 1 with specific message)
    # May fail for other reasons (no API key, etc.) but not for self-judging
    if result.exit_code != 0:
        assert "Self-judging" not in result.output


def test_invalid_list_choice():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "invalid"])
    assert result.exit_code != 0
