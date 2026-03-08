"""Tests for mbb.exceptions — hierarchy, inheritance, catch-all."""

from __future__ import annotations

import pytest

from mbb.exceptions import (
    ConfigError,
    JudgeError,
    ModelAdapterError,
    ParasiteError,
    ScoringError,
    SelfJudgingError,
    TaskLoadError,
)


class TestHierarchy:
    """All custom exceptions inherit from ParasiteError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [ConfigError, ModelAdapterError, JudgeError, TaskLoadError, ScoringError],
    )
    def test_direct_subclass(self, exc_cls: type[ParasiteError]) -> None:
        assert issubclass(exc_cls, ParasiteError)

    def test_self_judging_is_config_error(self) -> None:
        assert issubclass(SelfJudgingError, ConfigError)

    def test_self_judging_is_parasite_error(self) -> None:
        assert issubclass(SelfJudgingError, ParasiteError)


class TestCatchAll:
    """ParasiteError can be used as a broad catch."""

    def test_catch_config_error(self) -> None:
        with pytest.raises(ParasiteError):
            raise ConfigError("bad config")

    def test_catch_model_adapter_error(self) -> None:
        with pytest.raises(ParasiteError):
            raise ModelAdapterError("api failure")

    def test_catch_judge_error(self) -> None:
        with pytest.raises(ParasiteError):
            raise JudgeError("invalid score")

    def test_catch_task_load_error(self) -> None:
        with pytest.raises(ParasiteError):
            raise TaskLoadError("missing YAML")

    def test_catch_scoring_error(self) -> None:
        with pytest.raises(ParasiteError):
            raise ScoringError("aggregation failed")

    def test_catch_self_judging_error(self) -> None:
        with pytest.raises(ConfigError):
            raise SelfJudgingError("same provider")


class TestMessages:
    """Exception messages are preserved."""

    def test_message_preserved(self) -> None:
        exc = ModelAdapterError("Install openai")
        assert str(exc) == "Install openai"

    def test_config_error_message(self) -> None:
        exc = ConfigError("weights must sum to 1.0")
        assert "weights" in str(exc)

    def test_self_judging_message(self) -> None:
        exc = SelfJudgingError("gpt-4o cannot judge gpt-4o-mini")
        assert "gpt-4o" in str(exc)


class TestNotBuiltin:
    """Custom exceptions are distinct from built-in types."""

    def test_not_value_error(self) -> None:
        assert not issubclass(ConfigError, ValueError)

    def test_not_import_error(self) -> None:
        assert not issubclass(ModelAdapterError, ImportError)

    def test_not_file_not_found(self) -> None:
        assert not issubclass(TaskLoadError, FileNotFoundError)
