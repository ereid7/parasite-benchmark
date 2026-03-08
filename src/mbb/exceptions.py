"""Custom exception hierarchy for the PARASITE benchmark."""

from __future__ import annotations


class ParasiteError(Exception):
    """Base exception for all PARASITE benchmark errors."""


class ConfigError(ParasiteError):
    """Invalid configuration: weights, categories, judge setup, etc."""


class SelfJudgingError(ConfigError):
    """Same-provider judging detected — causes self-enhancement bias."""


class ModelAdapterError(ParasiteError):
    """Model adapter failure: API error, timeout, missing SDK, invalid response."""


class JudgeError(ParasiteError):
    """Judge evaluation failure: timeout, invalid score, parse error."""


class TaskLoadError(ParasiteError):
    """Task loading failure: YAML error, missing file, validation failure."""


class ScoringError(ParasiteError):
    """Scoring or aggregation computation error."""
