"""Tests for judge/ensemble.py — weights, aggregate, self-enhancement."""

from __future__ import annotations

import logging

import pytest

from parasite_benchmark.exceptions import ConfigError
from parasite_benchmark.judge.ensemble import (
    JudgeScore,
    aggregate_ensemble,
    check_self_enhancement,
    compute_equal_weights,
    cyclic_judge_assignment,
    parse_weights,
)


def test_equal_weights():
    w = compute_equal_weights(["a", "b", "c"])
    assert abs(sum(w.values()) - 1.0) < 0.001
    assert all(abs(v - 1 / 3) < 0.001 for v in w.values())


def test_parse_weights_none():
    w = parse_weights(["a", "b"], None)
    assert abs(sum(w.values()) - 1.0) < 0.001


def test_parse_weights_valid():
    w = parse_weights(["a", "b"], [0.6, 0.4])
    assert w["a"] == 0.6
    assert w["b"] == 0.4


def test_parse_weights_wrong_count():
    with pytest.raises(ConfigError, match="Number of weights"):
        parse_weights(["a", "b"], [0.5])


def test_parse_weights_wrong_sum():
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        parse_weights(["a", "b"], [0.5, 0.6])


def test_aggregate_basic(sample_judge_scores):
    weights = {"judge-a": 1 / 3, "judge-b": 1 / 3, "judge-c": 1 / 3}
    result = aggregate_ensemble(sample_judge_scores, weights)
    assert 0.0 <= result.score <= 1.0
    assert not result.high_disagreement


def test_aggregate_empty():
    result = aggregate_ensemble([], {"a": 1.0})
    assert result.score == 0.0
    assert result.std == 0.0


def test_aggregate_high_disagreement():
    scores = [
        JudgeScore(judge_id="a", score=0.1, reasoning="low"),
        JudgeScore(judge_id="b", score=0.9, reasoning="high"),
    ]
    weights = {"a": 0.5, "b": 0.5}
    result = aggregate_ensemble(scores, weights)
    assert result.high_disagreement


def test_check_self_enhancement_warning(caplog):
    with caplog.at_level(logging.WARNING):
        check_self_enhancement(["gpt-4o", "claude-sonnet-4"], "gpt-4-turbo")
    assert "Self-enhancement bias" in caplog.text


def test_cyclic_judge_assignment():
    judges = ["a", "b", "c"]
    assignments = [cyclic_judge_assignment(judges, i) for i in range(6)]
    assert assignments == ["a", "b", "c", "a", "b", "c"]
