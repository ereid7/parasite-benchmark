"""Tests for config.py — load_config, build_config, validate."""

from __future__ import annotations

from mbb.config import BenchmarkConfig, EvaluationConfig, JudgeConfig, build_config, load_config


def test_load_config_full():
    cfg = load_config("configs/full.yaml")
    assert isinstance(cfg, dict)


def test_load_config_quick():
    cfg = load_config("configs/quick.yaml")
    assert isinstance(cfg, dict)


def test_load_config_missing():
    cfg = load_config(None)
    assert cfg == {}


def test_build_config_defaults():
    cfg = build_config()
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.judge.runs == 3


def test_validate_valid():
    cfg = build_config()
    errors = cfg.validate()
    assert errors == []


def test_validate_invalid_categories():
    cfg = build_config({"weights": {"Z": 1.0}})
    errors = cfg.validate()
    assert any("Unknown category" in e for e in errors)


def test_judge_config_defaults():
    jc = JudgeConfig()
    assert jc.model == "glm-4.7-flash"
    assert jc.runs == 3
    assert jc.temperature == 0.3


def test_evaluation_config_defaults():
    ec = EvaluationConfig()
    assert ec.temperature == 0.0
    assert ec.max_tokens == 2048
    assert ec.max_concurrent == 5


def test_benchmark_config_models():
    cfg = build_config({"judge": {"models": ["a", "b"], "weights": {"a": 0.5, "b": 0.5}}})
    assert cfg.judge.models == ["a", "b"]
    assert cfg.judge.effective_model_string == "a,b"


def test_build_config_from_yaml():
    overrides = {
        "judge": {"model": "custom-judge", "runs": 5},
        "evaluation": {"temperature": 0.5},
        "seed": 42,
    }
    cfg = build_config(overrides)
    assert cfg.judge.model == "custom-judge"
    assert cfg.judge.runs == 5
    assert cfg.evaluation.temperature == 0.5
    assert cfg.seed == 42


def test_config_categories_property():
    cfg = build_config()
    assert len(cfg.weights) == 8


def test_config_weights_property():
    cfg = build_config()
    assert abs(sum(cfg.weights.values()) - 1.0) < 0.001
