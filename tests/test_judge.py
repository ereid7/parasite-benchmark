"""Tests for judge/judge.py — JudgeResult, evaluate (mock), fallback."""

from __future__ import annotations

import pytest

from mbb.judge.judge import JudgeEvaluation, JudgeResult


def _eval(score: float = 0.3) -> JudgeEvaluation:
    return JudgeEvaluation(
        test_id="A1",
        variant_id="v1",
        score=score,
        reasoning="",
        evidence=[],
        confidence=0.8,
    )


class TestJudgeResult:
    def test_mean_score(self):
        evals = [_eval(0.2), _eval(0.4), _eval(0.6)]
        jr = JudgeResult(test_id="A1", variant_id="v1", evaluations=evals)
        assert abs(jr.mean_score - 0.4) < 0.001

    def test_std(self):
        evals = [_eval(0.2), _eval(0.8)]
        jr = JudgeResult(test_id="A1", variant_id="v1", evaluations=evals)
        assert jr.score_std > 0.0

    def test_empty(self):
        jr = JudgeResult(test_id="A1", variant_id="v1", evaluations=[])
        assert jr.mean_score == 0.0
        assert jr.score_std == 0.0

    def test_single(self):
        jr = JudgeResult(test_id="A1", variant_id="v1", evaluations=[_eval(0.5)])
        assert jr.mean_score == 0.5
        assert jr.score_std == 0.0

    def test_score_clamping(self):
        assert _eval(0.0).score >= 0.0
        assert _eval(1.0).score <= 1.0

    def test_ensemble_score_override(self):
        from mbb.judge.ensemble import EnsembleScore, JudgeScore

        ensemble = EnsembleScore(
            score=0.5,
            std=0.1,
            high_disagreement=False,
            judge_scores=[JudgeScore(judge_id="j1", score=0.5, reasoning="")],
            weights={"j1": 1.0},
        )
        jr = JudgeResult(
            test_id="A1",
            variant_id="v1",
            evaluations=[_eval(0.2)],
            ensemble=ensemble,
        )
        assert jr.mean_score == 0.5

    def test_to_dict_via_ensemble(self):
        from mbb.judge.ensemble import EnsembleScore, JudgeScore

        ensemble = EnsembleScore(
            score=0.4,
            std=0.05,
            high_disagreement=False,
            judge_scores=[JudgeScore(judge_id="j1", score=0.4, reasoning="ok")],
            weights={"j1": 1.0},
        )
        d = ensemble.to_dict()
        assert "judges" in d
        assert "weights" in d


@pytest.mark.asyncio
async def test_evaluate_with_mock(mock_adapter):
    """Evaluate with a mock adapter — tests the scoring pipeline."""
    import mbb.judge.judge as judge_module
    from mbb.judge.judge import Judge

    original_create = judge_module.create_adapter

    def mock_create(model_id, **kwargs):
        return mock_adapter

    judge_module.create_adapter = mock_create
    try:
        j = Judge(
            judge_model="mock-judge",
            n_runs=2,
            anonymize=False,
            length_normalize=False,
        )
        result = await j.evaluate(
            test_id="A1",
            variant_id="v1",
            category="A",
            scenario="Test scenario",
            model_response="Test response",
        )
        assert isinstance(result, JudgeResult)
        assert len(result.evaluations) == 2
        assert 0.0 <= result.mean_score <= 1.0
    finally:
        judge_module.create_adapter = original_create


@pytest.mark.asyncio
async def test_multiple_runs(mock_adapter):
    """Multiple judge runs produce multiple evaluations."""
    import mbb.judge.judge as judge_module
    from mbb.judge.judge import Judge

    original_create = judge_module.create_adapter

    def mock_create(model_id, **kwargs):
        return mock_adapter

    judge_module.create_adapter = mock_create
    try:
        j = Judge(
            judge_model="mock-judge",
            n_runs=5,
            anonymize=False,
            length_normalize=False,
        )
        result = await j.evaluate(
            test_id="A1",
            variant_id="v1",
            category="A",
            scenario="Test",
            model_response="Response",
        )
        assert len(result.evaluations) == 5
    finally:
        judge_module.create_adapter = original_create
