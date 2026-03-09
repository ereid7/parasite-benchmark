"""Tests for judge/debiasing.py — anonymize, normalize, CoT prompt."""

from __future__ import annotations

from mbb.constants import LENGTH_NORMALIZATION_BASELINE, LENGTH_NORMALIZATION_PENALTY
from mbb.judge.debiasing import (
    _CALIBRATION_ANCHORS,
    ANCHOR_TOLERANCE,
    anonymize_response,
    build_cot_judge_prompt,
    compute_style_ratio,
    debias_score,
    normalize_score_by_length,
)


def test_anonymize_strips_model_names():
    text = "I'm ChatGPT, made by OpenAI. I can help you."
    result = anonymize_response(text)
    assert "ChatGPT" not in result
    assert "OpenAI" not in result


def test_anonymize_preserves_content():
    text = "The answer is 42. This is a factual statement."
    result = anonymize_response(text)
    assert "42" in result
    assert "factual" in result


def test_normalize_neutral():
    """Baseline-length response should have zero penalty."""
    result = normalize_score_by_length(0.5, 200, baseline_length=200)
    assert abs(result.length_penalty) < 0.001
    assert abs(result.normalized_score - 0.5) < 0.001


def test_normalize_long():
    """Longer-than-baseline should be corrected downward (reducing verbosity inflation)."""
    result = normalize_score_by_length(0.3, 800, baseline_length=200)
    assert result.length_penalty > 0.0  # penalty is positive for long responses
    assert result.normalized_score < 0.3  # but subtracted, so score decreases


def test_cot_prompt_includes_rubric():
    messages = build_cot_judge_prompt("A1", "A", "scenario", "response", "rubric text")
    combined = " ".join(m["content"] for m in messages)
    assert "rubric text" in combined


def test_cot_prompt_includes_scenario():
    messages = build_cot_judge_prompt("A1", "A", "test scenario", "response", "")
    combined = " ".join(m["content"] for m in messages)
    assert "test scenario" in combined


def test_calibration_anchors():
    anchors = [
        {
            "expected_score": 0.05,
            "classification": "mutualistic",
            "scenario": "s",
            "response": "r",
            "reasoning": "x",
        }
    ]
    messages = build_cot_judge_prompt(
        "A1",
        "A",
        "scenario",
        "response",
        "",
        calibration_examples=anchors,
    )
    combined = " ".join(m["content"] for m in messages)
    assert "Calibration" in combined


def test_style_ratio():
    response = "Great question! Let me think about that. ### Header\n- bullet point"
    ratio = compute_style_ratio(response)
    assert ratio > 0.0


def test_style_ratio_empty():
    assert compute_style_ratio("") == 0.0


def test_debias_identity():
    """With no normalization, final_score should equal raw_score."""
    result = debias_score(0.5, "short response", anonymize=False, length_normalize=False)
    assert abs(result.final_score - 0.5) < 0.001
    assert result.raw_score == 0.5


def test_length_constants_used():
    """H5: normalize_score_by_length uses constants from constants.py."""
    result = normalize_score_by_length(0.5, LENGTH_NORMALIZATION_BASELINE)
    assert abs(result.length_penalty) < 0.001
    assert LENGTH_NORMALIZATION_BASELINE == 200
    assert LENGTH_NORMALIZATION_PENALTY == 0.1


def test_calibration_anchors_defined():
    """H7: Calibration anchors exist with expected levels."""
    levels = {a["level"] for a in _CALIBRATION_ANCHORS}
    assert levels == {"low", "mid", "high"}
    assert ANCHOR_TOLERANCE == 0.10


def test_anchor_validation_mock():
    """H7: Mock judge returning expected scores → all within_tolerance."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from mbb.judge.debiasing import load_calibration_anchors, validate_calibration_anchors

    mock_judge = MagicMock()
    expected_scores = {
        str(anchor["id"]): float(anchor["expected_score"]) for anchor in load_calibration_anchors()
    }

    async def mock_evaluate(**kwargs):
        vid = kwargs.get("variant_id", "")
        jr = MagicMock()
        jr.mean_score = expected_scores.get(vid, 0.5)
        return jr

    mock_judge.evaluate = AsyncMock(side_effect=mock_evaluate)

    results = asyncio.run(validate_calibration_anchors(mock_judge, category="A"))
    per_anchor = {k: v for k, v in results.items() if not k.startswith("_")}
    assert all(r["within_tolerance"] for r in per_anchor.values())
    assert results["_summary"]["within_tolerance_rate"] == 1.0
