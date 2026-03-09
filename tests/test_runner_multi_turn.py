"""Tests for multi_turn.py — conversation building, placeholder replacement."""

from __future__ import annotations

import pytest

from parasite_benchmark.benchmark.multi_turn import (
    MODEL_RESPONSE_PLACEHOLDER,
    run_multi_turn_variant,
)


@pytest.fixture
def mock_judge(mock_adapter):
    """Create a mock Judge that returns canned JudgeResult."""
    from unittest.mock import AsyncMock, MagicMock

    from parasite_benchmark.judge.core import JudgeEvaluation, JudgeResult

    judge = MagicMock()
    judge.evaluate = AsyncMock(
        return_value=JudgeResult(
            test_id="G1",
            variant_id="G1_v1",
            evaluations=[
                JudgeEvaluation(
                    test_id="G1",
                    variant_id="G1_v1",
                    score=0.3,
                    raw_score=0.3,
                    reasoning="test",
                    evidence=[],
                    confidence=0.8,
                ),
            ],
        )
    )
    return judge


@pytest.mark.asyncio
async def test_basic_multi_turn(mock_adapter, mock_judge):
    turns = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": MODEL_RESPONSE_PLACEHOLDER},
        {"role": "user", "content": "Tell me more."},
    ]
    result = await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="Test rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    assert "response" in result
    assert "conversation" in result
    assert "score" in result


@pytest.mark.asyncio
async def test_placeholder_replaced(mock_adapter, mock_judge):
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": MODEL_RESPONSE_PLACEHOLDER},
        {"role": "user", "content": "Thanks"},
    ]
    result = await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    for msg in result["conversation"]:
        assert MODEL_RESPONSE_PLACEHOLDER not in msg["content"]


@pytest.mark.asyncio
async def test_final_user_turn(mock_adapter, mock_judge):
    """If the last turn is a user turn, the model should respond to it."""
    turns = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": MODEL_RESPONSE_PLACEHOLDER},
        {"role": "user", "content": "Question 2"},
    ]
    result = await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    # Should have user, assistant, user, assistant
    assert len(result["conversation"]) == 4
    assert result["conversation"][-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_conversation_length(mock_adapter, mock_judge):
    turns = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Fixed response"},
        {"role": "user", "content": "Turn 2"},
    ]
    result = await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    # 3 turns from the list + 1 model response for the final user turn
    assert len(result["conversation"]) == 4


@pytest.mark.asyncio
async def test_judge_called(mock_adapter, mock_judge):
    turns = [
        {"role": "user", "content": "Hello"},
    ]
    await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    mock_judge.evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_result_keys(mock_adapter, mock_judge):
    turns = [{"role": "user", "content": "Hi"}]
    result = await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    assert set(result.keys()) == {"response", "conversation", "judge_result", "score"}


@pytest.mark.asyncio
async def test_final_placeholder_does_not_double_call(mock_adapter, mock_judge):
    turns = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Context"},
        {"role": "user", "content": MODEL_RESPONSE_PLACEHOLDER},
    ]
    await run_multi_turn_variant(
        model=mock_adapter,
        turns=turns,
        judge=mock_judge,
        judge_criteria="rubric",
        test_id="G1",
        variant_id="G1_v1",
    )
    mock_adapter.complete.assert_awaited_once()
