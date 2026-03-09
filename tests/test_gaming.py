"""Tests for benchmark/gaming.py — detect_gaming_vectors, individual vectors."""

from __future__ import annotations

from parasite_benchmark.benchmark.gaming import (
    detect_gaming_vectors,
    intermittent_reinforcement_score,
    trauma_bonding_coefficient,
)


def test_no_flags_on_clean_data(make_observation):
    """Clean observations with uniform scores should not flag any gaming vectors."""
    observations = [
        make_observation(
            test_id="A1",
            variant_id=f"A1_v{i}",
            score=0.3,
            variant_type="standard",
            response="A clean response.",
            metadata={"user_tone": "neutral"},
        )
        for i in range(5)
    ]
    result = detect_gaming_vectors(observations, "gpt-4o")
    assert result["gaming_susceptibility"] < 0.3


def test_canary_detection(make_observation):
    """Canary variants with very different scores should flag canary_disagreement."""
    observations = [
        make_observation(test_id="A1", variant_id="A1_v1", variant_type="standard", score=0.2),
        make_observation(test_id="A1", variant_id="A1_v2", variant_type="standard", score=0.2),
        make_observation(test_id="A1", variant_id="A1_canary", variant_type="canary", score=0.8),
    ]
    result = detect_gaming_vectors(observations, "gpt-4o")
    assert result["vectors"]["canary_disagreement"]["score"] > 0.0


def test_position_effect(make_observation):
    """Different position buckets with different scores should flag position effect."""
    observations = [
        make_observation(
            variant_id=f"v{i}",
            score=0.2 if i < 3 else 0.8,
            metadata={"position_bucket": "first" if i < 3 else "last"},
        )
        for i in range(6)
    ]
    result = detect_gaming_vectors(observations, "gpt-4o")
    assert result["vectors"]["position_effect"]["score"] > 0.0


def test_self_enhancement(make_observation):
    """Own-family judges scoring lower should flag self_enhancement."""
    observations = [
        make_observation(
            variant_id=f"v{i}",
            score=0.3,
            judge_scores={"gpt-4o": 0.1, "claude-sonnet-4": 0.5},
        )
        for i in range(5)
    ]
    result = detect_gaming_vectors(observations, "gpt-4o")
    assert result["vectors"]["self_enhancement"]["score"] > 0.0


def test_tone_matching(make_observation):
    """Strong tone matching should produce a high tone score."""
    observations = [
        make_observation(
            variant_id=f"v{i}",
            score=0.3,
            metadata={"user_tone": "friendly"},
            response="Great! Excellent! Happy to help! Awesome!",
        )
        for i in range(5)
    ]
    result = detect_gaming_vectors(observations, "gpt-4o")
    assert "tone_matching" in result["vectors"]


def test_empty_observations():
    result = detect_gaming_vectors([], "gpt-4o")
    assert result["gaming_susceptibility"] == 0.0
    assert result["gaming_penalty_multiplier"] == 1.0


def test_summary_structure(make_observation):
    observations = [make_observation() for _ in range(3)]
    result = detect_gaming_vectors(observations, "gpt-4o")
    assert "vectors" in result
    assert "gaming_susceptibility" in result
    assert "gaming_penalty_multiplier" in result


def test_intermittent_reinforcement_short():
    result = intermittent_reinforcement_score([0.5])
    assert result["score"] == 0.0


def test_trauma_bonding_short(make_observation):
    observations = [make_observation(sequence_index=0), make_observation(sequence_index=1)]
    result = trauma_bonding_coefficient(observations)
    assert result["coefficient"] == 0.0


def test_multiple_flags(make_observation):
    """Multiple gaming vectors can flag simultaneously."""
    observations = [
        make_observation(
            test_id="A1",
            variant_id=f"A1_v{i}",
            variant_type="standard",
            score=0.2,
            metadata={"position_bucket": "first", "framing": "neutral"},
        )
        for i in range(3)
    ] + [
        make_observation(
            test_id="A1",
            variant_id="A1_canary",
            variant_type="canary",
            score=0.9,
            metadata={"position_bucket": "last", "framing": "urgent"},
        ),
    ]
    result = detect_gaming_vectors(observations, "gpt-4o")
    flagged = [k for k, v in result["vectors"].items() if v.get("flag")]
    # At least canary should be flagged
    assert len(flagged) >= 1
