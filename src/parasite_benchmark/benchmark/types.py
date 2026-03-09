"""Shared datatypes for PARASITE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VariantObservation:
    test_id: str
    category: str
    variant_id: str
    variant_type: str
    score: float
    response: str
    response_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    judge_scores: dict[str, float] = field(default_factory=dict)
    judge_scores_raw: dict[str, float] = field(default_factory=dict)
    sequence_index: int = 0
    sequence_total: int = 0
    judge_details: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observation to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariantObservation:
        """Reconstruct an observation from JSON data."""
        return cls(
            test_id=str(data.get("test_id", "")),
            category=str(data.get("category", "")),
            variant_id=str(data.get("variant_id", "")),
            variant_type=str(data.get("variant_type", "standard")),
            score=float(data.get("score", 0.0)),
            response=str(data.get("response", "")),
            response_length=int(data.get("response_length", 0)),
            metadata=dict(data.get("metadata", {})),
            judge_scores={k: float(v) for k, v in data.get("judge_scores", {}).items()},
            judge_scores_raw={
                k: float(v) for k, v in data.get("judge_scores_raw", {}).items()
            },
            sequence_index=int(data.get("sequence_index", 0)),
            sequence_total=int(data.get("sequence_total", 0)),
            judge_details={
                str(k): dict(v) for k, v in data.get("judge_details", {}).items()
            },
        )
