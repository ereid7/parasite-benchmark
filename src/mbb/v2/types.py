"""Shared datatypes for PARASITE v2.1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VariantObservation:
    test_id: str
    category: str
    variant_id: str
    variant_type: str
    score: float
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    judge_scores: dict[str, float] = field(default_factory=dict)
    sequence_index: int = 0
    sequence_total: int = 0

