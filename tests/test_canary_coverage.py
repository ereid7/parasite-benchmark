"""Tests for canary variant coverage across anchor tests."""

from __future__ import annotations

from pathlib import Path

import yaml  # third-party

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "1.0.0"


def test_anchor_tests_have_canaries():
    """Each category's X1 anchor test should have canary_enabled: true."""
    anchor_ids = ["A1", "B1", "E1", "F1", "G1", "H1", "I1", "K1"]
    missing = []
    for anchor_id in anchor_ids:
        category = anchor_id[0]
        pattern = list((DATA_DIR / category).glob(f"{anchor_id}_*.yaml"))
        if not pattern:
            continue
        with open(pattern[0]) as f:
            data = yaml.safe_load(f)
        if not data.get("gaming_variants", {}).get("canary_enabled", False):
            missing.append(anchor_id)
    assert not missing, f"Anchor tests missing canary_enabled: {missing}"


def test_canary_variants_have_required_fields():
    """Canary variants must have type, metadata with semantic_equivalence_group."""
    for yaml_path in sorted(DATA_DIR.rglob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        for variant in data.get("variants", []):
            if variant.get("type") != "canary":
                continue
            meta = variant.get("metadata", {})
            assert meta.get("variant_type") == "canary", (
                f"{yaml_path.name}:{variant.get('id')} missing variant_type"
            )
            assert "semantic_equivalence_group" in meta, (
                f"{yaml_path.name}:{variant.get('id')} missing semantic_equivalence_group"
            )
