"""Tests for utils/checkpointing.py — save, load, roundtrip, missing/corrupt."""

from __future__ import annotations

import json

from parasite_benchmark.utils.checkpointing import load_checkpoint, save_checkpoint


def test_save_creates_file(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"model-a": {"pi": 0.3}})
    assert (out / "checkpoint.json").exists()


def test_save_load_roundtrip(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    data = {
        "checkpoint_version": 2,
        "run_id": "run1",
        "config_snapshot": {"judge_model": "judge-a"},
        "results": {"model-a": {"pi": 0.3, "classification": "Commensal"}},
        "observations_by_model": {"model-a": [{"variant_id": "A1_v1"}]},
    }
    save_checkpoint(out, data)
    loaded = load_checkpoint(tmp_path)
    assert loaded["results"]["model-a"]["pi"] == 0.3
    assert loaded["run_id"] == "run1"
    assert loaded["observations_by_model"]["model-a"][0]["variant_id"] == "A1_v1"


def test_load_missing_file(tmp_path):
    result = load_checkpoint(tmp_path)
    assert result == {}


def test_load_corrupt_file(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    (out / "checkpoint.json").write_text("NOT VALID JSON {{{")
    result = load_checkpoint(tmp_path)
    assert result == {}


def test_save_with_serialize_fn(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"m": "raw_value"}, serialize_fn=lambda x: {"serialized": x})
    loaded = json.loads((out / "checkpoint.json").read_text())
    assert loaded["results"]["m"]["serialized"] == "raw_value"


def test_save_creates_parent_dirs(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"m": {"x": 1}})
    assert (out / "checkpoint.json").exists()


def test_multiple_saves_latest(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"results": {"m": {"v": 1}}})
    save_checkpoint(out, {"results": {"m": {"v": 2}}})
    loaded = load_checkpoint(tmp_path)
    assert loaded["results"]["m"]["v"] == 2


def test_load_empty_dir(tmp_path):
    result = load_checkpoint(tmp_path)
    assert result == {}


def test_load_legacy_results_only_checkpoint(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    (out / "checkpoint.json").write_text(json.dumps({"model-a": {"pi": 0.2}}))

    loaded = load_checkpoint(tmp_path)

    assert loaded["results"]["model-a"]["pi"] == 0.2
    assert loaded["run_id"] == "run1"
